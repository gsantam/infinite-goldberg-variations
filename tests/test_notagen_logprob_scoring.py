import unittest


try:
    import torch
    from transformers import GPT2Config

    from scripts.custom_grpo_notagen import (
        PATCH_SIZE,
        batched_trajectory_logprobs,
        patch_logprobs,
        trajectory_logprob_chunks,
    )
    from utils import NotaGenLMHeadModel
except ModuleNotFoundError as exc:
    torch = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


def _tiny_notagen():
    patch_config = GPT2Config(
        num_hidden_layers=2,
        max_length=64,
        max_position_embeddings=64,
        n_embd=32,
        num_attention_heads=4,
        vocab_size=1,
    )
    byte_config = GPT2Config(
        num_hidden_layers=1,
        max_length=PATCH_SIZE + 1,
        max_position_embeddings=PATCH_SIZE + 1,
        n_embd=32,
        num_attention_heads=4,
        vocab_size=128,
    )
    model = NotaGenLMHeadModel(encoder_config=patch_config, decoder_config=byte_config)
    model.eval()
    return model


def _chunked_logprobs(model, prompt_ids, generated_patches, chunk_patches):
    return torch.cat(
        list(
            trajectory_logprob_chunks(
                model,
                prompt_ids,
                generated_patches,
                precision="fp32",
                replay_context_patches=0,
                target_chunk_patches=chunk_patches,
            )
        )
    )


def _sequential_logprobs(model, prompt_ids, generated_patches, chunk_patches, replay_context_patches=0):
    chunks = list(
        trajectory_logprob_chunks(
            model,
            prompt_ids,
            generated_patches,
            precision="fp32",
            replay_context_patches=replay_context_patches,
            target_chunk_patches=chunk_patches,
        )
    )
    return torch.cat(chunks) if chunks else torch.empty(0)


@unittest.skipIf(torch is None, f"NotaGen torch dependencies unavailable: {IMPORT_ERROR}")
class NotaGenLogprobScoringTests(unittest.TestCase):
    def assert_chunked_matches_tokenwise(self, prompt_ids, generated_patches):
        torch.manual_seed(0)
        model = _tiny_notagen()
        with torch.no_grad():
            tokenwise = torch.stack(patch_logprobs(model, prompt_ids, generated_patches, "fp32"))
            for chunk_patches in (0, 1, 2, 4):
                chunked = _chunked_logprobs(model, prompt_ids, generated_patches, chunk_patches)
                self.assertEqual(tokenwise.shape, chunked.shape)
                self.assertTrue(
                    torch.allclose(tokenwise, chunked, atol=1e-5, rtol=1e-5),
                    msg=f"chunk_patches={chunk_patches} max_abs={(tokenwise - chunked).abs().max().item()}",
                )

    def test_chunked_logprobs_match_tokenwise_for_aligned_prompt(self):
        prompt_ids = [3 + (i % 80) for i in range(PATCH_SIZE * 3)]
        generated_patches = [
            [11 + ((patch_idx * 17 + i) % 50) for i in range(PATCH_SIZE)]
            for patch_idx in range(4)
        ]
        self.assert_chunked_matches_tokenwise(prompt_ids, generated_patches)

    def test_chunked_logprobs_match_tokenwise_for_unaligned_prompt(self):
        prompt_ids = [3 + (i % 80) for i in range(PATCH_SIZE * 3 + 7)]
        generated_patches = [
            [11 + (i % 50) for i in range(PATCH_SIZE - 7)],
            [23 + (i % 50) for i in range(PATCH_SIZE)],
            [37 + (i % 50) for i in range(PATCH_SIZE)],
        ]
        self.assert_chunked_matches_tokenwise(prompt_ids, generated_patches)

    def assert_batched_matches_sequential(self, prompt_ids, generated_batch, chunk_patches, replay_context_patches=0):
        torch.manual_seed(0)
        model = _tiny_notagen()
        with torch.no_grad():
            expected = [
                _sequential_logprobs(
                    model,
                    prompt_ids,
                    generated_patches,
                    chunk_patches,
                    replay_context_patches=replay_context_patches,
                )
                for generated_patches in generated_batch
            ]
            actual = batched_trajectory_logprobs(
                model,
                prompt_ids,
                generated_batch,
                precision="fp32",
                replay_context_patches=replay_context_patches,
                target_chunk_patches=chunk_patches,
            )
        self.assertEqual(len(expected), len(actual))
        for sample_idx, (exp, got) in enumerate(zip(expected, actual, strict=True)):
            self.assertEqual(exp.shape, got.shape, msg=f"sample_idx={sample_idx}")
            self.assertTrue(
                torch.allclose(exp, got, atol=1e-5, rtol=1e-5),
                msg=f"sample_idx={sample_idx} chunk_patches={chunk_patches} max_abs={(exp - got).abs().max().item()}",
            )

    def test_batched_logprobs_match_sequential_for_variable_aligned_trajectories(self):
        prompt_ids = [3 + (i % 80) for i in range(PATCH_SIZE * 4)]
        generated_batch = [
            [[11 + ((patch_idx * 17 + i) % 50) for i in range(PATCH_SIZE)] for patch_idx in range(1)],
            [[19 + ((patch_idx * 13 + i) % 60) for i in range(PATCH_SIZE)] for patch_idx in range(3)],
            [[23 + ((patch_idx * 7 + i) % 40) for i in range(PATCH_SIZE)] for patch_idx in range(5)],
        ]
        for chunk_patches in (0, 1, 2, 4):
            self.assert_batched_matches_sequential(prompt_ids, generated_batch, chunk_patches)

    def test_batched_logprobs_match_sequential_for_unaligned_prompt_prefix(self):
        prompt_ids = [3 + (i % 80) for i in range(PATCH_SIZE * 3 + 5)]
        first_patch_len = PATCH_SIZE - 5
        generated_batch = [
            [[11 + (i % 50) for i in range(first_patch_len)]]
            + [[23 + ((patch_idx * 11 + i) % 50) for i in range(PATCH_SIZE)] for patch_idx in range(2)],
            [[31 + (i % 40) for i in range(first_patch_len)]]
            + [[41 + ((patch_idx * 9 + i) % 40) for i in range(PATCH_SIZE)] for patch_idx in range(4)],
        ]
        for chunk_patches in (0, 1, 3):
            self.assert_batched_matches_sequential(prompt_ids, generated_batch, chunk_patches)

    def test_batched_logprobs_match_sequential_with_replay_context(self):
        prompt_ids = [3 + (i % 80) for i in range(PATCH_SIZE * 6)]
        generated_batch = [
            [[11 + ((patch_idx * 17 + i) % 50) for i in range(PATCH_SIZE)] for patch_idx in range(6)],
            [[19 + ((patch_idx * 13 + i) % 60) for i in range(PATCH_SIZE)] for patch_idx in range(9)],
        ]
        for chunk_patches in (2, 4):
            self.assert_batched_matches_sequential(
                prompt_ids,
                generated_batch,
                chunk_patches,
                replay_context_patches=5,
            )


if __name__ == "__main__":
    unittest.main()
