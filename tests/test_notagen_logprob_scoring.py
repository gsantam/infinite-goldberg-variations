import unittest


try:
    import torch
    from transformers import GPT2Config

    from scripts.custom_grpo_notagen import PATCH_SIZE, patch_logprobs, trajectory_logprob_chunks
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


if __name__ == "__main__":
    unittest.main()
