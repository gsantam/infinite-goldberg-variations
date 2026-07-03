import unittest


try:
    import numpy as np
    import torch
    from transformers import GPT2Config

    from grpo.notagen_cached_generation import (
        PATCH_SIZE,
        CachedNotaGenPatchGenerator,
        normalize_patch_for_context,
    )
    from grpo.notagen_cached_generation_batch import _accept_patches_batch, _BatchContext
    from grpo.notagen_wrapper import NotaGenLMHeadModel
except ModuleNotFoundError as exc:
    np = None
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


def _full_prefix_last_hidden(model, flat_ids):
    rem = len(flat_ids) % PATCH_SIZE
    prefix_ids = flat_ids[:-rem] if rem else flat_ids
    prefix_tensor = torch.tensor(prefix_ids, dtype=torch.long).reshape(1, -1, PATCH_SIZE)
    with torch.inference_mode():
        return model.patch_level_decoder(prefix_tensor)["last_hidden_state"][0, -1]


@unittest.skipIf(torch is None, f"NotaGen torch dependencies unavailable: {IMPORT_ERROR}")
class CachedNotaGenPatchGeneratorTests(unittest.TestCase):
    def test_cached_patch_hidden_matches_full_prefix_replay(self):
        torch.manual_seed(0)
        model = _tiny_notagen()
        flat_ids = [3 + (i % 80) for i in range(PATCH_SIZE * 3 + 5)]
        generator = CachedNotaGenPatchGenerator(model)

        state = generator.reset(flat_ids)
        self.assertTrue(torch.allclose(state.last_patch_hidden, _full_prefix_last_hidden(model, flat_ids), atol=1e-5))

        first_patch = [11 + (i % 50) for i in range(PATCH_SIZE - 5)]
        state = generator.accept_patch(first_patch)
        flat_ids = flat_ids + normalize_patch_for_context(
            first_patch,
            eos_token_id=model.eos_token_id,
            special_token_id=model.special_token_id,
        )
        self.assertTrue(torch.allclose(state.last_patch_hidden, _full_prefix_last_hidden(model, flat_ids), atol=1e-5))

        second_patch = [31 + (i % 40) for i in range(PATCH_SIZE)]
        state = generator.accept_patch(second_patch)
        flat_ids = flat_ids + normalize_patch_for_context(
            second_patch,
            eos_token_id=model.eos_token_id,
            special_token_id=model.special_token_id,
        )
        self.assertTrue(torch.allclose(state.last_patch_hidden, _full_prefix_last_hidden(model, flat_ids), atol=1e-5))

    def test_cached_sampling_matches_uncached_generate_for_same_seed(self):
        torch.manual_seed(1)
        model = _tiny_notagen()
        flat_ids = [3 + (i % 90) for i in range(PATCH_SIZE * 2 + 7)]

        uncached_input = torch.tensor([flat_ids], dtype=torch.long).reshape(1, -1)
        np.random.seed(123)
        uncached = model.generate(uncached_input.unsqueeze(0), top_k=0, top_p=1.0, temperature=1.0)

        generator = CachedNotaGenPatchGenerator(model)
        generator.reset(flat_ids)
        np.random.seed(123)
        cached = generator.generate_patch(top_k=0, top_p=1.0, temperature=1.0)

        self.assertEqual(cached, uncached)
        self.assertEqual(len(cached), PATCH_SIZE - 7)

    def test_batched_accept_matches_sequential_accept(self):
        torch.manual_seed(2)
        model = _tiny_notagen()
        flat_ids = [3 + (i % 80) for i in range(PATCH_SIZE * 3 + 5)]
        patches = [[11 + ((row * 7 + i) % 50) for i in range(PATCH_SIZE - 5)] for row in range(3)]

        sequential_generators = []
        batched_contexts = []
        for seed in range(3):
            sequential = CachedNotaGenPatchGenerator(model)
            sequential.reset(flat_ids)
            sequential_generators.append(sequential)

            batched = CachedNotaGenPatchGenerator(model)
            batched.reset(flat_ids)
            batched_contexts.append(
                _BatchContext(
                    generator=batched,
                    rng=np.random.default_rng(seed),
                    prompt_stream_lines=1,
                    target_total_stream_lines=32,
                    byte_list=[],
                    generated_patches=[],
                    start_time=0.0,
                    cut_index=None,
                    resets=1,
                )
            )

        for generator, patch in zip(sequential_generators, patches, strict=True):
            generator.accept_patch(patch)
        _accept_patches_batch(list(zip(batched_contexts, patches, strict=True)), precision="fp32")

        for sequential, batched in zip(sequential_generators, batched_contexts, strict=True):
            self.assertEqual(sequential.state.flat_ids, batched.generator.state.flat_ids)
            self.assertEqual(sequential.state.cached_patch_count, batched.generator.state.cached_patch_count)
            self.assertEqual(sequential.state.partial_ids, batched.generator.state.partial_ids)
            self.assertTrue(
                torch.allclose(
                    sequential.state.last_patch_hidden,
                    batched.generator.state.last_patch_hidden,
                    atol=1e-5,
                )
            )


if __name__ == "__main__":
    unittest.main()
