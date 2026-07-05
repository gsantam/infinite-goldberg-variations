import unittest

try:
    import torch
    from transformers import GPT2Config

    from scripts.custom_ppo_notagen import (
        PatchValueHead,
        discounted_returns,
        ppo_clipped_loss,
        terminal_returns,
        trajectory_patch_logprobs_values,
    )
    from scripts.custom_grpo_notagen import PATCH_SIZE
    from utils import NotaGenLMHeadModel
except ModuleNotFoundError as exc:
    torch = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


def _tiny_notagen():
    patch_config = GPT2Config(
        num_hidden_layers=1,
        max_length=32,
        max_position_embeddings=32,
        n_embd=32,
        num_attention_heads=4,
        vocab_size=1,
    )
    byte_config = GPT2Config(
        num_hidden_layers=1,
        max_length=PATCH_SIZE + 1,
        max_position_embeddings=PATCH_SIZE + 1,
        hidden_size=32,
        num_attention_heads=4,
        vocab_size=128,
    )
    model = NotaGenLMHeadModel(encoder_config=patch_config, decoder_config=byte_config)
    model.eval()
    return model


@unittest.skipIf(torch is None, f"NotaGen torch dependencies unavailable: {IMPORT_ERROR}")
class NotaGenPPOTests(unittest.TestCase):
    def test_patch_replay_returns_one_logprob_and_value_per_aligned_patch(self):
        torch.manual_seed(0)
        model = _tiny_notagen()
        value_head = PatchValueHead(32)
        prompt_ids = [3 + (i % 80) for i in range(PATCH_SIZE * 4)]
        generated_patches = [
            [11 + ((patch_idx * 17 + i) % 50) for i in range(PATCH_SIZE)]
            for patch_idx in range(3)
        ]

        replay = trajectory_patch_logprobs_values(
            model,
            value_head,
            prompt_ids,
            generated_patches,
            precision="fp32",
            replay_context_patches=4,
            target_chunk_patches=2,
        )

        self.assertEqual(replay.logprobs.shape, (3,))
        self.assertEqual(replay.values.shape, (3,))
        self.assertTrue(torch.isfinite(replay.logprobs).all())
        self.assertTrue(torch.isfinite(replay.values).all())

    def test_patch_replay_handles_unaligned_prompt_prefix(self):
        torch.manual_seed(0)
        model = _tiny_notagen()
        value_head = PatchValueHead(32)
        prompt_ids = [3 + (i % 80) for i in range(PATCH_SIZE * 3 + 5)]
        first_patch_len = PATCH_SIZE - 5
        generated_patches = [
            [11 + (i % 50) for i in range(first_patch_len)],
            [23 + (i % 50) for i in range(PATCH_SIZE)],
            [31 + (i % 40) for i in range(PATCH_SIZE)],
        ]

        replay = trajectory_patch_logprobs_values(
            model,
            value_head,
            prompt_ids,
            generated_patches,
            precision="fp32",
            replay_context_patches=4,
            target_chunk_patches=1,
        )

        self.assertEqual(replay.logprobs.shape, (3,))
        self.assertEqual(replay.values.shape, (3,))
        self.assertTrue(torch.isfinite(replay.logprobs).all())
        self.assertTrue(torch.isfinite(replay.values).all())

    def test_ppo_clipped_loss_is_finite(self):
        old_logprobs = torch.tensor([-4.0, -3.0, -2.0])
        new_logprobs = torch.tensor([-3.9, -3.2, -2.1], requires_grad=True)
        old_values = torch.tensor([0.2, 0.1, -0.1])
        values = torch.tensor([0.3, 0.0, -0.2], requires_grad=True)
        returns = terminal_returns(1.5, 3, gamma=1.0, device=torch.device("cpu"))

        payload = ppo_clipped_loss(
            new_logprobs=new_logprobs,
            old_logprobs=old_logprobs,
            values=values,
            old_values=old_values,
            returns=returns,
            clip_range=0.2,
            value_loss_coef=0.5,
        )

        self.assertTrue(torch.isfinite(payload.loss))
        payload.loss.backward()
        self.assertIsNotNone(new_logprobs.grad)
        self.assertIsNotNone(values.grad)

    def test_discounted_returns_accumulate_patch_rewards(self):
        rewards = torch.tensor([1.0, 2.0, 3.0])
        returns = discounted_returns(rewards, gamma=0.5)
        self.assertTrue(torch.allclose(returns, torch.tensor([2.75, 3.5, 3.0])))


if __name__ == "__main__":
    unittest.main()
