import tempfile
import unittest

try:
    import torch
    from transformers import GPT2Config

    from scripts.custom_ppo_notagen import (
        PatchValueHead,
        RewardEvent,
        _dtw_metric_reward_events,
        _project_reward_events_to_patches,
        _stream_line_end_patch_indices,
        _stream_line_spans,
        batch_trajectory_returns_advantages,
        discounted_returns,
        generalized_advantage_estimates,
        load_value_head_checkpoint,
        normalize_advantages,
        ppo_clipped_loss,
        save_value_head_checkpoint,
        terminal_returns,
        trajectory_patch_hidden_states,
        trajectory_patch_logprobs_values,
        trajectory_patch_values,
        value_mse_loss,
        value_prediction_metrics,
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

    def test_value_only_replay_returns_one_value_per_aligned_patch(self):
        torch.manual_seed(0)
        model = _tiny_notagen()
        value_head = PatchValueHead(32, value_hidden_size=16)
        prompt_ids = [3 + (i % 80) for i in range(PATCH_SIZE * 4)]
        generated_patches = [
            [11 + ((patch_idx * 17 + i) % 50) for i in range(PATCH_SIZE)]
            for patch_idx in range(3)
        ]

        values = trajectory_patch_values(
            model,
            value_head,
            prompt_ids,
            generated_patches,
            precision="fp32",
            replay_context_patches=4,
            target_chunk_patches=2,
        )

        self.assertEqual(values.shape, (3,))
        self.assertTrue(torch.isfinite(values).all())

    def test_hidden_state_replay_matches_value_replay(self):
        torch.manual_seed(0)
        model = _tiny_notagen()
        value_head = PatchValueHead(32, value_hidden_size=16)
        prompt_ids = [3 + (i % 80) for i in range(PATCH_SIZE * 4 + 5)]
        generated_patches = [
            [11 + ((patch_idx * 17 + i) % 50) for i in range(PATCH_SIZE)]
            for patch_idx in range(3)
        ]

        hidden_states = trajectory_patch_hidden_states(
            model,
            prompt_ids,
            generated_patches,
            precision="fp32",
            replay_context_patches=4,
            target_chunk_patches=2,
        )
        values = trajectory_patch_values(
            model,
            value_head,
            prompt_ids,
            generated_patches,
            precision="fp32",
            replay_context_patches=4,
            target_chunk_patches=2,
        )

        self.assertEqual(hidden_states.shape, (3, 32))
        self.assertTrue(torch.allclose(value_head(hidden_states), values))

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
        value_targets = terminal_returns(1.5, 3, gamma=1.0, device=torch.device("cpu"))
        advantages = value_targets - old_values

        payload = ppo_clipped_loss(
            new_logprobs=new_logprobs,
            old_logprobs=old_logprobs,
            values=values,
            old_values=old_values,
            advantages=advantages,
            value_targets=value_targets,
            clip_range=0.2,
            value_loss_coef=0.5,
        )

        self.assertTrue(torch.isfinite(payload.loss))
        payload.loss.backward()
        self.assertIsNotNone(new_logprobs.grad)
        self.assertIsNotNone(values.grad)

    def test_ppo_microbatch_loss_matches_full_batch_normalization(self):
        old_logprobs = torch.tensor([-4.0, -3.0, -2.0, -2.5, -3.5])
        new_logprobs = torch.tensor([-3.9, -3.2, -2.1, -2.7, -3.4], requires_grad=True)
        old_values = torch.tensor([0.2, 0.1, -0.1, 0.0, 0.5])
        values = torch.tensor([0.3, 0.0, -0.2, 0.1, 0.4], requires_grad=True)
        value_targets = torch.tensor([1.5, 0.7, 0.2, -0.1, 0.4])
        advantages = value_targets - old_values

        full = ppo_clipped_loss(
            new_logprobs=new_logprobs,
            old_logprobs=old_logprobs,
            values=values,
            old_values=old_values,
            advantages=advantages,
            value_targets=value_targets,
            clip_range=0.2,
            value_loss_coef=0.5,
            normalize_value_loss=True,
            value_loss_scale_min=1.0,
        )

        normalized_advantages, adv_mean, adv_std = normalize_advantages(advantages)
        value_loss_scale = torch.clamp(value_targets.std(unbiased=False), min=1.0)
        weighted = {}
        for start, end in [(0, 2), (2, 5)]:
            payload = ppo_clipped_loss(
                new_logprobs=new_logprobs[start:end],
                old_logprobs=old_logprobs[start:end],
                values=values[start:end],
                old_values=old_values[start:end],
                advantages=advantages[start:end],
                value_targets=value_targets[start:end],
                clip_range=0.2,
                value_loss_coef=0.5,
                normalize_advantage=False,
                normalize_value_loss=True,
                value_loss_scale_min=1.0,
                normalized_advantages=normalized_advantages[start:end],
                advantages_mean=adv_mean,
                advantages_std=adv_std,
                fixed_value_loss_scale=value_loss_scale,
            )
            weight = (end - start) / len(advantages)
            for name in ("loss", "policy_loss", "value_loss", "raw_value_loss", "approx_kl", "clip_fraction"):
                weighted[name] = weighted.get(name, torch.zeros(())) + getattr(payload, name).detach() * weight

        for name in weighted:
            self.assertTrue(torch.allclose(weighted[name], getattr(full, name).detach(), atol=1e-6), name)

    def test_value_mse_loss_normalization_keeps_raw_loss_visible(self):
        values = torch.tensor([0.0, 0.0])
        value_targets = torch.tensor([0.0, 4.0])

        scaled_loss, raw_loss, scale = value_mse_loss(
            values,
            value_targets,
            normalize_value_loss=True,
        )

        self.assertAlmostEqual(float(raw_loss), 8.0)
        self.assertAlmostEqual(float(scale), 2.0)
        self.assertAlmostEqual(float(scaled_loss), 2.0)

    def test_value_mse_loss_scale_min_clamps_tiny_target_variance(self):
        values = torch.tensor([0.0, 0.0])
        value_targets = torch.tensor([0.0, 0.2])

        scaled_loss, raw_loss, scale = value_mse_loss(
            values,
            value_targets,
            normalize_value_loss=True,
            scale_min=1.0,
        )

        self.assertAlmostEqual(float(scale), 1.0)
        self.assertAlmostEqual(float(scaled_loss), float(raw_loss))

    def test_value_prediction_metrics_reports_correlation_and_explained_variance(self):
        values = torch.tensor([0.0, 1.0, 2.0])
        targets = torch.tensor([0.0, 1.0, 2.0])

        metrics = value_prediction_metrics(values, targets)

        self.assertEqual(metrics["count"], 3)
        self.assertAlmostEqual(metrics["mse"], 0.0)
        self.assertAlmostEqual(metrics["mae"], 0.0)
        self.assertAlmostEqual(metrics["explained_variance"], 1.0)
        self.assertAlmostEqual(metrics["correlation"], 1.0)

    def test_value_head_checkpoint_roundtrip(self):
        torch.manual_seed(0)
        value_head = PatchValueHead(32, value_hidden_size=16)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/value_head.pt"
            save_value_head_checkpoint(value_head, path)
            loaded = PatchValueHead(32, value_hidden_size=16)
            meta = load_value_head_checkpoint(loaded, path, torch.device("cpu"))

        self.assertEqual(meta["config"]["hidden_size"], 32)
        self.assertEqual(meta["config"]["value_hidden_size"], 16)
        for original, restored in zip(value_head.parameters(), loaded.parameters(), strict=True):
            self.assertTrue(torch.allclose(original, restored))

    def test_discounted_returns_accumulate_patch_rewards(self):
        rewards = torch.tensor([1.0, 2.0, 3.0])
        returns = discounted_returns(rewards, gamma=0.5)
        self.assertTrue(torch.allclose(returns, torch.tensor([2.75, 3.5, 3.0])))

    def test_gae_lambda_one_matches_discounted_returns(self):
        rewards = torch.tensor([1.0, 2.0, 3.0])
        values = torch.tensor([0.2, 0.4, 0.6])

        advantages, value_targets = generalized_advantage_estimates(
            rewards,
            values,
            gamma=0.5,
            gae_lambda=1.0,
        )

        returns = discounted_returns(rewards, gamma=0.5)
        self.assertTrue(torch.allclose(value_targets, returns))
        self.assertTrue(torch.allclose(advantages, returns - values))

    def test_gae_lambda_zero_uses_one_step_td_errors(self):
        rewards = torch.tensor([1.0, 2.0, 3.0])
        values = torch.tensor([0.2, 0.4, 0.6])

        advantages, value_targets = generalized_advantage_estimates(
            rewards,
            values,
            gamma=0.5,
            gae_lambda=0.0,
        )

        expected_advantages = torch.tensor([
            1.0 + 0.5 * 0.4 - 0.2,
            2.0 + 0.5 * 0.6 - 0.4,
            3.0 - 0.6,
        ])
        self.assertTrue(torch.allclose(advantages, expected_advantages))
        self.assertTrue(torch.allclose(value_targets, expected_advantages + values))

    def test_batched_trajectory_returns_reset_at_boundaries(self):
        reward_tensors = [torch.tensor([1.0, 2.0]), torch.tensor([10.0])]
        value_tensors = [torch.zeros(2), torch.zeros(1)]

        payload = batch_trajectory_returns_advantages(
            reward_tensors=reward_tensors,
            value_tensors=value_tensors,
            gamma=1.0,
            gae_lambda=1.0,
        )

        self.assertTrue(torch.allclose(payload.patch_rewards, torch.tensor([1.0, 2.0, 10.0])))
        self.assertTrue(torch.allclose(payload.returns, torch.tensor([3.0, 2.0, 10.0])))
        self.assertTrue(torch.allclose(payload.advantages, payload.returns))
        self.assertTrue(torch.allclose(payload.value_targets, payload.returns))

    def test_stream_line_end_patch_indices_maps_line_boundaries(self):
        completion = "[r:0/1][V:1]abc|\n[r:1/0][V:1]def|\n"
        patch_texts = ["[r:0/1][V:1]", "abc|\n[r:1", "/0][V:1]def|\n"]
        self.assertEqual(_stream_line_end_patch_indices(completion, patch_texts), [1, 2])

    def test_stream_line_spans_follow_countdown_markers(self):
        completion = "[r:0/1][V:1]abc|[r:1/0][V:1]def|"
        self.assertEqual(_stream_line_spans(completion), [(0, 16), (16, len(completion))])

    def test_reward_events_are_distributed_by_patch_overlap(self):
        patch_texts = ["abcdefghij", "klmnopqrst", "uvwxyz"]
        events = [RewardEvent(start=5, end=25, value=2.0, name="line")]

        rewards = _project_reward_events_to_patches(events, patch_texts)

        self.assertEqual(len(rewards), 3)
        self.assertAlmostEqual(sum(rewards), 2.0)
        self.assertAlmostEqual(rewards[0], 0.5)
        self.assertAlmostEqual(rewards[1], 1.0)
        self.assertAlmostEqual(rewards[2], 0.5)

    def test_dtw_metric_reward_events_sum_to_metric_value(self):
        events = _dtw_metric_reward_events(
            name="root_dtw",
            reference=[0, 7, 2],
            candidate=[0, 2],
            candidate_spans=[(0, 10), (10, 20)],
            similarity_fn=lambda left, right: 1.0 if left == right else 0.0,
            total_value=0.9,
            band_ratio=1.0,
        )

        self.assertGreater(len(events), 0)
        self.assertAlmostEqual(sum(event.value for event in events), 0.9)
        self.assertTrue(all(event.name == "root_dtw" for event in events))


if __name__ == "__main__":
    unittest.main()
