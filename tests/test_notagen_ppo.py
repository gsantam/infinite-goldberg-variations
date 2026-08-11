import json
import multiprocessing as mp
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

try:
    import torch
    from transformers import GPT2Config

    from rewards.strict_similarity import (
        STRICT_SYMBOLIC_COMPONENT_WEIGHTS,
        STRICT_SYMBOLIC_COMPONENT_Z_KEY,
        strict_similarity_global_base_z_scores,
        strict_symbolic_similarity,
    )
    from scripts.custom_ppo_notagen import (
        PatchRewardTrace,
        PatchValueHead,
        PPORewardScoringOptions,
        PPORolloutPayload,
        PromptStructuralTarget,
        RewardEvent,
        RewardScore,
        _dtw_metric_reward_events,
        _score_total_reward_from_structural_breakdown,
        _strict_symbolic_reward_events,
        _terminal_similarity_component_rewards,
        _project_reward_events_to_patches,
        _stream_line_end_patch_indices,
        _stream_line_spans,
        build_fixed_eval_prompt_batch,
        build_prompt_batch_for_repeated_slot,
        build_prompt_batch_for_step,
        build_prompt_batch_for_slots,
        batched_trajectory_patch_logprobs_values,
        batched_trajectory_patch_hidden_states,
        batched_trajectory_token_log_dists,
        batched_trajectory_patch_values,
        batch_trajectory_returns_advantages,
        discounted_returns,
        exact_categorical_kl,
        fixed_eval_event_index_after_step,
        fixed_eval_event_index_before_training,
        fixed_eval_should_run_after_step,
        filter_prompts_by_bar_target,
        generalized_advantage_estimates,
        load_prompt_structural_targets,
        load_value_head_checkpoint,
        normalize_advantages,
        normalize_advantages_token_weighted,
        prompt_cycle_order,
        patch_rewards_single_pass,
        patch_rewards_simple_test,
        patch_rewards_terminal,
        ppo_clipped_loss,
        save_value_head_checkpoint,
        sample_ppo_rollouts,
        score_ppo_rollout_payloads,
        score_ppo_rollout_payloads_from_payload_context,
        select_prompt_for_update,
        terminal_returns,
        token_patch_indices_from_counts,
        trajectory_patch_hidden_states,
        trajectory_patch_logprobs_values,
        trajectory_patch_values,
        train_value_head_on_detached_returns,
        value_mse_loss,
        value_prediction_metrics,
    )
    from scripts.notagen_ppo_diagnostics import (
        advantage_distribution_summary,
        component_group_sums,
        component_lambda_return_tensors,
        component_reward_tensors,
        logprob_advantage_diagnostics,
        per_patch_diagnostic_records,
    )
    from scripts.train_notagen_ppo_value_head_offline import (
        PreparedValueSample as OfflinePreparedValueSample,
        _split_samples as split_offline_value_samples,
    )
    from scripts.summarize_ppo_advantages import summarize_steps
    from scripts.custom_grpo_notagen import (
        PATCH_SIZE,
        GoldbergRewardConfig,
        SimilarityRewardWeights,
        _rollout_seed,
        build_rollout_prefix,
    )
    from rewards.rewards import StructuralTarget
    from utils import NotaGenLMHeadModel, Patchilizer
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


def _generated_patches_from_text(text: str) -> list[list[int]]:
    patchilizer = Patchilizer(stream=True)
    return [[ord(char) for char in patch] for patch in patchilizer.split_patches(text, patch_size=PATCH_SIZE)]


@unittest.skipIf(torch is None, f"NotaGen torch dependencies unavailable: {IMPORT_ERROR}")
class NotaGenPPOTests(unittest.TestCase):
    def test_similarity_reward_is_decoupled_from_structural_gate(self):
        class FakeBreakdown:
            total_reward = 1.5

            def to_json(self):
                return {
                    "completion_reward": 0.0,
                    "bar_count_reward": 0.0,
                    "total_reward": self.total_reward,
                }

        with patch(
            "scripts.notagen_ppo_rewards.score_similarity_reward",
            return_value={"similarity_reward": 1.25, "aria_chroma_harmonic_hist": 1.25},
        ):
            score = _score_total_reward_from_structural_breakdown(
                prompt_text="X:1\n",
                completion_text="[r:0/0][V:1]C|\n",
                structural_breakdown=FakeBreakdown(),
                similarity_weights=SimilarityRewardWeights(aria_chroma=1.0),
                aria_similarity_ref=None,
                similarity_chroma_bins=8,
                similarity_band_ratio=0.25,
                similarity_timeout_s=1.0,
                max_similarity_reward=2.0,
            )

        self.assertEqual(score.breakdown["similarity_validity_gate"], 0.0)
        self.assertEqual(score.breakdown["active_similarity_reward"], 1.25)
        self.assertEqual(score.breakdown["effective_similarity_reward"], 1.25)
        self.assertEqual(score.total, 2.75)

    def test_strict_symbolic_similarity_component_is_terminal(self):
        score = RewardScore(
            total=1.5,
            breakdown={
                "raw_similarity_reward": 1.5,
                "clipped_similarity_reward": 1.5,
                f"aria_{STRICT_SYMBOLIC_COMPONENT_Z_KEY}": 1.5,
            },
        )

        components = _terminal_similarity_component_rewards(
            final_score=score,
            similarity_weights=SimilarityRewardWeights(aria_strict_symbolic=1.0),
            patch_count=3,
        )

        rewards = components[f"aria_{STRICT_SYMBOLIC_COMPONENT_Z_KEY}_active"]
        self.assertEqual(rewards, [0.0, 0.0, 1.5])

    def test_strict_symbolic_similarity_is_in_grouped_diagnostics(self):
        groups = component_group_sums(
            {
                "completion_reward": 0.25,
                f"aria_{STRICT_SYMBOLIC_COMPONENT_Z_KEY}_active": 1.5,
                "aria_strict_aligned_root_bass_active": 0.2,
                "aria_strict_cadence_root_bass_active": 0.3,
            }
        )

        self.assertEqual(groups["structural_total_reward"], 0.25)
        self.assertEqual(groups["aria_strict_symbolic_active"], 2.0)
        self.assertEqual(groups["active_similarity_reward"], 2.0)
        self.assertEqual(groups["effective_similarity_reward"], 2.0)
        self.assertEqual(groups["total_reward"], 2.25)

    def test_strict_symbolic_similarity_is_in_grouped_patch_diagnostics(self):
        trace = PatchRewardTrace(
            rewards=[0.0, 1.75],
            prefix_totals=[0.0, 1.75],
            final_score=RewardScore(total=1.75, breakdown={}),
            component_rewards={
                "completion_reward": [0.0, 0.25],
                f"aria_{STRICT_SYMBOLIC_COMPONENT_Z_KEY}_active": [0.0, 1.5],
                "aria_strict_aligned_root_bass_active": [0.2, 0.0],
                "aria_strict_cadence_root_bass_active": [0.0, 0.3],
            },
            component_prefix_totals={},
        )

        tensors = component_reward_tensors([trace], device=torch.device("cpu"))

        self.assertTrue(torch.allclose(tensors["structural_total_reward"], torch.tensor([0.0, 0.25])))
        self.assertTrue(torch.allclose(tensors["aria_strict_symbolic_active"], torch.tensor([0.2, 1.8])))
        self.assertTrue(torch.allclose(tensors["active_similarity_reward"], torch.tensor([0.2, 1.8])))
        self.assertTrue(torch.allclose(tensors["total_reward"], torch.tensor([0.2, 2.05])))

    def test_strict_symbolic_similarity_can_be_dense_events(self):
        reference_harmony = [
            {"root": 0, "bass": 0, "quality": "maj", "top_midi": 64},
            {"root": 7, "bass": 7, "quality": "maj", "top_midi": 67},
            {"root": 9, "bass": 9, "quality": "min", "top_midi": 69},
            {"root": 0, "bass": 0, "quality": "maj", "top_midi": 72},
        ]
        candidate_harmony = [dict(item) for item in reference_harmony]
        candidate_harmony[0] = {"root": 2, "bass": 2, "quality": "min", "top_midi": 62}
        candidate_spans = [(0, 4), (4, 8), (8, 12), (12, 16)]
        strict_scores = strict_symbolic_similarity(reference_harmony, candidate_harmony, band_ratio=0.05)
        strict_scores.update(strict_similarity_global_base_z_scores(strict_scores))
        expected_total = sum(
            STRICT_SYMBOLIC_COMPONENT_WEIGHTS[name] * strict_scores[f"{name}_global_base_z"]
            for name in STRICT_SYMBOLIC_COMPONENT_WEIGHTS
        )
        breakdown = {
            "raw_similarity_reward": expected_total,
            "clipped_similarity_reward": expected_total,
        }
        breakdown.update({f"aria_{name}": value for name, value in strict_scores.items()})

        events = _strict_symbolic_reward_events(
            reference_harmony=reference_harmony,
            candidate_harmony=candidate_harmony,
            candidate_spans=candidate_spans,
            similarity_weights=SimilarityRewardWeights(aria_strict_symbolic=1.0),
            final_score=RewardScore(total=expected_total, breakdown=breakdown),
            band_ratio=0.05,
        )

        event_names = {event.name for event in events}
        self.assertIn("aria_strict_aligned_root_bass_active", event_names)
        self.assertIn("aria_strict_harmony_dtw_narrow_active", event_names)
        self.assertIn("aria_strict_root_dtw_narrow_active", event_names)
        self.assertIn("aria_strict_bass_dtw_narrow_active", event_names)
        self.assertIn("aria_strict_root_bass_bigram_weighted_jaccard_active", event_names)
        self.assertIn("aria_strict_root_bass_fourgram_weighted_jaccard_active", event_names)
        self.assertIn("aria_strict_cadence_root_bass_active", event_names)
        self.assertAlmostEqual(sum(event.value for event in events), expected_total)
        self.assertTrue(
            any(
                event.value < 0.0
                for event in events
                if event.name
                in {
                    "aria_strict_aligned_root_bass_active",
                    "aria_strict_harmony_dtw_narrow_active",
                    "aria_strict_root_dtw_narrow_active",
                    "aria_strict_bass_dtw_narrow_active",
                    "aria_strict_cadence_root_bass_active",
                }
            )
        )
        self.assertTrue(any(event.end < candidate_spans[-1][1] for event in events))
        self.assertTrue(
            all(
                event.start >= candidate_spans[-1][1] - 1
                for event in events
                if event.name
                in {
                    "aria_strict_root_bass_bigram_weighted_jaccard_active",
                    "aria_strict_root_bass_fourgram_weighted_jaccard_active",
                }
            )
        )

    def test_offline_value_split_can_be_saved_and_reused(self):
        samples = [
            OfflinePreparedValueSample(
                hidden_states=torch.zeros(2, 3),
                targets=torch.ones(2),
                meta={
                    "source_json": "/tmp/rollouts.json",
                    "step": idx,
                    "prompt_index": idx % 2,
                    "prompt_name": f"prompt-{idx}",
                    "trajectory_index": idx,
                    "rollout_seed": 100 + idx,
                    "patch_count": 2,
                    "reward": float(idx),
                },
            )
            for idx in range(4)
        ]
        train, eval_samples, split_meta = split_offline_value_samples(
            samples,
            holdout_last_step=False,
            eval_fraction=0.5,
            seed=7,
        )
        self.assertEqual(len(train), 2)
        self.assertEqual(len(eval_samples), 2)
        self.assertIn("train_keys", split_meta)
        self.assertIn("eval_keys", split_meta)

        with tempfile.TemporaryDirectory() as tmp:
            split_path = Path(tmp) / "split.json"
            split_path.write_text(json.dumps({"dataset_split": split_meta}), encoding="utf-8")
            train_reused, eval_reused, reused_meta = split_offline_value_samples(
                list(reversed(samples)),
                holdout_last_step=False,
                eval_fraction=0.0,
                seed=999,
                dataset_split_json=split_path,
            )

        self.assertEqual([sample.meta["trajectory_index"] for sample in train_reused], [
            sample.meta["trajectory_index"] for sample in train
        ])
        self.assertEqual([sample.meta["trajectory_index"] for sample in eval_reused], [
            sample.meta["trajectory_index"] for sample in eval_samples
        ])
        self.assertEqual(reused_meta["mode"], "stored")

    def test_offline_value_split_keys_distinguish_same_filename_inputs(self):
        samples = [
            OfflinePreparedValueSample(
                hidden_states=torch.zeros(1, 3),
                targets=torch.ones(1),
                meta={
                    "source_json": f"/tmp/gpu{idx}/result.json",
                    "step": 1,
                    "prompt_index": 0,
                    "prompt_name": "prompt",
                    "trajectory_index": 0,
                    "rollout_seed": 123,
                    "patch_count": 1,
                    "reward": 1.0,
                },
            )
            for idx in range(2)
        ]

        _train, _eval_samples, split_meta = split_offline_value_samples(
            samples,
            holdout_last_step=False,
            eval_fraction=0.0,
            seed=0,
        )

        self.assertEqual(len(split_meta["train_keys"]), 2)
        self.assertEqual(len(set(split_meta["train_keys"])), 2)
        self.assertIn("gpu0/result.json", split_meta["train_keys"][0])
        self.assertIn("gpu1/result.json", split_meta["train_keys"][1])

    def test_ordered_prompt_schedule_cycles_by_update_index(self):
        selections = [
            select_prompt_for_update(
                update_index=update_index,
                prompt_count=3,
                selection="ordered",
                seed=123,
            )
            for update_index in range(8)
        ]

        self.assertEqual([item.prompt_idx for item in selections], [0, 1, 2, 0, 1, 2, 0, 1])
        self.assertEqual([item.cycle for item in selections], [0, 0, 0, 1, 1, 1, 2, 2])
        self.assertEqual([item.cycle_position for item in selections], [0, 1, 2, 0, 1, 2, 0, 1])

    def test_random_prompt_schedule_is_seeded_shuffle_cycle(self):
        first_cycle = [
            select_prompt_for_update(
                update_index=update_index,
                prompt_count=30,
                selection="random",
                seed=11,
            ).prompt_idx
            for update_index in range(30)
        ]
        repeated_first_cycle = prompt_cycle_order(30, selection="random", seed=11, cycle=0)
        second_cycle = [
            select_prompt_for_update(
                update_index=30 + update_index,
                prompt_count=30,
                selection="random",
                seed=11,
            ).prompt_idx
            for update_index in range(30)
        ]

        self.assertEqual(first_cycle, repeated_first_cycle)
        self.assertEqual(sorted(first_cycle), list(range(30)))
        self.assertEqual(sorted(second_cycle), list(range(30)))
        self.assertNotEqual(first_cycle, second_cycle)

    def test_random_prompt_schedule_honors_step_offset_position(self):
        selection = select_prompt_for_update(
            update_index=37,
            prompt_count=30,
            selection="random",
            seed=19,
        )
        cycle_order = prompt_cycle_order(30, selection="random", seed=19, cycle=1)

        self.assertEqual(selection.cycle, 1)
        self.assertEqual(selection.cycle_position, 7)
        self.assertEqual(selection.prompt_idx, cycle_order[7])

    def test_prompt_batch_consumes_slots_across_cycle_boundary(self):
        prompts = [{"name": f"p{idx}", "prompt": f"prompt {idx}\n"} for idx in range(30)]
        prompt_targets = [
            PromptStructuralTarget(
                target=StructuralTarget(expected_bars=idx + 1, expected_structure_bars=idx + 1),
                structure_path=f"target_{idx}.abc",
                source_key="test",
            )
            for idx in range(30)
        ]

        batch = build_prompt_batch_for_slots(
            prompts=prompts,
            prompt_targets=prompt_targets,
            selection="ordered",
            seed=0,
            start_slot=28,
            count=5,
        )

        self.assertEqual([item.prompt_idx for item in batch], [28, 29, 0, 1, 2])
        self.assertEqual([item.schedule.cycle for item in batch], [0, 0, 1, 1, 1])
        self.assertEqual([item.schedule.cycle_position for item in batch], [28, 29, 0, 1, 2])
        self.assertEqual([item.target_stream_lines for item in batch], [29, 30, 1, 2, 3])

    def test_repeated_prompt_batch_keeps_one_prompt_for_all_trajectories(self):
        prompts = [{"name": f"p{idx}", "prompt": f"prompt {idx}\n"} for idx in range(5)]
        prompt_targets = [
            PromptStructuralTarget(
                target=StructuralTarget(expected_bars=idx + 1, expected_structure_bars=idx + 1),
                structure_path=f"target_{idx}.abc",
                source_key="test",
            )
            for idx in range(5)
        ]

        batch = build_prompt_batch_for_repeated_slot(
            prompts=prompts,
            prompt_targets=prompt_targets,
            selection="ordered",
            seed=0,
            slot_index=3,
            count=4,
        )

        self.assertEqual([item.trajectory_index for item in batch], [0, 1, 2, 3])
        self.assertEqual([item.prompt_idx for item in batch], [3, 3, 3, 3])
        self.assertEqual([item.prompt_name for item in batch], ["p3", "p3", "p3", "p3"])
        self.assertEqual([item.target_stream_lines for item in batch], [4, 4, 4, 4])
        self.assertEqual([item.schedule.slot_index for item in batch], [3, 3, 3, 3])

    def test_step_prompt_batch_mode_repeats_prompt_per_step(self):
        prompts = [{"name": f"p{idx}", "prompt": f"prompt {idx}\n"} for idx in range(3)]
        prompt_targets = [
            PromptStructuralTarget(
                target=StructuralTarget(expected_bars=idx + 1, expected_structure_bars=idx + 1),
                structure_path=f"target_{idx}.abc",
                source_key="test",
            )
            for idx in range(3)
        ]
        args = SimpleNamespace(
            trajectories_per_step=4,
            prompt_batch_mode="step",
            prompt_selection="ordered",
            seed=0,
        )

        step_two = build_prompt_batch_for_step(
            prompts=prompts,
            prompt_targets=prompt_targets,
            args=args,
            step_idx=2,
        )

        self.assertEqual([item.prompt_idx for item in step_two], [1, 1, 1, 1])
        self.assertEqual([item.target_stream_lines for item in step_two], [2, 2, 2, 2])

    def test_fixed_eval_cadence_uses_every_steps(self):
        args = SimpleNamespace(fixed_eval_trajectories=4, fixed_eval_every_steps=3)

        self.assertFalse(fixed_eval_should_run_after_step(args, 1))
        self.assertFalse(fixed_eval_should_run_after_step(args, 2))
        self.assertTrue(fixed_eval_should_run_after_step(args, 3))
        self.assertFalse(fixed_eval_should_run_after_step(args, 4))
        self.assertTrue(fixed_eval_should_run_after_step(args, 6))
        self.assertEqual(fixed_eval_event_index_after_step(args, 3), 0)
        self.assertEqual(fixed_eval_event_index_after_step(args, 6), 1)

        disabled = SimpleNamespace(fixed_eval_trajectories=4, fixed_eval_every_steps=0)
        self.assertFalse(fixed_eval_should_run_after_step(disabled, 3))
        no_eval = SimpleNamespace(fixed_eval_trajectories=0, fixed_eval_every_steps=1)
        self.assertFalse(fixed_eval_should_run_after_step(no_eval, 1))

    def test_fixed_eval_before_training_event_index_accounts_for_resume_offset(self):
        fresh = SimpleNamespace(step_offset=0, fixed_eval_every_steps=5)
        resumed = SimpleNamespace(step_offset=12, fixed_eval_every_steps=5)
        disabled = SimpleNamespace(step_offset=12, fixed_eval_every_steps=0)

        self.assertEqual(fixed_eval_event_index_before_training(fresh), 0)
        self.assertEqual(fixed_eval_event_index_before_training(resumed), 2)
        self.assertEqual(fixed_eval_event_index_before_training(disabled), 0)

    def test_fixed_eval_prompt_batch_has_independent_ordered_rotation(self):
        prompts = [{"name": f"p{idx}", "prompt": f"prompt {idx}\n"} for idx in range(30)]
        prompt_targets = [
            PromptStructuralTarget(
                target=StructuralTarget(expected_bars=idx + 1, expected_structure_bars=idx + 1),
                structure_path=f"target_{idx}.abc",
                source_key="test",
            )
            for idx in range(30)
        ]
        args = SimpleNamespace(
            fixed_eval_trajectories=4,
            fixed_eval_prompt_selection="ordered",
            fixed_eval_prompt_batch_mode="trajectory",
            prompt_selection="random",
            prompt_batch_mode="trajectory",
            seed=3,
            fixed_eval_prompt_seed_offset=100,
            fixed_eval_reuse_prompt_batch=False,
        )

        first_eval = build_fixed_eval_prompt_batch(
            prompts=prompts,
            prompt_targets=prompt_targets,
            args=args,
            event_index=0,
        )
        second_eval = build_fixed_eval_prompt_batch(
            prompts=prompts,
            prompt_targets=prompt_targets,
            args=args,
            event_index=1,
        )

        self.assertEqual([item.prompt_idx for item in first_eval], [0, 1, 2, 3])
        self.assertEqual([item.prompt_idx for item in second_eval], [4, 5, 6, 7])
        self.assertEqual([item.target_stream_lines for item in first_eval], [1, 2, 3, 4])
        self.assertTrue(all(item.schedule.selection == "ordered" for item in first_eval + second_eval))

    def test_fixed_eval_prompt_batch_reuses_same_batch_by_default(self):
        prompts = [{"name": f"p{idx}", "prompt": f"prompt {idx}\n"} for idx in range(30)]
        prompt_targets = [
            PromptStructuralTarget(
                target=StructuralTarget(expected_bars=idx + 1, expected_structure_bars=idx + 1),
                structure_path=f"target_{idx}.abc",
                source_key="test",
            )
            for idx in range(30)
        ]
        args = SimpleNamespace(
            fixed_eval_trajectories=4,
            fixed_eval_prompt_selection="ordered",
            fixed_eval_prompt_batch_mode="trajectory",
            prompt_selection="random",
            prompt_batch_mode="trajectory",
            seed=3,
            fixed_eval_prompt_seed_offset=100,
        )

        first_eval = build_fixed_eval_prompt_batch(
            prompts=prompts,
            prompt_targets=prompt_targets,
            args=args,
            event_index=0,
        )
        later_eval = build_fixed_eval_prompt_batch(
            prompts=prompts,
            prompt_targets=prompt_targets,
            args=args,
            event_index=4,
        )

        self.assertEqual([item.prompt_idx for item in first_eval], [0, 1, 2, 3])
        self.assertEqual([item.prompt_idx for item in later_eval], [0, 1, 2, 3])
        self.assertEqual([item.target_stream_lines for item in later_eval], [1, 2, 3, 4])

    def test_fixed_eval_same_selection_uses_independent_shuffle_seed(self):
        prompts = [{"name": f"p{idx}", "prompt": f"prompt {idx}\n"} for idx in range(30)]
        prompt_targets = [
            PromptStructuralTarget(
                target=StructuralTarget(expected_bars=idx + 1, expected_structure_bars=idx + 1),
                structure_path=f"target_{idx}.abc",
                source_key="test",
            )
            for idx in range(30)
        ]
        args = SimpleNamespace(
            fixed_eval_trajectories=5,
            fixed_eval_prompt_selection="same",
            fixed_eval_prompt_batch_mode="same",
            prompt_selection="random",
            prompt_batch_mode="trajectory",
            seed=17,
            fixed_eval_prompt_seed_offset=99,
        )

        fixed_eval = build_fixed_eval_prompt_batch(
            prompts=prompts,
            prompt_targets=prompt_targets,
            args=args,
            event_index=0,
        )
        training_order = prompt_cycle_order(30, selection="random", seed=17, cycle=0)
        eval_order = prompt_cycle_order(30, selection="random", seed=116, cycle=0)

        self.assertEqual([item.prompt_idx for item in fixed_eval], eval_order[:5])
        self.assertNotEqual([item.prompt_idx for item in fixed_eval], training_order[:5])

    def test_fixed_eval_event_prompt_batch_mode_repeats_prompt_per_event(self):
        prompts = [{"name": f"p{idx}", "prompt": f"prompt {idx}\n"} for idx in range(4)]
        prompt_targets = [
            PromptStructuralTarget(
                target=StructuralTarget(expected_bars=idx + 1, expected_structure_bars=idx + 1),
                structure_path=f"target_{idx}.abc",
                source_key="test",
            )
            for idx in range(4)
        ]
        args = SimpleNamespace(
            fixed_eval_trajectories=3,
            fixed_eval_prompt_selection="ordered",
            fixed_eval_prompt_batch_mode="event",
            prompt_selection="random",
            prompt_batch_mode="trajectory",
            seed=0,
            fixed_eval_prompt_seed_offset=100,
            fixed_eval_reuse_prompt_batch=False,
        )

        second_eval = build_fixed_eval_prompt_batch(
            prompts=prompts,
            prompt_targets=prompt_targets,
            args=args,
            event_index=1,
        )

        self.assertEqual([item.prompt_idx for item in second_eval], [1, 1, 1])
        self.assertEqual([item.target_stream_lines for item in second_eval], [2, 2, 2])

    def test_mixed_prompt_rollout_passes_per_row_target_lengths_to_batch_sampler(self):
        prompts = [{"name": "p0", "prompt": "prompt 0\n"}, {"name": "p1", "prompt": "prompt 1\n"}]
        prompt_targets = [
            PromptStructuralTarget(
                target=StructuralTarget(expected_bars=3, expected_structure_bars=3),
                structure_path="target_0.abc",
                source_key="test",
            ),
            PromptStructuralTarget(
                target=StructuralTarget(expected_bars=5, expected_structure_bars=5),
                structure_path="target_1.abc",
                source_key="test",
            ),
        ]
        prompt_batch = build_prompt_batch_for_slots(
            prompts=prompts,
            prompt_targets=prompt_targets,
            selection="ordered",
            seed=0,
            start_slot=0,
            count=2,
        )

        def fake_batch(**kwargs):
            self.assertEqual(kwargs["prompts"], ["prompt 0\n", "prompt 1\n"])
            self.assertEqual(kwargs["target_stream_lines"], [3, 5])
            return [
                SimpleNamespace(
                    ok=True,
                    full_text="prompt 0\n[r:0/2][V:1]C|\n",
                    generated_patches=[[ord("C")]],
                    meta={"stop_reason": "target_stream_lines"},
                    error=None,
                ),
                SimpleNamespace(
                    ok=True,
                    full_text="prompt 1\n[r:0/4][V:1]D|\n",
                    generated_patches=[[ord("D")]],
                    meta={"stop_reason": "target_stream_lines"},
                    error=None,
                ),
            ]

        args = SimpleNamespace(
            trajectories_per_step=2,
            rollout_batch_size=2,
            cached_rollout=True,
            rollout_retries=1,
            rollout_failure_policy="error",
            rollout_spares_percent=10.0,
            seed=7,
            temperature=1.0,
            top_k=8,
            top_p=0.95,
            max_chars=100,
            max_generated_patches=10,
            timeout_s=5,
            precision="fp32",
        )
        with patch("scripts.custom_ppo_notagen.sample_completions_cached_batch", side_effect=fake_batch):
            payloads = sample_ppo_rollouts(
                policy_model=object(),
                policy_shape=object(),
                step_idx=1,
                args=args,
                prompt_batch=prompt_batch,
            )

        self.assertEqual([payload.prompt_idx for payload in payloads], [0, 1])
        self.assertEqual([payload.target_stream_lines for payload in payloads], [3, 5])
        self.assertEqual([payload.meta["rollout_target_stream_lines"] for payload in payloads], [3, 5])

    def test_rollout_seed_scope_run_reuses_seed_set_across_steps(self):
        prompts = [{"name": "p0", "prompt": "prompt 0\n"}, {"name": "p1", "prompt": "prompt 1\n"}]
        prompt_targets = [
            PromptStructuralTarget(
                target=StructuralTarget(expected_bars=3, expected_structure_bars=3),
                structure_path="target_0.abc",
                source_key="test",
            ),
            PromptStructuralTarget(
                target=StructuralTarget(expected_bars=5, expected_structure_bars=5),
                structure_path="target_1.abc",
                source_key="test",
            ),
        ]
        prompt_batch = build_prompt_batch_for_slots(
            prompts=prompts,
            prompt_targets=prompt_targets,
            selection="ordered",
            seed=0,
            start_slot=0,
            count=2,
        )
        captured_seeds: list[list[int]] = []

        def fake_batch(**kwargs):
            captured_seeds.append(list(kwargs["seeds"]))
            return [
                SimpleNamespace(
                    ok=True,
                    full_text=f"{prompt_batch[idx].prompt}[r:0/{prompt_batch[idx].target_stream_lines - 1}][V:1]C|\n",
                    generated_patches=[[ord("C")]],
                    meta={"stop_reason": "target_stream_lines"},
                    error=None,
                )
                for idx in range(2)
            ]

        args = SimpleNamespace(
            trajectories_per_step=2,
            rollout_batch_size=2,
            cached_rollout=True,
            rollout_retries=1,
            rollout_failure_policy="error",
            rollout_spares_percent=10.0,
            rollout_seed_scope="run",
            seed=7,
            temperature=1.0,
            top_k=8,
            top_p=0.95,
            max_chars=100,
            max_generated_patches=10,
            timeout_s=5,
            precision="fp32",
        )
        with patch("scripts.custom_ppo_notagen.sample_completions_cached_batch", side_effect=fake_batch):
            sample_ppo_rollouts(
                policy_model=object(),
                policy_shape=object(),
                step_idx=1,
                args=args,
                prompt_batch=prompt_batch,
            )
            sample_ppo_rollouts(
                policy_model=object(),
                policy_shape=object(),
                step_idx=2,
                args=args,
                prompt_batch=prompt_batch,
            )

        self.assertEqual(captured_seeds[0], captured_seeds[1])

    def test_mixed_prompt_scorer_uses_each_payload_target_context(self):
        prompts = [{"name": "p0", "prompt": "prompt 0\n"}, {"name": "p1", "prompt": "prompt 1\n"}]
        prompt_targets = [
            PromptStructuralTarget(
                target=StructuralTarget(expected_bars=3, expected_structure_bars=3),
                structure_path="target_0.abc",
                source_key="test0",
            ),
            PromptStructuralTarget(
                target=StructuralTarget(expected_bars=5, expected_structure_bars=5),
                structure_path="target_1.abc",
                source_key="test1",
            ),
        ]
        prompt_batch = build_prompt_batch_for_slots(
            prompts=prompts,
            prompt_targets=prompt_targets,
            selection="ordered",
            seed=0,
            start_slot=0,
            count=2,
        )
        payloads = [
            PPORolloutPayload(
                trajectory_index=item.trajectory_index,
                rollout_seed=10 + item.trajectory_index,
                full_text=item.prompt + "abc",
                generated_patches=[[ord("a") + item.trajectory_index]],
                meta={"stop_reason": "max_generated_patches"},
                prompt_idx=item.prompt_idx,
                prompt_name=item.prompt_name,
                prompt=item.prompt,
                prompt_target=item.prompt_target,
                target=item.target,
                target_stream_lines=item.target_stream_lines,
                prompt_schedule=item.schedule,
            )
            for item in prompt_batch
        ]
        args = SimpleNamespace(
            similarity_chroma_bins=8,
            similarity_band_ratio=0.25,
            similarity_timeout_s=5.0,
            max_similarity_reward=2.0,
            patch_reward_attribution="single_pass",
            reward_mode="length",
            simple_reward_note="G",
            simple_reward_max_count=64.0,
            simple_reward_length_unit="patches",
            simple_reward_length_target=1.0,
            simple_reward_scale=1.0,
            reward_workers=0,
        )

        scored = score_ppo_rollout_payloads_from_payload_context(
            rollout_payloads=payloads,
            reward_config=GoldbergRewardConfig(parse_validation_mode="abc-tokenize"),
            similarity_weights=SimilarityRewardWeights(),
            aria_similarity_ref=None,
            args=args,
            step_idx=1,
            candidate_name_prefix="mixed_prompt_score",
        )

        self.assertEqual(
            [log["reward_breakdown"]["target_stream_lines"] for log in scored.trajectory_logs],
            [3, 5],
        )
        self.assertEqual(
            [log["reward_breakdown"]["target_structure_source_key"] for log in scored.trajectory_logs],
            ["test0", "test1"],
        )

    def test_rollout_seeds_do_not_collide_for_batched_spares_across_steps(self):
        seeds = [
            _rollout_seed(base_seed=0, step_idx=step_idx, group_idx=candidate_idx, retry_idx=retry_idx)
            for step_idx in range(1, 14)
            for candidate_idx in range(18)
            for retry_idx in range(3)
        ]

        self.assertEqual(len(seeds), len(set(seeds)))

    def test_prompt_structural_targets_prefer_row_source_over_fallback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            target_json = tmp / "target.json"
            target_json.write_text("[{}]", encoding="utf-8")
            fallback_abc = tmp / "fallback.abc"
            fallback_abc.write_text("[r:0/0][V:1]C|\n", encoding="utf-8")
            source_abc = tmp / "source.abc"
            source_abc.write_text("[r:0/1][V:1]C|\n[r:1/0][V:1]D|\n", encoding="utf-8")
            prompts_jsonl = tmp / "prompts.jsonl"

            args = SimpleNamespace(
                prompts_jsonl=str(prompts_jsonl),
                target_json=str(target_json),
                target_structure_abc=str(fallback_abc),
            )
            prompt_targets = load_prompt_structural_targets(
                [
                    {"prompt": "", "source": "source.abc"},
                    {"prompt": ""},
                ],
                args,
            )

        self.assertEqual(prompt_targets[0].source_key, "source")
        self.assertEqual(prompt_targets[0].target.expected_structure_bars, 2)
        self.assertEqual(prompt_targets[0].target.expected_reward_bars, 1)
        self.assertEqual(prompt_targets[1].source_key, "fallback_target_structure_abc")
        self.assertEqual(prompt_targets[1].target.expected_structure_bars, 1)
        self.assertEqual(prompt_targets[1].target.expected_reward_bars, 1)

    def test_aria_matching_prompt_bar_filter_keeps_only_matching_source_structures(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            matching = tmp / "matching.abc"
            matching.write_text(
                "\n".join(
                    [
                        "X:1",
                        "M:3/4",
                        "L:1/8",
                        "K:C",
                        "V:1",
                        "[V:1]C2D2E2|",
                        "[V:1]F2G2A2:|",
                    ]
                ),
                encoding="utf-8",
            )
            mismatched = tmp / "mismatched.abc"
            mismatched.write_text(
                "\n".join(
                    [
                        "X:1",
                        "M:12/8",
                        "L:1/8",
                        "K:C",
                        "V:1",
                        "[V:1]C12:|",
                    ]
                ),
                encoding="utf-8",
            )
            prompts = [
                {"name": "matching", "prompt": "", "source": str(matching)},
                {"name": "mismatched", "prompt": "", "source": str(mismatched)},
            ]
            prompt_targets = [
                PromptStructuralTarget(
                    target=StructuralTarget(expected_bars=2, expected_structure_bars=2),
                    structure_path=str(matching),
                    source_key="source",
                ),
                PromptStructuralTarget(
                    target=StructuralTarget(expected_bars=2, expected_structure_bars=1),
                    structure_path=str(mismatched),
                    source_key="source",
                ),
            ]
            args = SimpleNamespace(prompt_bar_filter="aria-matching", prompt_bar_filter_tolerance=1e-6)

            filtered_prompts, filtered_targets, metadata = filter_prompts_by_bar_target(prompts, prompt_targets, args)

        self.assertEqual([row["name"] for row in filtered_prompts], ["matching"])
        self.assertEqual([Path(item.structure_path).name for item in filtered_targets], ["matching.abc"])
        self.assertEqual(metadata["input_count"], 2)
        self.assertEqual(metadata["kept_count"], 1)
        self.assertEqual(metadata["excluded_count"], 1)
        self.assertEqual(metadata["kept"][0]["source_format"], "raw-abc")
        self.assertEqual(metadata["excluded"][0]["prompt_name"], "mismatched")

    def test_parallel_rollout_scoring_matches_serial_exactly(self):
        if "fork" not in mp.get_all_start_methods():
            self.skipTest("fork multiprocessing context is unavailable")

        prompt = "X:1\nT:Parallel reward test\nM:3/4\nL:1/8\nK:C\nV:1\n%%score 1\n"
        completions = [
            "[r:0/1][V:1]C2 D2 E2|\n[r:1/0][V:1]F2 G2 A2|\n",
            "[r:0/1][V:1]G2 A2 B2|\n[r:1/0][V:1]c2 B2 A2|\n",
        ]
        rollout_payloads = [
            PPORolloutPayload(
                trajectory_index=index,
                rollout_seed=100 + index,
                full_text=prompt + completion,
                generated_patches=_generated_patches_from_text(completion),
                meta={
                    "cached_rollout": False,
                    "batched_rollout": False,
                    "rollout_batch_size": 1,
                    "rollout_target_stream_lines": 2,
                    "stop_reason": "target_stream_lines",
                },
            )
            for index, completion in enumerate(completions)
        ]
        target = StructuralTarget(expected_bars=2, expected_structure_bars=2)
        prompt_target = PromptStructuralTarget(
            target=target,
            structure_path="<test>",
            source_key="parallel_test",
        )
        common_args = {
            "similarity_chroma_bins": 8,
            "similarity_band_ratio": 0.25,
            "similarity_timeout_s": 5.0,
            "max_similarity_reward": 2.0,
        }
        serial = score_ppo_rollout_payloads(
            prompt=prompt,
            prompt_idx=0,
            prompt_name="parallel_test",
            prompt_target=prompt_target,
            target=target,
            target_stream_lines=2,
            rollout_payloads=rollout_payloads,
            reward_config=GoldbergRewardConfig(parse_validation_mode="abc-tokenize"),
            similarity_weights=SimilarityRewardWeights(),
            aria_similarity_ref=None,
            args=SimpleNamespace(**common_args, reward_workers=0),
            step_idx=0,
            candidate_name_prefix="parallel_exact",
        )
        parallel = score_ppo_rollout_payloads(
            prompt=prompt,
            prompt_idx=0,
            prompt_name="parallel_test",
            prompt_target=prompt_target,
            target=target,
            target_stream_lines=2,
            rollout_payloads=rollout_payloads,
            reward_config=GoldbergRewardConfig(parse_validation_mode="abc-tokenize"),
            similarity_weights=SimilarityRewardWeights(),
            aria_similarity_ref=None,
            args=SimpleNamespace(**common_args, reward_workers=2, reward_worker_start_method="fork"),
            step_idx=0,
            candidate_name_prefix="parallel_exact",
        )

        self.assertEqual(parallel.reward_summary, serial.reward_summary)
        self.assertEqual(parallel.trajectory_logs, serial.trajectory_logs)
        self.assertEqual(parallel.reward_traces, serial.reward_traces)

    def test_single_pass_reward_scores_rollout_prefix_not_raw_prompt(self):
        prompt = "X:1\nT:Prefix scoring test\nM:3/4\nL:1/8\nK:C\nV:1\n%%score 1\n"
        completion = "[r:0/0][V:1]C2 D2 E2|\n"
        generated_patches = _generated_patches_from_text(completion)
        target = StructuralTarget(expected_bars=1, expected_structure_bars=1)
        prompt_target = PromptStructuralTarget(
            target=target,
            structure_path="<test>",
            source_key="prefix_scoring_test",
        )
        payload = PPORolloutPayload(
            trajectory_index=0,
            rollout_seed=7,
            full_text=build_rollout_prefix(prompt, 1) + completion,
            generated_patches=generated_patches,
            meta={"stop_reason": "target_stream_lines"},
        )
        seen: dict[str, str] = {}

        def fake_prefix_rewards(**kwargs):
            seen["prompt_text"] = kwargs["prompt_text"]
            return PatchRewardTrace(
                rewards=[1.0],
                prefix_totals=[1.0],
                final_score=RewardScore(total=1.0, breakdown={"total_reward": 1.0}),
                component_rewards={"fake_reward": [1.0]},
                component_prefix_totals={"fake_reward": [1.0]},
            )

        with patch("scripts.custom_ppo_notagen.patch_rewards_from_prefix_deltas", side_effect=fake_prefix_rewards):
            score_ppo_rollout_payloads(
                prompt=prompt,
                prompt_idx=0,
                prompt_name="prefix_scoring_test",
                prompt_target=prompt_target,
                target=target,
                target_stream_lines=1,
                rollout_payloads=[payload],
                reward_config=GoldbergRewardConfig(parse_validation_mode="abc-tokenize"),
                similarity_weights=SimilarityRewardWeights(),
                aria_similarity_ref=None,
                args=SimpleNamespace(
                    similarity_chroma_bins=8,
                    similarity_band_ratio=0.25,
                    similarity_timeout_s=5.0,
                    max_similarity_reward=2.0,
                    reward_workers=0,
                    patch_reward_attribution="single_pass",
                ),
                step_idx=0,
                candidate_name_prefix="prefix_scoring",
            )

        self.assertEqual(seen["prompt_text"], build_rollout_prefix(prompt, 1))
        self.assertNotEqual(seen["prompt_text"], prompt)

    def test_zero_policy_records_failed_batched_rollout_without_retrying(self):
        prompt = "X:1\nT:Zero failed rollout test\nM:3/4\nL:1/8\nK:C\n"

        def fake_batch(**kwargs):
            return [
                SimpleNamespace(
                    ok=True,
                    full_text=prompt + "[r:0/1][V:1]C2 D2 E2|\n",
                    generated_patches=[[ord("C")]],
                    meta={"stop_reason": "target_stream_lines"},
                    error=None,
                ),
                SimpleNamespace(
                    ok=False,
                    full_text=None,
                    generated_patches=None,
                    meta={},
                    error="early eos",
                ),
            ]

        args = SimpleNamespace(
            trajectories_per_step=2,
            rollout_batch_size=2,
            cached_rollout=True,
            rollout_retries=3,
            rollout_failure_policy="zero",
            seed=7,
            temperature=1.0,
            top_k=8,
            top_p=0.95,
            max_chars=100,
            max_generated_patches=10,
            timeout_s=5,
            precision="fp32",
        )
        with patch("scripts.custom_ppo_notagen.sample_completions_cached_batch", side_effect=fake_batch) as mocked:
            payloads = sample_ppo_rollouts(
                policy_model=object(),
                policy_shape=object(),
                prompt=prompt,
                target_stream_lines=1,
                step_idx=2,
                args=args,
            )

        self.assertEqual(mocked.call_count, 1)
        self.assertEqual([payload.trajectory_index for payload in payloads], [0, 1])
        self.assertFalse(payloads[0].meta.get("rollout_failed", False))
        self.assertEqual(payloads[0].generated_patches, [[ord("C")]])
        self.assertTrue(payloads[1].meta["rollout_failed"])
        self.assertTrue(payloads[1].meta["zero_contribution_rollout"])
        self.assertEqual(payloads[1].generated_patches, [])
        self.assertEqual(payloads[1].full_text, prompt)

    def test_zero_policy_preserves_partial_failed_batched_rollout(self):
        prompt = "X:1\nT:Partial zero failed rollout test\nM:3/4\nL:1/8\nK:C\n"
        completion = "[r:0/0][V:1]C2 D2 E2|\n"
        generated_patches = _generated_patches_from_text(completion)

        def fake_batch(**kwargs):
            return [
                SimpleNamespace(
                    ok=False,
                    full_text=prompt + completion,
                    generated_patches=generated_patches,
                    meta={"stop_reason": "timeout"},
                    error="generation exceeded 1s",
                ),
            ]

        args = SimpleNamespace(
            trajectories_per_step=1,
            rollout_batch_size=2,
            cached_rollout=True,
            rollout_retries=3,
            rollout_failure_policy="zero",
            seed=7,
            temperature=1.0,
            top_k=8,
            top_p=0.95,
            max_chars=100,
            max_generated_patches=10,
            timeout_s=5,
            precision="fp32",
        )
        with patch("scripts.custom_ppo_notagen.sample_completions_cached_batch", side_effect=fake_batch):
            payloads = sample_ppo_rollouts(
                policy_model=object(),
                policy_shape=object(),
                prompt=prompt,
                target_stream_lines=1,
                step_idx=2,
                args=args,
            )

        self.assertEqual(len(payloads), 1)
        self.assertTrue(payloads[0].meta["rollout_failed"])
        self.assertFalse(payloads[0].meta["zero_contribution_rollout"])
        self.assertEqual(payloads[0].meta["stop_reason"], "timeout")
        self.assertEqual(payloads[0].generated_patches, generated_patches)
        self.assertEqual(payloads[0].full_text, prompt + completion)

    def test_spares_policy_fills_scheduled_slots_without_retrying(self):
        prompt = "X:1\nT:Spares rollout test\nM:3/4\nL:1/8\nK:C\n"

        def fake_batch(**kwargs):
            self.assertEqual(len(kwargs["seeds"]), 3)
            return [
                SimpleNamespace(
                    ok=False,
                    full_text=None,
                    generated_patches=None,
                    meta={},
                    error="early eos",
                ),
                SimpleNamespace(
                    ok=True,
                    full_text=prompt + "[r:0/1][V:1]C2 D2 E2|\n",
                    generated_patches=[[ord("C")]],
                    meta={"stop_reason": "target_stream_lines"},
                    error=None,
                ),
                SimpleNamespace(
                    ok=True,
                    full_text=prompt + "[r:0/1][V:1]F2 G2 A2|\n",
                    generated_patches=[[ord("F")]],
                    meta={"stop_reason": "target_stream_lines"},
                    error=None,
                ),
            ]

        args = SimpleNamespace(
            trajectories_per_step=2,
            rollout_batch_size=2,
            cached_rollout=True,
            rollout_retries=3,
            rollout_failure_policy="spares",
            rollout_spares_percent=50.0,
            seed=7,
            temperature=1.0,
            top_k=8,
            top_p=0.95,
            max_chars=100,
            max_generated_patches=10,
            timeout_s=5,
            precision="fp32",
        )
        with patch("scripts.custom_ppo_notagen.sample_completions_cached_batch", side_effect=fake_batch) as mocked:
            payloads = sample_ppo_rollouts(
                policy_model=object(),
                policy_shape=object(),
                prompt=prompt,
                target_stream_lines=1,
                step_idx=2,
                args=args,
            )

        self.assertEqual(mocked.call_count, 1)
        self.assertEqual([payload.trajectory_index for payload in payloads], [0, 1])
        self.assertEqual([payload.meta["rollout_candidate_index"] for payload in payloads], [2, 1])
        self.assertEqual([payload.meta["rollout_spare_attempt"] for payload in payloads], [1, 0])
        self.assertEqual([payload.generated_patches for payload in payloads], [[[ord("F")]], [[ord("C")]]])
        self.assertEqual(payloads[0].meta["rollout_sampled_candidates"], 3)
        self.assertEqual(payloads[0].meta["rollout_success_candidates"], 2)
        self.assertEqual(payloads[0].meta["rollout_failed_candidates"], 1)
        self.assertEqual(payloads[0].meta["rollout_dropped_candidates"], 1)
        self.assertEqual(payloads[0].meta["rollout_dropped_success_candidates"], 0)
        self.assertEqual(payloads[0].meta["rollout_effective_batch_size"], 3)

    def test_failed_rollout_scores_terminal_penalty_even_when_empty(self):
        prompt = "X:1\nT:Failed rollout score test\nM:3/4\nL:1/8\nK:C\n"
        target = StructuralTarget(expected_bars=1, expected_structure_bars=1)
        prompt_target = PromptStructuralTarget(
            target=target,
            structure_path="<test>",
            source_key="failed_rollout_test",
        )
        payload = PPORolloutPayload(
            trajectory_index=3,
            rollout_seed=44,
            full_text=prompt,
            generated_patches=[],
            meta={
                "cached_rollout": True,
                "batched_rollout": True,
                "rollout_batch_size": 2,
                "rollout_target_stream_lines": 1,
                "rollout_failed": True,
                "zero_contribution_rollout": True,
                "stop_reason": "rollout_failed",
                "error": "early eos",
            },
        )

        scored = score_ppo_rollout_payloads(
            prompt=prompt,
            prompt_idx=0,
            prompt_name="failed_rollout_test",
            prompt_target=prompt_target,
            target=target,
            target_stream_lines=1,
            rollout_payloads=[payload],
            reward_config=GoldbergRewardConfig(parse_validation_mode="abc-tokenize"),
            similarity_weights=SimilarityRewardWeights(),
            aria_similarity_ref=None,
            args=SimpleNamespace(
                similarity_chroma_bins=8,
                similarity_band_ratio=0.25,
                similarity_timeout_s=5.0,
                max_similarity_reward=2.0,
                reward_workers=0,
                rollout_failure_terminal_reward=-2.5,
            ),
            step_idx=0,
            candidate_name_prefix="failed_rollout",
        )

        self.assertEqual(scored.reward_summary["sample_rewards"], [-2.5])
        self.assertEqual(scored.reward_summary["reward_sum"], -2.5)
        self.assertEqual(scored.reward_traces[0].rewards, [])
        self.assertEqual(scored.reward_traces[0].final_score.total, -2.5)
        self.assertEqual(scored.trajectory_logs[0]["reward"], -2.5)
        self.assertEqual(scored.trajectory_logs[0]["patch_rewards"], [])
        self.assertEqual(scored.trajectory_logs[0]["reward_breakdown"]["rollout_failure_terminal_reward"], -2.5)
        self.assertTrue(scored.trajectory_logs[0]["reward_breakdown"]["rollout_failed"])
        self.assertTrue(scored.trajectory_logs[0]["reward_breakdown"]["zero_contribution_rollout"])

    def test_failed_rollout_with_partial_patches_gets_terminal_patch_penalty(self):
        prompt = "X:1\nT:Partial failed rollout score test\nM:3/4\nL:1/8\nK:C\n"
        completion = "[r:0/0][V:1]C2 D2 E2|\n"
        generated_patches = _generated_patches_from_text(completion)
        target = StructuralTarget(expected_bars=1, expected_structure_bars=1)
        prompt_target = PromptStructuralTarget(
            target=target,
            structure_path="<test>",
            source_key="partial_failed_rollout_test",
        )
        payload = PPORolloutPayload(
            trajectory_index=4,
            rollout_seed=45,
            full_text=prompt + completion,
            generated_patches=generated_patches,
            meta={
                "cached_rollout": True,
                "batched_rollout": True,
                "rollout_batch_size": 2,
                "rollout_target_stream_lines": 1,
                "rollout_failed": True,
                "zero_contribution_rollout": False,
                "stop_reason": "timeout",
                "error": "generation exceeded 1s",
            },
        )

        scored = score_ppo_rollout_payloads(
            prompt=prompt,
            prompt_idx=0,
            prompt_name="partial_failed_rollout_test",
            prompt_target=prompt_target,
            target=target,
            target_stream_lines=1,
            rollout_payloads=[payload],
            reward_config=GoldbergRewardConfig(parse_validation_mode="abc-tokenize"),
            similarity_weights=SimilarityRewardWeights(),
            aria_similarity_ref=None,
            args=SimpleNamespace(
                similarity_chroma_bins=8,
                similarity_band_ratio=0.25,
                similarity_timeout_s=5.0,
                max_similarity_reward=2.0,
                reward_workers=0,
                rollout_failure_terminal_reward=-3.0,
            ),
            step_idx=0,
            candidate_name_prefix="partial_failed_rollout",
        )

        self.assertEqual(scored.reward_summary["sample_rewards"], [-3.0])
        self.assertAlmostEqual(sum(scored.reward_traces[0].rewards), -3.0)
        self.assertEqual(scored.reward_traces[0].rewards[:-1], [0.0 for _idx in generated_patches[:-1]])
        self.assertEqual(scored.reward_traces[0].rewards[-1], -3.0)
        self.assertEqual(
            scored.reward_traces[0].component_rewards["rollout_failure_terminal_reward"],
            [0.0 for _idx in generated_patches[:-1]] + [-3.0],
        )
        self.assertFalse(scored.trajectory_logs[0]["reward_breakdown"]["zero_contribution_rollout"])
        self.assertEqual(
            scored.trajectory_logs[0]["patch_reward_group_sums"]["structural_total_reward"],
            -3.0,
        )

    def test_simple_note_count_reward_can_be_dense_or_terminal(self):
        completion = "[r:0/0][V:1]G G A g|\n"
        generated_patches = _generated_patches_from_text(completion)
        self.assertGreater(len(generated_patches), 1)

        dense = patch_rewards_simple_test(
            generated_patches=generated_patches,
            scoring_options=PPORewardScoringOptions(
                similarity_chroma_bins=8,
                similarity_band_ratio=0.25,
                similarity_timeout_s=5.0,
                max_similarity_reward=2.0,
                reward_mode="note_count",
                simple_reward_note="G",
                simple_reward_max_count=4.0,
                simple_reward_scale=2.0,
            ),
        )
        terminal = patch_rewards_simple_test(
            generated_patches=generated_patches,
            scoring_options=PPORewardScoringOptions(
                similarity_chroma_bins=8,
                similarity_band_ratio=0.25,
                similarity_timeout_s=5.0,
                max_similarity_reward=2.0,
                patch_reward_attribution="terminal",
                reward_mode="note_count",
                simple_reward_note="G",
                simple_reward_max_count=4.0,
                simple_reward_scale=2.0,
            ),
        )

        self.assertAlmostEqual(dense.final_score.total, 1.5)
        self.assertAlmostEqual(sum(dense.rewards), 1.5)
        self.assertEqual(dense.final_score.breakdown["simple_reward_note_count"], 3.0)
        self.assertTrue(any(reward > 0.0 for reward in dense.rewards[:-1]))
        self.assertAlmostEqual(terminal.final_score.total, 1.5)
        self.assertEqual(terminal.rewards[:-1], [0.0 for _idx in terminal.rewards[:-1]])
        self.assertAlmostEqual(terminal.rewards[-1], 1.5)

    def test_simple_note_fraction_reward_is_terminal_and_length_normalized(self):
        completion = "[r:0/0][V:1]G A [I:staff -1]g C|\n"
        generated_patches = _generated_patches_from_text(completion)
        self.assertGreater(len(generated_patches), 1)

        trace = patch_rewards_simple_test(
            generated_patches=generated_patches,
            scoring_options=PPORewardScoringOptions(
                similarity_chroma_bins=8,
                similarity_band_ratio=0.25,
                similarity_timeout_s=5.0,
                max_similarity_reward=2.0,
                patch_reward_attribution="terminal",
                reward_mode="note_fraction",
                simple_reward_note="G",
                simple_reward_scale=2.0,
            ),
        )

        self.assertAlmostEqual(trace.final_score.total, 1.0)
        self.assertEqual(trace.final_score.breakdown["simple_reward_note_count"], 2.0)
        self.assertEqual(trace.final_score.breakdown["simple_reward_total_note_count"], 4.0)
        self.assertAlmostEqual(trace.final_score.breakdown["simple_reward_fraction"], 0.5)
        self.assertEqual(trace.rewards[:-1], [0.0 for _idx in trace.rewards[:-1]])
        self.assertAlmostEqual(trace.rewards[-1], 1.0)

        with self.assertRaisesRegex(RuntimeError, "requires terminal"):
            patch_rewards_simple_test(
                generated_patches=generated_patches,
                scoring_options=PPORewardScoringOptions(
                    similarity_chroma_bins=8,
                    similarity_band_ratio=0.25,
                    similarity_timeout_s=5.0,
                    max_similarity_reward=2.0,
                    reward_mode="note_fraction",
                    simple_reward_note="G",
                ),
            )

    def test_score_rollout_payloads_supports_simple_note_count_mode(self):
        prompt = "X:1\nT:Simple note count reward test\nM:3/4\nL:1/8\nK:C\nV:1\n%%score 1\n"
        completion = "[r:0/0][V:1]G G A g|\n"
        target = StructuralTarget(expected_bars=1, expected_structure_bars=1)
        prompt_target = PromptStructuralTarget(
            target=target,
            structure_path="<test>",
            source_key="simple_note_count_test",
        )
        payload = PPORolloutPayload(
            trajectory_index=0,
            rollout_seed=10,
            full_text=prompt + completion,
            generated_patches=_generated_patches_from_text(completion),
            meta={"stop_reason": "target_stream_lines"},
        )

        scored = score_ppo_rollout_payloads(
            prompt=prompt,
            prompt_idx=0,
            prompt_name="simple_note_count_test",
            prompt_target=prompt_target,
            target=target,
            target_stream_lines=1,
            rollout_payloads=[payload],
            reward_config=GoldbergRewardConfig(parse_validation_mode="abc-tokenize"),
            similarity_weights=SimilarityRewardWeights(),
            aria_similarity_ref=None,
            args=SimpleNamespace(
                similarity_chroma_bins=8,
                similarity_band_ratio=0.25,
                similarity_timeout_s=5.0,
                max_similarity_reward=2.0,
                reward_workers=0,
                patch_reward_attribution="terminal",
                reward_mode="note_count",
                simple_reward_note="G",
                simple_reward_max_count=4.0,
                simple_reward_length_unit="patches",
                simple_reward_length_target=160.0,
                simple_reward_scale=2.0,
            ),
            step_idx=0,
            candidate_name_prefix="simple_note_count",
        )

        log = scored.trajectory_logs[0]
        self.assertAlmostEqual(log["reward"], 1.5)
        self.assertEqual(log["reward_breakdown"]["reward_mode"], "note_count")
        self.assertEqual(log["reward_breakdown"]["patch_reward_mode"], "simple_note_count_terminal")
        self.assertAlmostEqual(log["patch_reward_component_sums"]["simple_note_count_reward"], 1.5)
        self.assertAlmostEqual(sum(log["patch_rewards"]), 1.5)

    def test_score_rollout_payloads_supports_simple_note_fraction_mode(self):
        prompt = "X:1\nT:Simple note fraction reward test\nM:3/4\nL:1/8\nK:C\nV:1\n%%score 1\n"
        completion = "[r:0/0][V:1]G A [I:staff -1]g C|\n"
        target = StructuralTarget(expected_bars=1, expected_structure_bars=1)
        prompt_target = PromptStructuralTarget(
            target=target,
            structure_path="<test>",
            source_key="simple_note_fraction_test",
        )
        payload = PPORolloutPayload(
            trajectory_index=0,
            rollout_seed=10,
            full_text=prompt + completion,
            generated_patches=_generated_patches_from_text(completion),
            meta={"stop_reason": "target_stream_lines"},
        )

        scored = score_ppo_rollout_payloads(
            prompt=prompt,
            prompt_idx=0,
            prompt_name="simple_note_fraction_test",
            prompt_target=prompt_target,
            target=target,
            target_stream_lines=1,
            rollout_payloads=[payload],
            reward_config=GoldbergRewardConfig(parse_validation_mode="abc-tokenize"),
            similarity_weights=SimilarityRewardWeights(),
            aria_similarity_ref=None,
            args=SimpleNamespace(
                similarity_chroma_bins=8,
                similarity_band_ratio=0.25,
                similarity_timeout_s=5.0,
                max_similarity_reward=2.0,
                reward_workers=0,
                patch_reward_attribution="terminal",
                reward_mode="note_fraction",
                simple_reward_note="G",
                simple_reward_max_count=4.0,
                simple_reward_length_unit="patches",
                simple_reward_length_target=160.0,
                simple_reward_scale=2.0,
            ),
            step_idx=0,
            candidate_name_prefix="simple_note_fraction",
        )

        log = scored.trajectory_logs[0]
        self.assertAlmostEqual(log["reward"], 1.0)
        self.assertEqual(log["reward_breakdown"]["reward_mode"], "note_fraction")
        self.assertEqual(log["reward_breakdown"]["patch_reward_mode"], "simple_note_fraction_terminal")
        self.assertAlmostEqual(log["patch_reward_component_sums"]["simple_note_fraction_reward"], 1.0)
        self.assertAlmostEqual(sum(log["patch_rewards"]), 1.0)

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
        self.assertEqual(replay.token_counts.shape, (3,))
        self.assertEqual(replay.token_logprobs.shape, (PATCH_SIZE * 3,))
        self.assertEqual(
            replay.token_log_dists.shape,
            (PATCH_SIZE * 3, model.char_level_decoder.base.transformer.wte.weight.shape[0]),
        )
        self.assertTrue(torch.isfinite(replay.logprobs).all())
        self.assertTrue(torch.isfinite(replay.values).all())
        self.assertTrue(torch.isfinite(replay.token_log_dists).all())
        self.assertTrue(torch.allclose(replay.token_log_dists.logsumexp(dim=-1), torch.zeros(PATCH_SIZE * 3), atol=1e-5))
        split_tokens = torch.split(replay.token_logprobs, replay.token_counts.detach().cpu().tolist())
        self.assertTrue(torch.allclose(torch.stack([item.sum() for item in split_tokens]), replay.logprobs))

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

    def test_batched_hidden_state_replay_matches_serial_for_multiple_trajectories(self):
        torch.manual_seed(0)
        model = _tiny_notagen()
        model.eval()
        prompt_ids = [3 + (i % 80) for i in range(PATCH_SIZE * 3 + 5)]
        first_patch_len = PATCH_SIZE - 5
        generated_batch = [
            [
                [11 + (i % 50) for i in range(first_patch_len)],
                [23 + (i % 50) for i in range(PATCH_SIZE)],
                [31 + (i % 40) for i in range(PATCH_SIZE)],
            ],
            [
                [17 + (i % 45) for i in range(first_patch_len)],
                [29 + (i % 35) for i in range(PATCH_SIZE)],
            ],
            [],
        ]

        serial = [
            trajectory_patch_hidden_states(
                model,
                prompt_ids,
                generated_patches,
                precision="fp32",
                replay_context_patches=4,
                target_chunk_patches=1,
                detach_policy=True,
            )
            for generated_patches in generated_batch
        ]
        batched = batched_trajectory_patch_hidden_states(
            model,
            prompt_ids,
            generated_batch,
            precision="fp32",
            replay_context_patches=4,
            target_chunk_patches=1,
            detach_policy=True,
        )

        self.assertEqual(len(batched), len(serial))
        for serial_hidden, batched_hidden in zip(serial, batched, strict=True):
            self.assertEqual(batched_hidden.shape, serial_hidden.shape)
            self.assertFalse(batched_hidden.requires_grad)
            self.assertTrue(torch.allclose(batched_hidden, serial_hidden, atol=1e-6))

    def test_value_head_training_on_detached_returns_reduces_error(self):
        torch.manual_seed(0)
        value_head = PatchValueHead(32, value_hidden_size=0)
        optimizer = torch.optim.AdamW(value_head.parameters(), lr=5e-2)
        hidden_state_tensors = [
            torch.randn(6, 32),
            torch.randn(4, 32),
        ]
        for hidden_states in hidden_state_tensors:
            hidden_states.requires_grad_(False)
        return_tensors = [
            torch.zeros(6),
            torch.zeros(4),
        ]
        args = SimpleNamespace(
            normalize_value_loss=False,
            value_loss_eps=1e-6,
            value_loss_scale_min=1e-6,
            max_grad_norm=10.0,
        )
        targets = torch.cat(return_tensors)
        with torch.no_grad():
            before_values = value_head(torch.cat(hidden_state_tensors))
            before_loss = torch.mean((before_values - targets) ** 2).item()

        log = train_value_head_on_detached_returns(
            value_head=value_head,
            value_optimizer=optimizer,
            hidden_state_tensors=hidden_state_tensors,
            return_tensors=return_tensors,
            epochs=20,
            args=args,
        )

        with torch.no_grad():
            after_values = value_head(torch.cat(hidden_state_tensors))
            after_loss = torch.mean((after_values - targets) ** 2).item()
        self.assertEqual(log["epochs"], 20)
        self.assertTrue(log["hidden_states_detached"])
        self.assertEqual(log["hidden_state_source"], "current_policy_after_ppo_update")
        self.assertEqual(log["patch_count"], 10)
        self.assertLess(after_loss, before_loss)

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
        self.assertEqual(replay.token_counts.detach().cpu().tolist(), [first_patch_len, PATCH_SIZE, PATCH_SIZE])
        self.assertEqual(
            replay.token_log_dists.shape,
            (first_patch_len + PATCH_SIZE * 2, model.char_level_decoder.base.transformer.wte.weight.shape[0]),
        )
        self.assertTrue(torch.isfinite(replay.logprobs).all())
        self.assertTrue(torch.isfinite(replay.values).all())
        self.assertTrue(torch.isfinite(replay.token_log_dists).all())
        split_tokens = torch.split(replay.token_logprobs, replay.token_counts.detach().cpu().tolist())
        self.assertTrue(torch.allclose(torch.stack([item.sum() for item in split_tokens]), replay.logprobs))

    def test_batched_patch_replay_matches_serial_for_multiple_trajectories(self):
        torch.manual_seed(0)
        model = _tiny_notagen()
        value_head = PatchValueHead(32)
        model.eval()
        value_head.eval()
        prompt_ids = [3 + (i % 80) for i in range(PATCH_SIZE * 4)]
        generated_batch = [
            [
                [11 + ((patch_idx * 17 + i) % 50) for i in range(PATCH_SIZE)]
                for patch_idx in range(3)
            ],
            [
                [19 + ((patch_idx * 13 + i) % 45) for i in range(PATCH_SIZE)]
                for patch_idx in range(5)
            ],
            [],
        ]

        serial = [
            trajectory_patch_logprobs_values(
                model,
                value_head,
                prompt_ids,
                generated_patches,
                precision="fp32",
                replay_context_patches=4,
                target_chunk_patches=2,
            )
            for generated_patches in generated_batch
        ]
        batched = batched_trajectory_patch_logprobs_values(
            model,
            value_head,
            prompt_ids,
            generated_batch,
            precision="fp32",
            replay_context_patches=4,
            target_chunk_patches=2,
        )

        self.assertEqual(len(batched), len(serial))
        for serial_replay, batched_replay in zip(serial, batched, strict=True):
            self.assertEqual(batched_replay.logprobs.shape, serial_replay.logprobs.shape)
            self.assertEqual(batched_replay.values.shape, serial_replay.values.shape)
            self.assertEqual(batched_replay.token_logprobs.shape, serial_replay.token_logprobs.shape)
            self.assertEqual(batched_replay.token_log_dists.shape, serial_replay.token_log_dists.shape)
            self.assertEqual(batched_replay.token_counts.shape, serial_replay.token_counts.shape)
            self.assertTrue(torch.allclose(batched_replay.logprobs, serial_replay.logprobs, atol=1e-5))
            self.assertTrue(torch.allclose(batched_replay.values, serial_replay.values, atol=1e-6))
            self.assertTrue(torch.allclose(batched_replay.token_logprobs, serial_replay.token_logprobs, atol=1e-5))
            self.assertTrue(torch.allclose(batched_replay.token_log_dists, serial_replay.token_log_dists, atol=1e-5))
            self.assertTrue(torch.equal(batched_replay.token_counts, serial_replay.token_counts))

    def test_batched_patch_replay_handles_short_terminal_patches_without_extra_tokens(self):
        torch.manual_seed(0)
        model = _tiny_notagen()
        value_head = PatchValueHead(32)
        model.eval()
        value_head.eval()
        prompt_ids = [3 + (i % 80) for i in range(PATCH_SIZE * 4)]
        eos = model.eos_token_id
        generated_batch = [
            [[11, eos]],
            [[19 + ((patch_idx * 13 + i) % 45) for i in range(PATCH_SIZE)] for patch_idx in range(2)]
            + [[23, eos]],
        ]

        serial = [
            trajectory_patch_logprobs_values(
                model,
                value_head,
                prompt_ids,
                generated_patches,
                precision="fp32",
                replay_context_patches=4,
                target_chunk_patches=1,
            )
            for generated_patches in generated_batch
        ]
        batched = batched_trajectory_patch_logprobs_values(
            model,
            value_head,
            prompt_ids,
            generated_batch,
            precision="fp32",
            replay_context_patches=4,
            target_chunk_patches=1,
            replay_batch_size=2,
        )

        self.assertEqual([item.token_counts.detach().cpu().tolist() for item in batched], [[2], [PATCH_SIZE, PATCH_SIZE, 2]])
        for serial_replay, batched_replay in zip(serial, batched, strict=True):
            self.assertEqual(batched_replay.logprobs.shape, serial_replay.logprobs.shape)
            self.assertEqual(batched_replay.values.shape, serial_replay.values.shape)
            self.assertEqual(batched_replay.token_logprobs.shape, serial_replay.token_logprobs.shape)
            self.assertEqual(batched_replay.token_log_dists.shape, serial_replay.token_log_dists.shape)
            self.assertTrue(torch.allclose(batched_replay.logprobs, serial_replay.logprobs, atol=1e-5))
            self.assertTrue(torch.allclose(batched_replay.values, serial_replay.values, atol=1e-6))
            self.assertTrue(torch.allclose(batched_replay.token_logprobs, serial_replay.token_logprobs, atol=1e-5))
            self.assertTrue(torch.allclose(batched_replay.token_log_dists, serial_replay.token_log_dists, atol=1e-5))

    def test_distribution_only_replay_matches_generic_replay(self):
        torch.manual_seed(0)
        model = _tiny_notagen()
        value_head = PatchValueHead(32)
        model.eval()
        value_head.eval()
        prompt_ids = [3 + (i % 80) for i in range(PATCH_SIZE * 3 + 5)]
        first_patch_len = PATCH_SIZE - 5
        generated_batch = [
            [
                [11 + (i % 50) for i in range(first_patch_len)],
                [23 + (i % 50) for i in range(PATCH_SIZE)],
                [31 + (i % 40) for i in range(PATCH_SIZE)],
            ],
            [
                [17 + (i % 45) for i in range(first_patch_len)],
                [29 + (i % 35) for i in range(PATCH_SIZE)],
            ],
            [],
        ]

        generic = batched_trajectory_patch_logprobs_values(
            model,
            value_head,
            prompt_ids,
            generated_batch,
            precision="fp32",
            replay_context_patches=4,
            target_chunk_patches=1,
        )
        distribution_only = batched_trajectory_token_log_dists(
            model,
            prompt_ids,
            generated_batch,
            precision="fp32",
            replay_context_patches=4,
            target_chunk_patches=1,
        )

        self.assertEqual(len(distribution_only), len(generic))
        for distribution_replay, generic_replay in zip(distribution_only, generic, strict=True):
            self.assertEqual(distribution_replay.token_log_dists.shape, generic_replay.token_log_dists.shape)
            self.assertEqual(distribution_replay.token_counts.shape, generic_replay.token_counts.shape)
            self.assertTrue(torch.equal(distribution_replay.token_counts, generic_replay.token_counts))
            self.assertTrue(torch.allclose(distribution_replay.token_log_dists, generic_replay.token_log_dists, atol=1e-5))

    def test_batched_value_replay_matches_serial_for_multiple_trajectories(self):
        torch.manual_seed(0)
        model = _tiny_notagen()
        value_head = PatchValueHead(32, value_hidden_size=16)
        model.eval()
        value_head.eval()
        prompt_ids = [3 + (i % 80) for i in range(PATCH_SIZE * 4)]
        generated_batch = [
            [
                [11 + ((patch_idx * 17 + i) % 50) for i in range(PATCH_SIZE)]
                for patch_idx in range(3)
            ],
            [
                [19 + ((patch_idx * 13 + i) % 45) for i in range(PATCH_SIZE)]
                for patch_idx in range(5)
            ],
            [],
        ]

        serial = [
            trajectory_patch_values(
                model,
                value_head,
                prompt_ids,
                generated_patches,
                precision="fp32",
                replay_context_patches=4,
                target_chunk_patches=2,
            )
            for generated_patches in generated_batch
        ]
        batched = batched_trajectory_patch_values(
            model,
            value_head,
            prompt_ids,
            generated_batch,
            precision="fp32",
            replay_context_patches=4,
            target_chunk_patches=2,
        )

        self.assertEqual(len(batched), len(serial))
        for serial_values, batched_values in zip(serial, batched, strict=True):
            self.assertEqual(batched_values.shape, serial_values.shape)
            self.assertTrue(torch.allclose(batched_values, serial_values, atol=1e-6))

    def test_batched_patch_replay_matches_serial_with_unaligned_prompt_prefix(self):
        torch.manual_seed(0)
        model = _tiny_notagen()
        value_head = PatchValueHead(32)
        model.eval()
        value_head.eval()
        prompt_ids = [3 + (i % 80) for i in range(PATCH_SIZE * 3 + 5)]
        first_patch_len = PATCH_SIZE - 5
        generated_batch = [
            [
                [11 + (i % 50) for i in range(first_patch_len)],
                [23 + (i % 50) for i in range(PATCH_SIZE)],
                [31 + (i % 40) for i in range(PATCH_SIZE)],
            ],
            [
                [17 + (i % 45) for i in range(first_patch_len)],
                [29 + (i % 35) for i in range(PATCH_SIZE)],
            ],
        ]

        serial = [
            trajectory_patch_logprobs_values(
                model,
                value_head,
                prompt_ids,
                generated_patches,
                precision="fp32",
                replay_context_patches=4,
                target_chunk_patches=1,
            )
            for generated_patches in generated_batch
        ]
        batched = batched_trajectory_patch_logprobs_values(
            model,
            value_head,
            prompt_ids,
            generated_batch,
            precision="fp32",
            replay_context_patches=4,
            target_chunk_patches=1,
        )

        for serial_replay, batched_replay in zip(serial, batched, strict=True):
            self.assertEqual(batched_replay.logprobs.shape, serial_replay.logprobs.shape)
            self.assertEqual(batched_replay.values.shape, serial_replay.values.shape)
            self.assertEqual(batched_replay.token_logprobs.shape, serial_replay.token_logprobs.shape)
            self.assertEqual(batched_replay.token_log_dists.shape, serial_replay.token_log_dists.shape)
            self.assertEqual(batched_replay.token_counts.shape, serial_replay.token_counts.shape)
            self.assertTrue(torch.allclose(batched_replay.logprobs, serial_replay.logprobs, atol=1e-5))
            self.assertTrue(torch.allclose(batched_replay.values, serial_replay.values, atol=1e-6))
            self.assertTrue(torch.allclose(batched_replay.token_logprobs, serial_replay.token_logprobs, atol=1e-5))
            self.assertTrue(torch.allclose(batched_replay.token_log_dists, serial_replay.token_log_dists, atol=1e-5))
            self.assertTrue(torch.equal(batched_replay.token_counts, serial_replay.token_counts))

    def test_ppo_clipped_loss_is_finite(self):
        old_logprobs = torch.tensor([-4.0, -3.0, -2.0])
        new_logprobs = torch.tensor([-3.9, -3.2, -2.1], requires_grad=True)
        old_values = torch.tensor([0.2, 0.1, -0.1])
        values = torch.tensor([0.3, 0.0, -0.2], requires_grad=True)
        value_targets = terminal_returns(1.5, 3, gamma=1.0, device=torch.device("cpu"))
        advantages = value_targets - old_values
        token_counts = torch.ones(3, dtype=torch.long)

        payload = ppo_clipped_loss(
            new_logprobs=new_logprobs,
            old_logprobs=old_logprobs,
            values=values,
            old_values=old_values,
            advantages=advantages,
            value_targets=value_targets,
            clip_range=0.2,
            value_loss_coef=0.5,
            policy_patch_indices=token_patch_indices_from_counts(token_counts),
            value_token_counts=token_counts,
        )

        self.assertTrue(torch.isfinite(payload.loss))
        payload.loss.backward()
        self.assertIsNotNone(new_logprobs.grad)
        self.assertIsNotNone(values.grad)

    def test_ppo_token_clipped_loss_repeats_patch_advantages(self):
        old_token_logprobs = torch.tensor([-4.0, -3.0, -2.5, -2.0, -1.5, -1.0])
        new_token_logprobs = torch.tensor([-3.9, -3.2, -2.4, -2.2, -1.4, -1.1], requires_grad=True)
        token_counts = torch.tensor([2, 1, 3])
        policy_patch_indices = token_patch_indices_from_counts(token_counts)
        old_values = torch.tensor([0.2, 0.1, -0.1])
        values = torch.tensor([0.3, 0.0, -0.2], requires_grad=True)
        value_targets = torch.tensor([1.0, 0.2, -0.3])
        advantages = value_targets - old_values

        normalized_advantages, mean, std = normalize_advantages_token_weighted(advantages, token_counts)
        repeated_advantages = normalized_advantages[policy_patch_indices]
        self.assertAlmostEqual(
            float(repeated_advantages.mean()),
            0.0,
            places=6,
        )
        self.assertAlmostEqual(
            float(repeated_advantages.std(unbiased=False)),
            1.0,
            places=6,
        )

        payload = ppo_clipped_loss(
            new_logprobs=new_token_logprobs,
            old_logprobs=old_token_logprobs,
            values=values,
            old_values=old_values,
            advantages=advantages,
            value_targets=value_targets,
            clip_range=0.2,
            value_loss_coef=0.5,
            normalized_advantages=normalized_advantages,
            advantages_mean=mean,
            advantages_std=std,
            policy_patch_indices=policy_patch_indices,
        )

        log_ratio = new_token_logprobs - old_token_logprobs
        ratio = torch.exp(log_ratio)
        expected_policy_loss = -torch.minimum(
            ratio * repeated_advantages,
            torch.clamp(ratio, 0.8, 1.2) * repeated_advantages,
        ).mean()
        repeated_values = values[policy_patch_indices]
        repeated_value_targets = value_targets[policy_patch_indices]
        expected_value_loss = torch.nn.functional.mse_loss(repeated_values, repeated_value_targets)
        self.assertTrue(torch.allclose(payload.policy_loss, expected_policy_loss, atol=1e-6))
        self.assertTrue(torch.allclose(payload.value_loss, expected_value_loss, atol=1e-6))
        payload.loss.backward()
        self.assertIsNotNone(new_token_logprobs.grad)
        self.assertIsNotNone(values.grad)

    def test_ppo_reference_kl_loss_uses_full_token_distribution(self):
        policy_logits = torch.tensor(
            [
                [2.0, 0.5, -1.0, 0.0],
                [-0.5, 1.0, 0.25, 2.0],
            ],
            requires_grad=True,
        )
        reference_logits = torch.tensor(
            [
                [0.0, 1.0, 0.5, -0.5],
                [1.5, -0.25, 0.0, 0.5],
            ]
        )
        policy_log_dists = torch.log_softmax(policy_logits, dim=-1)
        reference_log_dists = torch.log_softmax(reference_logits, dim=-1)
        selected_tokens = torch.tensor([0, 3])
        new_logprobs = policy_log_dists[torch.arange(selected_tokens.numel()), selected_tokens]
        old_logprobs = new_logprobs.detach().clone()
        token_counts = torch.ones(selected_tokens.numel(), dtype=torch.long)

        payload = ppo_clipped_loss(
            new_logprobs=new_logprobs,
            old_logprobs=old_logprobs,
            values=torch.zeros(2),
            old_values=torch.zeros(2),
            advantages=torch.zeros(2),
            value_targets=torch.zeros(2),
            clip_range=0.2,
            value_loss_coef=0.0,
            normalize_advantage=False,
            policy_patch_indices=token_patch_indices_from_counts(token_counts),
            value_token_counts=token_counts,
            new_log_dists=policy_log_dists,
            old_log_dists=policy_log_dists.detach().clone(),
            reference_log_dists=reference_log_dists,
            reference_kl_coef=0.25,
        )

        expected_kl = exact_categorical_kl(policy_log_dists, reference_log_dists)
        self.assertTrue(torch.allclose(payload.reference_exact_kl, expected_kl, atol=1e-7))
        self.assertTrue(torch.allclose(payload.reference_kl_loss, 0.25 * expected_kl, atol=1e-7))
        self.assertAlmostEqual(float(payload.old_policy_exact_kl.detach()), 0.0, places=6)
        payload.loss.backward()
        self.assertIsNotNone(policy_logits.grad)
        self.assertGreater(float(policy_logits.grad.abs().sum()), 0.0)

    def test_logprob_advantage_diagnostics_reports_split_sign_hit_rates(self):
        diagnostics = logprob_advantage_diagnostics(
            old_logprobs=torch.zeros(5),
            post_step_logprobs=torch.tensor([0.1, -0.2, 0.3, 0.4, -0.5]),
            raw_advantages=torch.tensor([1.0, 2.0, -1.0, -2.0, -3.0]),
            normalized_advantages=torch.tensor([0.5, 1.0, -0.25, -0.75, -1.0]),
            patch_rewards=torch.zeros(5),
            returns=torch.zeros(5),
            value_targets=torch.zeros(5),
            old_values=torch.zeros(5),
            trajectory_lengths=[2, 3],
            trajectory_logs=[{"trajectory_index": 0}, {"trajectory_index": 1}],
            clip_range=0.2,
        )

        self.assertAlmostEqual(
            diagnostics["positive_advantage_positive_log_ratio_fraction"],
            0.5,
        )
        self.assertAlmostEqual(
            diagnostics["negative_advantage_negative_log_ratio_fraction"],
            1.0 / 3.0,
        )
        self.assertAlmostEqual(diagnostics["sign_alignment_fraction"], 0.4)
        self.assertAlmostEqual(
            diagnostics["per_trajectory"][0]["positive_advantage_positive_log_ratio_fraction"],
            0.5,
        )
        self.assertIsNone(diagnostics["per_trajectory"][0]["negative_advantage_negative_log_ratio_fraction"])
        self.assertAlmostEqual(
            diagnostics["per_trajectory"][1]["negative_advantage_negative_log_ratio_fraction"],
            1.0 / 3.0,
        )
        self.assertIn("advantage_summary", diagnostics)
        self.assertAlmostEqual(diagnostics["advantage_summary"]["positive_fraction"], 0.4)
        self.assertAlmostEqual(diagnostics["advantage_summary"]["negative_fraction"], 0.6)

    def test_logprob_advantage_diagnostics_reports_relative_patch_position_bins(self):
        diagnostics = logprob_advantage_diagnostics(
            old_logprobs=torch.zeros(5),
            post_step_logprobs=torch.tensor([0.1, -0.2, 0.3, 0.4, -0.5]),
            raw_advantages=torch.tensor([1.0, 2.0, -1.0, -2.0, -3.0]),
            normalized_advantages=torch.tensor([0.5, 1.0, -0.25, -0.75, -1.0]),
            patch_rewards=torch.tensor([0.2, 0.1, -0.1, 0.0, 0.3]),
            returns=torch.zeros(5),
            value_targets=torch.zeros(5),
            old_values=torch.zeros(5),
            trajectory_lengths=[2, 3],
            trajectory_logs=[{"trajectory_index": 0}, {"trajectory_index": 1}],
            clip_range=0.2,
            position_bins=2,
        )

        bins = diagnostics["by_relative_patch_position"]
        self.assertEqual(len(bins), 2)
        self.assertEqual(bins[0]["count"], 2)
        self.assertEqual(bins[1]["count"], 3)
        self.assertAlmostEqual(bins[0]["positive_advantage_positive_log_ratio_fraction"], 1.0)
        self.assertAlmostEqual(bins[0]["negative_advantage_negative_log_ratio_fraction"], 0.0)
        self.assertAlmostEqual(bins[1]["positive_advantage_positive_log_ratio_fraction"], 0.0)
        self.assertAlmostEqual(bins[1]["negative_advantage_negative_log_ratio_fraction"], 0.5)

    def test_advantage_distribution_summary_reports_trajectory_sums(self):
        summary = advantage_distribution_summary(
            raw_advantages=torch.tensor([1.0, 2.0, -1.0, -3.0]),
            normalized_advantages=torch.tensor([0.5, 1.0, -0.5, -1.0]),
            trajectory_lengths=[2, 2],
        )

        self.assertAlmostEqual(summary["positive_fraction"], 0.5)
        self.assertAlmostEqual(summary["negative_fraction"], 0.5)
        self.assertAlmostEqual(summary["positive_mean"], 1.5)
        self.assertAlmostEqual(summary["negative_mean"], -2.0)
        self.assertAlmostEqual(summary["by_trajectory"]["raw_sum"]["mean"], -0.5)
        self.assertAlmostEqual(summary["by_trajectory"]["raw_sum"]["min"], -4.0)
        self.assertAlmostEqual(summary["by_trajectory"]["raw_sum"]["max"], 3.0)

    def test_summarize_ppo_advantages_reads_result_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result_path = Path(tmpdir) / "result.json"
            result_path.write_text(
                json.dumps(
                    {
                        "steps": [
                            {
                                "step": 1,
                                "reward_mean": 6.0,
                                "fixed_eval": {"reward_mean": 6.1},
                                "post_step_approx_kl": 0.01,
                                "post_step_clip_fraction": 0.02,
                                "advantage_summary": {
                                    "raw": {"mean": 0.5, "std": 0.25, "min": -1.0, "p05": -0.5, "p50": 0.4, "p95": 1.1, "max": 1.2},
                                    "normalized": {"mean": 0.0, "std": 1.0},
                                    "positive_fraction": 0.75,
                                    "negative_fraction": 0.25,
                                    "zero_fraction": 0.0,
                                    "positive_mean": 0.8,
                                    "negative_mean": -0.4,
                                    "abs_mean": 0.7,
                                    "by_trajectory": {
                                        "raw_mean": {"mean": 0.2, "std": 0.1},
                                        "raw_sum": {"mean": 1.0, "std": 0.5},
                                    },
                                },
                                "logprob_advantage_diagnostics": {
                                    "advantage_log_ratio_correlation": 0.3,
                                    "sign_alignment_fraction": 0.6,
                                },
                            }
                        ]
                    }
                )
            )

            rows = summarize_steps(result_path)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["step"], 1)
        self.assertAlmostEqual(rows[0]["raw_advantage_mean"], 0.5)
        self.assertAlmostEqual(rows[0]["positive_advantage_fraction"], 0.75)
        self.assertAlmostEqual(rows[0]["trajectory_raw_advantage_sum_mean"], 1.0)

    def test_per_patch_diagnostic_records_include_position_and_raw_fields(self):
        records = per_patch_diagnostic_records(
            old_logprobs=torch.zeros(3),
            post_step_logprobs=torch.tensor([0.1, -0.2, 0.3]),
            raw_advantages=torch.tensor([1.0, -1.0, 2.0]),
            normalized_advantages=torch.tensor([0.5, -0.5, 1.0]),
            patch_rewards=torch.tensor([0.2, -0.1, 0.3]),
            returns=torch.tensor([0.4, 0.2, 0.3]),
            value_targets=torch.tensor([0.5, 0.1, 0.2]),
            old_values=torch.tensor([0.0, 0.0, 0.0]),
            trajectory_lengths=[1, 2],
            component_rewards={
                "structural_total_reward": torch.tensor([0.2, -0.1, 0.3]),
                    "aria_chroma_harmonic_hist_active": torch.tensor([0.0, 0.0, 0.4]),
            },
            component_lambda_returns={
                "structural_total_reward": torch.tensor([0.2, 0.185, 0.3]),
                    "aria_chroma_harmonic_hist_active": torch.tensor([0.0, 0.38, 0.4]),
            },
        )

        self.assertEqual(len(records), 3)
        self.assertEqual(records[0]["trajectory_index"], 0)
        self.assertEqual(records[0]["trajectory_patch_index"], 0)
        self.assertAlmostEqual(records[0]["trajectory_relative_position"], 0.0)
        self.assertEqual(records[1]["trajectory_index"], 1)
        self.assertEqual(records[1]["trajectory_patch_index"], 0)
        self.assertAlmostEqual(records[1]["trajectory_relative_position"], 0.0)
        self.assertEqual(records[2]["trajectory_index"], 1)
        self.assertEqual(records[2]["trajectory_patch_index"], 1)
        self.assertAlmostEqual(records[2]["trajectory_relative_position"], 1.0)
        self.assertAlmostEqual(records[2]["post_step_log_ratio"], 0.3)
        self.assertAlmostEqual(records[2]["raw_advantage"], 2.0)
        self.assertAlmostEqual(records[2]["structural_total_reward__reward"], 0.3)
        self.assertAlmostEqual(records[2]["structural_total_reward__lambda_return"], 0.3)
        self.assertAlmostEqual(records[1]["aria_chroma_harmonic_hist_active__reward"], 0.0)
        self.assertAlmostEqual(records[1]["aria_chroma_harmonic_hist_active__lambda_return"], 0.38)

    def test_component_reward_tensors_include_derived_reward_families(self):
        traces = [
            PatchRewardTrace(
                rewards=[1.2, 2.3],
                prefix_totals=[1.2, 3.5],
                final_score=RewardScore(total=3.5, breakdown={}),
                component_rewards={
                    "parse_reward": [0.0, 0.25],
                    "bar_count_reward": [0.5, 0.5],
                    "aria_chroma_harmonic_hist_active": [0.0, 1.0],
                    "aria_chroma_top_hist_active": [0.3, 0.0],
                    "aria_harmony_root_dtw_active": [0.2, 0.0],
                    "aria_harmony_aligned_bass_active": [0.1, 0.0],
                    "other_residual": [0.1, 0.55],
                },
                component_prefix_totals={},
            )
        ]

        tensors = component_reward_tensors(traces, device=torch.device("cpu"))

        self.assertTrue(torch.allclose(tensors["parse_reward"], torch.tensor([0.0, 0.25])))
        self.assertTrue(torch.allclose(tensors["structural_total_reward"], torch.tensor([0.5, 0.75])))
        self.assertTrue(torch.allclose(tensors["aria_harmony_dtw_active"], torch.tensor([0.2, 0.0])))
        self.assertTrue(torch.allclose(tensors["aria_harmony_aligned_active"], torch.tensor([0.1, 0.0])))
        self.assertTrue(torch.allclose(tensors["aria_chroma_top_hist_active"], torch.tensor([0.3, 0.0])))
        self.assertTrue(torch.allclose(tensors["active_similarity_reward"], torch.tensor([0.6, 1.0])))
        self.assertTrue(torch.allclose(tensors["total_reward"], torch.tensor([1.2, 2.3])))

        lambda_returns = component_lambda_return_tensors(
            traces,
            gamma=1.0,
            gae_lambda=0.5,
            device=torch.device("cpu"),
        )
        self.assertTrue(torch.allclose(lambda_returns["structural_total_reward"], torch.tensor([0.875, 0.75])))

    def test_ppo_microbatch_loss_matches_full_batch_normalization(self):
        token_counts = torch.tensor([1, 3, 2, 1, 4])
        policy_patch_indices = token_patch_indices_from_counts(token_counts)
        old_logprobs = torch.tensor([-4.0, -3.0, -2.8, -2.6, -2.0, -1.9, -2.5, -3.5, -3.4, -3.3, -3.2])
        new_logprobs = torch.tensor(
            [-3.9, -3.2, -2.7, -2.5, -2.1, -1.8, -2.7, -3.4, -3.5, -3.1, -3.0],
            requires_grad=True,
        )
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
            policy_patch_indices=policy_patch_indices,
        )

        normalized_advantages, adv_mean, adv_std = normalize_advantages_token_weighted(advantages, token_counts)
        value_loss_scale = torch.clamp(torch.repeat_interleave(value_targets, token_counts).std(unbiased=False), min=1.0)
        weighted = {}
        token_offsets = [0, 1, 4, 6, 7, 11]
        for start, end in [(0, 2), (2, 5)]:
            token_start = token_offsets[start]
            token_end = token_offsets[end]
            payload = ppo_clipped_loss(
                new_logprobs=new_logprobs[token_start:token_end],
                old_logprobs=old_logprobs[token_start:token_end],
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
                policy_patch_indices=token_patch_indices_from_counts(token_counts[start:end]),
            )
            token_weight = (token_end - token_start) / int(token_counts.sum())
            for name in ("policy_loss", "entropy_loss", "approx_kl", "clip_fraction"):
                weighted[name] = weighted.get(name, torch.zeros(())) + getattr(payload, name).detach() * token_weight
            for name in ("value_loss", "raw_value_loss"):
                weighted[name] = weighted.get(name, torch.zeros(())) + getattr(payload, name).detach() * token_weight

        weighted["loss"] = weighted["policy_loss"] + 0.5 * weighted["value_loss"] + weighted["entropy_loss"]

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

    def test_terminal_patch_reward_attribution_matches_final_score(self):
        prompt = "X:1\nT:Terminal reward test\nM:4/4\nL:1/4\nK:C\nV:1\n%%score 1\n"
        completion = "[r:0/1][V:1] C D E F |\n[r:1/0][V:1] G A B c |\n"
        generated_patches = _generated_patches_from_text(completion)
        target = StructuralTarget(expected_bars=2, expected_structure_bars=2)
        reward_config = GoldbergRewardConfig(parse_validation_mode="abc-tokenize")
        common_kwargs = {
            "prompt_text": prompt,
            "generated_patches": generated_patches,
            "target": target,
            "reward_config": reward_config,
            "candidate_name": "terminal_reward_test",
            "similarity_weights": SimilarityRewardWeights(),
            "aria_similarity_ref": None,
            "similarity_chroma_bins": 8,
            "similarity_band_ratio": 0.25,
            "similarity_timeout_s": 5.0,
            "max_similarity_reward": 2.0,
        }

        single_pass = patch_rewards_single_pass(**common_kwargs)
        terminal = patch_rewards_terminal(**common_kwargs)

        self.assertGreater(len(generated_patches), 1)
        self.assertAlmostEqual(terminal.final_score.total, single_pass.final_score.total)
        self.assertAlmostEqual(sum(terminal.rewards), single_pass.final_score.total)
        self.assertTrue(all(abs(value) < 1e-7 for value in terminal.rewards[:-1]))
        self.assertAlmostEqual(terminal.rewards[-1], terminal.final_score.total)

    def test_score_rollout_payloads_supports_terminal_patch_reward_attribution(self):
        prompt = "X:1\nT:Terminal rollout score test\nM:4/4\nL:1/4\nK:C\nV:1\n%%score 1\n"
        completion = "[r:0/1][V:1] C D E F |\n[r:1/0][V:1] G A B c |\n"
        generated_patches = _generated_patches_from_text(completion)
        target = StructuralTarget(expected_bars=2, expected_structure_bars=2)
        prompt_target = PromptStructuralTarget(
            target=target,
            structure_path="<test>",
            source_key="terminal_attribution_test",
        )
        payload = PPORolloutPayload(
            trajectory_index=0,
            rollout_seed=123,
            full_text=prompt + completion,
            generated_patches=generated_patches,
            meta={"stop_reason": "target_stream_lines"},
        )

        scored = score_ppo_rollout_payloads(
            prompt=prompt,
            prompt_idx=0,
            prompt_name="terminal_attribution_test",
            prompt_target=prompt_target,
            target=target,
            target_stream_lines=2,
            rollout_payloads=[payload],
            reward_config=GoldbergRewardConfig(parse_validation_mode="abc-tokenize"),
            similarity_weights=SimilarityRewardWeights(),
            aria_similarity_ref=None,
            args=SimpleNamespace(
                similarity_chroma_bins=8,
                similarity_band_ratio=0.25,
                similarity_timeout_s=5.0,
                max_similarity_reward=2.0,
                patch_reward_attribution="terminal",
                reward_workers=0,
            ),
            step_idx=0,
            candidate_name_prefix="terminal_attribution",
        )

        trajectory_log = scored.trajectory_logs[0]
        self.assertEqual(trajectory_log["reward_breakdown"]["patch_reward_mode"], "terminal_total_reward")
        self.assertAlmostEqual(sum(trajectory_log["patch_rewards"]), trajectory_log["reward"])
        self.assertTrue(all(abs(value) < 1e-7 for value in trajectory_log["patch_rewards"][:-1]))
        self.assertAlmostEqual(trajectory_log["patch_rewards"][-1], trajectory_log["reward"])


if __name__ == "__main__":
    unittest.main()
