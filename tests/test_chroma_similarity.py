from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from evaluation.chroma_similarity import chroma_features, load_chroma_feature_set, parse_piece_tonic
from evaluation.harmony_similarity import compare_harmony, harmony_from_text
from evaluation.strict_similarity import STRICT_SYMBOLIC_COMPONENT_Z_KEY
from evaluation.similarity_rewards import (
    SimilarityReference,
    SimilarityRewardWeights,
    finalize_similarity_reward_fields,
    score_similarity_reward,
)


class ChromaSimilarityTests(unittest.TestCase):
    def test_parse_piece_tonic_uses_header_key(self):
        self.assertEqual(parse_piece_tonic("K:G\n[V:1]G|\n"), "G")

    def test_top_and_bass_modes_select_expected_extremes_with_key_normalization(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "triad.abc"
            path.write_text(
                "\n".join(
                    [
                        "X:1",
                        "M:4/4",
                        "L:1/4",
                        "K:G",
                        "V:1",
                        "[V:1][G,Bd]4|",
                    ]
                ),
                encoding="utf-8",
            )

            bass = chroma_features(path, bins=1, mode="bass")
            top = chroma_features(path, bins=1, mode="top")

            self.assertEqual(int(np.argmax(bass.hist)), 0)  # G normalized to tonic C.
            self.assertEqual(int(np.argmax(top.hist)), 7)  # D normalized to G.

    def test_load_chroma_feature_set_matches_legacy_per_mode_computation_exactly(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "piece.abc"
            path.write_text(
                "\n".join(
                    [
                        "X:1",
                        "M:4/4",
                        "L:1/4",
                        "K:G",
                        "V:1",
                        "[V:1][G,Bd]2 [FAc]2|",
                        "[V:1]G A B c|",
                    ]
                ),
                encoding="utf-8",
            )

            legacy = {
                mode: chroma_features(path, bins=8, mode=mode, normalize_key=True)
                for mode in ("full", "bass", "top")
            }
            optimized = load_chroma_feature_set(path, bins=8, normalize_key=True)

            self.assertEqual(set(optimized), set(legacy))
            for mode in ("full", "bass", "top"):
                self.assertTrue(np.array_equal(optimized[mode].hist, legacy[mode].hist), mode)
                self.assertTrue(np.array_equal(optimized[mode].sequence, legacy[mode].sequence), mode)
                self.assertEqual(optimized[mode].frames, legacy[mode].frames)
                self.assertEqual(optimized[mode].duration_quarters, legacy[mode].duration_quarters)
                self.assertEqual(optimized[mode].tonic, legacy[mode].tonic)

    def test_similarity_reward_matches_legacy_chroma_feature_reference_exactly(self):
        with tempfile.TemporaryDirectory() as tmp:
            aria_path = Path(tmp) / "aria.abc"
            aria_path.write_text(
                "\n".join(
                    [
                        "X:1",
                        "M:4/4",
                        "L:1/4",
                        "K:G",
                        "V:1",
                        "[V:1]G A B c|",
                    ]
                ),
                encoding="utf-8",
            )
            candidate = "\n".join(
                [
                    "X:1",
                    "M:4/4",
                    "L:1/4",
                    "K:G",
                    "V:1",
                    "[V:1]G B d g|",
                ]
            )
            reference = SimilarityReference(
                path=aria_path,
                chroma={
                    mode: chroma_features(aria_path, bins=8, mode=mode, normalize_key=True)
                    for mode in ("full", "bass", "top")
                },
                harmony=None,
            )

            def legacy_chroma_feature_set(path: str | Path, *, bins: int = 128, normalize_key: bool = True):
                return {
                    mode: chroma_features(path, bins=bins, mode=mode, normalize_key=normalize_key)
                    for mode in ("full", "bass", "top")
                }

            with patch("evaluation.similarity_rewards.load_chroma_feature_set", side_effect=legacy_chroma_feature_set):
                legacy_payload = score_similarity_reward(
                    prompt_text="",
                    completion_text=candidate,
                    weights=SimilarityRewardWeights(aria_chroma=1.0),
                    aria=reference,
                    variation=None,
                    bins=8,
                    band_ratio=0.25,
                    timeout_s=5.0,
                )
            optimized_payload = score_similarity_reward(
                prompt_text="",
                completion_text=candidate,
                weights=SimilarityRewardWeights(aria_chroma=1.0),
                aria=SimilarityReference(
                    path=aria_path,
                    chroma=load_chroma_feature_set(aria_path, bins=8, normalize_key=True),
                    harmony=None,
                ),
                variation=None,
                bins=8,
                band_ratio=0.25,
                timeout_s=5.0,
            )

            self.assertEqual(optimized_payload, legacy_payload)

    def test_top_hist_similarity_weight_is_separate_from_harmonic_hist(self):
        with patch(
            "evaluation.similarity_rewards._chroma_scores",
            return_value={
                "similarity_chroma_valid": True,
                "aria_chroma_harmonic_hist": 0.40,
                "aria_chroma_top_hist": 0.70,
            },
        ) as mocked_chroma:
            payload = score_similarity_reward(
                prompt_text="",
                completion_text="X:1\nM:4/4\nL:1/4\nK:G\nG A B c|\n",
                weights=SimilarityRewardWeights(aria_chroma=2.0, aria_chroma_top=3.0),
                aria=None,
                variation=None,
                bins=8,
                band_ratio=0.25,
                timeout_s=5.0,
            )

        mocked_chroma.assert_called_once()
        self.assertAlmostEqual(payload["similarity_reward"], 2.0 * 0.40 + 3.0 * 0.70)

    def test_aligned_harmony_weights_are_separate_active_similarity_terms(self):
        with patch(
            "evaluation.similarity_rewards._harmony_scores",
            return_value={
                "similarity_harmony_valid": True,
                "aria_harmony_dtw_combined": 0.50,
                "aria_harmony_aligned_root": 0.25,
                "aria_harmony_aligned_bass": 0.75,
                "aria_harmony_aligned_top": 0.40,
            },
        ) as mocked_harmony:
            payload = score_similarity_reward(
                prompt_text="",
                completion_text="X:1\nM:4/4\nL:1/4\nK:G\nG A B c|\n",
                weights=SimilarityRewardWeights(
                    aria_harmony=2.0,
                    aria_harmony_aligned_root=3.0,
                    aria_harmony_aligned_bass=5.0,
                    aria_harmony_aligned_top=7.0,
                ),
                aria=None,
                variation=None,
                bins=8,
                band_ratio=0.25,
                timeout_s=5.0,
            )

        mocked_harmony.assert_called_once()
        self.assertAlmostEqual(payload["similarity_reward"], 2.0 * 0.50 + 3.0 * 0.25 + 5.0 * 0.75 + 7.0 * 0.40)

    def test_aria_harmony_scores_use_written_reference_not_repeat_expanded_reference(self):
        aria_harmony = [{"root": idx, "bass": idx, "quality": "maj"} for idx in range(64)]
        candidate_harmony = [{"root": idx, "bass": idx, "quality": "maj"} for idx in range(32)]

        with (
            patch("evaluation.similarity_rewards.harmony_from_text", return_value=candidate_harmony),
            patch(
                "evaluation.similarity_rewards.compare_harmony",
                return_value={"dtw_combined": 0.75},
            ) as compare,
        ):
            payload = score_similarity_reward(
                prompt_text="",
                completion_text="X:1\nM:4/4\nL:1/4\nK:C\n[V:1]C|\n",
                weights=SimilarityRewardWeights(aria_harmony=1.0),
                aria=SimilarityReference(path=Path("aria.abc"), harmony=aria_harmony),
                variation=None,
                bins=8,
                band_ratio=0.25,
                timeout_s=1.0,
            )

        reference_arg = compare.call_args.args[0]
        self.assertEqual(len(reference_arg), 32)
        self.assertEqual([item["root"] for item in reference_arg[:2]], [0, 1])
        self.assertEqual([item["root"] for item in reference_arg[16:18]], [32, 33])
        self.assertEqual(payload["aria_harmony_dtw_combined"], 0.75)

    def test_legacy_harmony_combined_key_still_scores(self):
        with patch(
            "evaluation.similarity_rewards._harmony_scores",
            return_value={
                "similarity_harmony_valid": True,
                "aria_harmony_combined": 0.50,
            },
        ):
            payload = score_similarity_reward(
                prompt_text="",
                completion_text="X:1\nM:4/4\nL:1/4\nK:G\nG A B c|\n",
                weights=SimilarityRewardWeights(aria_harmony=2.0),
                aria=None,
                variation=None,
                bins=8,
                band_ratio=0.25,
                timeout_s=5.0,
            )

        self.assertAlmostEqual(payload["similarity_reward"], 1.0)

    def test_strict_symbolic_similarity_adds_component_z_reward(self):
        aria_text = "\n".join(
            [
                "X:1",
                "M:4/4",
                "L:1/4",
                "K:C",
                "V:1",
                "[V:1]C E G c|",
                "[V:1]F A c f|",
                "[V:1]G B d g|",
                "[V:1]C E G c|",
            ]
        )
        prompt = "X:1\nM:4/4\nL:1/4\nK:C\nV:1\n"
        completion = "\n".join(
            [
                "[r:0/3][V:1]C E G c|",
                "[r:1/2][V:1]F A c f|",
                "[r:2/1][V:1]G B d g|",
                "[r:3/0][V:1]C E G c|",
            ]
        )

        payload = score_similarity_reward(
            prompt_text=prompt,
            completion_text=completion,
            weights=SimilarityRewardWeights(aria_strict_symbolic=0.5),
            aria=SimilarityReference(path=Path("aria.abc"), harmony=harmony_from_text(aria_text)),
            variation=None,
            bins=8,
            band_ratio=0.25,
            timeout_s=1.0,
        )

        active_key = f"aria_{STRICT_SYMBOLIC_COMPONENT_Z_KEY}"
        self.assertTrue(payload["similarity_harmony_valid"])
        self.assertIn(active_key, payload)
        self.assertIn("aria_strict_aligned_root_bass_global_base_z", payload)
        expected_component_z = (
            0.30 * payload["aria_strict_aligned_root_bass_global_base_z"]
            + 0.25 * payload["aria_strict_dtw_combined_narrow_global_base_z"]
            + 0.20 * payload["aria_strict_root_bass_bigram_weighted_jaccard_global_base_z"]
            + 0.15 * payload["aria_strict_root_bass_fourgram_weighted_jaccard_global_base_z"]
            + 0.10 * payload["aria_strict_cadence_root_bass_global_base_z"]
        )
        self.assertAlmostEqual(payload[active_key], expected_component_z)
        self.assertAlmostEqual(payload["similarity_reward"], 0.5 * payload[active_key])

    def test_strict_symbolic_similarity_does_not_emit_legacy_harmony_scores(self):
        aria_text = "\n".join(
            [
                "X:1",
                "M:4/4",
                "L:1/4",
                "K:C",
                "V:1",
                "[V:1]C E G c|",
                "[V:1]F A c f|",
            ]
        )
        completion = "[r:0/1][V:1]C E G c|\n[r:1/0][V:1]F A c f|\n"

        payload = score_similarity_reward(
            prompt_text="X:1\nM:4/4\nL:1/4\nK:C\nV:1\n",
            completion_text=completion,
            weights=SimilarityRewardWeights(aria_strict_symbolic=1.0),
            aria=SimilarityReference(path=Path("aria.abc"), harmony=harmony_from_text(aria_text)),
            variation=None,
            bins=8,
            band_ratio=0.25,
            timeout_s=1.0,
        )

        self.assertTrue(payload["similarity_harmony_valid"])
        self.assertIn(f"aria_{STRICT_SYMBOLIC_COMPONENT_Z_KEY}", payload)
        self.assertNotIn("aria_harmony_harmony_dtw", payload)
        self.assertNotIn("aria_harmony_aligned_root", payload)
        self.assertNotIn("aria_harmony_combined", payload)

    def test_final_similarity_reward_fields_do_not_apply_diagnostic_gate(self):
        fields = finalize_similarity_reward_fields(
            similarity_payload={"similarity_reward": 1.25},
            structural_total_reward=2.0,
            completion_reward=0.0,
            bar_count_reward=0.0,
            max_similarity_reward=2.0,
        )

        self.assertEqual(fields["similarity_validity_gate"], 0.0)
        self.assertEqual(fields["active_similarity_reward"], 1.25)
        self.assertEqual(fields["effective_similarity_reward"], 1.25)
        self.assertEqual(fields["total_reward"], 3.25)

    def test_harmony_reward_emits_dtw_combined_name(self):
        aria_text = "X:1\nM:4/4\nL:1/4\nK:C\n[V:1][CEG]4|[DFA]4|\n"
        completion_text = "X:1\nM:4/4\nL:1/4\nK:C\n[V:1][CEG]4|[DFA]4|\n"
        aria = SimilarityReference(path=Path("aria.abc"), harmony=harmony_from_text(aria_text))

        payload = score_similarity_reward(
            prompt_text="",
            completion_text=completion_text,
            weights=SimilarityRewardWeights(aria_harmony=1.0),
            aria=aria,
            variation=None,
            bins=8,
            band_ratio=0.25,
            timeout_s=5.0,
        )

        self.assertIn("aria_harmony_dtw_combined", payload)
        self.assertIn("aria_harmony_combined", payload)
        self.assertEqual(payload["aria_harmony_dtw_combined"], payload["aria_harmony_combined"])
        self.assertAlmostEqual(payload["similarity_reward"], payload["aria_harmony_dtw_combined"])

    def test_same_bar_top_alignment_compares_highest_pitch_class_not_root(self):
        reference = harmony_from_text("X:1\nM:4/4\nL:1/4\nK:C\n[V:1][CEG]4|[DFA]4|\n")
        candidate = harmony_from_text("X:1\nM:4/4\nL:1/4\nK:C\n[V:1][CEG]4|[DFAc]4|\n")

        scores = compare_harmony(reference, candidate, band_ratio=0.25)

        self.assertEqual(scores["aligned_root"], 1.0)
        self.assertEqual(scores["aligned_bass"], 1.0)
        self.assertEqual(scores["aligned_top"], 0.5)

    def test_load_chroma_feature_set_reuses_one_note_event_parse_for_all_modes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "piece.abc"
            path.write_text(
                "\n".join(
                    [
                        "X:1",
                        "M:4/4",
                        "L:1/4",
                        "K:G",
                        "V:1",
                        "[V:1]G A B c|",
                    ]
                ),
                encoding="utf-8",
            )
            events = [
                (0.0, 1.0, 67, 7),
                (1.0, 1.0, 71, 11),
                (2.0, 1.0, 74, 2),
            ]

            with patch("evaluation.chroma_similarity._note_events", return_value=(events, 3.0)) as note_events:
                features = load_chroma_feature_set(path, bins=4, normalize_key=True)

            self.assertEqual(note_events.call_count, 1)
            self.assertEqual(set(features), {"full", "bass", "top"})


if __name__ == "__main__":
    unittest.main()
