from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from evaluation.chroma_similarity import chroma_features, load_chroma_feature_set, parse_piece_tonic
from evaluation.similarity_rewards import (
    SimilarityReference,
    SimilarityRewardWeights,
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
