from fractions import Fraction
import time
import unittest

from grpo.notagen_abc_postprocess import expand_notagen_rest_omitted_voice_segments
from grpo.rewards import (
    GoldbergRewardConfig,
    StructuralBarTarget,
    StructuralTarget,
    _bar_count_reward,
    _countdown_reward,
    _extract_header_context,
    _extract_stream_line_features,
    _abc_grammar_metrics,
    _parse_length_multiplier,
    score_candidate_text,
    _total_reward,
    _validated_bar_metrics,
)


class GoldbergRewardTests(unittest.TestCase):
    def test_parse_length_multiplier_accepts_abc_shorthand_fraction(self):
        self.assertEqual(_parse_length_multiplier("/"), Fraction(1, 2))
        self.assertEqual(_parse_length_multiplier("3/"), Fraction(3, 2))

    def test_absurd_duration_skips_music21_parse(self):
        target = StructuralTarget(
            expected_bars=1,
            bars=[
                StructuralBarTarget(
                    bar_index=1,
                    chord_root="C",
                    bass_pitch_class="C",
                    bass_midi=None,
                    cadence_bar=False,
                )
            ],
        )
        text = "\n".join(
            [
                "M:3/4",
                "L:1/8",
                "[r:0/0][V:1]C2222224737|",
            ]
        )

        start = time.perf_counter()
        breakdown = score_candidate_text(text, target, GoldbergRewardConfig(music21_parse_timeout_s=1.0))

        self.assertFalse(breakdown.parse_valid)
        self.assertLess(time.perf_counter() - start, 1.0)

    def test_meter_validation_tracks_inline_meter_changes(self):
        text = "\n".join(
            [
                "M:3/4",
                "L:1/8",
                "[r:0/2][V:1]C2D2E2|[V:2]x6|",
                "[r:1/1][V:1][M:12/8]C2D2E2F2G2A2|[V:2]x12|",
                "[r:2/0][V:1]A2G2F2E2D2C2|[V:2]x12|",
            ]
        )

        header = _extract_header_context(text)
        metrics = _validated_bar_metrics(_extract_stream_line_features(text), header)

        self.assertEqual(header.meter, Fraction(3, 4))
        self.assertEqual(metrics.validated_bars, 3)
        self.assertEqual(metrics.strict_validated_bars, 3)
        self.assertEqual(metrics.meter_alignment_reward, 1.0)
        self.assertEqual(metrics.meter_duration_closeness_reward, 1.0)
        self.assertEqual(metrics.bar_meter_consistency_reward, 1.0)
        self.assertEqual(metrics.strict_bar_meter_consistency_reward, 1.0)

    def test_countdown_reward_accepts_notagen_index_total_tags(self):
        text = "\n".join(
            [
                "M:3/4",
                "L:1/8",
                "[r:1/2][V:1]C2D2E2|",
                "[r:2/2][V:1]E2D2C2|",
            ]
        )

        self.assertEqual(_countdown_reward(_extract_stream_line_features(text)), 1.0)

    def test_meter_validation_rejects_duration_that_mismatches_active_meter(self):
        text = "\n".join(
            [
                "M:3/4",
                "L:1/8",
                "[r:0/1][V:1]C2D2E2|[V:2]x6|",
                "[r:1/0][V:1][M:12/8]C2D2E2|[V:2]x6|",
            ]
        )

        metrics = _validated_bar_metrics(
            _extract_stream_line_features(text),
            _extract_header_context(text),
        )

        self.assertEqual(metrics.validated_bars, 1)
        self.assertEqual(metrics.strict_validated_bars, 1)
        self.assertEqual(metrics.meter_alignment_reward, 0.5)
        self.assertEqual(metrics.meter_duration_closeness_reward, 0.75)
        self.assertEqual(metrics.bar_meter_consistency_reward, 0.5)
        self.assertEqual(metrics.strict_bar_meter_consistency_reward, 0.5)

    def test_strict_meter_validation_requires_all_populated_voices(self):
        text = "\n".join(
            [
                "M:3/4",
                "L:1/8",
                "[r:0/0][V:1]C2D2E2|[V:2]x4|",
            ]
        )

        metrics = _validated_bar_metrics(
            _extract_stream_line_features(text),
            _extract_header_context(text),
        )

        self.assertEqual(metrics.validated_bars, 1)
        self.assertEqual(metrics.strict_validated_bars, 0)
        self.assertEqual(metrics.bar_meter_consistency_reward, 1.0)
        self.assertEqual(metrics.strict_bar_meter_consistency_reward, 0.0)

    def test_grammar_metrics_reject_undeclared_and_unscored_voices(self):
        text = "\n".join(
            [
                "%%score ( 1 )",
                "M:3/4",
                "L:1/8",
                "V:1 treble",
                "[r:0/0][V:1]C2D2E2|[V:4]x6|",
            ]
        )

        metrics = _abc_grammar_metrics(
            _extract_stream_line_features(text),
            _extract_header_context(text),
        )

        self.assertEqual(metrics.voice_declaration_reward, 0.5)
        self.assertEqual(metrics.score_voice_reward, 0.5)

    def test_grammar_metrics_penalize_unmatched_repeat_endings(self):
        text = "\n".join(
            [
                "M:3/4",
                "L:1/8",
                "[r:0/0][V:1]C2D2E2|1",
            ]
        )

        metrics = _abc_grammar_metrics(
            _extract_stream_line_features(text),
            _extract_header_context(text),
        )

        self.assertLess(metrics.repeat_syntax_reward, 1.0)

    def test_expand_notagen_rest_omitted_voice_segments_adds_missing_declared_voices(self):
        text = "\n".join(
            [
                "%%score ( 1 2 ) ( 4 )",
                "M:3/4",
                "L:1/8",
                "V:1 treble",
                "V:2 treble",
                "V:4 bass",
                "[r:0/0][V:1]C2D2E2|[V:4]G,6|",
            ]
        )

        expanded = expand_notagen_rest_omitted_voice_segments(text)

        self.assertIn("[V:2]x6|", expanded)
        self.assertIn("[r:0/0][V:1]C2D2E2|[V:2]x6|[V:4]G,6|", expanded)

    def test_expand_notagen_rest_omitted_voice_segments_uses_inline_meter(self):
        text = "\n".join(
            [
                "%%score ( 1 2 )",
                "M:3/4",
                "L:1/8",
                "V:1 treble",
                "V:2 treble",
                "[r:0/0][V:1][M:2/2]C4D4|",
            ]
        )

        expanded = expand_notagen_rest_omitted_voice_segments(text)

        self.assertIn("[V:2]x8|", expanded)

    def test_expand_notagen_rest_omitted_voice_segments_uses_partial_bar_duration(self):
        text = "\n".join(
            [
                "%%score ( 1 2 )",
                "M:2/2",
                "L:1/8",
                "V:1 treble",
                "V:2 treble",
                "[V:1]C|",
            ]
        )

        expanded = expand_notagen_rest_omitted_voice_segments(text)

        self.assertIn("[V:2]x|", expanded)
        self.assertNotIn("[V:2]x16|", expanded)

    def test_expand_notagen_rest_omitted_voice_segments_preserves_numbered_repeat_endings(self):
        text = "\n".join(
            [
                "%%score ( 1 2 )",
                "M:3/4",
                "L:1/8",
                "V:1 treble",
                "V:2 treble",
                "[V:1]C2D2E2:|2",
            ]
        )

        expanded = expand_notagen_rest_omitted_voice_segments(text)

        self.assertIn("[V:2]x6:|2", expanded)

    def test_validated_bars_dominate_zero_bar_harmonic_guess(self):
        config = GoldbergRewardConfig()

        zero_bar_with_harmony = _total_reward(
            config=config,
            expected_bars=32,
            validated_bars=0,
            parse_reward=1.0,
            countdown_reward=0.9,
            line_closure_reward=1.0,
            bar_token_reward=1.0,
            meter_alignment_reward=0.0,
            meter_duration_closeness_reward=0.0,
            bar_meter_consistency_reward=0.0,
            bar_count_reward=_bar_count_reward(0, 32),
            voice_declaration_reward=1.0,
            score_voice_reward=1.0,
            repeat_syntax_reward=1.0,
            root_similarity_reward=1.0,
            bass_pitch_class_reward=1.0,
            cadence_root_reward=1.0,
            cadence_bass_reward=1.0,
        )

        four_bar_without_harmony = _total_reward(
            config=config,
            expected_bars=32,
            validated_bars=4,
            parse_reward=1.0,
            countdown_reward=0.9,
            line_closure_reward=1.0,
            bar_token_reward=1.0,
            meter_alignment_reward=0.1,
            meter_duration_closeness_reward=0.2,
            bar_meter_consistency_reward=0.1,
            bar_count_reward=_bar_count_reward(4, 32),
            voice_declaration_reward=1.0,
            score_voice_reward=1.0,
            repeat_syntax_reward=1.0,
            root_similarity_reward=0.0,
            bass_pitch_class_reward=0.0,
            cadence_root_reward=0.0,
            cadence_bass_reward=0.0,
        )

        self.assertGreater(four_bar_without_harmony, zero_bar_with_harmony)

    def test_harmonic_rewards_still_apply_for_complete_variation(self):
        config = GoldbergRewardConfig()
        base = _total_reward(
            config=config,
            expected_bars=32,
            validated_bars=32,
            parse_reward=1.0,
            countdown_reward=1.0,
            line_closure_reward=1.0,
            bar_token_reward=1.0,
            meter_alignment_reward=1.0,
            meter_duration_closeness_reward=1.0,
            bar_meter_consistency_reward=1.0,
            bar_count_reward=_bar_count_reward(32, 32),
            voice_declaration_reward=1.0,
            score_voice_reward=1.0,
            repeat_syntax_reward=1.0,
            root_similarity_reward=0.0,
            bass_pitch_class_reward=0.0,
            cadence_root_reward=0.0,
            cadence_bass_reward=0.0,
        )
        with_harmony = _total_reward(
            config=config,
            expected_bars=32,
            validated_bars=32,
            parse_reward=1.0,
            countdown_reward=1.0,
            line_closure_reward=1.0,
            bar_token_reward=1.0,
            meter_alignment_reward=1.0,
            meter_duration_closeness_reward=1.0,
            bar_meter_consistency_reward=1.0,
            bar_count_reward=_bar_count_reward(32, 32),
            voice_declaration_reward=1.0,
            score_voice_reward=1.0,
            repeat_syntax_reward=1.0,
            root_similarity_reward=0.5,
            bass_pitch_class_reward=0.5,
            cadence_root_reward=0.5,
            cadence_bass_reward=0.5,
        )

        self.assertGreater(with_harmony, base)


if __name__ == "__main__":
    unittest.main()
