from fractions import Fraction
import time
import unittest

from preprocessing.notagen_abc import (
    expand_notagen_rest_omitted_voice_segments,
    preprocess_notagen_abc,
    strip_dangling_terminal_ties,
)
from evaluation.rewards import (
    GoldbergRewardConfig,
    StructuralTarget,
    _bar_count_reward,
    _countdown_reward,
    _extract_header_context,
    _extract_stream_line_features,
    _abc_grammar_metrics,
    count_notagen_structure_lines,
    _parse_length_multiplier,
    score_candidate_text,
    _total_reward,
    _validated_bar_metrics,
)


class GoldbergRewardTests(unittest.TestCase):
    def test_count_notagen_structure_lines_accepts_plain_and_stream_tagged_abc(self):
        plain = "\n".join(
            [
                "X:1",
                "M:12/8",
                "L:1/8",
                "K:G",
                "V:1 treble",
                "[V:1]C2D2E2F2G2A2|",
                "[V:1]A2G2F2E2D2C2|",
            ]
        )
        tagged = "\n".join(
            [
                "M:12/8",
                "L:1/8",
                "[r:0/1][V:1]C2D2E2F2G2A2|",
                "[r:1/0][V:1]A2G2F2E2D2C2|",
            ]
        )

        self.assertEqual(count_notagen_structure_lines(plain), 2)
        self.assertEqual(count_notagen_structure_lines(tagged), 2)

    def test_bar_count_uses_target_notagen_structure_length_when_available(self):
        target = StructuralTarget(
            expected_bars=32,
            expected_structure_bars=2,
        )
        text = "\n".join(
            [
                "M:12/8",
                "L:1/8",
                "[r:0/1][V:1]C2D2E2F2G2A2|",
                "[r:1/0][V:1]A2G2F2E2D2C2|",
            ]
        )

        breakdown = score_candidate_text(text, target, GoldbergRewardConfig(music21_parse_timeout_s=1.0))

        self.assertEqual(breakdown.validated_bars, 2)
        self.assertEqual(breakdown.bar_count_reward, 1.0)

    def test_parse_length_multiplier_accepts_abc_shorthand_fraction(self):
        self.assertEqual(_parse_length_multiplier("/"), Fraction(1, 2))
        self.assertEqual(_parse_length_multiplier("3/"), Fraction(3, 2))

    def test_absurd_duration_skips_music21_parse(self):
        target = StructuralTarget(
            expected_bars=1,
            expected_structure_bars=1,
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

    def test_preprocess_notagen_abc_strips_unsupported_instructions(self):
        text = "\n".join(
            [
                "%%score ( 1 2 )",
                "M:3/4",
                "L:1/8",
                "V:1 treble",
                "V:2 treble",
                "[V:1]C2!courtesy!D2!trill(!E2|",
            ]
        )

        preprocessed = preprocess_notagen_abc(text)

        self.assertNotIn("!courtesy!", preprocessed)
        self.assertNotIn("!trill(!", preprocessed)
        self.assertIn("[V:2]x6|", preprocessed)

    def test_strip_dangling_terminal_ties_removes_tie_into_rest(self):
        text = "\n".join(
            [
                "%%score ( 1 )",
                "M:3/4",
                "L:1/8",
                "V:1 treble",
                "[V:1]C2D2E2-|",
                "[V:1]x6|",
            ]
        )

        preprocessed = strip_dangling_terminal_ties(text)

        self.assertIn("[V:1]C2D2E2|", preprocessed)
        self.assertNotIn("E2-|", preprocessed)

    def test_strip_dangling_terminal_ties_preserves_tie_into_note(self):
        text = "\n".join(
            [
                "%%score ( 1 )",
                "M:3/4",
                "L:1/8",
                "V:1 treble",
                "[V:1]C2D2E2-|",
                "[V:1]E2D2C2|",
            ]
        )

        preprocessed = strip_dangling_terminal_ties(text)

        self.assertIn("E2-|", preprocessed)

    def test_validated_bars_dominate_zero_bar_output(self):
        config = GoldbergRewardConfig()

        zero_bar = _total_reward(
            config=config,
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
        )

        four_bar = _total_reward(
            config=config,
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
        )

        self.assertGreater(four_bar, zero_bar)

    def test_total_reward_combines_structural_terms(self):
        config = GoldbergRewardConfig()
        strong = _total_reward(
            config=config,
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
        )
        weak = _total_reward(
            config=config,
            parse_reward=1.0,
            countdown_reward=1.0,
            line_closure_reward=1.0,
            bar_token_reward=1.0,
            meter_alignment_reward=0.5,
            meter_duration_closeness_reward=0.5,
            bar_meter_consistency_reward=0.5,
            bar_count_reward=_bar_count_reward(32, 32),
            voice_declaration_reward=1.0,
            score_voice_reward=1.0,
        )

        self.assertGreater(strong, weak)


if __name__ == "__main__":
    unittest.main()
