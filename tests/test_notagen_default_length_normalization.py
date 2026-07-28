from __future__ import annotations

from fractions import Fraction

from preprocessing.notagen_abc import (
    _extract_header_context,
    _split_voice_segments,
    _voice_segment_duration,
    normalize_abc_default_length,
    preprocess_notagen_abc,
)


def _first_voice_duration(text: str, voice: int = 1) -> Fraction:
    header = _extract_header_context(text)
    for line in text.splitlines():
        if "[V:" not in line:
            continue
        for segment_voice, segment in _split_voice_segments(line):
            if segment_voice == voice:
                return _voice_segment_duration(
                    segment,
                    header.voice_lengths.get(segment_voice, header.default_length),
                )
    raise AssertionError(f"voice {voice} was not found")


def test_normalize_l_one_sixteenth_rewrites_suffixes_to_preserve_meter_duration() -> None:
    source = "\n".join(
        [
            "X:1",
            "M:3/4",
            "L:1/16",
            "K:G",
            "V:1",
            "[r:0/0][V:1]B2 c2 B2|[V:2][B,D]6|",
        ]
    )

    normalized = normalize_abc_default_length(source, "1/8")

    assert "L:1/8" in normalized
    assert "[V:1]B c B|" in normalized
    assert "[V:2][B,D]3|" in normalized
    assert _first_voice_duration(source, 1) == Fraction(3, 8)
    assert _first_voice_duration(normalized, 1) == Fraction(3, 8)


def test_normalize_bare_l_one_sixteenth_notes_use_half_length_suffix() -> None:
    source = "\n".join(
        [
            "M:3/4",
            "L:1/16",
            "K:G",
            "V:1",
            "[r:0/0][V:1]z DEF GFGA BAGB|",
        ]
    )

    normalized = normalize_abc_default_length(source, "1/8")

    assert "[V:1]z/2 D/2E/2F/2 G/2F/2G/2A/2 B/2A/2G/2B/2|" in normalized
    assert _first_voice_duration(source) == Fraction(3, 4)
    assert _first_voice_duration(normalized) == Fraction(3, 4)


def test_normalize_l_one_quarter_expands_suffixes_to_preserve_meter_duration() -> None:
    source = "\n".join(
        [
            "M:2/2",
            "L:1/4",
            "K:G",
            "V:1",
            "[r:0/0][V:1]G2 A2|",
        ]
    )

    normalized = normalize_abc_default_length(source, "1/8")

    assert "L:1/8" in normalized
    assert "[V:1]G4 A4|" in normalized
    assert _first_voice_duration(source) == Fraction(1, 1)
    assert _first_voice_duration(normalized) == Fraction(1, 1)


def test_normalize_respects_voice_specific_default_lengths() -> None:
    source = "\n".join(
        [
            "M:3/4",
            "K:G",
            "V:1",
            "L:1/16",
            "V:2",
            "L:1/4",
            "[r:0/0][V:1]B2 c2 B2|[V:2]G2 A|",
        ]
    )

    normalized = normalize_abc_default_length(source, "1/8")

    assert normalized.count("L:1/8") == 2
    assert "[V:1]B c B|" in normalized
    assert "[V:2]G4 A2|" in normalized
    assert _first_voice_duration(source, 1) == _first_voice_duration(normalized, 1)
    assert _first_voice_duration(source, 2) == _first_voice_duration(normalized, 2)


def test_preprocess_can_opt_into_default_length_normalization() -> None:
    source = "\n".join(
        [
            "M:3/4",
            "L:1/16",
            "K:G",
            "V:1",
            "[r:0/0][V:1]B2 c2 B2|",
        ]
    )

    normalized = preprocess_notagen_abc(source, target_default_length="1/8")

    assert "L:1/8" in normalized
    assert "[V:1]B c B|" in normalized


def test_normalize_resets_context_for_concatenated_abc_blocks() -> None:
    source = "\n".join(
        [
            "%%score 1",
            "M:3/4",
            "L:1/8",
            "K:G",
            "V:1",
            "[V:1]C2D2E2|",
            "%%score 1",
            "M:3/4",
            "L:1/16",
            "K:G",
            "V:1",
            "[V:1]B2 c2 B2|",
        ]
    )

    normalized = normalize_abc_default_length(source, "1/8")

    assert normalized.count("L:1/8") == 2
    assert "L:1/16" not in normalized
    assert "[V:1]B c B|" in normalized
