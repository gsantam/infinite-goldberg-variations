from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True)
class AbcHeaderContext:
    meter: Fraction
    default_length: Fraction
    voice_lengths: dict[int, Fraction]
    score_voices: set[int]
    has_score: bool


def _parse_fraction_token(token: str, fallback: Fraction) -> Fraction:
    token = token.strip()
    if not token:
        return fallback
    try:
        if "/" in token:
            num, den = token.split("/", 1)
            if num == "":
                num = "1"
            if den == "":
                den = "2"
            return Fraction(int(num), int(den))
        return Fraction(int(token), 1)
    except Exception:
        return fallback


def _extract_header_context(text: str) -> AbcHeaderContext:
    meter = Fraction(3, 4)
    default_length = Fraction(1, 8)
    current_voice: int | None = None
    voice_lengths: dict[int, Fraction] = {}
    score_voices: set[int] = set()
    has_score = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("[r:"):
            break
        if line.startswith("%%score"):
            has_score = True
            score_voices.update(int(item) for item in re.findall(r"\d+", line))
        if line.startswith("M:"):
            meter = _parse_fraction_token(line[2:], meter)
        elif line.startswith("L:"):
            if current_voice is not None:
                voice_lengths[current_voice] = _parse_fraction_token(line[2:], default_length)
            else:
                default_length = _parse_fraction_token(line[2:], default_length)
        elif line.startswith("V:"):
            match = re.match(r"^V:(\d+)", line)
            if match:
                current_voice = int(match.group(1))
                voice_lengths.setdefault(current_voice, default_length)
    return AbcHeaderContext(
        meter=meter,
        default_length=default_length,
        voice_lengths=voice_lengths,
        score_voices=score_voices,
        has_score=has_score,
    )


def _target_header_voices(header: AbcHeaderContext) -> set[int]:
    if header.has_score and header.score_voices:
        return set(header.score_voices)
    return set(header.voice_lengths)


def _rest_token_for_duration(duration: Fraction, base_length: Fraction) -> str:
    if duration <= 0 or base_length <= 0:
        return "x"
    units = duration / base_length
    if units.denominator == 1:
        return "x" if units.numerator == 1 else f"x{units.numerator}"
    return f"x{units.numerator}/{units.denominator}"


def _parse_length_multiplier(token: str | None) -> Fraction:
    if not token:
        return Fraction(1, 1)
    if token == "/":
        return Fraction(1, 2)
    if token.startswith("/"):
        den = token[1:]
        if not den:
            return Fraction(1, 2)
        return Fraction(1, int(den))
    if "/" in token:
        num, den = token.split("/", 1)
        if not den:
            return Fraction(int(num), 2)
        return Fraction(int(num), int(den))
    return Fraction(int(token), 1)


def _voice_segment_duration(segment: str, base_length: Fraction) -> Fraction:
    cleaned = re.sub(r'"[^"\n]*"', " ", segment)
    cleaned = re.sub(r"![^!\n]*!", " ", cleaned)
    cleaned = re.sub(r"\[[A-Za-z]:[^\]]*\]", " ", cleaned)
    cleaned = re.sub(r"\{[^}]*\}", " ", cleaned)
    cleaned = re.sub(r"[{}<>~$PMHSTuvw]", " ", cleaned)
    total = Fraction(0, 1)
    token_pattern = re.compile(r"(\(\d+)|(\[[^\]]+\]|[_=^]*[A-Ga-gxz][,']*)(\d+(?:/\d*)?|/\d+|/)?")
    tuplet_notes_left = 0
    tuplet_ratio = Fraction(1, 1)
    for match in token_pattern.finditer(cleaned):
        tuplet_marker = match.group(1)
        if tuplet_marker:
            tuplet_count = int(tuplet_marker[1:])
            if tuplet_count > 0:
                tuplet_notes_left = tuplet_count
                tuplet_ratio = Fraction(2, tuplet_count) if tuplet_count == 3 else Fraction(1, 1)
            continue
        multiplier = _parse_length_multiplier(match.group(3))
        if tuplet_notes_left > 0:
            multiplier *= tuplet_ratio
            tuplet_notes_left -= 1
            if tuplet_notes_left == 0:
                tuplet_ratio = Fraction(1, 1)
        total += base_length * multiplier
    return total


def _body_meter(body: str, current_meter: Fraction) -> tuple[Fraction, Fraction]:
    active_meter = current_meter
    first_meter = current_meter
    saw_meter = False
    for match in re.finditer(r"\[M:([^\]]+)\]", body):
        active_meter = _parse_fraction_token(match.group(1), active_meter)
        if not saw_meter:
            first_meter = active_meter
            saw_meter = True
    return first_meter, active_meter


def _split_voice_segments(body: str) -> list[tuple[int | None, str]]:
    parts = re.split(r"(\[V:\d+\])", body)
    segments: list[tuple[int | None, str]] = []
    current_voice: int | None = None
    for part in parts:
        if not part:
            continue
        voice_match = re.fullmatch(r"\[V:(\d+)\]", part)
        if voice_match:
            current_voice = int(voice_match.group(1))
            continue
        segments.append((current_voice, part))
    return segments or [(None, body)]


def _segment_closing_barline(segment: str) -> str:
    stripped = segment.rstrip()
    match = re.search(r"(:\|\d+|\|\d+|\|:|::|:\||\|\]|\|\||\||:)$", stripped)
    if match:
        return match.group(1)
    # xml2abc occasionally serializes a numbered ending as a digit after a
    # voice-local note without a plain barline in the rest of the segment.
    match = re.search(r"(\d+)$", stripped)
    if match and ":|" in stripped:
        return f":|{match.group(1)}"
    return "|"


def _segment_has_barline(segment: str) -> bool:
    stripped = segment.rstrip()
    return "|" in stripped or stripped.endswith(":")


def _split_optional_stream_tag(line: str) -> tuple[str, str] | None:
    stream_match = re.match(r"^(\[r:\d+/\d+\])(.*)$", line.strip())
    if stream_match is not None:
        return stream_match.group(1), stream_match.group(2)
    if "[V:" in line and _segment_has_barline(line):
        return "", line.strip()
    return None


def _line_closing_barline(body: str) -> str:
    for _voice, segment in reversed(_split_voice_segments(body)):
        if _segment_has_barline(segment):
            return _segment_closing_barline(segment)
    return "|"


def _line_rest_duration(
    body: str,
    header: AbcHeaderContext,
    fallback_meter: Fraction,
) -> Fraction:
    durations = []
    for voice, segment in _split_voice_segments(body):
        if voice is None or not _segment_has_barline(segment):
            continue
        base_length = header.voice_lengths.get(voice, header.default_length)
        duration = _voice_segment_duration(segment, base_length)
        if duration > 0:
            durations.append(duration)
    if not durations:
        return fallback_meter
    return max(durations)


def _rebuild_body_with_missing_rests(
    body: str,
    expected_voices: set[int],
    missing_voices: set[int],
    rest_by_voice: dict[int, str],
) -> str:
    by_voice: dict[int, list[str]] = {}
    unvoiced: list[str] = []
    for voice, segment in _split_voice_segments(body):
        if voice is None:
            unvoiced.append(segment)
        else:
            by_voice.setdefault(voice, []).append(segment)

    ordered_parts = unvoiced[:]
    voice_order = sorted(expected_voices | set(by_voice))
    for voice in voice_order:
        if voice in by_voice:
            ordered_parts.extend(f"[V:{voice}]{segment}" for segment in by_voice[voice])
        elif voice in missing_voices:
            ordered_parts.append(f"[V:{voice}]{rest_by_voice[voice]}")
    return "".join(ordered_parts)


def expand_notagen_rest_omitted_voice_segments(text: str) -> str:
    """Add rest-only voice segments that NotaGen preprocessing omits.

    NotaGen trains on a compact augmented representation where voices with a
    full-bar rest are absent from that stream line. ABC renderers such as
    abc2midi are stricter about declared voices spanning the tune, so this
    helper reconstructs only missing all-rest segments at render/eval time.
    """

    header = _extract_header_context(text)
    expected_voices = _target_header_voices(header)
    if not expected_voices:
        return text

    active_meter = header.meter
    output_lines: list[str] = []
    changed = False
    for raw_line in text.splitlines(keepends=True):
        line_body = raw_line[:-1] if raw_line.endswith("\n") else raw_line
        newline = "\n" if raw_line.endswith("\n") else ""
        parsed = _split_optional_stream_tag(line_body)
        if parsed is None:
            output_lines.append(raw_line)
            continue

        prefix, body = parsed
        line_meter, active_meter = _body_meter(body, active_meter)
        if not _segment_has_barline(body):
            output_lines.append(raw_line)
            continue

        used_voices = {
            voice
            for voice, segment in _split_voice_segments(body)
            if voice is not None and _segment_has_barline(segment)
        }
        missing_voices = sorted(expected_voices - used_voices)
        if not used_voices or not missing_voices:
            output_lines.append(raw_line)
            continue

        barline = _line_closing_barline(body)
        rest_duration = _line_rest_duration(body, header, line_meter)
        rest_by_voice = {}
        for voice in missing_voices:
            base_length = header.voice_lengths.get(voice, header.default_length)
            rest_by_voice[voice] = f"{_rest_token_for_duration(rest_duration, base_length)}{barline}"
        expanded_body = _rebuild_body_with_missing_rests(
            body=body,
            expected_voices=expected_voices,
            missing_voices=set(missing_voices),
            rest_by_voice=rest_by_voice,
        )
        output_lines.append(f"{prefix}{expanded_body}{newline}")
        changed = True

    if not changed:
        return text
    return "".join(output_lines)
