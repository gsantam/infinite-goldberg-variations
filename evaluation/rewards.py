from __future__ import annotations

import json
import math
import re
import signal
import threading
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from fractions import Fraction
from pathlib import Path
import tempfile

from music21 import abcFormat, converter

from evaluation.stream_tags import (
    StreamLine,
    StreamTag,
    extract_stream_lines,
    stream_line_closed,
    stream_tag_sequence_reward,
)


@dataclass
class StructuralTarget:
    expected_bars: int
    expected_structure_bars: int
    musical_bar_unit: Fraction = Fraction(3, 4)
    expected_expanded_bars: int | None = None

    @property
    def expected_repeat_expanded_bars(self) -> int:
        return self.expected_expanded_bars if self.expected_expanded_bars is not None else self.expected_bars * 2

    @property
    def expected_reward_bars(self) -> int:
        # Rollout/stopping and structural scoring should use the target score
        # measure count, not the source variation's serialized line count.
        return self.expected_bars


@dataclass
class GoldbergRewardConfig:
    completion_weight: float = 0.25
    expanded_completion_weight: float = 0.25
    parse_weight: float = 0.25
    syntax_penalty_weight: float = 0.25
    termination_penalty_weight: float = 0.0
    countdown_weight: float = 0.25
    line_closure_weight: float = 0.25
    bar_token_weight: float = 0.10
    note_bearing_line_weight: float = 0.25
    meter_alignment_weight: float = 0.75
    meter_duration_closeness_weight: float = 0.75
    bar_meter_consistency_weight: float = 0.75
    bar_count_weight: float = 1.0
    expanded_bar_count_weight: float = 1.0
    voice_declaration_weight: float = 1.0
    score_voice_weight: float = 0.5
    parse_validation_mode: str = "music21"
    music21_parse_timeout_s: float = 5.0
    max_music21_meter_component: int = 128
    max_music21_duration_component: int = 512


@dataclass
class RewardBreakdown:
    candidate_path: str
    parse_valid: bool
    clearly_malformed_syntax: bool
    observed_stream_lines: int
    observed_bars: float
    observed_written_bars: int
    observed_repeat_expanded_bars: float
    primary_validated_bars: float
    validated_bars: float
    strict_validated_bars: int
    completion_reward: float
    expanded_completion_reward: float
    parse_reward: float
    parse_balanced_construct_reward: float
    parse_inline_field_reward: float
    parse_duration_sanity_reward: float
    parse_tokenizer_reward: float
    parse_music21_reward: float
    syntax_penalty_reward: float
    termination_penalty_reward: float
    countdown_reward: float
    line_closure_reward: float
    bar_token_reward: float
    note_bearing_line_reward: float
    meter_alignment_reward: float
    meter_duration_closeness_reward: float
    bar_meter_consistency_reward: float
    strict_bar_meter_consistency_reward: float
    bar_count_reward: float
    expanded_bar_count_reward: float
    voice_declaration_reward: float
    score_voice_reward: float
    structural_validity_gate_reward: float
    ungated_total_reward: float
    structural_validity_gate_adjustment: float
    total_reward: float

    def to_json(self) -> dict:
        return asdict(self)


@dataclass
class StreamLineFeatures:
    index: int
    tag_marker: int
    body: str
    has_bar_token: bool
    closed: bool

    @property
    def remaining(self) -> int:
        return self.tag_marker


@dataclass
class HeaderContext:
    meter: Fraction
    default_length: Fraction
    voice_lengths: dict[int, Fraction]
    score_voices: set[int]
    has_score: bool


@dataclass(frozen=True)
class MeterValidationMetrics:
    meter_alignment_reward: float
    meter_duration_closeness_reward: float
    validated_bars: int
    observed_musical_bars: float
    written_bars: int
    written_bar_units: float
    strict_validated_bars: int
    bar_meter_consistency_reward: float
    strict_bar_meter_consistency_reward: float


@dataclass(frozen=True)
class ParseValidationMetrics:
    parse_valid: bool
    clearly_malformed_syntax: bool
    parse_reward: float
    balanced_construct_reward: float
    inline_field_reward: float
    duration_sanity_reward: float
    tokenizer_reward: float
    music21_reward: float


@dataclass(frozen=True)
class AbcGrammarMetrics:
    voice_declaration_reward: float
    score_voice_reward: float


@dataclass(frozen=True)
class StreamLineLocalMetrics:
    meter_alignment_reward: list[float]
    meter_duration_closeness_reward: list[float]
    bar_meter_consistency_reward: list[float]
    note_bearing_line_reward: list[float]
    musical_bar_units: list[float]
    written_bar_units: list[float]
    voice_declaration_reward: list[float]
    score_voice_reward: list[float]


@dataclass(frozen=True)
class StreamLineMetricBundle:
    meter_metrics: MeterValidationMetrics
    grammar_metrics: AbcGrammarMetrics
    local_metrics: StreamLineLocalMetrics


@dataclass(frozen=True)
class CandidateStructuralScore:
    breakdown: RewardBreakdown
    stream_lines: list[StreamLineFeatures]
    local_metrics: StreamLineLocalMetrics


@dataclass(frozen=True)
class SourceStructureBarCounts:
    observed_bars: float
    observed_repeat_expanded_bars: float
    observed_lines: int
    source_format: str


def count_notagen_structure_lines(text: str) -> int:
    stream_lines = extract_stream_lines(text)
    if stream_lines:
        return len(stream_lines)

    count = 0
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("%"):
            continue
        if re.match(r"^[A-Za-z]:", line):
            continue
        if "[V:" in line or "|" in line:
            count += 1
    return count


def load_structural_target(
    path: str | Path,
    *,
    structure_path: str | Path,
) -> StructuralTarget:
    expected_structure_bars = count_notagen_structure_lines(Path(structure_path).read_text(encoding="utf-8"))

    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    return StructuralTarget(
        expected_bars=len(rows),
        expected_structure_bars=expected_structure_bars,
    )


def _safe_fraction(matches: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return matches / total


def _bar_token_count(text: str) -> int:
    # Remove repeat punctuation so a terminal :| or |] still counts as one closure.
    normalized = text.replace(":|", "|").replace("|:", "|").replace("|]", "|").replace("[|", "|")
    return normalized.count("|")


def _extract_stream_line_features(text: str) -> list[StreamLineFeatures]:
    features: list[StreamLineFeatures] = []
    for line in extract_stream_lines(text):
        body = line.body
        features.append(
            StreamLineFeatures(
                index=line.tag.index,
                tag_marker=line.tag.marker,
                body=body,
                has_bar_token="|" in body or "::" in body,
                closed=stream_line_closed(line),
            )
        )
    return features


def _extract_raw_score_line_features(text: str) -> list[StreamLineFeatures]:
    features: list[StreamLineFeatures] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("%"):
            continue
        if re.match(r"^[A-Za-z]:", line):
            continue
        if "|" not in line and "[V:" not in line:
            continue
        features.append(
            StreamLineFeatures(
                index=len(features),
                tag_marker=0,
                body=line,
                has_bar_token="|" in line or "::" in line,
                closed="|" in line or "::" in line,
            )
        )
    return features


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


def _extract_header_context(text: str) -> HeaderContext:
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
    return HeaderContext(
        meter=meter,
        default_length=default_length,
        voice_lengths=voice_lengths,
        score_voices=score_voices,
        has_score=has_score,
    )


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


def _parse_length_multiplier(token: str | None) -> Fraction:
    try:
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
    except Exception:
        return Fraction(0, 1)


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
                # ABC shorthand: (3abc means three notes in the time of two.
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


def _strip_non_note_context(segment: str) -> str:
    cleaned = re.sub(r'"[^"\n]*"', " ", segment)
    cleaned = re.sub(r"![^!\n]*!", " ", cleaned)
    cleaned = re.sub(r"\[[A-Za-z]:[^\]]*\]", " ", cleaned)
    cleaned = re.sub(r"\{[^}]*\}", " ", cleaned)
    return cleaned


_ABC_NOTE_RE = re.compile(r"[_=^]*[A-Ga-g][,']*(?:\d+(?:/\d*)?|/\d+|/)?")


def _line_has_note(segment: str) -> bool:
    return _ABC_NOTE_RE.search(_strip_non_note_context(segment)) is not None


class _Music21ParseTimeout(TimeoutError):
    pass


@contextmanager
def _music21_parse_time_limit(timeout_s: float):
    if (
        timeout_s <= 0
        or threading.current_thread() is not threading.main_thread()
        or not hasattr(signal, "SIGALRM")
    ):
        yield
        return

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, timeout_s)

    def _raise_timeout(_signum, _frame):
        raise _Music21ParseTimeout(f"music21 parse exceeded {timeout_s}s")

    signal.signal(signal.SIGALRM, _raise_timeout)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, previous_timer[0], previous_timer[1])
        signal.signal(signal.SIGALRM, previous_handler)


def _fraction_component_too_large(value: Fraction, limit: int) -> bool:
    return abs(value.numerator) > limit or abs(value.denominator) > limit


def _balanced_abc_construct_guard_tripped(text: str) -> bool:
    for raw_line in text.splitlines():
        if raw_line.count('"') % 2 != 0:
            return True
        line = re.sub(r'"[^"\n]*"', " ", raw_line)
        bracket_line = line.replace("|]", "|").replace("[|", "|")
        bracket_line = re.sub(r"\[\d+", " ", bracket_line)
        if bracket_line.count("[") != bracket_line.count("]"):
            return True
        if line.count("{") != line.count("}"):
            return True
        if line.count("!") % 2 != 0:
            return True
    return False


def _inline_field_guard_tripped(field: str, value: str, config: GoldbergRewardConfig) -> bool:
    value = value.strip()
    if not value:
        return True
    if field == "V":
        return bool(re.search(r"[\[\]\n]", value))
    if field in {"M", "L"}:
        parsed = _parse_fraction_token(value, Fraction(0, 1))
        return parsed <= 0 or _fraction_component_too_large(parsed, config.max_music21_meter_component)
    if field == "K":
        return bool(re.search(r"[\[\]\n]", value))
    # Keep the preflight conservative: these are the inline fields this reward
    # path expects to see in generated NotaGen ABC.
    return field not in {"I", "P", "Q"}


def _bracket_token_guard_tripped(content: str, config: GoldbergRewardConfig) -> bool:
    if not content or "[" in content or "]" in content or "\n" in content:
        return True
    field_match = re.match(r"^([A-Za-z]):(.*)$", content)
    if field_match:
        return _inline_field_guard_tripped(field_match.group(1), field_match.group(2), config)
    if ":" in content or "|" in content:
        return True
    if not re.search(r"[_=^]*[A-Ga-gxz]", content):
        return True

    note_pattern = re.compile(r"[_=^]*[A-Ga-gxz][,']*(?:\d+(?:/\d*)?|/\d+|/)?")
    remainder = note_pattern.sub(" ", content)
    remainder = re.sub(r"[\s,\-/_=^']+", " ", remainder)
    return bool(remainder.strip())


def _length_token_reasonable(token: str | None, config: GoldbergRewardConfig) -> bool:
    try:
        multiplier = _parse_length_multiplier(token)
    except Exception:
        return False
    return (
        multiplier > 0
        and not _fraction_component_too_large(multiplier, config.max_music21_duration_component)
    )


def _inline_field_sanity_guard_tripped(
    abc_text: str,
    stream_lines: list[StreamLineFeatures],
    config: GoldbergRewardConfig,
) -> bool:
    for raw_line in abc_text.splitlines():
        line = raw_line.strip()
        header_match = re.match(r"^([VMLK]):(.*)$", line)
        if header_match and _inline_field_guard_tripped(header_match.group(1), header_match.group(2), config):
            return True

    bracket_token_pattern = re.compile(r"\[([^\]\n]*)\](\d+(?:/\d*)?|/\d+|/)?")
    for line in stream_lines:
        cleaned = re.sub(r'"[^"\n]*"', " ", line.body)
        cleaned = re.sub(r"![^!\n]*!", " ", cleaned)
        for match in bracket_token_pattern.finditer(cleaned):
            if _bracket_token_guard_tripped(match.group(1), config):
                return True
            if not _length_token_reasonable(match.group(2), config):
                return True
    return False


def _duration_sanity_guard_tripped(
    stream_lines: list[StreamLineFeatures],
    config: GoldbergRewardConfig,
) -> bool:
    token_pattern = re.compile(r"(\[[^\]]+\]|[_=^]*[A-Ga-gxz][,']*)(\d+(?:/\d*)?|/\d+|/)?")
    bracket_token_pattern = re.compile(r"\[([^\]\n]*)\](\d+(?:/\d*)?|/\d+|/)?")
    bracket_note_pattern = re.compile(r"[_=^]*[A-Ga-gxz][,']*(\d+(?:/\d*)?|/\d+|/)?")
    malformed_fraction_pattern = re.compile(r"[_=^]*[A-Ga-gxz][,']*//+\d*")
    for line in stream_lines:
        cleaned = re.sub(r'"[^"\n]*"', " ", line.body)
        cleaned = re.sub(r"![^!\n]*!", " ", cleaned)
        if malformed_fraction_pattern.search(cleaned):
            return True
        cleaned_without_inline_fields = re.sub(r"\[[A-Za-z]:[^\]]*\]", " ", cleaned)
        for match in bracket_token_pattern.finditer(cleaned_without_inline_fields):
            if not _length_token_reasonable(match.group(2), config):
                return True
            content = match.group(1)
            if not re.search(r"[_=^]*[A-Ga-gxz]", content):
                continue
            for note_match in bracket_note_pattern.finditer(content):
                if not _length_token_reasonable(note_match.group(1), config):
                    return True
        for match in token_pattern.finditer(cleaned_without_inline_fields):
            if not _length_token_reasonable(match.group(2), config):
                return True
    return False


def _abc_tokenize_valid(renderable_abc_text: str) -> bool:
    try:
        abcFormat.ABCFile().readstr(renderable_abc_text)
        return True
    except Exception:
        return False


def _music21_parse_guard_tripped(
    abc_text: str,
    stream_lines: list[StreamLineFeatures],
    config: GoldbergRewardConfig,
) -> bool:
    return (
        _balanced_abc_construct_guard_tripped(abc_text)
        or _inline_field_sanity_guard_tripped(abc_text, stream_lines, config)
        or _duration_sanity_guard_tripped(stream_lines, config)
    )


def _extract_parse_validation_metrics(
    abc_text: str,
    stream_lines: list[StreamLineFeatures],
    config: GoldbergRewardConfig,
) -> ParseValidationMetrics:
    mode = config.parse_validation_mode.replace("_", "-")
    if mode == "none":
        return ParseValidationMetrics(
            parse_valid=True,
            clearly_malformed_syntax=False,
            parse_reward=1.0,
            balanced_construct_reward=1.0,
            inline_field_reward=1.0,
            duration_sanity_reward=1.0,
            tokenizer_reward=1.0,
            music21_reward=1.0,
        )
    if mode not in {"music21", "abc-tokenize"}:
        raise ValueError(f"unsupported parse_validation_mode: {config.parse_validation_mode}")

    balanced_construct_reward = 0.0 if _balanced_abc_construct_guard_tripped(abc_text) else 1.0
    inline_field_reward = 0.0 if _inline_field_sanity_guard_tripped(abc_text, stream_lines, config) else 1.0
    duration_sanity_reward = 0.0 if _duration_sanity_guard_tripped(stream_lines, config) else 1.0
    clearly_malformed_syntax = (
        balanced_construct_reward == 0.0
        or inline_field_reward == 0.0
        or duration_sanity_reward == 0.0
    )

    tokenizer_reward = 0.0
    music21_reward = 0.0
    parse_valid = False
    checked_rewards = [
        balanced_construct_reward,
        inline_field_reward,
        duration_sanity_reward,
    ]

    renderable_abc = _ensure_renderable_abc(abc_text)
    if not clearly_malformed_syntax:
        tokenizer_reward = 1.0 if _abc_tokenize_valid(renderable_abc) else 0.0
    checked_rewards.append(tokenizer_reward)
    if mode == "abc-tokenize":
        parse_valid = tokenizer_reward == 1.0
    elif tokenizer_reward == 1.0:
        try:
            with _music21_parse_time_limit(config.music21_parse_timeout_s):
                converter.parseData(renderable_abc, format="abc")
            music21_reward = 1.0
        except Exception:
            music21_reward = 0.0
        checked_rewards.append(music21_reward)
        parse_valid = music21_reward == 1.0
    else:
        checked_rewards.append(music21_reward)

    parse_reward = sum(checked_rewards) / len(checked_rewards) if checked_rewards else 0.0
    return ParseValidationMetrics(
        parse_valid=parse_valid,
        clearly_malformed_syntax=clearly_malformed_syntax,
        parse_reward=parse_reward,
        balanced_construct_reward=balanced_construct_reward,
        inline_field_reward=inline_field_reward,
        duration_sanity_reward=duration_sanity_reward,
        tokenizer_reward=tokenizer_reward,
        music21_reward=music21_reward,
    )


def _extract_music21_candidate_features(
    abc_text: str,
    stream_lines: list[StreamLineFeatures],
    config: GoldbergRewardConfig,
) -> bool:
    return _extract_parse_validation_metrics(abc_text, stream_lines, config).parse_valid


def _segment_active_meter(segment: str, current_meter: Fraction) -> tuple[Fraction, Fraction]:
    active_meter = current_meter
    for match in re.finditer(r"\[M:([^\]]+)\]", segment):
        active_meter = _parse_fraction_token(match.group(1), active_meter)
    return active_meter, active_meter


_BARLINE_RE = re.compile(r"::|:\|\d*|\|:\d*|\|\]|\[\||\|\d*")


def _closed_bar_chunks(segment: str) -> list[str]:
    chunks: list[str] = []
    start = 0
    for match in _BARLINE_RE.finditer(segment):
        chunks.append(segment[start : match.start()])
        start = match.end()
    return chunks


def _segment_bar_durations(
    segment: str,
    *,
    base_length: Fraction,
    active_meter: Fraction,
) -> tuple[list[tuple[Fraction, Fraction]], Fraction]:
    events, current_meter = _segment_bar_events(
        segment,
        base_length=base_length,
        active_meter=active_meter,
    )
    return [(duration, meter) for duration, meter, _barline in events if duration > 0], current_meter


def _segment_bar_events(
    segment: str,
    *,
    base_length: Fraction,
    active_meter: Fraction,
) -> tuple[list[tuple[Fraction, Fraction, str]], Fraction]:
    events: list[tuple[Fraction, Fraction, str]] = []
    start = 0
    current_meter = active_meter
    for match in _BARLINE_RE.finditer(segment):
        chunk = segment[start : match.start()]
        chunk_meter, current_meter = _segment_active_meter(chunk, current_meter)
        duration = _voice_segment_duration(chunk, base_length)
        events.append((duration, chunk_meter, match.group(0)))
        start = match.end()

    tail = segment[start:]
    _tail_meter, current_meter = _segment_active_meter(tail, current_meter)
    return events, current_meter


def _musical_bar_units_by_stream_line(
    stream_lines: list[StreamLineFeatures],
    header: HeaderContext,
    *,
    musical_bar_unit: Fraction,
) -> tuple[list[float], float, int, list[float], float]:
    # Historical name: these are score-measure counts, not duration-normalized
    # aria units. Meter/default length validation happens in the meter rewards.
    if musical_bar_unit <= 0:
        empty = [0.0 for _line in stream_lines]
        return empty, 0.0, 0, empty, 0.0

    voice_totals: dict[int | None, Fraction] = {}
    voice_written_units: dict[int | None, Fraction] = {}
    voice_written_bars: dict[int | None, int] = {}
    voice_repeat_starts: dict[int | None, Fraction] = {}
    voice_meters: dict[int | None, Fraction] = {}
    line_units: list[float] = []
    line_written_units: list[float] = []

    for stream_line in stream_lines:
        increments: dict[int | None, Fraction] = {}
        written_increments: dict[int | None, Fraction] = {}
        for voice, segment in _split_voice_segments(stream_line.body):
            active_meter = voice_meters.get(voice, header.meter)
            base_length = header.voice_lengths.get(voice, header.default_length) if voice is not None else header.default_length
            events, active_meter = _segment_bar_events(
                segment,
                base_length=base_length,
                active_meter=active_meter,
            )
            voice_meters[voice] = active_meter
            for duration, _meter, barline in events:
                if duration > 0:
                    units = Fraction(1, 1)
                    voice_totals[voice] = voice_totals.get(voice, Fraction(0, 1)) + units
                    increments[voice] = increments.get(voice, Fraction(0, 1)) + units
                    voice_written_units[voice] = voice_written_units.get(voice, Fraction(0, 1)) + units
                    voice_written_bars[voice] = voice_written_bars.get(voice, 0) + 1
                    written_increments[voice] = written_increments.get(voice, Fraction(0, 1)) + units

                current_total = voice_totals.get(voice, Fraction(0, 1))
                repeat_start = voice_repeat_starts.get(voice, Fraction(0, 1))
                if barline.startswith("|:"):
                    voice_repeat_starts[voice] = current_total
                elif barline.startswith(":|") or barline == "::":
                    repeat_units = max(Fraction(0, 1), current_total - repeat_start)
                    if repeat_units > 0:
                        voice_totals[voice] = current_total + repeat_units
                        increments[voice] = increments.get(voice, Fraction(0, 1)) + repeat_units
                        current_total = voice_totals[voice]
                    if barline == "::":
                        voice_repeat_starts[voice] = current_total
        line_units.append(float(max(increments.values())) if increments else 0.0)
        line_written_units.append(float(max(written_increments.values())) if written_increments else 0.0)

    total_units = float(max(voice_totals.values())) if voice_totals else 0.0
    written_bars = max(voice_written_bars.values()) if voice_written_bars else 0
    written_units = float(max(voice_written_units.values())) if voice_written_units else 0.0
    return line_units, total_units, written_bars, line_written_units, written_units


def score_source_structure_bars(
    abc_text: str,
    *,
    musical_bar_unit: Fraction = Fraction(3, 4),
) -> SourceStructureBarCounts:
    """Count written and repeat-expanded score measures in source ABC.

    Generated completions are represented as NotaGen `[r:i/j]` stream lines,
    while PPO prompt source files may be ordinary ABC body lines. This helper
    supports both formats and intentionally computes only bar counts, not full
    structural rewards.
    """

    stream_lines = _extract_stream_line_features(abc_text)
    source_format = "notagen-stream"
    if not stream_lines:
        stream_lines = _extract_raw_score_line_features(abc_text)
        source_format = "raw-abc"
    header = _extract_header_context(abc_text)
    (
        _line_units,
        observed_repeat_expanded_bars,
        _observed_written_bars,
        _line_written_units,
        observed_bars,
    ) = _musical_bar_units_by_stream_line(
        stream_lines,
        header,
        musical_bar_unit=musical_bar_unit,
    )
    return SourceStructureBarCounts(
        observed_bars=float(observed_bars),
        observed_repeat_expanded_bars=float(observed_repeat_expanded_bars),
        observed_lines=len(stream_lines),
        source_format=source_format,
    )


def _duration_closeness(duration: Fraction, meter: Fraction) -> float:
    if duration <= 0 or meter <= 0:
        return 0.0
    return max(0.0, 1.0 - float(abs(duration - meter) / meter))


def _validated_bar_metrics(
    stream_lines: list[StreamLineFeatures],
    header: HeaderContext,
    musical_bar_unit: Fraction = Fraction(3, 4),
) -> MeterValidationMetrics:
    total_voice_bars = 0
    aligned_voice_bars = 0
    duration_closeness_sum = 0.0
    total_stream_bars = 0
    validated_stream_bars = 0
    strict_validated_bars = 0
    active_meter = header.meter

    for stream_line in stream_lines:
        voice_segments = _split_voice_segments(stream_line.body)
        populated = 0
        aligned = 0
        for voice, segment in voice_segments:
            if "|" not in segment:
                _segment_meter, active_meter = _segment_active_meter(segment, active_meter)
                continue
            base_length = header.voice_lengths.get(voice, header.default_length) if voice is not None else header.default_length
            events, active_meter = _segment_bar_durations(
                segment,
                base_length=base_length,
                active_meter=active_meter,
            )
            for duration, segment_meter in events:
                if duration == 0:
                    continue
                populated += 1
                total_voice_bars += 1
                duration_closeness_sum += _duration_closeness(duration, segment_meter)
                if duration == segment_meter:
                    aligned += 1
                    aligned_voice_bars += 1
        # Real Goldberg targets often mix a meter-aligned primary voice with
        # shorter accompaniment fragments or longer sustained lower voices in
        # the same streamed line. Count the line as a validated bar when at
        # least one populated voice cleanly spans the active meter, while the
        # separate meter_alignment_reward still measures the per-voice quality.
        if populated > 0 and aligned > 0:
            validated_stream_bars += 1
        if populated > 0 and aligned == populated:
            strict_validated_bars += 1
        if populated > 0:
            total_stream_bars += 1

    (
        _line_units,
        observed_musical_bars,
        observed_written_bars,
        _line_written_units,
        observed_written_bar_units,
    ) = _musical_bar_units_by_stream_line(
        stream_lines,
        header,
        musical_bar_unit=musical_bar_unit,
    )
    meter_alignment_reward = _safe_fraction(aligned_voice_bars, total_voice_bars)
    meter_duration_closeness_reward = duration_closeness_sum / total_voice_bars if total_voice_bars > 0 else 0.0
    bar_meter_consistency_reward = _safe_fraction(validated_stream_bars, total_stream_bars)
    strict_bar_meter_consistency_reward = _safe_fraction(strict_validated_bars, total_stream_bars)
    return MeterValidationMetrics(
        meter_alignment_reward=meter_alignment_reward,
        meter_duration_closeness_reward=meter_duration_closeness_reward,
        validated_bars=validated_stream_bars,
        observed_musical_bars=observed_musical_bars,
        written_bars=observed_written_bars,
        written_bar_units=observed_written_bar_units,
        strict_validated_bars=strict_validated_bars,
        bar_meter_consistency_reward=bar_meter_consistency_reward,
        strict_bar_meter_consistency_reward=strict_bar_meter_consistency_reward,
    )


def _abc_grammar_metrics(stream_lines: list[StreamLineFeatures], header: HeaderContext) -> AbcGrammarMetrics:
    used_voices = {
        voice
        for line in stream_lines
        for voice, _segment in _split_voice_segments(line.body)
        if voice is not None
    }
    declared_voices = set(header.voice_lengths)

    if used_voices:
        voice_declaration_reward = _safe_fraction(len(used_voices & declared_voices), len(used_voices))
        score_voice_reward = (
            _safe_fraction(len(used_voices & header.score_voices), len(used_voices)) if header.has_score else 1.0
        )
    else:
        voice_declaration_reward = 1.0
        score_voice_reward = 1.0

    return AbcGrammarMetrics(
        voice_declaration_reward=voice_declaration_reward,
        score_voice_reward=score_voice_reward,
    )


def _stream_line_metric_bundle(
    stream_lines: list[StreamLineFeatures],
    header: HeaderContext,
    musical_bar_unit: Fraction = Fraction(3, 4),
) -> StreamLineMetricBundle:
    global_total_voice_bars = 0
    global_aligned_voice_bars = 0
    global_duration_closeness_sum = 0.0
    global_total_stream_bars = 0
    global_validated_bars = 0
    global_strict_validated_bars = 0
    global_active_meter = header.meter
    global_used_voices: set[int] = set()

    local_meter_alignment: list[float] = []
    local_meter_duration: list[float] = []
    local_bar_meter: list[float] = []
    local_note_bearing: list[float] = []
    local_voice_declaration: list[float] = []
    local_score_voice: list[float] = []
    (
        local_musical_units,
        observed_musical_bars,
        observed_written_bars,
        local_written_units,
        observed_written_bar_units,
    ) = _musical_bar_units_by_stream_line(
        stream_lines,
        header,
        musical_bar_unit=musical_bar_unit,
    )

    declared_voices = set(header.voice_lengths)
    for stream_line in stream_lines:
        voice_segments = _split_voice_segments(stream_line.body)
        local_note_bearing.append(1.0 if any(_line_has_note(segment) for _voice, segment in voice_segments) else 0.0)
        used_voices = {voice for voice, _segment in voice_segments if voice is not None}
        global_used_voices.update(used_voices)

        if used_voices:
            local_voice_declaration.append(_safe_fraction(len(used_voices & declared_voices), len(used_voices)))
            local_score_voice.append(
                _safe_fraction(len(used_voices & header.score_voices), len(used_voices)) if header.has_score else 1.0
            )
        else:
            local_voice_declaration.append(1.0)
            local_score_voice.append(1.0)

        local_total_voice_bars = 0
        local_aligned_voice_bars = 0
        local_duration_closeness_sum = 0.0
        local_populated = 0
        local_aligned = 0
        local_active_meter = header.meter

        global_populated = 0
        global_aligned = 0

        for voice, segment in voice_segments:
            if "|" not in segment:
                _global_segment_meter, global_active_meter = _segment_active_meter(segment, global_active_meter)
                _local_segment_meter, local_active_meter = _segment_active_meter(segment, local_active_meter)
                continue
            base_length = header.voice_lengths.get(voice, header.default_length) if voice is not None else header.default_length
            global_events, global_active_meter = _segment_bar_durations(
                segment,
                base_length=base_length,
                active_meter=global_active_meter,
            )
            local_events, local_active_meter = _segment_bar_durations(
                segment,
                base_length=base_length,
                active_meter=local_active_meter,
            )
            for event_idx, (duration, global_segment_meter) in enumerate(global_events):
                if duration == 0:
                    continue
                local_segment_meter = (
                    local_events[event_idx][1] if event_idx < len(local_events) else global_segment_meter
                )

                global_populated += 1
                global_total_voice_bars += 1
                global_duration_closeness_sum += _duration_closeness(duration, global_segment_meter)
                if duration == global_segment_meter:
                    global_aligned += 1
                    global_aligned_voice_bars += 1

                local_populated += 1
                local_total_voice_bars += 1
                local_duration_closeness_sum += _duration_closeness(duration, local_segment_meter)
                if duration == local_segment_meter:
                    local_aligned += 1
                    local_aligned_voice_bars += 1

        if global_populated > 0 and global_aligned > 0:
            global_validated_bars += 1
        if global_populated > 0 and global_aligned == global_populated:
            global_strict_validated_bars += 1
        if global_populated > 0:
            global_total_stream_bars += 1

        local_meter_alignment.append(_safe_fraction(local_aligned_voice_bars, local_total_voice_bars))
        local_meter_duration.append(
            local_duration_closeness_sum / local_total_voice_bars if local_total_voice_bars > 0 else 0.0
        )
        local_bar_meter.append(1.0 if local_populated > 0 and local_aligned > 0 else 0.0)

    if global_used_voices:
        voice_declaration_reward = _safe_fraction(len(global_used_voices & declared_voices), len(global_used_voices))
        score_voice_reward = (
            _safe_fraction(len(global_used_voices & header.score_voices), len(global_used_voices))
            if header.has_score
            else 1.0
        )
    else:
        voice_declaration_reward = 1.0
        score_voice_reward = 1.0

    meter_metrics = MeterValidationMetrics(
        meter_alignment_reward=_safe_fraction(global_aligned_voice_bars, global_total_voice_bars),
        meter_duration_closeness_reward=(
            global_duration_closeness_sum / global_total_voice_bars if global_total_voice_bars > 0 else 0.0
        ),
        validated_bars=global_validated_bars,
        observed_musical_bars=observed_musical_bars,
        written_bars=observed_written_bars,
        written_bar_units=observed_written_bar_units,
        strict_validated_bars=global_strict_validated_bars,
        bar_meter_consistency_reward=_safe_fraction(global_validated_bars, global_total_stream_bars),
        strict_bar_meter_consistency_reward=_safe_fraction(global_strict_validated_bars, global_total_stream_bars),
    )
    grammar_metrics = AbcGrammarMetrics(
        voice_declaration_reward=voice_declaration_reward,
        score_voice_reward=score_voice_reward,
    )
    local_metrics = StreamLineLocalMetrics(
        meter_alignment_reward=local_meter_alignment,
        meter_duration_closeness_reward=local_meter_duration,
        bar_meter_consistency_reward=local_bar_meter,
        note_bearing_line_reward=local_note_bearing,
        musical_bar_units=local_musical_units,
        written_bar_units=local_written_units,
        voice_declaration_reward=local_voice_declaration,
        score_voice_reward=local_score_voice,
    )
    return StreamLineMetricBundle(
        meter_metrics=meter_metrics,
        grammar_metrics=grammar_metrics,
        local_metrics=local_metrics,
    )


def _stream_line_local_metrics(
    stream_lines: list[StreamLineFeatures],
    header: HeaderContext,
    musical_bar_unit: Fraction = Fraction(3, 4),
) -> StreamLineLocalMetrics:
    return _stream_line_metric_bundle(stream_lines, header, musical_bar_unit=musical_bar_unit).local_metrics


def _countdown_reward(stream_lines: list[StreamLineFeatures]) -> float:
    if not stream_lines:
        return 0.0
    return stream_tag_sequence_reward(
        [
            StreamLine(
                tag=StreamTag(index=line.index, marker=line.tag_marker),
                body=line.body,
                raw=f"[r:{line.index}/{line.tag_marker}]{line.body}",
            )
            for line in stream_lines
        ]
    )


def _line_closure_reward(stream_lines: list[StreamLineFeatures]) -> float:
    if not stream_lines:
        return 0.0
    return sum(1 for line in stream_lines if line.closed) / len(stream_lines)


def _bar_token_reward(stream_lines: list[StreamLineFeatures]) -> float:
    if not stream_lines:
        return 0.0
    return sum(1 for line in stream_lines if line.has_bar_token) / len(stream_lines)


def _note_bearing_line_reward(stream_lines: list[StreamLineFeatures]) -> float:
    if not stream_lines:
        return 0.0
    return sum(1 for line in stream_lines if _line_has_note(line.body)) / len(stream_lines)


def _bar_count_reward(observed_bars: float, expected_bars: float) -> float:
    if expected_bars <= 0:
        return 0.0
    return max(0.0, 1.0 - abs(observed_bars - expected_bars) / expected_bars)


def _completion_reward(observed_bars: float, expected_bars: float) -> float:
    if expected_bars <= 0:
        return 0.0
    return 1.0 if observed_bars >= expected_bars else 0.0


def _termination_penalty_reward(_observed_bars: float, _expected_bars: float) -> float:
    return 0.0


def _syntax_penalty_reward(clearly_malformed_syntax: bool) -> float:
    return -1.0 if clearly_malformed_syntax else 0.0


def _ungated_total_reward(
    *,
    config: GoldbergRewardConfig,
    parse_reward: float,
    countdown_reward: float,
    line_closure_reward: float,
    bar_token_reward: float,
    note_bearing_line_reward: float,
    meter_alignment_reward: float,
    meter_duration_closeness_reward: float,
    bar_meter_consistency_reward: float,
    bar_count_reward: float,
    expanded_bar_count_reward: float,
    voice_declaration_reward: float,
    score_voice_reward: float,
    completion_reward: float = 0.0,
    expanded_completion_reward: float = 0.0,
    syntax_penalty_reward: float = 0.0,
    termination_penalty_reward: float = 0.0,
) -> float:
    return (
        config.completion_weight * completion_reward
        + config.expanded_completion_weight * expanded_completion_reward
        + config.parse_weight * parse_reward
        + config.syntax_penalty_weight * syntax_penalty_reward
        + config.termination_penalty_weight * termination_penalty_reward
        + config.countdown_weight * countdown_reward
        + config.line_closure_weight * line_closure_reward
        + config.bar_token_weight * bar_token_reward
        + config.note_bearing_line_weight * note_bearing_line_reward
        + config.meter_alignment_weight * meter_alignment_reward
        + config.meter_duration_closeness_weight * meter_duration_closeness_reward
        + config.bar_meter_consistency_weight * bar_meter_consistency_reward
        + config.bar_count_weight * bar_count_reward
        + config.expanded_bar_count_weight * expanded_bar_count_reward
        + config.voice_declaration_weight * voice_declaration_reward
        + config.score_voice_weight * score_voice_reward
    )


def _total_reward(
    *,
    config: GoldbergRewardConfig,
    parse_reward: float,
    countdown_reward: float,
    line_closure_reward: float,
    bar_token_reward: float,
    meter_alignment_reward: float,
    meter_duration_closeness_reward: float,
    bar_meter_consistency_reward: float,
    bar_count_reward: float,
    voice_declaration_reward: float,
    score_voice_reward: float,
    note_bearing_line_reward: float = 1.0,
    completion_reward: float = 0.0,
    expanded_completion_reward: float = 0.0,
    expanded_bar_count_reward: float = 0.0,
    syntax_penalty_reward: float = 0.0,
    termination_penalty_reward: float = 0.0,
) -> float:
    return _ungated_total_reward(
        config=config,
        completion_reward=completion_reward,
        expanded_completion_reward=expanded_completion_reward,
        parse_reward=parse_reward,
        syntax_penalty_reward=syntax_penalty_reward,
        termination_penalty_reward=termination_penalty_reward,
        countdown_reward=countdown_reward,
        line_closure_reward=line_closure_reward,
        bar_token_reward=bar_token_reward,
        note_bearing_line_reward=note_bearing_line_reward,
        meter_alignment_reward=meter_alignment_reward,
        meter_duration_closeness_reward=meter_duration_closeness_reward,
        bar_meter_consistency_reward=bar_meter_consistency_reward,
        bar_count_reward=bar_count_reward,
        expanded_bar_count_reward=expanded_bar_count_reward,
        voice_declaration_reward=voice_declaration_reward,
        score_voice_reward=score_voice_reward,
    )


def score_candidate_file(
    candidate_path: str | Path,
    target: StructuralTarget,
    config: GoldbergRewardConfig | None = None,
) -> RewardBreakdown:
    config = config or GoldbergRewardConfig()
    candidate_path = Path(candidate_path)
    candidate_text = candidate_path.read_text(encoding="utf-8")
    stream_lines = _extract_stream_line_features(candidate_text)
    header = _extract_header_context(candidate_text)
    parse_metrics = _extract_parse_validation_metrics(candidate_text, stream_lines, config)

    observed_stream_lines = len(stream_lines)
    meter_metrics = _validated_bar_metrics(stream_lines, header, target.musical_bar_unit)
    grammar_metrics = _abc_grammar_metrics(stream_lines, header)
    meter_alignment_reward = meter_metrics.meter_alignment_reward
    primary_validated_bars = meter_metrics.validated_bars
    validated_bars = primary_validated_bars
    observed_repeat_expanded_bars = meter_metrics.observed_musical_bars
    observed_bars = float(meter_metrics.written_bar_units)
    parse_reward = parse_metrics.parse_reward
    countdown_reward = _countdown_reward(stream_lines)
    line_closure_reward = _line_closure_reward(stream_lines)
    bar_token_reward = _bar_token_reward(stream_lines)
    note_bearing_line_reward = _note_bearing_line_reward(stream_lines)
    expected_reward_bars = target.expected_bars
    expected_expanded_bars = target.expected_repeat_expanded_bars
    bar_count_reward = _bar_count_reward(observed_bars, expected_reward_bars)
    expanded_bar_count_reward = _bar_count_reward(observed_repeat_expanded_bars, expected_expanded_bars)
    completion_reward = _completion_reward(observed_bars, expected_reward_bars)
    expanded_completion_reward = _completion_reward(observed_repeat_expanded_bars, expected_expanded_bars)
    syntax_penalty_reward = _syntax_penalty_reward(parse_metrics.clearly_malformed_syntax)
    termination_penalty_reward = _termination_penalty_reward(observed_bars, expected_reward_bars)

    ungated_total_reward = _ungated_total_reward(
        config=config,
        completion_reward=completion_reward,
        parse_reward=parse_reward,
        syntax_penalty_reward=syntax_penalty_reward,
        termination_penalty_reward=termination_penalty_reward,
        countdown_reward=countdown_reward,
        line_closure_reward=line_closure_reward,
        bar_token_reward=bar_token_reward,
        note_bearing_line_reward=note_bearing_line_reward,
        meter_alignment_reward=meter_alignment_reward,
        meter_duration_closeness_reward=meter_metrics.meter_duration_closeness_reward,
        bar_meter_consistency_reward=meter_metrics.bar_meter_consistency_reward,
        bar_count_reward=bar_count_reward,
        expanded_bar_count_reward=expanded_bar_count_reward,
        voice_declaration_reward=grammar_metrics.voice_declaration_reward,
        score_voice_reward=grammar_metrics.score_voice_reward,
        expanded_completion_reward=expanded_completion_reward,
    )
    structural_validity_gate_reward = 1.0 if parse_metrics.parse_valid else 0.0
    total_reward = ungated_total_reward
    structural_validity_gate_adjustment = 0.0

    return RewardBreakdown(
        candidate_path=str(candidate_path),
        parse_valid=parse_metrics.parse_valid,
        clearly_malformed_syntax=parse_metrics.clearly_malformed_syntax,
        observed_stream_lines=observed_stream_lines,
        observed_bars=observed_bars,
        observed_written_bars=meter_metrics.written_bars,
        observed_repeat_expanded_bars=observed_repeat_expanded_bars,
        primary_validated_bars=primary_validated_bars,
        validated_bars=validated_bars,
        strict_validated_bars=meter_metrics.strict_validated_bars,
        completion_reward=completion_reward,
        expanded_completion_reward=expanded_completion_reward,
        parse_reward=parse_reward,
        parse_balanced_construct_reward=parse_metrics.balanced_construct_reward,
        parse_inline_field_reward=parse_metrics.inline_field_reward,
        parse_duration_sanity_reward=parse_metrics.duration_sanity_reward,
        parse_tokenizer_reward=parse_metrics.tokenizer_reward,
        parse_music21_reward=parse_metrics.music21_reward,
        syntax_penalty_reward=syntax_penalty_reward,
        termination_penalty_reward=termination_penalty_reward,
        countdown_reward=countdown_reward,
        line_closure_reward=line_closure_reward,
        bar_token_reward=bar_token_reward,
        note_bearing_line_reward=note_bearing_line_reward,
        meter_alignment_reward=meter_alignment_reward,
        meter_duration_closeness_reward=meter_metrics.meter_duration_closeness_reward,
        bar_meter_consistency_reward=meter_metrics.bar_meter_consistency_reward,
        strict_bar_meter_consistency_reward=meter_metrics.strict_bar_meter_consistency_reward,
        bar_count_reward=bar_count_reward,
        expanded_bar_count_reward=expanded_bar_count_reward,
        voice_declaration_reward=grammar_metrics.voice_declaration_reward,
        score_voice_reward=grammar_metrics.score_voice_reward,
        structural_validity_gate_reward=structural_validity_gate_reward,
        ungated_total_reward=ungated_total_reward,
        structural_validity_gate_adjustment=structural_validity_gate_adjustment,
        total_reward=total_reward,
    )


def _ensure_renderable_abc(text: str) -> str:
    text = text if text.endswith("\n") else text + "\n"
    header_prefix = []
    if not re.search(r"(?m)^X:", text):
        header_prefix.append("X:1")
    if not re.search(r"(?m)^L:", text):
        header_prefix.append("L:1/8")
    if not re.search(r"(?m)^M:", text):
        header_prefix.append("M:3/4")
    if not re.search(r"(?m)^K:", text):
        header_prefix.append("K:G")
    if header_prefix:
        text = "\n".join(header_prefix) + "\n" + text
    return text


def score_candidate_text(
    abc_text: str,
    target: StructuralTarget,
    config: GoldbergRewardConfig | None = None,
    candidate_name: str = "<memory>",
) -> RewardBreakdown:
    config = config or GoldbergRewardConfig()
    stream_lines = _extract_stream_line_features(abc_text)
    header = _extract_header_context(abc_text)
    parse_metrics = _extract_parse_validation_metrics(abc_text, stream_lines, config)

    observed_stream_lines = len(stream_lines)
    meter_metrics = _validated_bar_metrics(stream_lines, header, target.musical_bar_unit)
    grammar_metrics = _abc_grammar_metrics(stream_lines, header)
    meter_alignment_reward = meter_metrics.meter_alignment_reward
    primary_validated_bars = meter_metrics.validated_bars
    validated_bars = primary_validated_bars
    observed_repeat_expanded_bars = meter_metrics.observed_musical_bars
    observed_bars = float(meter_metrics.written_bar_units)
    parse_reward = parse_metrics.parse_reward
    countdown_reward = _countdown_reward(stream_lines)
    line_closure_reward = _line_closure_reward(stream_lines)
    bar_token_reward = _bar_token_reward(stream_lines)
    note_bearing_line_reward = _note_bearing_line_reward(stream_lines)
    expected_reward_bars = target.expected_bars
    expected_expanded_bars = target.expected_repeat_expanded_bars
    bar_count_reward = _bar_count_reward(observed_bars, expected_reward_bars)
    expanded_bar_count_reward = _bar_count_reward(observed_repeat_expanded_bars, expected_expanded_bars)
    completion_reward = _completion_reward(observed_bars, expected_reward_bars)
    expanded_completion_reward = _completion_reward(observed_repeat_expanded_bars, expected_expanded_bars)
    syntax_penalty_reward = _syntax_penalty_reward(parse_metrics.clearly_malformed_syntax)
    termination_penalty_reward = _termination_penalty_reward(observed_bars, expected_reward_bars)

    ungated_total_reward = _ungated_total_reward(
        config=config,
        completion_reward=completion_reward,
        parse_reward=parse_reward,
        syntax_penalty_reward=syntax_penalty_reward,
        termination_penalty_reward=termination_penalty_reward,
        countdown_reward=countdown_reward,
        line_closure_reward=line_closure_reward,
        bar_token_reward=bar_token_reward,
        note_bearing_line_reward=note_bearing_line_reward,
        meter_alignment_reward=meter_alignment_reward,
        meter_duration_closeness_reward=meter_metrics.meter_duration_closeness_reward,
        bar_meter_consistency_reward=meter_metrics.bar_meter_consistency_reward,
        bar_count_reward=bar_count_reward,
        expanded_bar_count_reward=expanded_bar_count_reward,
        voice_declaration_reward=grammar_metrics.voice_declaration_reward,
        score_voice_reward=grammar_metrics.score_voice_reward,
        expanded_completion_reward=expanded_completion_reward,
    )
    structural_validity_gate_reward = 1.0 if parse_metrics.parse_valid else 0.0
    total_reward = ungated_total_reward
    structural_validity_gate_adjustment = 0.0

    return RewardBreakdown(
        candidate_path=candidate_name,
        parse_valid=parse_metrics.parse_valid,
        clearly_malformed_syntax=parse_metrics.clearly_malformed_syntax,
        observed_stream_lines=observed_stream_lines,
        observed_bars=observed_bars,
        observed_written_bars=meter_metrics.written_bars,
        observed_repeat_expanded_bars=observed_repeat_expanded_bars,
        primary_validated_bars=primary_validated_bars,
        validated_bars=validated_bars,
        strict_validated_bars=meter_metrics.strict_validated_bars,
        completion_reward=completion_reward,
        expanded_completion_reward=expanded_completion_reward,
        parse_reward=parse_reward,
        parse_balanced_construct_reward=parse_metrics.balanced_construct_reward,
        parse_inline_field_reward=parse_metrics.inline_field_reward,
        parse_duration_sanity_reward=parse_metrics.duration_sanity_reward,
        parse_tokenizer_reward=parse_metrics.tokenizer_reward,
        parse_music21_reward=parse_metrics.music21_reward,
        syntax_penalty_reward=syntax_penalty_reward,
        termination_penalty_reward=termination_penalty_reward,
        countdown_reward=countdown_reward,
        line_closure_reward=line_closure_reward,
        bar_token_reward=bar_token_reward,
        note_bearing_line_reward=note_bearing_line_reward,
        meter_alignment_reward=meter_alignment_reward,
        meter_duration_closeness_reward=meter_metrics.meter_duration_closeness_reward,
        bar_meter_consistency_reward=meter_metrics.bar_meter_consistency_reward,
        strict_bar_meter_consistency_reward=meter_metrics.strict_bar_meter_consistency_reward,
        bar_count_reward=bar_count_reward,
        expanded_bar_count_reward=expanded_bar_count_reward,
        voice_declaration_reward=grammar_metrics.voice_declaration_reward,
        score_voice_reward=grammar_metrics.score_voice_reward,
        structural_validity_gate_reward=structural_validity_gate_reward,
        ungated_total_reward=ungated_total_reward,
        structural_validity_gate_adjustment=structural_validity_gate_adjustment,
        total_reward=total_reward,
    )


def score_candidate_text_with_local_metrics(
    abc_text: str,
    target: StructuralTarget,
    config: GoldbergRewardConfig | None = None,
    candidate_name: str = "<memory>",
) -> CandidateStructuralScore:
    config = config or GoldbergRewardConfig()
    stream_lines = _extract_stream_line_features(abc_text)
    header = _extract_header_context(abc_text)
    parse_metrics = _extract_parse_validation_metrics(abc_text, stream_lines, config)
    metric_bundle = _stream_line_metric_bundle(stream_lines, header, musical_bar_unit=target.musical_bar_unit)

    observed_stream_lines = len(stream_lines)
    meter_metrics = metric_bundle.meter_metrics
    grammar_metrics = metric_bundle.grammar_metrics
    meter_alignment_reward = meter_metrics.meter_alignment_reward
    primary_validated_bars = meter_metrics.validated_bars
    validated_bars = primary_validated_bars
    observed_repeat_expanded_bars = meter_metrics.observed_musical_bars
    observed_bars = float(meter_metrics.written_bar_units)
    parse_reward = parse_metrics.parse_reward
    countdown_reward = _countdown_reward(stream_lines)
    line_closure_reward = _line_closure_reward(stream_lines)
    bar_token_reward = _bar_token_reward(stream_lines)
    note_bearing_line_reward = _note_bearing_line_reward(stream_lines)
    expected_reward_bars = target.expected_bars
    expected_expanded_bars = target.expected_repeat_expanded_bars
    bar_count_reward = _bar_count_reward(observed_bars, expected_reward_bars)
    expanded_bar_count_reward = _bar_count_reward(observed_repeat_expanded_bars, expected_expanded_bars)
    completion_reward = _completion_reward(observed_bars, expected_reward_bars)
    expanded_completion_reward = _completion_reward(observed_repeat_expanded_bars, expected_expanded_bars)
    syntax_penalty_reward = _syntax_penalty_reward(parse_metrics.clearly_malformed_syntax)
    termination_penalty_reward = _termination_penalty_reward(observed_bars, expected_reward_bars)

    ungated_total_reward = _ungated_total_reward(
        config=config,
        completion_reward=completion_reward,
        parse_reward=parse_reward,
        syntax_penalty_reward=syntax_penalty_reward,
        termination_penalty_reward=termination_penalty_reward,
        countdown_reward=countdown_reward,
        line_closure_reward=line_closure_reward,
        bar_token_reward=bar_token_reward,
        note_bearing_line_reward=note_bearing_line_reward,
        meter_alignment_reward=meter_alignment_reward,
        meter_duration_closeness_reward=meter_metrics.meter_duration_closeness_reward,
        bar_meter_consistency_reward=meter_metrics.bar_meter_consistency_reward,
        bar_count_reward=bar_count_reward,
        expanded_bar_count_reward=expanded_bar_count_reward,
        voice_declaration_reward=grammar_metrics.voice_declaration_reward,
        score_voice_reward=grammar_metrics.score_voice_reward,
        expanded_completion_reward=expanded_completion_reward,
    )
    structural_validity_gate_reward = 1.0 if parse_metrics.parse_valid else 0.0
    total_reward = ungated_total_reward
    structural_validity_gate_adjustment = 0.0

    breakdown = RewardBreakdown(
        candidate_path=candidate_name,
        parse_valid=parse_metrics.parse_valid,
        clearly_malformed_syntax=parse_metrics.clearly_malformed_syntax,
        observed_stream_lines=observed_stream_lines,
        observed_bars=observed_bars,
        observed_written_bars=meter_metrics.written_bars,
        observed_repeat_expanded_bars=observed_repeat_expanded_bars,
        primary_validated_bars=primary_validated_bars,
        validated_bars=validated_bars,
        strict_validated_bars=meter_metrics.strict_validated_bars,
        completion_reward=completion_reward,
        expanded_completion_reward=expanded_completion_reward,
        parse_reward=parse_reward,
        parse_balanced_construct_reward=parse_metrics.balanced_construct_reward,
        parse_inline_field_reward=parse_metrics.inline_field_reward,
        parse_duration_sanity_reward=parse_metrics.duration_sanity_reward,
        parse_tokenizer_reward=parse_metrics.tokenizer_reward,
        parse_music21_reward=parse_metrics.music21_reward,
        syntax_penalty_reward=syntax_penalty_reward,
        termination_penalty_reward=termination_penalty_reward,
        countdown_reward=countdown_reward,
        line_closure_reward=line_closure_reward,
        bar_token_reward=bar_token_reward,
        note_bearing_line_reward=note_bearing_line_reward,
        meter_alignment_reward=meter_alignment_reward,
        meter_duration_closeness_reward=meter_metrics.meter_duration_closeness_reward,
        bar_meter_consistency_reward=meter_metrics.bar_meter_consistency_reward,
        strict_bar_meter_consistency_reward=meter_metrics.strict_bar_meter_consistency_reward,
        bar_count_reward=bar_count_reward,
        expanded_bar_count_reward=expanded_bar_count_reward,
        voice_declaration_reward=grammar_metrics.voice_declaration_reward,
        score_voice_reward=grammar_metrics.score_voice_reward,
        structural_validity_gate_reward=structural_validity_gate_reward,
        ungated_total_reward=ungated_total_reward,
        structural_validity_gate_adjustment=structural_validity_gate_adjustment,
        total_reward=total_reward,
    )
    return CandidateStructuralScore(
        breakdown=breakdown,
        stream_lines=stream_lines,
        local_metrics=metric_bundle.local_metrics,
    )


def score_prompt_completion_pair(
    prompt_text: str,
    completion_text: str,
    target: StructuralTarget,
    config: GoldbergRewardConfig | None = None,
    candidate_name: str = "<prompt+completion>",
) -> RewardBreakdown:
    return score_candidate_text(
        abc_text=prompt_text + completion_text,
        target=target,
        config=config,
        candidate_name=candidate_name,
    )


def make_trl_reward_func(
    target: StructuralTarget,
    config: GoldbergRewardConfig | None = None,
):
    config = config or GoldbergRewardConfig()

    def reward_func(prompts, completions, **kwargs):
        rewards = []
        for idx, (prompt, completion) in enumerate(zip(prompts, completions)):
            breakdown = score_prompt_completion_pair(
                prompt_text=prompt,
                completion_text=completion,
                target=target,
                config=config,
                candidate_name=f"sample-{idx}",
            )
            rewards.append(float(breakdown.total_reward))
        return rewards

    return reward_func


def compute_group_advantages(rewards: list[RewardBreakdown]) -> list[dict]:
    totals = [item.total_reward for item in rewards]
    if not totals:
        return []
    mean = sum(totals) / len(totals)
    variance = sum((x - mean) ** 2 for x in totals) / len(totals)
    std = math.sqrt(variance)
    denom = std if std > 1e-8 else 1.0
    rows = []
    for item in rewards:
        rows.append(
            {
                "candidate_path": item.candidate_path,
                "total_reward": item.total_reward,
                "advantage": (item.total_reward - mean) / denom,
            }
        )
    return rows
