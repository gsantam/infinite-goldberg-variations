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

    @property
    def expected_reward_bars(self) -> int:
        return self.expected_structure_bars


@dataclass
class GoldbergRewardConfig:
    parse_weight: float = 0.25
    countdown_weight: float = 0.25
    line_closure_weight: float = 0.25
    bar_token_weight: float = 0.10
    meter_alignment_weight: float = 0.75
    meter_duration_closeness_weight: float = 0.75
    bar_meter_consistency_weight: float = 0.75
    bar_count_weight: float = 3.0
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
    observed_stream_lines: int
    observed_bars: int
    primary_validated_bars: int
    validated_bars: int
    strict_validated_bars: int
    parse_reward: float
    countdown_reward: float
    line_closure_reward: float
    bar_token_reward: float
    meter_alignment_reward: float
    meter_duration_closeness_reward: float
    bar_meter_consistency_reward: float
    strict_bar_meter_consistency_reward: float
    bar_count_reward: float
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
    strict_validated_bars: int
    bar_meter_consistency_reward: float
    strict_bar_meter_consistency_reward: float


@dataclass(frozen=True)
class AbcGrammarMetrics:
    voice_declaration_reward: float
    score_voice_reward: float


@dataclass(frozen=True)
class StreamLineLocalMetrics:
    meter_alignment_reward: list[float]
    meter_duration_closeness_reward: list[float]
    bar_meter_consistency_reward: list[float]
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
                has_bar_token="|" in body,
                closed=stream_line_closed(line),
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


def _music21_parse_guard_tripped(
    abc_text: str,
    stream_lines: list[StreamLineFeatures],
    config: GoldbergRewardConfig,
) -> bool:
    meter_tokens = re.findall(r"(?:^|\n)M:([^\s\]]+)|\[M:([^\]]+)\]", abc_text)
    for header_token, inline_token in meter_tokens:
        token = header_token or inline_token
        meter = _parse_fraction_token(token, Fraction(0, 1))
        if meter <= 0 or _fraction_component_too_large(meter, config.max_music21_meter_component):
            return True

    token_pattern = re.compile(r"(\[[^\]]+\]|[_=^]*[A-Ga-gxz][,']*)(\d+(?:/\d*)?|/\d+|/)?")
    bracket_token_pattern = re.compile(r"\[([^\]\n]+)\](\d+(?:/\d*)?|/\d+|/)?")
    bracket_note_pattern = re.compile(r"[_=^]*[A-Ga-gxz][,']*(\d+(?:/\d*)?|/\d+|/)?")
    for line in stream_lines:
        cleaned = re.sub(r'"[^"\n]*"', " ", line.body)
        cleaned = re.sub(r"![^!\n]*!", " ", cleaned)
        cleaned = re.sub(r"\[[A-Za-z]:[^\]]*\]", " ", cleaned)
        for match in bracket_token_pattern.finditer(cleaned):
            content = match.group(1)
            outer_multiplier = _parse_length_multiplier(match.group(2))
            if _fraction_component_too_large(outer_multiplier, config.max_music21_duration_component):
                return True
            if not re.search(r"[_=^]*[A-Ga-gxz]", content):
                continue
            for note_match in bracket_note_pattern.finditer(content):
                multiplier = _parse_length_multiplier(note_match.group(1))
                if _fraction_component_too_large(multiplier, config.max_music21_duration_component):
                    return True
        for match in token_pattern.finditer(cleaned):
            multiplier = _parse_length_multiplier(match.group(2))
            if _fraction_component_too_large(multiplier, config.max_music21_duration_component):
                return True
    return False


def _extract_music21_candidate_features(
    abc_text: str,
    stream_lines: list[StreamLineFeatures],
    config: GoldbergRewardConfig,
) -> bool:
    mode = config.parse_validation_mode.replace("_", "-")
    if mode == "none":
        return True
    if mode not in {"music21", "abc-tokenize"}:
        raise ValueError(f"unsupported parse_validation_mode: {config.parse_validation_mode}")
    if _music21_parse_guard_tripped(abc_text, stream_lines, config):
        return False
    if mode == "abc-tokenize":
        try:
            abcFormat.ABCFile().readstr(_ensure_renderable_abc(abc_text))
            return True
        except Exception:
            return False
    try:
        with _music21_parse_time_limit(config.music21_parse_timeout_s):
            converter.parseData(_ensure_renderable_abc(abc_text), format="abc")
        return True
    except Exception:
        return False


def _segment_active_meter(segment: str, current_meter: Fraction) -> tuple[Fraction, Fraction]:
    active_meter = current_meter
    for match in re.finditer(r"\[M:([^\]]+)\]", segment):
        active_meter = _parse_fraction_token(match.group(1), active_meter)
    return active_meter, active_meter


def _duration_closeness(duration: Fraction, meter: Fraction) -> float:
    if duration <= 0 or meter <= 0:
        return 0.0
    return max(0.0, 1.0 - float(abs(duration - meter) / meter))


def _validated_bar_metrics(stream_lines: list[StreamLineFeatures], header: HeaderContext) -> MeterValidationMetrics:
    total_voice_bars = 0
    aligned_voice_bars = 0
    duration_closeness_sum = 0.0
    total_stream_bars = 0
    validated_bars = 0
    strict_validated_bars = 0
    active_meter = header.meter

    for stream_line in stream_lines:
        voice_segments = _split_voice_segments(stream_line.body)
        populated = 0
        aligned = 0
        for voice, segment in voice_segments:
            segment_meter, active_meter = _segment_active_meter(segment, active_meter)
            if "|" not in segment:
                continue
            base_length = header.voice_lengths.get(voice, header.default_length) if voice is not None else header.default_length
            duration = _voice_segment_duration(segment, base_length)
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
            validated_bars += 1
        if populated > 0 and aligned == populated:
            strict_validated_bars += 1
        if populated > 0:
            total_stream_bars += 1

    meter_alignment_reward = _safe_fraction(aligned_voice_bars, total_voice_bars)
    meter_duration_closeness_reward = duration_closeness_sum / total_voice_bars if total_voice_bars > 0 else 0.0
    bar_meter_consistency_reward = _safe_fraction(validated_bars, total_stream_bars)
    strict_bar_meter_consistency_reward = _safe_fraction(strict_validated_bars, total_stream_bars)
    return MeterValidationMetrics(
        meter_alignment_reward=meter_alignment_reward,
        meter_duration_closeness_reward=meter_duration_closeness_reward,
        validated_bars=validated_bars,
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


def _stream_line_metric_bundle(stream_lines: list[StreamLineFeatures], header: HeaderContext) -> StreamLineMetricBundle:
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
    local_voice_declaration: list[float] = []
    local_score_voice: list[float] = []

    declared_voices = set(header.voice_lengths)
    for stream_line in stream_lines:
        voice_segments = _split_voice_segments(stream_line.body)
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
            global_segment_meter, global_active_meter = _segment_active_meter(segment, global_active_meter)
            local_segment_meter, local_active_meter = _segment_active_meter(segment, local_active_meter)
            if "|" not in segment:
                continue
            base_length = header.voice_lengths.get(voice, header.default_length) if voice is not None else header.default_length
            duration = _voice_segment_duration(segment, base_length)
            if duration == 0:
                continue

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
        voice_declaration_reward=local_voice_declaration,
        score_voice_reward=local_score_voice,
    )
    return StreamLineMetricBundle(
        meter_metrics=meter_metrics,
        grammar_metrics=grammar_metrics,
        local_metrics=local_metrics,
    )


def _stream_line_local_metrics(stream_lines: list[StreamLineFeatures], header: HeaderContext) -> StreamLineLocalMetrics:
    return _stream_line_metric_bundle(stream_lines, header).local_metrics


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


def _bar_count_reward(observed_bars: int, expected_bars: int) -> float:
    if expected_bars <= 0:
        return 0.0
    return max(0.0, 1.0 - abs(observed_bars - expected_bars) / expected_bars)


def _ungated_total_reward(
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
) -> float:
    return (
        config.parse_weight * parse_reward
        + config.countdown_weight * countdown_reward
        + config.line_closure_weight * line_closure_reward
        + config.bar_token_weight * bar_token_reward
        + config.meter_alignment_weight * meter_alignment_reward
        + config.meter_duration_closeness_weight * meter_duration_closeness_reward
        + config.bar_meter_consistency_weight * bar_meter_consistency_reward
        + config.bar_count_weight * bar_count_reward
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
) -> float:
    return _ungated_total_reward(
        config=config,
        parse_reward=parse_reward,
        countdown_reward=countdown_reward,
        line_closure_reward=line_closure_reward,
        bar_token_reward=bar_token_reward,
        meter_alignment_reward=meter_alignment_reward,
        meter_duration_closeness_reward=meter_duration_closeness_reward,
        bar_meter_consistency_reward=bar_meter_consistency_reward,
        bar_count_reward=bar_count_reward,
        voice_declaration_reward=voice_declaration_reward,
        score_voice_reward=score_voice_reward,
    ) * parse_reward


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
    parse_valid = _extract_music21_candidate_features(candidate_text, stream_lines, config)

    observed_stream_lines = len(stream_lines)
    meter_metrics = _validated_bar_metrics(stream_lines, header)
    grammar_metrics = _abc_grammar_metrics(stream_lines, header)
    meter_alignment_reward = meter_metrics.meter_alignment_reward
    primary_validated_bars = meter_metrics.validated_bars
    validated_bars = primary_validated_bars
    observed_bars = validated_bars
    parse_reward = 1.0 if parse_valid else 0.0
    countdown_reward = _countdown_reward(stream_lines)
    line_closure_reward = _line_closure_reward(stream_lines)
    bar_token_reward = _bar_token_reward(stream_lines)
    expected_reward_bars = target.expected_reward_bars
    bar_count_reward = _bar_count_reward(observed_stream_lines, expected_reward_bars)

    ungated_total_reward = _ungated_total_reward(
        config=config,
        parse_reward=parse_reward,
        countdown_reward=countdown_reward,
        line_closure_reward=line_closure_reward,
        bar_token_reward=bar_token_reward,
        meter_alignment_reward=meter_alignment_reward,
        meter_duration_closeness_reward=meter_metrics.meter_duration_closeness_reward,
        bar_meter_consistency_reward=meter_metrics.bar_meter_consistency_reward,
        bar_count_reward=bar_count_reward,
        voice_declaration_reward=grammar_metrics.voice_declaration_reward,
        score_voice_reward=grammar_metrics.score_voice_reward,
    )
    structural_validity_gate_reward = parse_reward
    total_reward = ungated_total_reward * structural_validity_gate_reward
    structural_validity_gate_adjustment = total_reward - ungated_total_reward

    return RewardBreakdown(
        candidate_path=str(candidate_path),
        parse_valid=parse_valid,
        observed_stream_lines=observed_stream_lines,
        observed_bars=observed_bars,
        primary_validated_bars=primary_validated_bars,
        validated_bars=validated_bars,
        strict_validated_bars=meter_metrics.strict_validated_bars,
        parse_reward=parse_reward,
        countdown_reward=countdown_reward,
        line_closure_reward=line_closure_reward,
        bar_token_reward=bar_token_reward,
        meter_alignment_reward=meter_alignment_reward,
        meter_duration_closeness_reward=meter_metrics.meter_duration_closeness_reward,
        bar_meter_consistency_reward=meter_metrics.bar_meter_consistency_reward,
        strict_bar_meter_consistency_reward=meter_metrics.strict_bar_meter_consistency_reward,
        bar_count_reward=bar_count_reward,
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
    parse_valid = _extract_music21_candidate_features(abc_text, stream_lines, config)

    observed_stream_lines = len(stream_lines)
    meter_metrics = _validated_bar_metrics(stream_lines, header)
    grammar_metrics = _abc_grammar_metrics(stream_lines, header)
    meter_alignment_reward = meter_metrics.meter_alignment_reward
    primary_validated_bars = meter_metrics.validated_bars
    validated_bars = primary_validated_bars
    observed_bars = validated_bars
    parse_reward = 1.0 if parse_valid else 0.0
    countdown_reward = _countdown_reward(stream_lines)
    line_closure_reward = _line_closure_reward(stream_lines)
    bar_token_reward = _bar_token_reward(stream_lines)
    expected_reward_bars = target.expected_reward_bars
    bar_count_reward = _bar_count_reward(observed_stream_lines, expected_reward_bars)

    ungated_total_reward = _ungated_total_reward(
        config=config,
        parse_reward=parse_reward,
        countdown_reward=countdown_reward,
        line_closure_reward=line_closure_reward,
        bar_token_reward=bar_token_reward,
        meter_alignment_reward=meter_alignment_reward,
        meter_duration_closeness_reward=meter_metrics.meter_duration_closeness_reward,
        bar_meter_consistency_reward=meter_metrics.bar_meter_consistency_reward,
        bar_count_reward=bar_count_reward,
        voice_declaration_reward=grammar_metrics.voice_declaration_reward,
        score_voice_reward=grammar_metrics.score_voice_reward,
    )
    structural_validity_gate_reward = parse_reward
    total_reward = ungated_total_reward * structural_validity_gate_reward
    structural_validity_gate_adjustment = total_reward - ungated_total_reward

    return RewardBreakdown(
        candidate_path=candidate_name,
        parse_valid=parse_valid,
        observed_stream_lines=observed_stream_lines,
        observed_bars=observed_bars,
        primary_validated_bars=primary_validated_bars,
        validated_bars=validated_bars,
        strict_validated_bars=meter_metrics.strict_validated_bars,
        parse_reward=parse_reward,
        countdown_reward=countdown_reward,
        line_closure_reward=line_closure_reward,
        bar_token_reward=bar_token_reward,
        meter_alignment_reward=meter_alignment_reward,
        meter_duration_closeness_reward=meter_metrics.meter_duration_closeness_reward,
        bar_meter_consistency_reward=meter_metrics.bar_meter_consistency_reward,
        strict_bar_meter_consistency_reward=meter_metrics.strict_bar_meter_consistency_reward,
        bar_count_reward=bar_count_reward,
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
    parse_valid = _extract_music21_candidate_features(abc_text, stream_lines, config)
    metric_bundle = _stream_line_metric_bundle(stream_lines, header)

    observed_stream_lines = len(stream_lines)
    meter_metrics = metric_bundle.meter_metrics
    grammar_metrics = metric_bundle.grammar_metrics
    meter_alignment_reward = meter_metrics.meter_alignment_reward
    primary_validated_bars = meter_metrics.validated_bars
    validated_bars = primary_validated_bars
    observed_bars = validated_bars
    parse_reward = 1.0 if parse_valid else 0.0
    countdown_reward = _countdown_reward(stream_lines)
    line_closure_reward = _line_closure_reward(stream_lines)
    bar_token_reward = _bar_token_reward(stream_lines)
    expected_reward_bars = target.expected_reward_bars
    bar_count_reward = _bar_count_reward(observed_stream_lines, expected_reward_bars)

    ungated_total_reward = _ungated_total_reward(
        config=config,
        parse_reward=parse_reward,
        countdown_reward=countdown_reward,
        line_closure_reward=line_closure_reward,
        bar_token_reward=bar_token_reward,
        meter_alignment_reward=meter_alignment_reward,
        meter_duration_closeness_reward=meter_metrics.meter_duration_closeness_reward,
        bar_meter_consistency_reward=meter_metrics.bar_meter_consistency_reward,
        bar_count_reward=bar_count_reward,
        voice_declaration_reward=grammar_metrics.voice_declaration_reward,
        score_voice_reward=grammar_metrics.score_voice_reward,
    )
    structural_validity_gate_reward = parse_reward
    total_reward = ungated_total_reward * structural_validity_gate_reward
    structural_validity_gate_adjustment = total_reward - ungated_total_reward

    breakdown = RewardBreakdown(
        candidate_path=candidate_name,
        parse_valid=parse_valid,
        observed_stream_lines=observed_stream_lines,
        observed_bars=observed_bars,
        primary_validated_bars=primary_validated_bars,
        validated_bars=validated_bars,
        strict_validated_bars=meter_metrics.strict_validated_bars,
        parse_reward=parse_reward,
        countdown_reward=countdown_reward,
        line_closure_reward=line_closure_reward,
        bar_token_reward=bar_token_reward,
        meter_alignment_reward=meter_alignment_reward,
        meter_duration_closeness_reward=meter_metrics.meter_duration_closeness_reward,
        bar_meter_consistency_reward=meter_metrics.bar_meter_consistency_reward,
        strict_bar_meter_consistency_reward=meter_metrics.strict_bar_meter_consistency_reward,
        bar_count_reward=bar_count_reward,
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
