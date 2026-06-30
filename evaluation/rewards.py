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

from music21 import converter
from music21.pitch import Pitch

from grpo.stream_tags import (
    StreamLine,
    StreamTag,
    extract_stream_lines,
    stream_line_closed,
    stream_tag_sequence_reward,
)


CADENCE_BARS = {8, 16, 24, 32}
NOTE_TO_PC = {
    "C": 0,
    "^C": 1,
    "_D": 1,
    "D": 2,
    "^D": 3,
    "_E": 3,
    "E": 4,
    "F": 5,
    "^F": 6,
    "_G": 6,
    "G": 7,
    "^G": 8,
    "_A": 8,
    "A": 9,
    "^A": 10,
    "_B": 10,
    "B": 11,
}
PC_TO_NAME = {
    0: "C",
    1: "C#",
    2: "D",
    3: "D#",
    4: "E",
    5: "F",
    6: "F#",
    7: "G",
    8: "G#",
    9: "A",
    10: "A#",
    11: "B",
}


@dataclass
class StructuralBarTarget:
    bar_index: int
    chord_root: str | None
    bass_pitch_class: str | None
    bass_midi: int | None
    cadence_bar: bool


@dataclass
class StructuralTarget:
    expected_bars: int
    bars: list[StructuralBarTarget]


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
    repeat_syntax_weight: float = 0.0
    root_weight: float = 2.0
    bass_pc_weight: float = 2.0
    cadence_root_weight: float = 3.0
    cadence_bass_weight: float = 3.0
    harmonic_bar_progress_power: float = 1.0
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
    repeat_syntax_reward: float
    root_similarity_reward: float
    bass_pitch_class_reward: float
    cadence_root_reward: float
    cadence_bass_reward: float
    total_reward: float

    def to_json(self) -> dict:
        return asdict(self)


@dataclass
class CandidateBarFeatures:
    bar_index: int
    chord_root: str | None
    bass_pitch_class: str | None
    bass_midi: int | None


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
    repeat_syntax_reward: float


def load_structural_target(path: str | Path) -> StructuralTarget:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    bars = [StructuralBarTarget(**row) for row in rows]
    return StructuralTarget(expected_bars=len(bars), bars=bars)


def _pitch_name_from_midi(midi_value: int) -> str:
    pitch = Pitch()
    pitch.midi = midi_value
    return pitch.name


def _extract_candidate_features(score) -> list[CandidateBarFeatures]:
    chordified = score.chordify()
    measures = list(chordified.recurse().getElementsByClass("Measure"))
    features: list[CandidateBarFeatures] = []
    for idx, measure in enumerate(measures, start=1):
        chord_root = None
        for chord in measure.recurse().getElementsByClass("Chord"):
            if chord.pitches:
                root = chord.root()
                chord_root = root.name if root else None
                break

        lowest_midi = None
        for note in measure.recurse().notes:
            note_pitches = note.pitches if hasattr(note, "pitches") else [note.pitch]
            for pitch in note_pitches:
                if lowest_midi is None or pitch.midi < lowest_midi:
                    lowest_midi = pitch.midi

        features.append(
            CandidateBarFeatures(
                bar_index=idx,
                chord_root=chord_root,
                bass_pitch_class=_pitch_name_from_midi(lowest_midi) if lowest_midi is not None else None,
                bass_midi=lowest_midi,
            )
        )
    return features


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
    for line in stream_lines:
        cleaned = re.sub(r'"[^"\n]*"', " ", line.body)
        cleaned = re.sub(r"![^!\n]*!", " ", cleaned)
        cleaned = re.sub(r"\[[A-Za-z]:[^\]]*\]", " ", cleaned)
        for match in token_pattern.finditer(cleaned):
            multiplier = _parse_length_multiplier(match.group(2))
            if _fraction_component_too_large(multiplier, config.max_music21_duration_component):
                return True
    return False


def _extract_music21_candidate_features(
    abc_text: str,
    stream_lines: list[StreamLineFeatures],
    config: GoldbergRewardConfig,
) -> tuple[bool, list[CandidateBarFeatures]]:
    if _music21_parse_guard_tripped(abc_text, stream_lines, config):
        return False, []
    try:
        with _music21_parse_time_limit(config.music21_parse_timeout_s):
            score = converter.parseData(_ensure_renderable_abc(abc_text), format="abc")
        return True, _extract_candidate_features(score)
    except Exception:
        return False, []


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

    repeat_errors = 0
    open_repeats = 0
    repeat_text = "\n".join(line.body for line in stream_lines)
    repeat_text = re.sub(r'"[^"\n]*"', " ", repeat_text)
    repeat_text = re.sub(r"![^!\n]*!", " ", repeat_text)
    for match in re.finditer(r":\|\d+|\|\d+|\[\d+|\|:|:\||::", repeat_text):
        token = match.group(0)
        if token == "|:":
            open_repeats += 1
        elif token == "::":
            if open_repeats <= 0:
                repeat_errors += 1
            else:
                open_repeats -= 1
            open_repeats += 1
        elif token.startswith(":|"):
            if open_repeats <= 0:
                repeat_errors += 1
            else:
                open_repeats -= 1
        elif token.startswith("|") or token.startswith("["):
            if open_repeats <= 0:
                repeat_errors += 1

    repeat_errors += open_repeats
    repeat_syntax_reward = 1.0 / (1.0 + repeat_errors)
    return AbcGrammarMetrics(
        voice_declaration_reward=voice_declaration_reward,
        score_voice_reward=score_voice_reward,
        repeat_syntax_reward=repeat_syntax_reward,
    )


def _parse_abc_note_token(token: str) -> tuple[str, int] | None:
    match = re.match(r"([_=^]*)([A-Ga-g])([,']*)", token)
    if not match:
        return None
    accidental_raw, letter_raw, octave_marks = match.groups()
    accidental = accidental_raw.replace("=", "")
    if len(accidental) > 1:
        accidental = accidental[0]
    letter = letter_raw.upper()
    key = f"{accidental}{letter}" if accidental else letter
    if key not in NOTE_TO_PC:
        return None
    pc = NOTE_TO_PC[key]
    octave = 5 if letter_raw.islower() else 4
    for ch in octave_marks:
        octave += 1 if ch == "'" else -1
    midi = 12 * (octave + 1) + pc
    return PC_TO_NAME[pc], midi


def _extract_stream_line_candidate_features(stream_lines: list[StreamLineFeatures]) -> list[CandidateBarFeatures]:
    features: list[CandidateBarFeatures] = []
    for line_no, stream_line in enumerate(stream_lines, start=1):
        pitch_classes: set[str] = set()
        bass_midi: int | None = None
        bass_pc: str | None = None

        for voice, segment in _split_voice_segments(stream_line.body):
            cleaned = re.sub(r'"[^"\n]*"', " ", segment)
            cleaned = re.sub(r"![^!\n]*!", " ", cleaned)
            cleaned = re.sub(r"\[[A-Za-z]:[^\]]*\]", " ", cleaned)
            for note_match in re.finditer(r"([_=^]*[A-Ga-g][,']*)", cleaned):
                parsed = _parse_abc_note_token(note_match.group(1))
                if parsed is None:
                    continue
                pc_name, midi = parsed
                pitch_classes.add(pc_name)
                if bass_midi is None or midi < bass_midi:
                    bass_midi = midi
                    bass_pc = pc_name
                if voice is not None and voice >= 4:
                    if bass_midi is None or midi <= bass_midi:
                        bass_midi = midi
                        bass_pc = pc_name

        root_pc = _infer_root_from_pitch_classes(pitch_classes)
        features.append(
            CandidateBarFeatures(
                bar_index=line_no,
                chord_root=root_pc,
                bass_pitch_class=bass_pc,
                bass_midi=bass_midi,
            )
        )
    return features


def _infer_root_from_pitch_classes(pitch_classes: set[str]) -> str | None:
    if not pitch_classes:
        return None
    pcs = {pc for pc, name in PC_TO_NAME.items() if name in pitch_classes}
    if not pcs:
        return None

    triad_templates = [
        {0, 4, 7},
        {0, 3, 7},
        {0, 3, 6},
        {0, 4, 8},
        {0, 5, 7},
    ]

    best_root: int | None = None
    best_score = -1
    for root in range(12):
        score = 0
        if root in pcs:
            score += 2
        for template in triad_templates:
            hits = len({(root + interval) % 12 for interval in template} & pcs)
            score = max(score, hits)
        if score > best_score:
            best_score = score
            best_root = root

    return PC_TO_NAME[best_root] if best_root is not None else None


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


def _bar_progress(validated_bars: int, expected_bars: int, power: float = 1.0) -> float:
    if expected_bars <= 0:
        return 0.0
    progress = max(0.0, min(1.0, validated_bars / expected_bars))
    return progress**power


def _total_reward(
    *,
    config: GoldbergRewardConfig,
    expected_bars: int,
    validated_bars: int,
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
    repeat_syntax_reward: float,
    root_similarity_reward: float,
    bass_pitch_class_reward: float,
    cadence_root_reward: float,
    cadence_bass_reward: float,
) -> float:
    harmonic_progress = _bar_progress(
        validated_bars,
        expected_bars,
        power=config.harmonic_bar_progress_power,
    )
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
        + config.repeat_syntax_weight * repeat_syntax_reward
        + harmonic_progress
        * (
            config.root_weight * root_similarity_reward
            + config.bass_pc_weight * bass_pitch_class_reward
            + config.cadence_root_weight * cadence_root_reward
            + config.cadence_bass_weight * cadence_bass_reward
        )
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

    parse_valid, features = _extract_music21_candidate_features(candidate_text, stream_lines, config)

    if not features:
        features = _extract_stream_line_candidate_features(stream_lines)

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
    bar_count_reward = _bar_count_reward(validated_bars, target.expected_bars)

    overlap = min(len(features), len(target.bars))
    root_matches = 0
    root_total = 0
    bass_matches = 0
    bass_total = 0
    cadence_root_matches = 0
    cadence_root_total = 0
    cadence_bass_matches = 0
    cadence_bass_total = 0

    for idx in range(overlap):
        candidate_bar = features[idx]
        target_bar = target.bars[idx]

        if target_bar.chord_root is not None and candidate_bar.chord_root is not None:
            root_total += 1
            if candidate_bar.chord_root == target_bar.chord_root:
                root_matches += 1

        if target_bar.bass_pitch_class is not None and candidate_bar.bass_pitch_class is not None:
            bass_total += 1
            if candidate_bar.bass_pitch_class == target_bar.bass_pitch_class:
                bass_matches += 1

        if target_bar.cadence_bar:
            if target_bar.chord_root is not None and candidate_bar.chord_root is not None:
                cadence_root_total += 1
                if candidate_bar.chord_root == target_bar.chord_root:
                    cadence_root_matches += 1
            if target_bar.bass_pitch_class is not None and candidate_bar.bass_pitch_class is not None:
                cadence_bass_total += 1
                if candidate_bar.bass_pitch_class == target_bar.bass_pitch_class:
                    cadence_bass_matches += 1

    root_similarity_reward = _safe_fraction(root_matches, root_total)
    bass_pitch_class_reward = _safe_fraction(bass_matches, bass_total)
    cadence_root_reward = _safe_fraction(cadence_root_matches, cadence_root_total)
    cadence_bass_reward = _safe_fraction(cadence_bass_matches, cadence_bass_total)

    total_reward = _total_reward(
        config=config,
        expected_bars=target.expected_bars,
        validated_bars=validated_bars,
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
        repeat_syntax_reward=grammar_metrics.repeat_syntax_reward,
        root_similarity_reward=root_similarity_reward,
        bass_pitch_class_reward=bass_pitch_class_reward,
        cadence_root_reward=cadence_root_reward,
        cadence_bass_reward=cadence_bass_reward,
    )

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
        repeat_syntax_reward=grammar_metrics.repeat_syntax_reward,
        root_similarity_reward=root_similarity_reward,
        bass_pitch_class_reward=bass_pitch_class_reward,
        cadence_root_reward=cadence_root_reward,
        cadence_bass_reward=cadence_bass_reward,
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
    parse_valid, features = _extract_music21_candidate_features(abc_text, stream_lines, config)

    if not features:
        features = _extract_stream_line_candidate_features(stream_lines)

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
    bar_count_reward = _bar_count_reward(validated_bars, target.expected_bars)

    overlap = min(len(features), len(target.bars))
    root_matches = 0
    root_total = 0
    bass_matches = 0
    bass_total = 0
    cadence_root_matches = 0
    cadence_root_total = 0
    cadence_bass_matches = 0
    cadence_bass_total = 0

    for idx in range(overlap):
        candidate_bar = features[idx]
        target_bar = target.bars[idx]

        if target_bar.chord_root is not None and candidate_bar.chord_root is not None:
            root_total += 1
            if candidate_bar.chord_root == target_bar.chord_root:
                root_matches += 1

        if target_bar.bass_pitch_class is not None and candidate_bar.bass_pitch_class is not None:
            bass_total += 1
            if candidate_bar.bass_pitch_class == target_bar.bass_pitch_class:
                bass_matches += 1

        if target_bar.cadence_bar:
            if target_bar.chord_root is not None and candidate_bar.chord_root is not None:
                cadence_root_total += 1
                if candidate_bar.chord_root == target_bar.chord_root:
                    cadence_root_matches += 1
            if target_bar.bass_pitch_class is not None and candidate_bar.bass_pitch_class is not None:
                cadence_bass_total += 1
                if candidate_bar.bass_pitch_class == target_bar.bass_pitch_class:
                    cadence_bass_matches += 1

    root_similarity_reward = _safe_fraction(root_matches, root_total)
    bass_pitch_class_reward = _safe_fraction(bass_matches, bass_total)
    cadence_root_reward = _safe_fraction(cadence_root_matches, cadence_root_total)
    cadence_bass_reward = _safe_fraction(cadence_bass_matches, cadence_bass_total)

    total_reward = _total_reward(
        config=config,
        expected_bars=target.expected_bars,
        validated_bars=validated_bars,
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
        repeat_syntax_reward=grammar_metrics.repeat_syntax_reward,
        root_similarity_reward=root_similarity_reward,
        bass_pitch_class_reward=bass_pitch_class_reward,
        cadence_root_reward=cadence_root_reward,
        cadence_bass_reward=cadence_bass_reward,
    )

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
        repeat_syntax_reward=grammar_metrics.repeat_syntax_reward,
        root_similarity_reward=root_similarity_reward,
        bass_pitch_class_reward=bass_pitch_class_reward,
        cadence_root_reward=cadence_root_reward,
        cadence_bass_reward=cadence_bass_reward,
        total_reward=total_reward,
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
