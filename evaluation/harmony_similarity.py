from __future__ import annotations

import re
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


NOTE_RE = re.compile(
    r"(?P<acc>\^{1,2}|_{1,2}|=)?(?P<note>[A-Ga-g])(?P<oct>[,']*)(?P<dur>\d+(?:/\d+)?|/\d+|/+)?"
)
BASE_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
QUALITIES = {
    "maj": {0, 4, 7},
    "min": {0, 3, 7},
    "dim": {0, 3, 6},
    "aug": {0, 4, 8},
    "sus4": {0, 5, 7},
    "dom7": {0, 4, 7, 10},
    "min7": {0, 3, 7, 10},
    "maj7": {0, 4, 7, 11},
}


@dataclass(frozen=True)
class DtwAlignment:
    similarity: float
    path: list[tuple[int, int]]
    local_similarities: list[float]


def parse_duration(text: str | None) -> float:
    if not text:
        return 1.0
    if text.isdigit():
        return float(text)
    if text.startswith("/"):
        return 1.0 / (2 ** len(text)) if set(text) == {"/"} else 1.0 / float(text[1:])
    if "/" in text:
        numerator, denominator = text.split("/", maxsplit=1)
        return float(numerator or 1) / float(denominator or 2)
    return 1.0


def note_midi_pc(match: re.Match[str]) -> tuple[int, int, float]:
    accidental = match.group("acc") or ""
    note = match.group("note")
    octave_marks = match.group("oct") or ""
    pc = BASE_PC[note.upper()]
    if accidental.startswith("^"):
        pc += len(accidental)
    elif accidental.startswith("_"):
        pc -= len(accidental)
    pc %= 12
    octave = 4 if note.isupper() else 5
    octave += octave_marks.count("'") - octave_marks.count(",")
    midi = (octave + 1) * 12 + pc
    return midi, pc, parse_duration(match.group("dur"))


def strip_inline_tags(bar: str) -> str:
    bar = re.sub(r"\[V:[^\]]+\]", " ", bar)
    bar = re.sub(r"\[r:\d+/\d+\]", " ", bar)
    bar = re.sub(r"!.*?!", " ", bar)
    bar = re.sub(r'"[^"]*"', " ", bar)
    return bar


def parse_bar_notes(bar: str) -> list[tuple[int, int, float]]:
    return [note_midi_pc(match) for match in NOTE_RE.finditer(strip_inline_tags(bar))]


def voice_bars_from_text(text: str) -> list[list[str]]:
    voices: list[list[str]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line.startswith("[V:"):
            continue
        body = re.sub(r"^\[V:[^\]]+\]\s*", "", line)
        parts = re.split(r"\|+", body)
        bars = [part.strip(" []") for part in parts if part.strip(" []")]
        if bars:
            voices.append(bars)
    return voices


def stream_bars_from_text(text: str) -> list[str]:
    bars = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("[V:") and "|" in line:
            bars.append(line)
    return bars


def looks_like_stream_bars(text: str) -> bool:
    bars = stream_bars_from_text(text)
    if len(bars) < 4:
        return False
    voice_ids = []
    for line in bars:
        match = re.match(r"^\[V:([^\]]+)\]", line)
        if match:
            voice_ids.append(match.group(1))
    repeated_leading_voice = len(voice_ids) != len(set(voice_ids))
    mostly_single_bar_lines = sum(line.count("|") <= 2 for line in bars) >= len(bars) * 0.75
    multi_voice_stream_lines = sum(line.count("[V:") > 1 for line in bars) >= len(bars) * 0.25
    return repeated_leading_voice or mostly_single_bar_lines or multi_voice_stream_lines


def piece_bars_from_text(text: str) -> list[list[tuple[int, int, float]]]:
    if looks_like_stream_bars(text):
        return [parse_bar_notes(bar) for bar in stream_bars_from_text(text)]

    voices = voice_bars_from_text(text)
    bar_count = max((len(voice) for voice in voices), default=0)
    bars = []
    for bar_index in range(bar_count):
        notes: list[tuple[int, int, float]] = []
        for voice in voices:
            if bar_index < len(voice):
                notes.extend(parse_bar_notes(voice[bar_index]))
        bars.append(notes)
    return bars


def infer_harmony(notes: list[tuple[int, int, float]]) -> dict[str, Any]:
    if not notes:
        return {"root": None, "quality": None, "bass": None, "pcs": [], "note_count": 0, "score": 0.0}
    weighted = Counter()
    for _midi, pc, duration in notes:
        weighted[pc] += max(duration, 0.25)
    pcs = set(weighted)
    bass_pc = min(notes, key=lambda item: item[0])[1]
    best: tuple[float, int, str] | None = None
    for root in range(12):
        relative_pcs = {(pc - root) % 12 for pc in pcs}
        for quality, template_pcs in QUALITIES.items():
            template = set(template_pcs)
            intersection = len(relative_pcs & template)
            extra = len(relative_pcs - template)
            missing = len(template - relative_pcs)
            score = intersection - 0.35 * extra - 0.55 * missing
            bass_relative = (bass_pc - root) % 12
            if bass_relative == 0:
                score += 0.35
            elif bass_relative in (3, 4, 7):
                score += 0.15
            if best is None or score > best[0]:
                best = (score, root, quality)
    assert best is not None
    return {
        "root": best[1],
        "quality": best[2],
        "bass": bass_pc,
        "top_midi": max(midi for midi, _pc, _duration in notes),
        "bass_midi": min(midi for midi, _pc, _duration in notes),
        "pcs": sorted(pcs),
        "note_count": len(notes),
        "score": best[0],
    }


def harmony_from_text(text: str) -> list[dict[str, Any]]:
    return [infer_harmony(notes) for notes in piece_bars_from_text(text)]


def harmony_from_path(path: str | Path) -> list[dict[str, Any]]:
    return harmony_from_text(Path(path).read_text(encoding="utf-8"))


def token_similarity(left: dict[str, Any], right: dict[str, Any]) -> float:
    if left["root"] is None or right["root"] is None:
        return 0.0
    root = 1.0 if left["root"] == right["root"] else (0.5 if (left["root"] - right["root"]) % 12 in (5, 7) else 0.0)
    bass = 1.0 if left["bass"] == right["bass"] else (0.5 if (left["bass"] - right["bass"]) % 12 in (5, 7) else 0.0)
    quality = 1.0 if left["quality"] == right["quality"] else 0.0
    return 0.5 * root + 0.35 * bass + 0.15 * quality


def pitch_class_similarity(left: int | None, right: int | None) -> float:
    if left is None or right is None:
        return 0.0
    if left == right:
        return 1.0
    return 0.5 if (left - right) % 12 in (5, 7) else 0.0


def scalar_similarity(left: float, right: float, *, scale: float) -> float:
    return max(0.0, 1.0 - abs(left - right) / scale)


def contour_sequence(harmony: list[dict[str, Any]]) -> list[int]:
    sequence = []
    previous: int | None = None
    for item in harmony:
        current = item.get("top_midi")
        if current is None:
            continue
        if previous is not None:
            delta = current - previous
            if delta > 2:
                sequence.append(1)
            elif delta < -2:
                sequence.append(-1)
            else:
                sequence.append(0)
        previous = current
    return sequence


def generic_dtw_alignment(
    reference: list[Any],
    candidate: list[Any],
    similarity_fn: Callable[[Any, Any], float],
    *,
    band_ratio: float,
) -> DtwAlignment:
    n = len(reference)
    m = len(candidate)
    if n == 0 or m == 0:
        return DtwAlignment(similarity=0.0, path=[], local_similarities=[])
    inf = 1e9
    costs = [[inf] * (m + 1) for _ in range(n + 1)]
    traceback: list[list[tuple[int, int] | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    costs[0][0] = 0.0
    band = max(abs(n - m), int(max(n, m) * band_ratio))
    for i in range(1, n + 1):
        center = int(i * m / n)
        for j in range(max(1, center - band), min(m, center + band) + 1):
            cost = 1.0 - similarity_fn(reference[i - 1], candidate[j - 1])
            predecessors = (
                (costs[i - 1][j], (i - 1, j)),
                (costs[i][j - 1], (i, j - 1)),
                (costs[i - 1][j - 1], (i - 1, j - 1)),
            )
            prev_cost, prev_idx = min(predecessors, key=lambda item: item[0])
            costs[i][j] = cost + prev_cost
            traceback[i][j] = prev_idx

    if costs[n][m] >= inf:
        return DtwAlignment(similarity=0.0, path=[], local_similarities=[])

    path: list[tuple[int, int]] = []
    local_similarities: list[float] = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        local_similarities.append(similarity_fn(reference[i - 1], candidate[j - 1]))
        previous = traceback[i][j]
        if previous is None:
            break
        i, j = previous
    path.reverse()
    local_similarities.reverse()

    distance = costs[n][m] / (n + m)
    return DtwAlignment(similarity=1.0 / (1.0 + distance), path=path, local_similarities=local_similarities)


def generic_dtw_similarity(reference: list[Any], candidate: list[Any], similarity_fn, *, band_ratio: float) -> float:
    return generic_dtw_alignment(reference, candidate, similarity_fn, band_ratio=band_ratio).similarity


def aligned_similarity(reference: list[dict[str, Any]], candidate: list[dict[str, Any]]) -> dict[str, float | int]:
    root_scores: list[float] = []
    bass_scores: list[float] = []
    quality_scores: list[float] = []
    combined_scores: list[float] = []
    for left, right in zip(reference, candidate):
        if left["root"] is None or right["root"] is None:
            continue
        root_scores.append(1.0 if left["root"] == right["root"] else 0.0)
        bass_scores.append(1.0 if left["bass"] == right["bass"] else 0.0)
        quality_scores.append(1.0 if left["quality"] == right["quality"] else 0.0)
        combined_scores.append(token_similarity(left, right))
    return {
        "aligned_combined": statistics.mean(combined_scores) if combined_scores else 0.0,
        "aligned_root": statistics.mean(root_scores) if root_scores else 0.0,
        "aligned_bass": statistics.mean(bass_scores) if bass_scores else 0.0,
        "aligned_quality": statistics.mean(quality_scores) if quality_scores else 0.0,
        "aligned_compared_bars": len(combined_scores),
    }


def harmony_dtw_similarity(reference: list[dict[str, Any]], candidate: list[dict[str, Any]], *, band_ratio: float) -> float:
    return generic_dtw_similarity(reference, candidate, token_similarity, band_ratio=band_ratio)


def compare_harmony(reference: list[dict[str, Any]], candidate: list[dict[str, Any]], *, band_ratio: float) -> dict[str, float | int]:
    scores = aligned_similarity(reference, candidate)
    scores["harmony_dtw"] = harmony_dtw_similarity(reference, candidate, band_ratio=band_ratio)
    scores["root_dtw"] = generic_dtw_similarity(
        [item["root"] for item in reference],
        [item["root"] for item in candidate],
        pitch_class_similarity,
        band_ratio=band_ratio,
    )
    scores["bass_dtw"] = generic_dtw_similarity(
        [item["bass"] for item in reference],
        [item["bass"] for item in candidate],
        pitch_class_similarity,
        band_ratio=band_ratio,
    )
    scores["top_contour_dtw"] = generic_dtw_similarity(
        contour_sequence(reference),
        contour_sequence(candidate),
        lambda left, right: 1.0 if left == right else 0.0,
        band_ratio=band_ratio,
    )
    scores["density_dtw"] = generic_dtw_similarity(
        [float(item["note_count"]) for item in reference],
        [float(item["note_count"]) for item in candidate],
        lambda left, right: scalar_similarity(left, right, scale=12.0),
        band_ratio=band_ratio,
    )
    scores["combined"] = (
        float(scores["harmony_dtw"])
        + float(scores["root_dtw"])
        + float(scores["bass_dtw"])
    ) / 3.0
    return scores
