from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from music21 import converter


TONIC_TO_PC = {
    "C": 0,
    "B#": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "Fb": 4,
    "E#": 5,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
    "Cb": 11,
}


@dataclass(frozen=True)
class ChromaFeatures:
    hist: np.ndarray
    sequence: np.ndarray
    frames: int
    duration_quarters: float
    tonic: str | None


@dataclass(frozen=True)
class ChromaSimilarity:
    full_hist: float
    full_dtw: float
    bass_hist: float
    bass_dtw: float
    top_hist: float
    top_dtw: float

    @property
    def combined(self) -> float:
        return (
            self.full_hist
            + self.full_dtw
            + self.bass_hist
            + self.bass_dtw
            + self.top_hist
            + self.top_dtw
        ) / 6.0

    def to_json(self) -> dict[str, float]:
        payload = asdict(self)
        payload["combined"] = self.combined
        return payload


def parse_piece_tonic(text: str) -> str | None:
    for line in text.splitlines():
        match = re.match(r"^K:\s*([A-Ga-g][#b]?)", line.strip())
        if match:
            tonic = match.group(1)
            return tonic[0].upper() + tonic[1:]
    match = re.search(r"(?m)^%\s*original_key\s*:\s*([A-Ga-g][#b]?)", text)
    if match:
        tonic = match.group(1)
        return tonic[0].upper() + tonic[1:]
    return None


def tonic_pitch_class(tonic: str | None) -> int:
    if tonic is None:
        return 0
    return TONIC_TO_PC.get(tonic, 0)


def _note_events(path: Path) -> tuple[list[tuple[float, float, int, int]], float]:
    score = converter.parse(path)
    events: list[tuple[float, float, int, int]] = []
    for note in score.recurse().notes:
        pitches = note.pitches if hasattr(note, "pitches") else [note.pitch]
        try:
            offset = float(note.getOffsetInHierarchy(score))
        except Exception:
            offset = float(note.offset)
        duration = max(0.0, float(note.duration.quarterLength))
        for pitch in pitches:
            events.append((offset, duration, int(pitch.midi), int(pitch.pitchClass)))
    if not events:
        return [], 1.0
    total_duration = max(offset + duration for offset, duration, _midi, _pc in events)
    return events, max(total_duration, 1e-6)


def chroma_features(
    path: str | Path,
    *,
    bins: int = 128,
    mode: str = "full",
    normalize_key: bool = True,
) -> ChromaFeatures:
    path = Path(path)
    if bins <= 0:
        raise ValueError("bins must be positive")
    if mode not in {"full", "bass", "top"}:
        raise ValueError("mode must be one of: full, bass, top")

    text = path.read_text(encoding="utf-8")
    tonic = parse_piece_tonic(text)
    shift = (-tonic_pitch_class(tonic)) % 12 if normalize_key else 0
    events, total_duration = _note_events(path)

    frames = np.zeros((bins, 12), dtype=np.float64)
    for frame_idx in range(bins):
        start = total_duration * frame_idx / bins
        end = total_duration * (frame_idx + 1) / bins
        active: list[tuple[int, int, float]] = []
        for offset, duration, midi, pitch_class in events:
            overlap = max(0.0, min(end, offset + duration) - max(start, offset))
            if overlap > 0.0:
                active.append((midi, (pitch_class + shift) % 12, overlap))

        if mode == "bass" and active:
            selected_midi = min(midi for midi, _pc, _overlap in active)
            active = [item for item in active if item[0] == selected_midi]
        elif mode == "top" and active:
            selected_midi = max(midi for midi, _pc, _overlap in active)
            active = [item for item in active if item[0] == selected_midi]

        for _midi, pitch_class, overlap in active:
            frames[frame_idx, pitch_class] += overlap

    hist = frames.sum(axis=0)
    hist_norm = np.linalg.norm(hist)
    if hist_norm > 0.0:
        hist = hist / hist_norm

    frame_norms = np.linalg.norm(frames, axis=1, keepdims=True)
    sequence = np.divide(frames, frame_norms, out=np.zeros_like(frames), where=frame_norms > 0.0)
    nonempty = np.linalg.norm(sequence, axis=1) > 0.0
    sequence = sequence[nonempty] if np.any(nonempty) else sequence[:1]

    return ChromaFeatures(
        hist=hist,
        sequence=sequence,
        frames=int(sequence.shape[0]),
        duration_quarters=total_duration,
        tonic=tonic,
    )


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return float(np.clip(np.dot(a, b) / (norm_a * norm_b), -1.0, 1.0))


def dtw_distance(a: np.ndarray, b: np.ndarray, *, band_ratio: float = 0.25) -> float:
    n = int(a.shape[0])
    m = int(b.shape[0])
    if n == 0 or m == 0:
        return 1.0

    cost = 1.0 - np.clip(a @ b.T, -1.0, 1.0)
    inf = 1e12
    previous = np.full(m + 1, inf, dtype=np.float64)
    current = np.full(m + 1, inf, dtype=np.float64)
    previous[0] = 0.0
    band = max(abs(n - m), int(max(n, m) * band_ratio))

    for i in range(1, n + 1):
        current.fill(inf)
        center = int(i * m / n)
        start = max(1, center - band)
        end = min(m, center + band)
        for j in range(start, end + 1):
            current[j] = cost[i - 1, j - 1] + min(previous[j], current[j - 1], previous[j - 1])
        previous, current = current, previous

    return float(previous[m] / (n + m))


def dtw_similarity(a: np.ndarray, b: np.ndarray, *, band_ratio: float = 0.25) -> float:
    return 1.0 / (1.0 + dtw_distance(a, b, band_ratio=band_ratio))


def compare_chroma_features(
    reference: dict[str, ChromaFeatures],
    candidate: dict[str, ChromaFeatures],
    *,
    band_ratio: float = 0.25,
) -> ChromaSimilarity:
    return ChromaSimilarity(
        full_hist=cosine_similarity(reference["full"].hist, candidate["full"].hist),
        full_dtw=dtw_similarity(reference["full"].sequence, candidate["full"].sequence, band_ratio=band_ratio),
        bass_hist=cosine_similarity(reference["bass"].hist, candidate["bass"].hist),
        bass_dtw=dtw_similarity(reference["bass"].sequence, candidate["bass"].sequence, band_ratio=band_ratio),
        top_hist=cosine_similarity(reference["top"].hist, candidate["top"].hist),
        top_dtw=dtw_similarity(reference["top"].sequence, candidate["top"].sequence, band_ratio=band_ratio),
    )


def load_chroma_feature_set(
    path: str | Path,
    *,
    bins: int = 128,
    normalize_key: bool = True,
) -> dict[str, ChromaFeatures]:
    return {
        mode: chroma_features(path, bins=bins, mode=mode, normalize_key=normalize_key)
        for mode in ("full", "bass", "top")
    }
