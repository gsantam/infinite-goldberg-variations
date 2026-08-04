from __future__ import annotations

import statistics
from collections import Counter
from typing import Any

from rewards.harmony_similarity import (
    generic_dtw_similarity,
    pitch_class_similarity,
    token_similarity,
)
from rewards.strict_similarity_baseline_norms import (
    STRICT_SIMILARITY_BASELINE_METRICS,
    STRICT_SIMILARITY_GLOBAL_NORMS,
    STRICT_SIMILARITY_Z_STD_FLOORS,
)

STRICT_SIMILARITY_SCORE_KEYS = (
    "strict_aligned_root",
    "strict_aligned_bass",
    "strict_aligned_root_bass",
    "strict_aligned_top",
    "strict_aligned_quality",
    "strict_aligned_coverage",
    "strict_harmony_dtw_narrow",
    "strict_root_dtw_narrow",
    "strict_bass_dtw_narrow",
    "strict_dtw_combined_narrow",
    "strict_root_bass_bigram_jaccard",
    "strict_root_bass_bigram_weighted_jaccard",
    "strict_root_bass_fourgram_jaccard",
    "strict_root_bass_fourgram_weighted_jaccard",
    "strict_cadence_root_bass",
    "strict_symbolic_similarity",
    "strict_negative_reverse_similarity",
    "strict_negative_rotated_similarity",
    "strict_negative_mean_similarity",
    "strict_aria_specificity_mean_delta",
    "strict_reference_bars",
    "strict_candidate_bars",
)

STRICT_SYMBOLIC_COMPONENT_Z_KEY = "strict_symbolic_component_global_base_z"

STRICT_SYMBOLIC_COMPONENT_WEIGHTS = {
    "strict_aligned_root_bass": 0.30,
    "strict_dtw_combined_narrow": 0.25,
    "strict_root_bass_bigram_weighted_jaccard": 0.20,
    "strict_root_bass_fourgram_weighted_jaccard": 0.15,
    "strict_cadence_root_bass": 0.10,
}


def aggregate_weighted_scores(scores: dict[str, Any], weights: dict[str, float]) -> float:
    return sum(float(weight) * float(scores.get(metric, 0.0)) for metric, weight in weights.items())


def _valid_harmony(item: dict[str, Any]) -> bool:
    return item.get("root") is not None and item.get("bass") is not None


def _safe_mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _soft_root_bass_similarity(left: dict[str, Any], right: dict[str, Any]) -> float:
    if not _valid_harmony(left) or not _valid_harmony(right):
        return 0.0
    return 0.5 * pitch_class_similarity(left.get("root"), right.get("root")) + 0.5 * pitch_class_similarity(
        left.get("bass"), right.get("bass")
    )


def _aligned_by_reference(
    reference: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
) -> dict[str, float]:
    root_scores: list[float] = []
    bass_scores: list[float] = []
    root_bass_scores: list[float] = []
    top_scores: list[float] = []
    quality_scores: list[float] = []

    for idx, ref_item in enumerate(reference):
        cand_item = candidate[idx] if idx < len(candidate) else {}
        root_scores.append(pitch_class_similarity(ref_item.get("root"), cand_item.get("root")))
        bass_scores.append(pitch_class_similarity(ref_item.get("bass"), cand_item.get("bass")))
        root_bass_scores.append(_soft_root_bass_similarity(ref_item, cand_item))

        ref_top = ref_item.get("top_midi")
        cand_top = cand_item.get("top_midi")
        if ref_top is None or cand_top is None:
            top_scores.append(0.0)
        else:
            top_scores.append(1.0 if int(ref_top) % 12 == int(cand_top) % 12 else 0.0)

        ref_quality = ref_item.get("quality")
        cand_quality = cand_item.get("quality")
        quality_scores.append(1.0 if ref_quality is not None and ref_quality == cand_quality else 0.0)

    reference_bars = len(reference)
    candidate_bars = len(candidate)
    return {
        "strict_aligned_root": _safe_mean(root_scores),
        "strict_aligned_bass": _safe_mean(bass_scores),
        "strict_aligned_root_bass": _safe_mean(root_bass_scores),
        "strict_aligned_top": _safe_mean(top_scores),
        "strict_aligned_quality": _safe_mean(quality_scores),
        "strict_aligned_coverage": min(candidate_bars, reference_bars) / reference_bars if reference_bars else 0.0,
        "strict_reference_bars": float(reference_bars),
        "strict_candidate_bars": float(candidate_bars),
    }


def _root_bass_token(item: dict[str, Any]) -> tuple[int, int] | None:
    root = item.get("root")
    bass = item.get("bass")
    if root is None or bass is None:
        return None
    return int(root), int(bass)


def _ngrams(harmony: list[dict[str, Any]], n: int) -> list[tuple[tuple[int, int], ...]]:
    tokens = [_root_bass_token(item) for item in harmony]
    grams: list[tuple[tuple[int, int], ...]] = []
    if n <= 0 or len(tokens) < n:
        return grams
    for idx in range(len(tokens) - n + 1):
        window = tokens[idx : idx + n]
        if all(token is not None for token in window):
            grams.append(tuple(token for token in window if token is not None))
    return grams


def _jaccard(left: list[Any], right: list[Any]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set and not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def _weighted_jaccard(left: list[Any], right: list[Any]) -> float:
    left_counts = Counter(left)
    right_counts = Counter(right)
    keys = set(left_counts) | set(right_counts)
    if not keys:
        return 0.0
    numerator = sum(min(left_counts[key], right_counts[key]) for key in keys)
    denominator = sum(max(left_counts[key], right_counts[key]) for key in keys)
    return numerator / denominator if denominator else 0.0


def _ngram_scores(reference: list[dict[str, Any]], candidate: list[dict[str, Any]]) -> dict[str, float]:
    ref_bigrams = _ngrams(reference, 2)
    cand_bigrams = _ngrams(candidate, 2)
    ref_fourgrams = _ngrams(reference, 4)
    cand_fourgrams = _ngrams(candidate, 4)
    return {
        "strict_root_bass_bigram_jaccard": _jaccard(ref_bigrams, cand_bigrams),
        "strict_root_bass_bigram_weighted_jaccard": _weighted_jaccard(ref_bigrams, cand_bigrams),
        "strict_root_bass_fourgram_jaccard": _jaccard(ref_fourgrams, cand_fourgrams),
        "strict_root_bass_fourgram_weighted_jaccard": _weighted_jaccard(ref_fourgrams, cand_fourgrams),
    }


def _dtw_scores(
    reference: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    *,
    band_ratio: float,
) -> dict[str, float]:
    harmony_dtw = generic_dtw_similarity(reference, candidate, token_similarity, band_ratio=band_ratio)
    root_dtw = generic_dtw_similarity(
        [item.get("root") for item in reference],
        [item.get("root") for item in candidate],
        pitch_class_similarity,
        band_ratio=band_ratio,
    )
    bass_dtw = generic_dtw_similarity(
        [item.get("bass") for item in reference],
        [item.get("bass") for item in candidate],
        pitch_class_similarity,
        band_ratio=band_ratio,
    )
    return {
        "strict_harmony_dtw_narrow": harmony_dtw,
        "strict_root_dtw_narrow": root_dtw,
        "strict_bass_dtw_narrow": bass_dtw,
        "strict_dtw_combined_narrow": aggregate_weighted_scores(
            {
                "strict_harmony_dtw_narrow": harmony_dtw,
                "strict_root_dtw_narrow": root_dtw,
                "strict_bass_dtw_narrow": bass_dtw,
            },
            {
                "strict_harmony_dtw_narrow": 1.0 / 3.0,
                "strict_root_dtw_narrow": 1.0 / 3.0,
                "strict_bass_dtw_narrow": 1.0 / 3.0,
            },
        ),
    }


def _cadence_positions(reference_length: int) -> list[int]:
    if reference_length <= 0:
        return []
    positions = {reference_length - 1}
    for divisor in (4, 2):
        step = max(1, reference_length // divisor)
        for pos in range(step - 1, reference_length, step):
            positions.add(pos)
    return sorted(pos for pos in positions if 0 <= pos < reference_length)


def _cadence_similarity(reference: list[dict[str, Any]], candidate: list[dict[str, Any]]) -> float:
    scores: list[float] = []
    for pos in _cadence_positions(len(reference)):
        ref_item = reference[pos]
        cand_item = candidate[pos] if pos < len(candidate) else {}
        scores.append(_soft_root_bass_similarity(ref_item, cand_item))
    return _safe_mean(scores)


def _rotated_reference(reference: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not reference:
        return []
    pivot = len(reference) // 2
    return reference[pivot:] + reference[:pivot]


def written_harmony_reference(harmony: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the written A/B Goldberg form from a repeat-expanded reference."""

    if len(harmony) == 64:
        return harmony[:16] + harmony[32:48]
    return harmony


def _strict_symbolic_similarity_no_negatives(
    reference: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    *,
    band_ratio: float,
) -> dict[str, float]:
    scores = {
        **_aligned_by_reference(reference, candidate),
        **_ngram_scores(reference, candidate),
        **_dtw_scores(reference, candidate, band_ratio=band_ratio),
        "strict_cadence_root_bass": _cadence_similarity(reference, candidate),
    }
    scores["strict_symbolic_similarity"] = aggregate_weighted_scores(scores, STRICT_SYMBOLIC_COMPONENT_WEIGHTS)
    return scores


def strict_symbolic_similarity(
    reference: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    *,
    band_ratio: float = 0.05,
) -> dict[str, float]:
    """Aria-specific symbolic similarity.

    This uses stricter, mostly bar-local harmony comparisons so we can test
    whether the model is matching the Aria plan instead of only generic tonal
    material.
    """

    scores = _strict_symbolic_similarity_no_negatives(reference, candidate, band_ratio=band_ratio)
    reverse_scores = _strict_symbolic_similarity_no_negatives(list(reversed(reference)), candidate, band_ratio=band_ratio)
    rotated_scores = _strict_symbolic_similarity_no_negatives(_rotated_reference(reference), candidate, band_ratio=band_ratio)
    negative_mean = (
        reverse_scores["strict_symbolic_similarity"] + rotated_scores["strict_symbolic_similarity"]
    ) / 2.0
    scores["strict_negative_reverse_similarity"] = reverse_scores["strict_symbolic_similarity"]
    scores["strict_negative_rotated_similarity"] = rotated_scores["strict_symbolic_similarity"]
    scores["strict_negative_mean_similarity"] = negative_mean
    scores["strict_aria_specificity_mean_delta"] = scores["strict_symbolic_similarity"] - negative_mean
    scores["strict_aria_specificity_max_delta"] = scores["strict_symbolic_similarity"] - max(
        reverse_scores["strict_symbolic_similarity"],
        rotated_scores["strict_symbolic_similarity"],
    )
    return scores


def strict_similarity_global_base_z_scores(
    scores: dict[str, Any],
    *,
    metrics: tuple[str, ...] = STRICT_SIMILARITY_BASELINE_METRICS,
    z_clip: float | None = None,
) -> dict[str, float]:
    """Convert strict similarity metrics to global base-model z-scores.

    The baseline means/stds are computed from sampled NotaGen-large completions
    pooled across the Aria-matching PPO prompt set. This is intentionally an
    opt-in calibration helper, not part of the raw strict similarity metric.
    """

    normalized: dict[str, float] = {}
    for metric in metrics:
        value = scores.get(metric)
        if not isinstance(value, (int, float)):
            continue
        norm = STRICT_SIMILARITY_GLOBAL_NORMS.get(metric)
        if norm is None:
            continue
        std_floor = float(STRICT_SIMILARITY_Z_STD_FLOORS.get(metric, 1e-4))
        std = max(float(norm.get("std_safe", norm.get("std", 0.0))), std_floor)
        if std <= 0.0:
            continue
        z_value = (float(value) - float(norm["mean"])) / std
        if z_clip is not None:
            z_value = max(-float(z_clip), min(float(z_clip), z_value))
        normalized[f"{metric}_global_base_z"] = z_value
    z_weights = {f"{metric}_global_base_z": weight for metric, weight in STRICT_SYMBOLIC_COMPONENT_WEIGHTS.items()}
    if all(metric in normalized for metric in z_weights):
        normalized[STRICT_SYMBOLIC_COMPONENT_Z_KEY] = aggregate_weighted_scores(normalized, z_weights)
    return normalized
