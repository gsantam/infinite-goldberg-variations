from __future__ import annotations

import re
import signal
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evaluation.chroma_similarity import compare_chroma_features, load_chroma_feature_set
from evaluation.harmony_similarity import compare_harmony, harmony_from_path, harmony_from_text
from preprocessing.notagen_abc import preprocess_notagen_abc


HEADER_RE = re.compile(r"^(X|T|C|M|L|Q|K|V):")


@dataclass(frozen=True)
class SimilarityReference:
    path: Path
    chroma: dict[str, Any] | None = None
    harmony: list[dict[str, Any]] | None = None


@dataclass(frozen=True)
class SimilarityRewardWeights:
    aria_chroma: float = 0.0
    variation_chroma: float = 0.0
    aria_harmony: float = 0.0
    variation_harmony: float = 0.0

    @property
    def needs_chroma(self) -> bool:
        return self.aria_chroma != 0.0 or self.variation_chroma != 0.0

    @property
    def needs_harmony(self) -> bool:
        return self.aria_harmony != 0.0 or self.variation_harmony != 0.0

    @property
    def enabled(self) -> bool:
        return self.needs_chroma or self.needs_harmony


@contextmanager
def time_limit(seconds: float, label: str):
    if seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return

    previous = signal.signal(signal.SIGALRM, lambda _signum, _frame: (_ for _ in ()).throw(TimeoutError(label)))
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)


def continuation_for_similarity(text: str) -> str:
    """Build renderable ABC for generated NotaGen stream lines only.

    Aria-prompted samples may contain a full untagged aria before the generated
    `[r:i/j]` continuation. Similarity rewards should score the continuation,
    not re-score the conditioning prompt, so this keeps only the final header
    context plus generated stream-line bodies.
    """

    header: list[str] = []
    body: list[str] = []
    in_stream = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("[r:"):
            in_stream = True
            body.append(re.sub(r"^\[r:\d+/\d+\]", "", line))
            continue
        if in_stream:
            continue
        if not line:
            continue
        if line.startswith("%") or line.startswith("%%score") or HEADER_RE.match(line):
            header.append(line)

    if not any(line.startswith("X:") for line in header):
        header.insert(0, "X:1")
    if not any(line.startswith("L:") for line in header):
        header.append("L:1/8")
    if not any(line.startswith("M:") for line in header):
        header.append("M:3/4")
    if not any(line.startswith("K:") for line in header):
        header.append("K:G")
    if not body:
        body = [
            re.sub(r"^\[r:\d+/\d+\]", "", line.strip())
            for line in text.splitlines()
            if "[V:" in line and "|" in line
        ]
    return preprocess_notagen_abc("\n".join(header + body) + "\n")


def _add_prefixed_scores(payload: dict[str, Any], prefix: str, scores: dict[str, Any]) -> None:
    for key, value in scores.items():
        payload[f"{prefix}_{key}"] = float(value) if isinstance(value, (int, float)) else value


def _add_chroma_reward_aggregates(payload: dict[str, Any], prefix: str) -> None:
    full_hist = float(payload.get(f"{prefix}_full_hist", 0.0))
    bass_hist = float(payload.get(f"{prefix}_bass_hist", 0.0))
    top_hist = float(payload.get(f"{prefix}_top_hist", 0.0))
    payload[f"{prefix}_static_hist"] = (full_hist + bass_hist + top_hist) / 3.0
    payload[f"{prefix}_harmonic_hist"] = (full_hist + bass_hist) / 2.0


def _chroma_scores(
    *,
    candidate_abc: str,
    aria: SimilarityReference | None,
    variation: SimilarityReference | None,
    bins: int,
    band_ratio: float,
    timeout_s: float,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"similarity_chroma_valid": False}
    with tempfile.TemporaryDirectory(prefix="grpo_chroma_") as tmp:
        candidate_path = Path(tmp) / "candidate.abc"
        candidate_path.write_text(candidate_abc, encoding="utf-8")
        with time_limit(timeout_s, "chroma similarity timed out"):
            candidate = load_chroma_feature_set(candidate_path, bins=bins)
        payload["similarity_chroma_valid"] = True
        if aria is not None and aria.chroma is not None:
            _add_prefixed_scores(
                payload,
                "aria_chroma",
                compare_chroma_features(aria.chroma, candidate, band_ratio=band_ratio).to_json(),
            )
            _add_chroma_reward_aggregates(payload, "aria_chroma")
        if variation is not None and variation.chroma is not None:
            _add_prefixed_scores(
                payload,
                "variation_chroma",
                compare_chroma_features(variation.chroma, candidate, band_ratio=band_ratio).to_json(),
            )
            _add_chroma_reward_aggregates(payload, "variation_chroma")
    return payload


def _harmony_scores(
    *,
    candidate_abc: str,
    aria: SimilarityReference | None,
    variation: SimilarityReference | None,
    band_ratio: float,
    timeout_s: float,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"similarity_harmony_valid": False}
    with time_limit(timeout_s, "harmony similarity timed out"):
        candidate = harmony_from_text(candidate_abc)
        payload["similarity_harmony_valid"] = True
        if aria is not None and aria.harmony is not None:
            _add_prefixed_scores(payload, "aria_harmony", compare_harmony(aria.harmony, candidate, band_ratio=band_ratio))
        if variation is not None and variation.harmony is not None:
            _add_prefixed_scores(
                payload,
                "variation_harmony",
                compare_harmony(variation.harmony, candidate, band_ratio=band_ratio),
            )
    return payload


def load_similarity_reference(
    path: str | Path,
    *,
    load_chroma: bool,
    load_harmony: bool,
    bins: int,
) -> SimilarityReference:
    path = Path(path)
    return SimilarityReference(
        path=path,
        chroma=load_chroma_feature_set(path, bins=bins) if load_chroma else None,
        harmony=harmony_from_path(path) if load_harmony else None,
    )


def score_similarity_reward(
    *,
    prompt_text: str,
    completion_text: str,
    weights: SimilarityRewardWeights,
    aria: SimilarityReference | None,
    variation: SimilarityReference | None,
    bins: int = 128,
    band_ratio: float = 0.25,
    timeout_s: float = 20.0,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "similarity_reward": 0.0,
        "similarity_candidate_chars": 0,
    }
    if not weights.enabled:
        return payload

    candidate_abc = continuation_for_similarity(prompt_text + completion_text)
    payload["similarity_candidate_chars"] = len(candidate_abc)

    if weights.needs_chroma:
        try:
            payload.update(
                _chroma_scores(
                    candidate_abc=candidate_abc,
                    aria=aria,
                    variation=variation,
                    bins=bins,
                    band_ratio=band_ratio,
                    timeout_s=timeout_s,
                )
            )
        except Exception as exc:
            payload["similarity_chroma_valid"] = False
            payload["similarity_chroma_error"] = str(exc)

    if weights.needs_harmony:
        try:
            payload.update(
                _harmony_scores(
                    candidate_abc=candidate_abc,
                    aria=aria,
                    variation=variation,
                    band_ratio=band_ratio,
                    timeout_s=timeout_s,
                )
            )
        except Exception as exc:
            payload["similarity_harmony_valid"] = False
            payload["similarity_harmony_error"] = str(exc)

    reward = 0.0
    reward += weights.aria_chroma * float(payload.get("aria_chroma_harmonic_hist", 0.0))
    reward += weights.variation_chroma * float(payload.get("variation_chroma_harmonic_hist", 0.0))
    reward += weights.aria_harmony * float(payload.get("aria_harmony_combined", 0.0))
    reward += weights.variation_harmony * float(payload.get("variation_harmony_combined", 0.0))
    payload["similarity_reward"] = reward
    return payload
