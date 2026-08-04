#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.strict_similarity import (
    STRICT_SYMBOLIC_COMPONENT_Z_KEY,
    strict_similarity_global_base_z_scores,
    strict_symbolic_similarity,
    written_harmony_reference,
)
from evaluation.harmony_similarity import harmony_from_path, harmony_from_text, infer_harmony, parse_bar_notes, strip_stream_tag
from evaluation.similarity_rewards import HEADER_RE, continuation_for_similarity


DEFAULT_RESCORE_DIR = (
    REPO_ROOT
    / "data/processed/notagen/remote_runs"
    / "SFT_E3_L18_allvars_eval0_fixed60_chroma_harmony_exactkl_cap1024_b32_20260731T102931Z"
    / "current_reward_rescore_20260803_aria_matching_bars"
)
DEFAULT_ARIA_PATH = REPO_ROOT / "data/processed/goldberg/abc/aria-bwv-988.abc"


def discover_datasets(rescore_dir: Path) -> dict[str, Path]:
    datasets: dict[str, Path] = {
        "base": rescore_dir / "base" / "pretrained_baseline_rewards.jsonl",
    }
    for path in sorted((rescore_dir / "scores").glob("epoch*_rewards.jsonl")):
        epoch_token = path.stem.removeprefix("epoch").removesuffix("_rewards")
        try:
            epoch = int(epoch_token)
        except ValueError:
            continue
        datasets[f"epoch{epoch}"] = path
    datasets["GT"] = rescore_dir / "gt_variations_current_rewards.jsonl"
    return datasets


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _candidate_path(row: dict[str, Any]) -> Path:
    path = row.get("path")
    if not isinstance(path, str) or not path:
        raise ValueError(f"row does not contain a usable path: {row}")
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate


def _stream_continuation_text(text: str) -> str:
    header: list[str] = []
    stream_lines: list[str] = []
    in_stream = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("[r:"):
            in_stream = True
            stream_lines.append(line)
            continue
        if in_stream:
            continue
        if not line:
            continue
        if line.startswith("%") or line.startswith("%%score") or HEADER_RE.match(line):
            header.append(line)
    return "\n".join(header + stream_lines) + "\n" if stream_lines else text


def _candidate_harmony(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    stream_harmony = [
        infer_harmony(parse_bar_notes(strip_stream_tag(line.strip())))
        for line in text.splitlines()
        if line.strip().startswith("[r:")
    ]
    if stream_harmony:
        return stream_harmony
    harmony = harmony_from_text(_stream_continuation_text(text))
    if harmony:
        return harmony
    return harmony_from_text(continuation_for_similarity(text))


def score_dataset(
    *,
    label: str,
    path: Path,
    aria_harmony: list[dict[str, Any]],
    band_ratio: float,
    z_clip: float | None,
) -> list[dict[str, Any]]:
    scored_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(_read_jsonl(path)):
        candidate_path = _candidate_path(row)
        try:
            candidate_harmony = _candidate_harmony(candidate_path)
            scores = strict_symbolic_similarity(aria_harmony, candidate_harmony, band_ratio=band_ratio)
            scores.update(strict_similarity_global_base_z_scores(scores, z_clip=z_clip))
            error = ""
        except Exception as exc:
            scores = {}
            error = str(exc)

        scored_rows.append(
            {
                "label": label,
                "row_index": idx,
                "source_path": str(candidate_path),
                "prefix_name": row.get("prefix_name", ""),
                "parse_valid": row.get("parse_valid", ""),
                "active_similarity_reward": row.get("active_similarity_reward", ""),
                "structural_total_reward": row.get("structural_total_reward", ""),
                "error": error,
                **scores,
            }
        )
    return scored_rows


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
    return statistics.mean(values) if values else 0.0


def _minimum(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
    return min(values) if values else 0.0


def _maximum(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
    return max(values) if values else 0.0


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metric_keys = [
        "strict_symbolic_similarity",
        "strict_aria_specificity_mean_delta",
        "strict_aria_specificity_max_delta",
        "strict_aligned_root_bass",
        "strict_aligned_root",
        "strict_aligned_bass",
        "strict_dtw_combined_narrow",
        "strict_harmony_dtw_narrow",
        "strict_root_dtw_narrow",
        "strict_bass_dtw_narrow",
        "strict_root_bass_bigram_weighted_jaccard",
        "strict_root_bass_fourgram_weighted_jaccard",
        "strict_cadence_root_bass",
        "strict_symbolic_similarity_global_base_z",
        STRICT_SYMBOLIC_COMPONENT_Z_KEY,
        "strict_aligned_root_bass_global_base_z",
        "strict_dtw_combined_narrow_global_base_z",
        "strict_root_bass_bigram_weighted_jaccard_global_base_z",
        "strict_root_bass_fourgram_weighted_jaccard_global_base_z",
        "strict_cadence_root_bass_global_base_z",
        "strict_candidate_bars",
        "active_similarity_reward",
        "structural_total_reward",
    ]
    labels = []
    for row in rows:
        if row["label"] not in labels:
            labels.append(row["label"])

    summary_rows: list[dict[str, Any]] = []
    for label in labels:
        label_rows = [row for row in rows if row["label"] == label]
        summary: dict[str, Any] = {
            "label": label,
            "n": len(label_rows),
            "n_scored": sum(1 for row in label_rows if not row.get("error")),
            "n_errors": sum(1 for row in label_rows if row.get("error")),
        }
        for key in metric_keys:
            summary[f"{key}_mean"] = _mean(label_rows, key)
            summary[f"{key}_min"] = _minimum(label_rows, key)
            summary[f"{key}_max"] = _maximum(label_rows, key)
        summary_rows.append(summary)
    return summary_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rescore-dir", type=Path, default=DEFAULT_RESCORE_DIR)
    parser.add_argument("--aria-path", type=Path, default=DEFAULT_ARIA_PATH)
    parser.add_argument("--aria-representation", choices=("written", "rendered"), default="written")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--band-ratio", type=float, default=0.05)
    parser.add_argument("--z-clip", type=float, default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or args.rescore_dir / "strict_similarity"
    out_dir.mkdir(parents=True, exist_ok=True)

    aria_path = args.aria_path
    if not aria_path.is_absolute():
        aria_path = REPO_ROOT / aria_path
    aria_harmony = harmony_from_path(aria_path)
    if args.aria_representation == "written":
        aria_harmony = written_harmony_reference(aria_harmony)

    rows: list[dict[str, Any]] = []
    for label, dataset_path in discover_datasets(args.rescore_dir).items():
        if not dataset_path.exists():
            raise FileNotFoundError(dataset_path)
        rows.extend(
            score_dataset(
                label=label,
                path=dataset_path,
                aria_harmony=aria_harmony,
                band_ratio=args.band_ratio,
                z_clip=args.z_clip,
            )
        )

    rows_path = out_dir / "strict_similarity_rows.jsonl"
    rows_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    write_csv(out_dir / "strict_similarity_rows.csv", rows)

    summary_rows = summarize(rows)
    write_csv(out_dir / "strict_similarity_summary.csv", summary_rows)
    print(f"Wrote {rows_path}")
    print(f"Wrote {out_dir / 'strict_similarity_summary.csv'}")
    for row in summary_rows:
        print(
            "{label:>6} n={n_scored:>3}/{n:<3} "
            "strict={strict_symbolic_similarity_mean:.4f} "
            "strict_component_z={strict_symbolic_component_global_base_z_mean:.2f} "
            "delta_mean={strict_aria_specificity_mean_delta_mean:.4f} "
            "aligned_rb={strict_aligned_root_bass_mean:.4f} "
            "dtw={strict_dtw_combined_narrow_mean:.4f} "
            "bigram_wj={strict_root_bass_bigram_weighted_jaccard_mean:.4f} "
            "fourgram_wj={strict_root_bass_fourgram_weighted_jaccard_mean:.4f}".format(**row)
        )


if __name__ == "__main__":
    main()
