#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rewards.harmony_similarity import harmony_from_path, infer_harmony, parse_bar_notes, strip_stream_tag
from rewards.strict_similarity import STRICT_SIMILARITY_SCORE_KEYS, strict_symbolic_similarity, written_harmony_reference
from notagen_runtime.notagen_cached_generation_batch import sample_completions_cached_batch
from notagen_runtime.notagen_wrapper import build_model, count_stream_lines


DEFAULT_RUN_DIR = (
    REPO_ROOT
    / "data/processed/notagen/remote_runs"
    / "SFT_E3_L18_allvars_eval0_fixed60_chroma_harmony_exactkl_cap1024_b32_20260731T102931Z"
    / "current_reward_rescore_20260803_aria_matching_bars"
)
DEFAULT_WEIGHTS = (
    REPO_ROOT.parent
    / "NotaGen/weights/weights_notagen_pretrain-finetune_p_size_16_p_length_1024_p_layers_c_layers_6_20_h_size_1280_lr_1e-05_batch_1.pth"
)
DEFAULT_PROMPTS = REPO_ROOT / "data/processed/notagen/goldberg_ppo_prompts_e3_header_allvoices.jsonl"
DEFAULT_ELIGIBLE_PROMPTS = REPO_ROOT / "data/processed/goldberg/structure/aria_matching_prompt_names.txt"
DEFAULT_ARIA_PATH = REPO_ROOT / "data/processed/goldberg/abc/aria-bwv-988.abc"


STRICT_METRIC_KEYS = list(STRICT_SIMILARITY_SCORE_KEYS)


def read_prompts(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows.append(row)
    return rows


def read_eligible(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    values = {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
    return values


def candidate_harmony_from_text(text: str) -> list[dict[str, Any]]:
    stream_harmony = [
        infer_harmony(parse_bar_notes(strip_stream_tag(line.strip())))
        for line in text.splitlines()
        if line.strip().startswith("[r:")
    ]
    return stream_harmony


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


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return statistics.stdev(values)


def summarize_prompt_norms(rows: list[dict[str, Any]], *, eps: float) -> list[dict[str, Any]]:
    prompt_names = sorted({str(row["prefix_name"]) for row in rows if row.get("ok")})
    summaries: list[dict[str, Any]] = []
    for prefix_name in prompt_names:
        prompt_rows = [row for row in rows if row.get("ok") and row["prefix_name"] == prefix_name]
        summary: dict[str, Any] = {
            "prefix_name": prefix_name,
            "n": len(prompt_rows),
        }
        for key in STRICT_METRIC_KEYS:
            values = [float(row[key]) for row in prompt_rows if isinstance(row.get(key), (int, float))]
            mean = statistics.mean(values) if values else 0.0
            std = _std(values)
            summary[f"{key}_mean"] = mean
            summary[f"{key}_std"] = std
            summary[f"{key}_std_safe"] = max(std, eps)
        summaries.append(summary)
    return summaries


def write_norms_json(path: Path, rows: list[dict[str, Any]], *, eps: float, config: dict[str, Any]) -> None:
    prompt_summaries = summarize_prompt_norms(rows, eps=eps)
    payload = {
        "config": config,
        "metrics": STRICT_METRIC_KEYS,
        "prompts": {
            row["prefix_name"]: {
                key: {
                    "mean": row[f"{key}_mean"],
                    "std": row[f"{key}_std"],
                    "std_safe": row[f"{key}_std_safe"],
                    "n": row["n"],
                }
                for key in STRICT_METRIC_KEYS
            }
            for row in prompt_summaries
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--prompts-jsonl", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--eligible-prompts", type=Path, default=DEFAULT_ELIGIBLE_PROMPTS)
    parser.add_argument("--aria-path", type=Path, default=DEFAULT_ARIA_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_RUN_DIR / "base_strict_similarity_norms")
    parser.add_argument("--samples-per-prompt", type=int, default=8)
    parser.add_argument("--sampling-batch-size", type=int, default=16)
    parser.add_argument("--seed-base", type=int, default=880000)
    parser.add_argument("--target-stream-lines", type=int, default=32)
    parser.add_argument("--max-generated-patches", type=int, default=256)
    parser.add_argument("--max-chars", type=int, default=24000)
    parser.add_argument("--timeout-s", type=int, default=300)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--band-ratio", type=float, default=0.05)
    parser.add_argument("--std-eps", type=float, default=1e-4)
    parser.add_argument("--resume", action="store_true", help="Reuse existing sample rows in --out-dir/sample_rows.jsonl.")
    parser.add_argument("--quiet", action="store_true", help="Do not print one JSON row per sampled trajectory.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = args.out_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    eligible = read_eligible(args.eligible_prompts)
    prompt_rows = read_prompts(args.prompts_jsonl)
    if eligible is not None:
        prompt_rows = [row for row in prompt_rows if str(row.get("name")) in eligible]
    if not prompt_rows:
        raise ValueError("no prompts selected")

    aria_harmony = written_harmony_reference(harmony_from_path(args.aria_path))
    rows_path = args.out_dir / "sample_rows.jsonl"
    sample_rows: list[dict[str, Any]] = []
    if args.resume and rows_path.exists():
        sample_rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        model, model_shape = build_model(args.weights, precision=args.precision)
        specs: list[dict[str, Any]] = []
        for prompt_index, row in enumerate(prompt_rows):
            for sample_index in range(args.samples_per_prompt):
                specs.append(
                    {
                        "prefix_name": str(row["name"]),
                        "prompt": str(row["prompt"]),
                        "seed": args.seed_base + prompt_index * 10000 + sample_index,
                        "sample_index_for_prompt": sample_index,
                    }
                )

        for batch_start in range(0, len(specs), args.sampling_batch_size):
            batch = specs[batch_start : batch_start + args.sampling_batch_size]
            t0 = time.perf_counter()
            results = sample_completions_cached_batch(
                model=model,
                model_shape=model_shape,
                prompts=[row["prompt"] for row in batch],
                seeds=[int(row["seed"]) for row in batch],
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                target_stream_lines=args.target_stream_lines,
                target_new_stream_lines=False,
                max_chars=args.max_chars,
                max_generated_patches=args.max_generated_patches,
                timeout_s=args.timeout_s,
                precision=args.precision,
            )
            elapsed_s = time.perf_counter() - t0
            for local_idx, (spec, result) in enumerate(zip(batch, results, strict=True)):
                row: dict[str, Any] = {
                    "prefix_name": spec["prefix_name"],
                    "seed": spec["seed"],
                    "sample_index_for_prompt": spec["sample_index_for_prompt"],
                    "batch_start": batch_start,
                    "batch_elapsed_s": elapsed_s,
                    "elapsed_s": elapsed_s / max(1, len(batch)),
                    "ok": bool(result.ok),
                }
                if result.ok and result.full_text is not None:
                    sample_name = f"base_{spec['prefix_name']}_seed{spec['seed']}.abc"
                    sample_path = samples_dir / sample_name
                    sample_path.write_text(result.full_text, encoding="utf-8")
                    harmony = candidate_harmony_from_text(result.full_text)
                    scores = strict_symbolic_similarity(aria_harmony, harmony, band_ratio=args.band_ratio)
                    row.update(
                        {
                            "path": str(sample_path),
                            "generated_patches": len(result.generated_patches or []),
                            "chars": len(result.full_text),
                            "stream_lines": count_stream_lines(result.full_text),
                            **(result.meta or {}),
                            **scores,
                        }
                    )
                else:
                    row["error"] = result.error or "unknown sampling error"
                sample_rows.append(row)
                if not args.quiet:
                    print(json.dumps({"sample": batch_start + local_idx, **row}), flush=True)
            if args.quiet:
                ok_count = sum(1 for row in sample_rows if row.get("ok"))
                print(
                    json.dumps(
                        {
                            "batch_start": batch_start,
                            "batch_size": len(batch),
                            "samples_seen": len(sample_rows),
                            "ok_seen": ok_count,
                            "elapsed_s": elapsed_s,
                        }
                    ),
                    flush=True,
                )

        rows_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in sample_rows) + "\n", encoding="utf-8")

    write_csv(args.out_dir / "sample_rows.csv", sample_rows)
    prompt_norm_rows = summarize_prompt_norms(sample_rows, eps=args.std_eps)
    write_csv(args.out_dir / "prompt_strict_similarity_norms.csv", prompt_norm_rows)
    config = {
        "weights": str(args.weights),
        "prompts_jsonl": str(args.prompts_jsonl),
        "eligible_prompts": str(args.eligible_prompts) if args.eligible_prompts else None,
        "aria_path": str(args.aria_path),
        "aria_representation": "written",
        "samples_per_prompt": args.samples_per_prompt,
        "sampling_batch_size": args.sampling_batch_size,
        "seed_base": args.seed_base,
        "target_stream_lines": args.target_stream_lines,
        "max_generated_patches": args.max_generated_patches,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "band_ratio": args.band_ratio,
        "std_eps": args.std_eps,
    }
    write_norms_json(args.out_dir / "prompt_strict_similarity_norms.json", sample_rows, eps=args.std_eps, config=config)

    ok_rows = [row for row in sample_rows if row.get("ok")]
    print(
        json.dumps(
            {
                "out_dir": str(args.out_dir),
                "prompts": len(prompt_rows),
                "samples": len(sample_rows),
                "ok": len(ok_rows),
                "failed": len(sample_rows) - len(ok_rows),
                "mean_strict_symbolic_similarity": statistics.mean(
                    float(row["strict_symbolic_similarity"]) for row in ok_rows
                )
                if ok_rows
                else math.nan,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
