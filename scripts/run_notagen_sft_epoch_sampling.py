#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


def run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None, log: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    if log is None:
        subprocess.run(cmd, cwd=cwd, env=env, check=True)
        return
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=handle, stderr=subprocess.STDOUT, text=True)
        code = process.wait()
    if code != 0:
        raise subprocess.CalledProcessError(code, cmd)


def parse_loss_log(path: Path) -> dict[str, float | int | str]:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    result: dict[str, float | int | str] = {}
    for line in text.splitlines():
        if line.startswith("Epoch "):
            result["reported_epoch"] = int(line.split()[1])
        elif line.startswith("train_loss:"):
            result["train_loss"] = float(line.split(":", 1)[1].strip())
        elif line.startswith("eval_loss:"):
            result["eval_loss"] = float(line.split(":", 1)[1].strip())
        elif line.startswith("time:"):
            result["time"] = line.split(":", 1)[1].strip()
    return result


def mean_metric(rows: list[dict], key: str) -> float | None:
    values = [float(row[key]) for row in rows if key in row and row[key] is not None]
    if not values:
        return None
    return sum(values) / len(values)


def mean_metric_first(rows: list[dict], *keys: str) -> float | None:
    for key in keys:
        value = mean_metric(rows, key)
        if value is not None:
            return value
    return None


def load_jsonl_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def variation_name_from_index_row(row: dict) -> str:
    path = Path(str(row.get("path") or ""))
    match = re.search(r"variation-\d+", path.name)
    if match is None:
        match = re.search(r"variation-\d+", str(path))
    if match is None:
        raise ValueError(f"could not infer variation name from index row: {row}")
    return match.group(0)


def dedupe_index_rows(rows: list[dict]) -> list[dict]:
    seen: set[tuple[str, str]] = set()
    deduped = []
    for row in rows:
        key = (str(row.get("path")), str(row.get("key")))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def build_sft_variation_split_manifests(
    *,
    train_jsonl: Path,
    eval_jsonl: Path,
    output_dir: Path,
    eval_variation_count: int,
    eval_variation_seed: int,
) -> tuple[Path, Path, dict]:
    """Build run-local train/eval manifests split by variation name.

    eval_variation_count=0 means train on all rows. To keep NotaGen's eval-loss
    code alive, eval rows are also set to all rows; this eval loss is therefore
    in-sample and should not be treated as held-out validation.
    """

    source_rows = dedupe_index_rows(load_jsonl_rows(train_jsonl) + load_jsonl_rows(eval_jsonl))
    if not source_rows:
        raise ValueError("cannot build SFT split from empty train/eval manifests")

    variation_names = sorted({variation_name_from_index_row(row) for row in source_rows})
    if eval_variation_count < 0:
        raise ValueError("--sft-eval-variation-count must be non-negative")
    if eval_variation_count >= len(variation_names) and eval_variation_count != 0:
        raise ValueError(
            "--sft-eval-variation-count must be 0 or smaller than the number of variations "
            f"({len(variation_names)})"
        )

    if eval_variation_count == 0:
        eval_variations: set[str] = set()
        train_rows = source_rows
        eval_rows = source_rows
        split_mode = "train_all_eval_all_for_loss"
    else:
        rng = random.Random(eval_variation_seed)
        eval_variations = set(rng.sample(variation_names, eval_variation_count))
        train_rows = [row for row in source_rows if variation_name_from_index_row(row) not in eval_variations]
        eval_rows = [row for row in source_rows if variation_name_from_index_row(row) in eval_variations]
        split_mode = "heldout_variations"

    manifest_dir = output_dir / "manifests"
    train_out = manifest_dir / "augmented_train.jsonl"
    eval_out = manifest_dir / "augmented_eval.jsonl"
    write_jsonl_rows(train_out, train_rows)
    write_jsonl_rows(eval_out, eval_rows)

    split_manifest = {
        "mode": split_mode,
        "source_train_jsonl": str(train_jsonl),
        "source_eval_jsonl": str(eval_jsonl),
        "train_jsonl": str(train_out),
        "eval_jsonl": str(eval_out),
        "eval_variation_count": eval_variation_count,
        "eval_variation_seed": eval_variation_seed,
        "variation_count": len(variation_names),
        "train_row_count": len(train_rows),
        "eval_row_count": len(eval_rows),
        "train_variations": [
            name for name in variation_names if eval_variation_count == 0 or name not in eval_variations
        ],
        "eval_variations": sorted(eval_variations),
        "notes": [
            "Rows are split by variation name, preserving all key augmentations for each selected variation.",
            *(
                [
                    "eval_variation_count=0 trains on all variations; eval loss uses the same all-row manifest and is not held out.",
                ]
                if eval_variation_count == 0
                else []
            ),
        ],
    }
    (manifest_dir / "sft_variation_split_manifest.json").write_text(
        json.dumps(split_manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    (manifest_dir / "eval_variations.txt").write_text(
        "\n".join(sorted(eval_variations)) + ("\n" if eval_variations else ""),
        encoding="utf-8",
    )
    return train_out, eval_out, split_manifest


def _ratio_fraction(label: str) -> Fraction:
    if "/" in label:
        numerator, denominator = label.split("/", 1)
        return Fraction(int(numerator), int(denominator))
    return Fraction(int(label), 1)


def summarize_meter_duration_ratio_counts(counts: Counter[Fraction]) -> dict:
    total = sum(counts.values())
    if total <= 0:
        return {
            "voice_bar_count": 0,
            "half_bar_fraction": None,
            "exact_bar_fraction": None,
            "double_bar_fraction": None,
            "top_ratios": [],
        }
    return {
        "voice_bar_count": total,
        "half_bar_fraction": counts[Fraction(1, 2)] / total,
        "exact_bar_fraction": counts[Fraction(1, 1)] / total,
        "double_bar_fraction": counts[Fraction(2, 1)] / total,
        "top_ratios": [
            {"ratio": str(ratio), "count": count, "fraction": count / total}
            for ratio, count in counts.most_common(8)
        ],
    }


def meter_duration_ratio_counts_for_text(abc_text: str) -> Counter[Fraction]:
    from evaluation.rewards import (  # type: ignore
        _extract_header_context,
        _extract_stream_line_features,
        _segment_active_meter,
        _split_voice_segments,
        _voice_segment_duration,
    )

    header = _extract_header_context(abc_text)
    active_meter = header.meter
    counts: Counter[Fraction] = Counter()
    for stream_line in _extract_stream_line_features(abc_text):
        for voice, segment in _split_voice_segments(stream_line.body):
            segment_meter, active_meter = _segment_active_meter(segment, active_meter)
            if "|" not in segment:
                continue
            base_length = header.voice_lengths.get(voice, header.default_length) if voice is not None else header.default_length
            duration = _voice_segment_duration(segment, base_length)
            if duration > 0 and segment_meter > 0:
                counts[duration / segment_meter] += 1
    return counts


def aggregate_meter_duration_ratio_monitor(rows: list[dict]) -> dict:
    def row_counts(row: dict) -> Counter[Fraction]:
        return Counter(
            {
                _ratio_fraction(label): int(count)
                for label, count in (row.get("meter_duration_ratio_counts") or {}).items()
            }
        )

    overall: Counter[Fraction] = Counter()
    by_prompt: dict[str, Counter[Fraction]] = defaultdict(Counter)
    by_meter: dict[str, Counter[Fraction]] = defaultdict(Counter)
    by_default_length: dict[str, Counter[Fraction]] = defaultdict(Counter)
    group_metadata: dict[str, dict] = {}

    for row in rows:
        counts = row_counts(row)
        if not counts:
            continue
        overall.update(counts)
        prompt_name = str(row.get("prefix_name") or row.get("candidate_path") or "unknown")
        meter = str(row.get("prompt_meter") or "unknown")
        default_length = str(row.get("prompt_default_length") or "unknown")
        by_prompt[prompt_name].update(counts)
        by_meter[meter].update(counts)
        by_default_length[default_length].update(counts)
        group_metadata.setdefault(
            prompt_name,
            {
                "prompt_meter": row.get("prompt_meter"),
                "prompt_default_length": row.get("prompt_default_length"),
            },
        )

    def grouped(group_counts: dict[str, Counter[Fraction]], *, include_prompt_metadata: bool = False) -> list[dict]:
        result = []
        for name, counts in sorted(group_counts.items()):
            item = {"name": name, **summarize_meter_duration_ratio_counts(counts)}
            if include_prompt_metadata:
                item.update(group_metadata.get(name, {}))
            result.append(item)
        return result

    return {
        "overall": summarize_meter_duration_ratio_counts(overall),
        "by_prompt": grouped(by_prompt, include_prompt_metadata=True),
        "by_meter": grouped(by_meter),
        "by_default_length": grouped(by_default_length),
    }


def load_cached_sampler(project_dir: Path):
    if str(project_dir) not in sys.path:
        sys.path.insert(0, str(project_dir))
    from scripts.generate_notagen_cached_inference import sample_completion_cached  # type: ignore
    from notagen_runtime.notagen_cached_generation_batch import sample_completions_cached_batch  # type: ignore
    from notagen_runtime.notagen_wrapper import build_model, count_stream_lines, set_seed  # type: ignore

    return build_model, count_stream_lines, sample_completion_cached, sample_completions_cached_batch, set_seed


def shuffled_prefix_specs_for_epoch(
    prefix_specs: list[dict] | None,
    *,
    epoch: int,
    prefix_shuffle_seed: int | None,
) -> list[dict]:
    shuffled_prefix_specs = prefix_specs[:] if prefix_specs else []
    rng = random.Random(epoch if prefix_shuffle_seed is None else prefix_shuffle_seed)
    rng.shuffle(shuffled_prefix_specs)
    return shuffled_prefix_specs


def sample_cached_trajectories(
    *,
    project_dir: Path,
    weights: Path,
    prefix: Path | None,
    prefix_specs: list[dict] | None,
    out_dir: Path,
    samples_per_epoch: int,
    max_generation_attempts: int,
    target_stream_lines: int,
    temperature: str,
    top_k: str,
    top_p: str,
    timeout_s: str,
    max_chars: str,
    max_generated_patches: int,
    precision: str,
    epoch: int,
    sampling_batch_size: int,
    prefix_shuffle_seed: int | None,
) -> tuple[list[Path], list[dict], list[dict]]:
    (
        build_model,
        count_stream_lines,
        sample_completion_cached,
        sample_completions_cached_batch,
        set_seed,
    ) = load_cached_sampler(project_dir)
    model, model_shape = build_model(weights, precision=precision)
    candidates: list[Path] = []
    generation_failures: list[dict] = []
    sample_metadata: list[dict] = []
    shuffled_prefix_specs = shuffled_prefix_specs_for_epoch(
        prefix_specs,
        epoch=epoch,
        prefix_shuffle_seed=prefix_shuffle_seed,
    )

    def resolve_prefix_spec(attempt_seed: int) -> tuple[Path, str, str]:
        if shuffled_prefix_specs:
            prefix_spec = shuffled_prefix_specs[attempt_seed % len(shuffled_prefix_specs)]
            prefix_path = Path(prefix_spec["prefix"])
            if not prefix_path.is_absolute():
                prefix_path = project_dir / prefix_path
            prefix_name = prefix_path.stem
        else:
            if prefix is None:
                raise ValueError("prefix is required when prefix_specs is not provided")
            prefix_path = prefix
            prefix_name = prefix_path.stem
        return prefix_path, prefix_name, prefix_path.read_text(encoding="utf-8")

    attempt_seed = 0
    while attempt_seed < max_generation_attempts and len(candidates) < samples_per_epoch:
        if sampling_batch_size <= 1:
            prefix_path, prefix_name, prompt = resolve_prefix_spec(attempt_seed)
            sample_idx = len(candidates)
            output = out_dir / f"epoch{epoch:02d}_sample{sample_idx:02d}_{prefix_name}_seed{attempt_seed}.abc"
            try:
                set_seed(attempt_seed)
                t0 = time.perf_counter()
                full_text, generated_patches, meta = sample_completion_cached(
                    model=model,
                    model_shape=model_shape,
                    prompt=prompt,
                    temperature=float(temperature),
                    top_k=int(top_k),
                    top_p=float(top_p),
                    target_stream_lines=target_stream_lines,
                    target_new_stream_lines=False,
                    max_chars=int(max_chars),
                    timeout_s=int(timeout_s),
                    precision=precision,
                )
                elapsed_s = time.perf_counter() - t0
            except RuntimeError as exc:
                generation_failures.append({"seed": attempt_seed, "error": str(exc)})
                print(f"generation failed epoch={epoch} seed={attempt_seed}: {exc}; continuing", flush=True)
                attempt_seed += 1
                continue

            output.write_text(full_text, encoding="utf-8")
            stream_lines = count_stream_lines(full_text)
            metadata = {
                "seed": attempt_seed,
                "path": str(output),
                "prefix_path": str(prefix_path),
                "prefix_name": prefix_name,
                "generated_patches": len(generated_patches),
                "chars": len(full_text),
                "stream_lines": stream_lines,
                "new_stream_lines": max(0, stream_lines - int(meta["prompt_stream_lines"])),
                "elapsed_s": elapsed_s,
                **meta,
            }
            sample_metadata.append(metadata)
            candidates.append(output)
            print(json.dumps({"sample": sample_idx, **metadata}), flush=True)
            attempt_seed += 1
            continue

        batch_specs = []
        batch_count = min(sampling_batch_size, max_generation_attempts - attempt_seed)
        for offset in range(batch_count):
            seed = attempt_seed + offset
            prefix_path, prefix_name, prompt = resolve_prefix_spec(seed)
            batch_specs.append(
                {
                    "seed": seed,
                    "prefix_path": prefix_path,
                    "prefix_name": prefix_name,
                    "prompt": prompt,
                }
            )
        prompts = [row["prompt"] for row in batch_specs]
        seeds = [int(row["seed"]) for row in batch_specs]
        t0 = time.perf_counter()
        batch_results = sample_completions_cached_batch(
            model=model,
            model_shape=model_shape,
            prompts=prompts,
            seeds=seeds,
            temperature=float(temperature),
            top_k=int(top_k),
            top_p=float(top_p),
            target_stream_lines=target_stream_lines,
            target_new_stream_lines=False,
            max_chars=int(max_chars),
            max_generated_patches=max_generated_patches,
            timeout_s=int(timeout_s),
            precision=precision,
        )
        elapsed_s = time.perf_counter() - t0
        elapsed_per_sample = elapsed_s / max(1, len(batch_results))

        for spec, result in zip(batch_specs, batch_results, strict=True):
            if len(candidates) >= samples_per_epoch:
                break
            if not result.ok:
                generation_failures.append({"seed": spec["seed"], "error": result.error or "unknown batch generation error"})
                print(
                    f"generation failed epoch={epoch} seed={spec['seed']}: {result.error or 'unknown batch generation error'}; continuing",
                    flush=True,
                )
                continue

            assert result.full_text is not None
            assert result.generated_patches is not None
            assert result.meta is not None
            sample_idx = len(candidates)
            output = out_dir / f"epoch{epoch:02d}_sample{sample_idx:02d}_{spec['prefix_name']}_seed{spec['seed']}.abc"
            output.write_text(result.full_text, encoding="utf-8")
            stream_lines = count_stream_lines(result.full_text)
            metadata = {
                "seed": spec["seed"],
                "path": str(output),
                "prefix_path": str(spec["prefix_path"]),
                "prefix_name": str(spec["prefix_name"]),
                "generated_patches": len(result.generated_patches),
                "chars": len(result.full_text),
                "stream_lines": stream_lines,
                "new_stream_lines": max(0, stream_lines - int(result.meta["prompt_stream_lines"])),
                "elapsed_s": elapsed_per_sample,
                **result.meta,
            }
            sample_metadata.append(metadata)
            candidates.append(output)
            print(json.dumps({"sample": sample_idx, **metadata}), flush=True)
        attempt_seed += batch_count

    del model
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    return candidates, generation_failures, sample_metadata


def sample_cached_trajectories_isolated(
    *,
    project_dir: Path,
    venv_python: Path,
    weights: Path,
    prefix: Path | None,
    prefix_specs: list[dict] | None,
    out_dir: Path,
    samples_per_epoch: int,
    max_generation_attempts: int,
    target_stream_lines: int,
    temperature: str,
    top_k: str,
    top_p: str,
    timeout_s: str,
    max_chars: str,
    max_generated_patches: int,
    precision: str,
    epoch: int,
    prefix_shuffle_seed: int | None,
) -> tuple[list[Path], list[dict], list[dict]]:
    if str(project_dir) not in sys.path:
        sys.path.insert(0, str(project_dir))
    from notagen_runtime.notagen_wrapper import count_stream_lines  # type: ignore

    candidates: list[Path] = []
    generation_failures: list[dict] = []
    sample_metadata: list[dict] = []
    shuffled_prefix_specs = shuffled_prefix_specs_for_epoch(
        prefix_specs,
        epoch=epoch,
        prefix_shuffle_seed=prefix_shuffle_seed,
    )
    tmp_root = out_dir / ".isolated_tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)

    def resolve_prefix_spec(attempt_seed: int) -> tuple[Path, str]:
        if shuffled_prefix_specs:
            prefix_spec = shuffled_prefix_specs[attempt_seed % len(shuffled_prefix_specs)]
            prefix_path = Path(prefix_spec["prefix"])
            if not prefix_path.is_absolute():
                prefix_path = project_dir / prefix_path
            prefix_name = prefix_path.stem
        else:
            if prefix is None:
                raise ValueError("prefix is required when prefix_specs is not provided")
            prefix_path = prefix
            prefix_name = prefix_path.stem
        return prefix_path, prefix_name

    attempt_seed = 0
    while attempt_seed < max_generation_attempts and len(candidates) < samples_per_epoch:
        prefix_path, prefix_name = resolve_prefix_spec(attempt_seed)
        sample_idx = len(candidates)
        output = out_dir / f"epoch{epoch:02d}_sample{sample_idx:02d}_{prefix_name}_seed{attempt_seed}.abc"
        attempt_dir = tmp_root / f"seed{attempt_seed}"
        shutil.rmtree(attempt_dir, ignore_errors=True)
        attempt_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            str(venv_python),
            str(project_dir / "scripts" / "generate_notagen_cached_inference.py"),
            "--weights",
            str(weights),
            "--prefix",
            str(prefix_path),
            "--out-dir",
            str(attempt_dir),
            "--seeds",
            str(attempt_seed),
            "--precision",
            precision,
            "--temperature",
            temperature,
            "--top-k",
            top_k,
            "--top-p",
            top_p,
            "--target-stream-lines",
            str(target_stream_lines),
            "--max-chars",
            max_chars,
            "--timeout-s",
            timeout_s,
        ]
        try:
            result = subprocess.run(
                cmd,
                cwd=project_dir,
                check=True,
                text=True,
                capture_output=True,
                timeout=int(timeout_s) + 180,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            stdout = getattr(exc, "stdout", "") or ""
            stderr = getattr(exc, "stderr", "") or ""
            error = (stdout + "\n" + stderr).strip()
            if len(error) > 4000:
                error = error[-4000:]
            generation_failures.append({"seed": attempt_seed, "error": error or repr(exc)})
            print(f"generation failed epoch={epoch} seed={attempt_seed}: {error or exc!r}; continuing", flush=True)
            shutil.rmtree(attempt_dir, ignore_errors=True)
            attempt_seed += 1
            continue

        json_rows = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                json_rows.append(json.loads(line))
        if not json_rows:
            generation_failures.append({"seed": attempt_seed, "error": "isolated sampler produced no metadata"})
            print(f"generation failed epoch={epoch} seed={attempt_seed}: isolated sampler produced no metadata; continuing", flush=True)
            shutil.rmtree(attempt_dir, ignore_errors=True)
            attempt_seed += 1
            continue

        metadata = json_rows[-1]
        child_path = Path(metadata["path"])
        if output.exists():
            output.unlink()
        child_path.replace(output)
        shutil.rmtree(attempt_dir, ignore_errors=True)

        text = output.read_text(encoding="utf-8")
        stream_lines = count_stream_lines(text)
        metadata.update(
            {
                "seed": attempt_seed,
                "path": str(output),
                "prefix_path": str(prefix_path),
                "prefix_name": prefix_name,
                "chars": len(text),
                "stream_lines": stream_lines,
                "new_stream_lines": max(0, stream_lines - int(metadata["prompt_stream_lines"])),
            }
        )
        sample_metadata.append(metadata)
        candidates.append(output)
        print(json.dumps({"sample": sample_idx, **metadata}), flush=True)
        attempt_seed += 1

    shutil.rmtree(tmp_root, ignore_errors=True)
    return candidates, generation_failures, sample_metadata


def _flatten_patch_ids(patches: list[list[int]]) -> list[int]:
    return [int(token) for patch in patches for token in patch]


def _chunk_generated_flat_ids(generated_flat_ids: list[int], prompt_token_count: int) -> list[list[int]]:
    if not generated_flat_ids:
        return []
    first_len = 16 - (prompt_token_count % 16)
    if first_len == 16:
        first_len = 16
    chunks: list[list[int]] = []
    offset = 0
    while offset < len(generated_flat_ids):
        chunk_len = first_len if not chunks else 16
        chunks.append(generated_flat_ids[offset : offset + chunk_len])
        offset += chunk_len
    return chunks


def _effective_generation_prompt(prompt_text: str, target_stream_lines: int, count_stream_lines) -> str:
    if count_stream_lines(prompt_text) == 0:
        return prompt_text + f"[r:0/{target_stream_lines - 1}]"
    return prompt_text


def _kl_replay_inputs(
    *,
    prompt_text: str,
    full_text: str,
    target_stream_lines: int,
    patchilizer,
    count_stream_lines,
) -> tuple[list[int], list[list[int]], str]:
    effective_prompt = _effective_generation_prompt(prompt_text, target_stream_lines, count_stream_lines)
    prompt_flat = _flatten_patch_ids(patchilizer.encode_generate(effective_prompt))
    full_flat = _flatten_patch_ids(patchilizer.encode_generate(full_text))
    if full_flat[: len(prompt_flat)] != prompt_flat:
        raise RuntimeError(
            "encoded full sample does not start with the encoded effective prompt; "
            "cannot compute prompt-masked exact KL reliably"
        )
    generated_flat = full_flat[len(prompt_flat) :]
    return prompt_flat, _chunk_generated_flat_ids(generated_flat, len(prompt_flat)), effective_prompt


def compute_exact_reference_kl(
    *,
    project_dir: Path,
    policy_weights: Path,
    reference_weights: Path,
    samples: list[dict],
    target_stream_lines: int,
    precision: str,
    score_chunk_patches: int,
    replay_context_patches: int,
    replay_batch_size: int,
    out_path: Path,
) -> dict:
    if str(project_dir) not in sys.path:
        sys.path.insert(0, str(project_dir))

    import torch  # type: ignore
    from notagen_runtime.notagen_replay import (  # type: ignore
        batched_trajectory_token_log_dists,
        exact_categorical_kl,
    )
    from notagen_runtime.notagen_wrapper import PATCH_STREAM, Patchilizer, build_model, count_stream_lines  # type: ignore

    policy_model = None
    reference_model = None
    policy_model, _policy_shape = build_model(policy_weights, precision=precision)
    device = next(policy_model.parameters()).device
    reference_model, _reference_shape = build_model(reference_weights, device=device, precision=precision)
    patchilizer = Patchilizer(stream=PATCH_STREAM)

    rows: list[dict] = []
    failures: list[dict] = []
    weighted_kl_sum = 0.0
    total_tokens = 0
    total_patches = 0
    t0 = time.perf_counter()
    try:
        with torch.no_grad():
            for index, sample in enumerate(samples):
                try:
                    candidate_path = Path(sample["path"])
                    prefix_path = Path(sample["prefix_path"])
                    prompt_text = prefix_path.read_text(encoding="utf-8")
                    full_text = candidate_path.read_text(encoding="utf-8")
                    prompt_flat, generated_patches, effective_prompt = _kl_replay_inputs(
                        prompt_text=prompt_text,
                        full_text=full_text,
                        target_stream_lines=target_stream_lines,
                        patchilizer=patchilizer,
                        count_stream_lines=count_stream_lines,
                    )
                    if not generated_patches:
                        failures.append(
                            {
                                "sample_index": index,
                                "path": str(candidate_path),
                                "error": "no generated tokens after prompt masking",
                            }
                        )
                        continue
                    policy_replay = batched_trajectory_token_log_dists(
                        policy_model,
                        prompt_flat,
                        [generated_patches],
                        precision=precision,
                        replay_context_patches=replay_context_patches,
                        target_chunk_patches=score_chunk_patches,
                        replay_batch_size=replay_batch_size,
                    )[0]
                    reference_replay = batched_trajectory_token_log_dists(
                        reference_model,
                        prompt_flat,
                        [generated_patches],
                        precision=precision,
                        replay_context_patches=replay_context_patches,
                        target_chunk_patches=score_chunk_patches,
                        replay_batch_size=replay_batch_size,
                    )[0]
                    if policy_replay.token_log_dists.shape != reference_replay.token_log_dists.shape:
                        raise RuntimeError(
                            "token distribution shape mismatch: "
                            f"policy={tuple(policy_replay.token_log_dists.shape)} "
                            f"reference={tuple(reference_replay.token_log_dists.shape)}"
                        )
                    if not torch.equal(
                        policy_replay.token_counts.detach().cpu(),
                        reference_replay.token_counts.detach().cpu(),
                    ):
                        raise RuntimeError("token-count mismatch between policy and reference replay")
                    token_count = int(policy_replay.token_log_dists.shape[0])
                    if token_count <= 0:
                        failures.append(
                            {
                                "sample_index": index,
                                "path": str(candidate_path),
                                "error": "zero active replay tokens",
                            }
                        )
                        continue
                    kl = float(
                        exact_categorical_kl(policy_replay.token_log_dists, reference_replay.token_log_dists)
                        .detach()
                        .cpu()
                    )
                    patch_count = int(policy_replay.token_counts.numel())
                    weighted_kl_sum += kl * token_count
                    total_tokens += token_count
                    total_patches += patch_count
                    rows.append(
                        {
                            "sample_index": index,
                            "path": str(candidate_path),
                            "prefix_path": str(prefix_path),
                            "prefix_name": sample.get("prefix_name"),
                            "token_count": token_count,
                            "patch_count": patch_count,
                            "effective_prompt_chars": len(effective_prompt),
                            "exact_kl_to_reference": kl,
                        }
                    )
                except Exception as exc:  # keep per-sample context in the JSON report
                    failures.append(
                        {
                            "sample_index": index,
                            "path": sample.get("path"),
                            "prefix_path": sample.get("prefix_path"),
                            "error": str(exc),
                        }
                    )
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    finally:
        if policy_model is not None:
            del policy_model
        if reference_model is not None:
            del reference_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = {
        "ok": bool(rows),
        "policy_weights": str(policy_weights),
        "reference_weights": str(reference_weights),
        "trajectory_count": len(rows),
        "failed_trajectory_count": len(failures),
        "token_count": total_tokens,
        "patch_count": total_patches,
        "mean_exact_kl_to_reference": (weighted_kl_sum / total_tokens) if total_tokens else None,
        "per_trajectory": rows,
        "failures": failures,
        "score_chunk_patches": score_chunk_patches,
        "replay_context_patches": replay_context_patches,
        "replay_batch_size": replay_batch_size,
        "elapsed_s": time.perf_counter() - t0,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def score_rewards(
    *,
    project_dir: Path,
    prefix: Path | None,
    candidate_prefixes: dict[str, Path],
    reward_target_json: Path,
    reward_target_structure_abc: Path | None,
    candidates: list[Path],
    out_path: Path,
    aria_reference: Path,
    aria_chroma_reward_weight: float,
    aria_chroma_top_reward_weight: float,
    aria_harmony_reward_weight: float,
    aria_harmony_aligned_root_reward_weight: float,
    aria_harmony_aligned_bass_reward_weight: float,
    aria_harmony_aligned_top_reward_weight: float,
    aria_strict_symbolic_reward_weight: float,
    max_similarity_reward: float,
    similarity_chroma_bins: int,
    similarity_band_ratio: float,
    similarity_timeout_s: float,
) -> list[dict]:
    if str(project_dir) not in sys.path:
        sys.path.insert(0, str(project_dir))
    from evaluation.rewards import (  # type: ignore
        GoldbergRewardConfig,
        _extract_header_context,
        load_structural_target,
        score_prompt_completion_pair,
    )
    from evaluation.similarity_rewards import (  # type: ignore
        SimilarityRewardWeights,
        finalize_similarity_reward_fields,
        load_similarity_reference,
        score_similarity_reward,
    )

    target = load_structural_target(reward_target_json, structure_path=reward_target_structure_abc)
    config = GoldbergRewardConfig()
    similarity_weights = SimilarityRewardWeights(
        aria_chroma=aria_chroma_reward_weight,
        aria_chroma_top=aria_chroma_top_reward_weight,
        aria_harmony=aria_harmony_reward_weight,
        aria_harmony_aligned_root=aria_harmony_aligned_root_reward_weight,
        aria_harmony_aligned_bass=aria_harmony_aligned_bass_reward_weight,
        aria_harmony_aligned_top=aria_harmony_aligned_top_reward_weight,
        aria_strict_symbolic=aria_strict_symbolic_reward_weight,
    )
    aria_similarity_ref = None
    if similarity_weights.enabled:
        aria_similarity_ref = load_similarity_reference(
            aria_reference,
            load_chroma=similarity_weights.needs_chroma,
            load_harmony=similarity_weights.needs_harmony,
            bins=similarity_chroma_bins,
        )
    rows = []
    for candidate in candidates:
        prefix_path = candidate_prefixes.get(str(candidate), prefix)
        if prefix_path is None:
            raise ValueError(f"missing prefix for {candidate}")
        prompt = prefix_path.read_text(encoding="utf-8")
        text = candidate.read_text(encoding="utf-8")
        completion = text[len(prompt) :] if text.startswith(prompt) else text
        breakdown = score_prompt_completion_pair(
            prompt_text=prompt,
            completion_text=completion,
            target=target,
            config=config,
            candidate_name=candidate.stem,
        ).to_json()
        structural_total_reward = float(breakdown["total_reward"])
        breakdown["structural_total_reward"] = structural_total_reward
        if similarity_weights.enabled:
            similarity_payload = score_similarity_reward(
                prompt_text=prompt,
                completion_text=completion,
                weights=similarity_weights,
                aria=aria_similarity_ref,
                variation=None,
                bins=similarity_chroma_bins,
                band_ratio=similarity_band_ratio,
                timeout_s=similarity_timeout_s,
            )
            breakdown.update(similarity_payload)
            breakdown.update(
                finalize_similarity_reward_fields(
                    similarity_payload=similarity_payload,
                    structural_total_reward=structural_total_reward,
                    completion_reward=breakdown.get("completion_reward", 0.0),
                    bar_count_reward=breakdown.get("bar_count_reward", 0.0),
                    max_similarity_reward=max_similarity_reward,
                )
            )
        else:
            breakdown["similarity_reward"] = 0.0
            breakdown["active_similarity_reward"] = 0.0
            breakdown["effective_similarity_reward"] = 0.0
        prompt_header = _extract_header_context(prompt)
        duration_ratio_counts = meter_duration_ratio_counts_for_text(text)
        breakdown["path"] = str(candidate)
        breakdown["prefix_path"] = str(prefix_path)
        breakdown["prefix_name"] = prefix_path.stem
        breakdown["prompt_meter"] = str(prompt_header.meter)
        breakdown["prompt_default_length"] = str(prompt_header.default_length)
        breakdown["meter_duration_ratio_counts"] = {
            str(ratio): count for ratio, count in sorted(duration_ratio_counts.items(), key=lambda item: item[0])
        }
        breakdown["meter_duration_ratio_summary"] = summarize_meter_duration_ratio_counts(duration_ratio_counts)
        breakdown["completion_chars"] = len(completion)
        rows.append(breakdown)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return rows


def _eval_tag(row_type: str, epoch: int) -> str:
    if row_type == "pretrained_baseline":
        return "pretrained_baseline"
    return f"epoch{epoch:02d}"


def evaluate_checkpoint_samples(
    *,
    args: argparse.Namespace,
    row_type: str,
    epoch: int,
    checkpoint: Path,
    checkpoint_is_rolling: bool,
    rolling_checkpoint: Path | None,
    epoch_checkpoint: Path | None,
    losses: dict | None,
    prefix_specs: list[dict] | None,
    variation_refs: list[Path],
    exact_kl_reference_checkpoint: Path,
    max_generated_patches: int,
    samples_dir: Path,
    logs_dir: Path,
    scores_dir: Path,
) -> dict:
    tag = _eval_tag(row_type, epoch)
    if args.samples_per_epoch == 0:
        return {
            "epoch": epoch,
            "row_type": row_type,
            "checkpoint": str(checkpoint),
            "checkpoint_is_rolling": checkpoint_is_rolling,
            "rolling_checkpoint": None if rolling_checkpoint is None else str(rolling_checkpoint),
            "epoch_checkpoint": None if epoch_checkpoint is None else str(epoch_checkpoint),
            "losses": losses,
            "generation_failures": [],
            "samples": [],
            "meter_duration_ratio_monitor": None,
            "mean_clamp2_aria_similarity": None,
            "mean_clamp2_variation_centroid_similarity": None,
            "mean_reward": None,
            "mean_structural_total_reward": None,
            "mean_active_similarity_reward": None,
            "mean_effective_similarity_reward": None,
            "mean_aria_chroma_harmonic_hist": None,
            "mean_aria_chroma_top_hist": None,
            "mean_aria_harmony_dtw_combined": None,
            "exact_pretrained_kl": None,
            "mean_exact_kl_to_pretrained": None,
            "prefix_shuffle_seed": args.prefix_shuffle_seed,
        }

    print(f"===== {tag} sample =====", flush=True)
    epoch_samples_dir = samples_dir / tag
    epoch_samples_dir.mkdir(parents=True, exist_ok=True)
    sample_kwargs = {
        "project_dir": args.project_dir,
        "weights": checkpoint,
        "prefix": args.prefix if args.prefix_manifest is None else None,
        "prefix_specs": prefix_specs,
        "out_dir": epoch_samples_dir,
        "samples_per_epoch": args.samples_per_epoch,
        "max_generation_attempts": args.max_generation_attempts,
        "target_stream_lines": args.target_stream_lines,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "timeout_s": args.timeout_s,
        "max_chars": args.max_chars,
        "max_generated_patches": max_generated_patches,
        "precision": args.precision,
        "epoch": epoch,
        "prefix_shuffle_seed": args.prefix_shuffle_seed,
    }
    if args.isolated_sampling:
        candidates, generation_failures, sample_metadata = sample_cached_trajectories_isolated(
            venv_python=args.venv_python,
            **sample_kwargs,
        )
    else:
        candidates, generation_failures, sample_metadata = sample_cached_trajectories(
            sampling_batch_size=args.sampling_batch_size,
            **sample_kwargs,
        )
    if len(candidates) < args.samples_per_epoch:
        raise RuntimeError(
            f"{tag} produced {len(candidates)} successful samples "
            f"after {args.max_generation_attempts} attempts"
        )

    print(f"===== {tag} score =====", flush=True)
    rewards_jsonl = scores_dir / f"{tag}_rewards.jsonl"
    candidate_prefixes = {row["path"]: Path(row["prefix_path"]) for row in sample_metadata if "prefix_path" in row}
    reward_rows = score_rewards(
        project_dir=args.project_dir,
        prefix=args.prefix if args.prefix_manifest is None else None,
        candidate_prefixes=candidate_prefixes,
        reward_target_json=args.reward_target_json,
        reward_target_structure_abc=args.reward_target_structure_abc,
        candidates=candidates,
        out_path=rewards_jsonl,
        aria_reference=args.aria_reference,
        aria_chroma_reward_weight=args.aria_chroma_reward_weight,
        aria_chroma_top_reward_weight=args.aria_chroma_top_reward_weight,
        aria_harmony_reward_weight=args.aria_harmony_reward_weight,
        aria_harmony_aligned_root_reward_weight=args.aria_harmony_aligned_root_reward_weight,
        aria_harmony_aligned_bass_reward_weight=args.aria_harmony_aligned_bass_reward_weight,
        aria_harmony_aligned_top_reward_weight=args.aria_harmony_aligned_top_reward_weight,
        aria_strict_symbolic_reward_weight=args.aria_strict_symbolic_reward_weight,
        max_similarity_reward=args.max_similarity_reward,
        similarity_chroma_bins=args.similarity_chroma_bins,
        similarity_band_ratio=args.similarity_band_ratio,
        similarity_timeout_s=args.similarity_timeout_s,
    )

    metadata_by_path = {row["path"]: row for row in sample_metadata}
    reward_by_path = {row["path"]: row for row in reward_rows}
    scores = None
    variation_scores = None
    aria_by_path = {}
    variation_by_path = {}
    if not args.skip_clamp2:
        score_json = scores_dir / f"{tag}_clamp2_aria_similarity.json"
        run(
            [
                str(args.venv_python),
                str(args.project_dir / "scripts" / "score_clamp2_similarity.py"),
                "--clamp2-dir",
                str(args.clamp2_dir),
                "--reference",
                str(args.aria_reference),
                "--output-json",
                str(score_json),
                *[str(path) for path in candidates],
            ],
            cwd=args.project_dir,
            log=logs_dir / f"{tag}_clamp2.log",
        )

        variation_score_json = scores_dir / f"{tag}_clamp2_variation_centroid_similarity.json"
        run(
            [
                str(args.venv_python),
                str(args.project_dir / "scripts" / "score_clamp2_similarity.py"),
                "--clamp2-dir",
                str(args.clamp2_dir),
                "--reference",
                *[str(path) for path in variation_refs],
                "--output-json",
                str(variation_score_json),
                *[str(path) for path in candidates],
            ],
            cwd=args.project_dir,
            log=logs_dir / f"{tag}_clamp2_variation_centroid.log",
        )
        scores = json.loads(score_json.read_text(encoding="utf-8"))
        variation_scores = json.loads(variation_score_json.read_text(encoding="utf-8"))
        aria_by_path = {row["path"]: row for row in scores["rows"]}
        variation_by_path = {row["path"]: row for row in variation_scores["rows"]}

    sample_rows = []
    for candidate in candidates:
        path_key = str(candidate)
        row = {
            **metadata_by_path[path_key],
            "reward_breakdown": reward_by_path[path_key],
        }
        if not args.skip_clamp2:
            row["clamp2_aria"] = aria_by_path[path_key]
            row["clamp2_variation_centroid"] = variation_by_path[path_key]
        sample_rows.append(row)

    exact_pretrained_kl = None
    if args.exact_pretrained_kl:
        print(f"===== {tag} exact KL to pretrained =====", flush=True)
        exact_pretrained_kl = compute_exact_reference_kl(
            project_dir=args.project_dir,
            policy_weights=checkpoint,
            reference_weights=exact_kl_reference_checkpoint,
            samples=sample_rows,
            target_stream_lines=args.target_stream_lines,
            precision=args.precision,
            score_chunk_patches=args.exact_kl_score_chunk_patches,
            replay_context_patches=args.exact_kl_replay_context_patches,
            replay_batch_size=args.exact_kl_replay_batch_size,
            out_path=scores_dir / f"{tag}_exact_kl_to_pretrained.json",
        )

    meter_duration_ratio_monitor = aggregate_meter_duration_ratio_monitor(reward_rows)
    return {
        "epoch": epoch,
        "row_type": row_type,
        "checkpoint": str(checkpoint),
        "checkpoint_is_rolling": checkpoint_is_rolling,
        "rolling_checkpoint": None if rolling_checkpoint is None else str(rolling_checkpoint),
        "epoch_checkpoint": None if epoch_checkpoint is None else str(epoch_checkpoint),
        "losses": losses,
        "generation_failures": generation_failures,
        "samples": sample_rows,
        "meter_duration_ratio_monitor": meter_duration_ratio_monitor,
        "mean_clamp2_aria_similarity": mean_metric(scores["rows"], "cosine_similarity_to_reference") if scores is not None else None,
        "mean_clamp2_variation_centroid_similarity": mean_metric(variation_scores["rows"], "cosine_similarity_to_reference") if variation_scores is not None else None,
        "mean_reward": mean_metric(reward_rows, "total_reward"),
        "mean_structural_total_reward": mean_metric(reward_rows, "structural_total_reward"),
        "mean_active_similarity_reward": mean_metric(reward_rows, "active_similarity_reward"),
        "mean_effective_similarity_reward": mean_metric(reward_rows, "effective_similarity_reward"),
        "mean_aria_chroma_harmonic_hist": mean_metric(reward_rows, "aria_chroma_harmonic_hist"),
        "mean_aria_chroma_top_hist": mean_metric(reward_rows, "aria_chroma_top_hist"),
        "mean_aria_harmony_dtw_combined": mean_metric_first(
            reward_rows,
            "aria_harmony_dtw_combined",
            "aria_harmony_combined",
        ),
        "exact_pretrained_kl": exact_pretrained_kl,
        "mean_exact_kl_to_pretrained": (
            None if exact_pretrained_kl is None else exact_pretrained_kl.get("mean_exact_kl_to_reference")
        ),
        "prefix_shuffle_seed": args.prefix_shuffle_seed,
    }


def write_summary_row(summary_path: Path, row: dict) -> None:
    with summary_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def print_summary_row(row: dict) -> None:
    parts = [
        f"epoch={row['epoch']}",
        f"row_type={row.get('row_type', 'sft_epoch')}",
    ]
    losses = row.get("losses") or {}
    if losses:
        parts.append(f"train_loss={losses.get('train_loss')}")
        parts.append(f"eval_loss={losses.get('eval_loss')}")
    else:
        parts.append("train_loss=None")
        parts.append("eval_loss=None")
    if row["mean_clamp2_aria_similarity"] is not None:
        parts.append(f"mean_clamp2_aria_similarity={row['mean_clamp2_aria_similarity']:.6f}")
    if row["mean_clamp2_variation_centroid_similarity"] is not None:
        parts.append(
            f"mean_clamp2_variation_centroid_similarity={row['mean_clamp2_variation_centroid_similarity']:.6f}"
        )
    if row["mean_exact_kl_to_pretrained"] is not None:
        parts.append(f"mean_exact_kl_to_pretrained={row['mean_exact_kl_to_pretrained']:.6f}")
    if row["mean_reward"] is not None:
        parts.append(f"mean_reward={row['mean_reward']:.6f}")
    if row["mean_structural_total_reward"] is not None:
        parts.append(f"mean_structural={row['mean_structural_total_reward']:.6f}")
    active_similarity = row.get("mean_active_similarity_reward", row.get("mean_effective_similarity_reward"))
    if active_similarity is not None:
        parts.append(f"mean_similarity={active_similarity:.6f}")
    harmony_dtw = row.get("mean_aria_harmony_dtw_combined", row.get("mean_aria_harmony_combined"))
    if harmony_dtw is not None:
        parts.append(f"aria_harmony_dtw={harmony_dtw:.6f}")
    if row["mean_aria_chroma_harmonic_hist"] is not None:
        parts.append(f"aria_chroma={row['mean_aria_chroma_harmonic_hist']:.6f}")
    if row.get("mean_aria_chroma_top_hist") is not None:
        parts.append(f"aria_chroma_top={row['mean_aria_chroma_top_hist']:.6f}")
    meter_duration_ratio_monitor = row.get("meter_duration_ratio_monitor")
    if meter_duration_ratio_monitor is not None:
        ratio_overall = meter_duration_ratio_monitor["overall"]
        if ratio_overall["voice_bar_count"]:
            parts.append(f"meter_exact={ratio_overall['exact_bar_fraction']:.6f}")
            parts.append(f"meter_half={ratio_overall['half_bar_fraction']:.6f}")
            parts.append(f"meter_double={ratio_overall['double_bar_fraction']:.6f}")
    print(" ".join(parts), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--notagen-dir", type=Path, default=Path("/home/jl_fs/music-generation/NotaGen"))
    parser.add_argument("--project-dir", type=Path, default=Path("/home/jl_fs/music-generation/infinite-goldberg-variations"))
    parser.add_argument("--venv-python", type=Path, default=Path("/home/jl_fs/music-generation/.venvs/notagen-trl/bin/python"))
    parser.add_argument("--pretrained", type=Path, default=Path("/home/jl_fs/music-generation/models/weights_notagen_pretrain-finetune_p_size_16_p_length_1024_p_layers_c_layers_6_20_h_size_1280_lr_1e-05_batch_1.pth"))
    parser.add_argument("--train-jsonl", type=Path, default=Path("/home/jl_fs/music-generation/infinite-goldberg-variations/data/processed/notagen/goldberg_aria_conditioned/augmented_train.jsonl"))
    parser.add_argument("--eval-jsonl", type=Path, default=Path("/home/jl_fs/music-generation/infinite-goldberg-variations/data/processed/notagen/goldberg_aria_conditioned/augmented_eval.jsonl"))
    parser.add_argument(
        "--sft-eval-variation-count",
        type=int,
        default=None,
        help=(
            "Build run-local train/eval manifests by variation before SFT. "
            "Use 0 to train on all variations; eval loss then uses all rows too and is in-sample. "
            "When unset, use --train-jsonl and --eval-jsonl unchanged."
        ),
    )
    parser.add_argument(
        "--sft-eval-variation-seed",
        type=int,
        default=0,
        help="Deterministic seed used when --sft-eval-variation-count is positive.",
    )
    parser.add_argument("--train-prefix-mask-root", type=Path, default=None)
    parser.add_argument("--train-prefix-mask-source-root", type=Path, default=None)
    parser.add_argument("--train-prefix-mask-manifest", type=Path, default=None)
    parser.add_argument("--prefix", type=Path, default=Path("/home/jl_fs/music-generation/infinite-goldberg-variations/data/processed/notagen/aria_plus_variation01_setup_G.abc"))
    parser.add_argument("--prefix-manifest", type=Path, default=None)
    parser.add_argument("--aria-reference", type=Path, default=Path("/home/jl_fs/music-generation/infinite-goldberg-variations/data/processed/notagen/aria_prefix_G_streamed.abc"))
    parser.add_argument("--aria-chroma-reward-weight", type=float, default=0.0)
    parser.add_argument(
        "--aria-chroma-top-reward-weight",
        type=float,
        default=None,
        help="Separate top-voice chroma histogram reward weight. Defaults to --aria-chroma-reward-weight.",
    )
    parser.add_argument("--aria-harmony-reward-weight", type=float, default=0.0)
    parser.add_argument("--aria-harmony-aligned-root-reward-weight", type=float, default=None)
    parser.add_argument("--aria-harmony-aligned-bass-reward-weight", type=float, default=None)
    parser.add_argument("--aria-harmony-aligned-top-reward-weight", type=float, default=None)
    parser.add_argument(
        "--aria-strict-symbolic-reward-weight",
        type=float,
        default=0.0,
        help=(
            "Reward weight for strict symbolic Aria similarity, computed as fixed weights over individually "
            "global-base-z normalized aligned, DTW, n-gram, and cadence submetrics."
        ),
    )
    parser.add_argument("--max-similarity-reward", type=float, default=3.5)
    parser.add_argument("--similarity-chroma-bins", type=int, default=128)
    parser.add_argument("--similarity-band-ratio", type=float, default=0.25)
    parser.add_argument("--similarity-timeout-s", type=float, default=20.0)
    parser.add_argument("--clamp2-dir", type=Path, default=Path("/home/jl_fs/music-generation/NotaGen/clamp2"))
    parser.add_argument("--output-dir", type=Path, default=Path("/home/jl_fs/music-generation/outputs/large_sft10_epoch_sampling_clamp2"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--start-epoch", type=int, default=1)
    parser.add_argument("--samples-per-epoch", type=int, default=4)
    parser.add_argument("--max-generation-attempts", type=int, default=16)
    parser.add_argument("--rolling-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--score-pretrained-baseline",
        action="store_true",
        help=(
            "Before epoch 1 training, sample --pretrained with the same fixed eval prompts/seeds "
            "and write an epoch=0 row_type=pretrained_baseline row with rewards/similarity/KL."
        ),
    )
    parser.add_argument(
        "--save-epoch-checkpoints",
        action="store_true",
        help="Copy the rolling checkpoint to checkpoints/epochXX.pth after each epoch.",
    )
    parser.add_argument(
        "--prefix-shuffle-seed",
        type=int,
        default=None,
        help=(
            "Use a fixed shuffle seed for --prefix-manifest sampling across epochs. "
            "By default the epoch number is used, which changes the prompt order each epoch."
        ),
    )
    parser.add_argument("--delete-rolling-checkpoint-at-end", action="store_true")
    parser.add_argument("--lr", type=str, default="1e-6")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accumulation-steps", type=int, default=1)
    parser.add_argument("--target-stream-lines", type=int, default=32)
    parser.add_argument("--temperature", type=str, default="1.0")
    parser.add_argument("--top-k", type=str, default="8")
    parser.add_argument("--top-p", type=str, default="0.95")
    parser.add_argument("--timeout-s", type=str, default="300")
    parser.add_argument("--max-chars", type=str, default="24000")
    parser.add_argument(
        "--max-generated-patches",
        type=int,
        default=None,
        help="Maximum generated NotaGen patches for batched sampling. Defaults to ceil(max_chars / 16).",
    )
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--sampling-batch-size", type=int, default=1)
    parser.add_argument("--isolated-sampling", action="store_true")
    parser.add_argument("--enable-key-augmentation", action="store_true")
    parser.add_argument("--skip-clamp2", action="store_true")
    parser.add_argument(
        "--exact-pretrained-kl",
        action="store_true",
        help=(
            "After each epoch's sampling pass, replay generated tokens under the epoch checkpoint "
            "and the non-SFT/pretrained checkpoint and log exact full-vocabulary KL."
        ),
    )
    parser.add_argument(
        "--exact-kl-reference-checkpoint",
        type=Path,
        default=None,
        help="Reference checkpoint for --exact-pretrained-kl. Defaults to --pretrained.",
    )
    parser.add_argument(
        "--exact-kl-score-chunk-patches",
        type=int,
        default=64,
        help="Generated patch chunk size for exact KL replay.",
    )
    parser.add_argument(
        "--exact-kl-replay-context-patches",
        type=int,
        default=0,
        help="Patch context cap for exact KL replay. Use 0 for no cap.",
    )
    parser.add_argument(
        "--exact-kl-replay-batch-size",
        type=int,
        default=0,
        help="Trajectory batch size inside exact KL replay chunks. Use 0 for all active trajectories.",
    )
    parser.add_argument(
        "--variation-reference-glob",
        type=str,
        default="data/processed/notagen/goldberg_aria_conditioned_split2/interleaved/variation-*.abc",
    )
    parser.add_argument(
        "--reward-target-json",
        type=Path,
        default=Path("/home/jl_fs/music-generation/infinite-goldberg-variations/data/processed/goldberg/structure/aria_bar_skeleton.json"),
    )
    parser.add_argument(
        "--reward-target-structure-abc",
        type=Path,
        required=True,
        help="Reference NotaGen ABC whose body/stream-line count is used for the bar-count reward.",
    )
    args = parser.parse_args()
    if args.aria_chroma_top_reward_weight is None:
        args.aria_chroma_top_reward_weight = args.aria_chroma_reward_weight
    default_aligned_harmony_weight = 0.25 if args.aria_harmony_reward_weight != 0.0 else 0.0
    if args.aria_harmony_aligned_root_reward_weight is None:
        args.aria_harmony_aligned_root_reward_weight = default_aligned_harmony_weight
    if args.aria_harmony_aligned_bass_reward_weight is None:
        args.aria_harmony_aligned_bass_reward_weight = default_aligned_harmony_weight
    if args.aria_harmony_aligned_top_reward_weight is None:
        args.aria_harmony_aligned_top_reward_weight = 0.0
    exact_kl_reference_checkpoint = args.exact_kl_reference_checkpoint or args.pretrained
    max_generated_patches = args.max_generated_patches
    if max_generated_patches is None:
        max_generated_patches = max(1, (int(args.max_chars) + 15) // 16)
    if max_generated_patches <= 0:
        raise ValueError("--max-generated-patches must be positive")

    if args.sft_eval_variation_count is not None:
        train_jsonl, eval_jsonl, split_manifest = build_sft_variation_split_manifests(
            train_jsonl=args.train_jsonl,
            eval_jsonl=args.eval_jsonl,
            output_dir=args.output_dir,
            eval_variation_count=args.sft_eval_variation_count,
            eval_variation_seed=args.sft_eval_variation_seed,
        )
        args.train_jsonl = train_jsonl
        args.eval_jsonl = eval_jsonl
        print(
            "prepared SFT variation split "
            f"mode={split_manifest['mode']} "
            f"train_rows={split_manifest['train_row_count']} "
            f"eval_rows={split_manifest['eval_row_count']} "
            f"eval_variations={split_manifest['eval_variations']}",
            flush=True,
        )

    similarity_rewards_enabled = (
        args.aria_chroma_reward_weight != 0.0
        or args.aria_chroma_top_reward_weight != 0.0
        or args.aria_harmony_reward_weight != 0.0
        or args.aria_harmony_aligned_root_reward_weight != 0.0
        or args.aria_harmony_aligned_bass_reward_weight != 0.0
        or args.aria_harmony_aligned_top_reward_weight != 0.0
        or args.aria_strict_symbolic_reward_weight != 0.0
    )
    required_paths = [args.notagen_dir, args.project_dir, args.pretrained, args.train_jsonl, args.eval_jsonl, args.reward_target_json]
    required_paths.append(args.reward_target_structure_abc)
    if not args.skip_clamp2 or similarity_rewards_enabled:
        required_paths.append(args.aria_reference)
    if not args.skip_clamp2:
        required_paths.append(args.clamp2_dir)
    if args.train_prefix_mask_root is not None:
        required_paths.append(args.train_prefix_mask_root)
    if args.train_prefix_mask_source_root is not None:
        required_paths.append(args.train_prefix_mask_source_root)
    if args.train_prefix_mask_manifest is not None:
        required_paths.append(args.train_prefix_mask_manifest)
    if args.exact_pretrained_kl:
        required_paths.append(exact_kl_reference_checkpoint)
    if args.prefix_manifest is None:
        required_paths.append(args.prefix)
    else:
        required_paths.append(args.prefix_manifest)
    for required in required_paths:
        if not required.exists():
            raise FileNotFoundError(required)
    prefix_specs = None
    if args.prefix_manifest is not None:
        prefix_specs = [json.loads(line) for line in args.prefix_manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not prefix_specs:
            raise ValueError(f"empty prefix manifest: {args.prefix_manifest}")
    variation_refs: list[Path] = []
    if not args.skip_clamp2:
        variation_refs = sorted(args.project_dir.glob(args.variation_reference_glob))
        if not variation_refs:
            raise FileNotFoundError(f"no variation references matched {args.variation_reference_glob!r} under {args.project_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = args.output_dir / "checkpoints"
    samples_dir = args.output_dir / "samples"
    logs_dir = args.output_dir / "logs"
    scores_dir = args.output_dir / "scores"
    for path in [checkpoints_dir, samples_dir, logs_dir, scores_dir]:
        path.mkdir(parents=True, exist_ok=True)

    summary_path = args.output_dir / "summary.jsonl"
    if args.start_epoch <= 1 and summary_path.exists():
        summary_path.unlink()

    rolling_checkpoint = args.rolling_checkpoint or (checkpoints_dir / "current.pth")
    if args.start_epoch > 1 and not rolling_checkpoint.exists():
        raise FileNotFoundError(f"rolling checkpoint is required when resuming: {rolling_checkpoint}")
    if args.start_epoch <= 1 and rolling_checkpoint.exists():
        rolling_checkpoint.unlink()

    if args.score_pretrained_baseline and args.start_epoch <= 1:
        print("===== pretrained baseline eval =====", flush=True)
        baseline_row = evaluate_checkpoint_samples(
            args=args,
            row_type="pretrained_baseline",
            epoch=0,
            checkpoint=args.pretrained,
            checkpoint_is_rolling=False,
            rolling_checkpoint=None,
            epoch_checkpoint=None,
            losses=None,
            prefix_specs=prefix_specs,
            variation_refs=variation_refs,
            exact_kl_reference_checkpoint=exact_kl_reference_checkpoint,
            max_generated_patches=max_generated_patches,
            samples_dir=samples_dir,
            logs_dir=logs_dir,
            scores_dir=scores_dir,
        )
        write_summary_row(summary_path, baseline_row)
        print_summary_row(baseline_row)

    for epoch in range(args.start_epoch, args.epochs + 1):
        print(f"===== epoch {epoch}/{args.epochs} train =====", flush=True)
        loss_log = logs_dir / f"epoch{epoch:02d}_loss.log"
        train_stdout = logs_dir / f"epoch{epoch:02d}_train_stdout.log"

        env = os.environ.copy()
        env.update(
            {
                "PYTHONPATH": str(args.notagen_dir / "finetune" / "finetune"),
                "NOTAGEN_DATA_TRAIN_INDEX_PATH": str(args.train_jsonl),
                "NOTAGEN_DATA_EVAL_INDEX_PATH": str(args.eval_jsonl),
                "NOTAGEN_PATCH_LENGTH": "1024",
                "NOTAGEN_PATCH_NUM_LAYERS": "20",
                "NOTAGEN_CHAR_NUM_LAYERS": "6",
                "NOTAGEN_HIDDEN_SIZE": "1280",
                "NOTAGEN_BATCH_SIZE": str(args.batch_size),
                "NOTAGEN_LEARNING_RATE": args.lr,
                "NOTAGEN_NUM_EPOCHS": str(epoch),
                "NOTAGEN_ACCUMULATION_STEPS": str(args.grad_accumulation_steps),
                "NOTAGEN_DISABLE_KEY_AUGMENTATION": "false" if args.enable_key_augmentation else "true",
                "NOTAGEN_PRETRAINED_PATH": str(args.pretrained),
                "NOTAGEN_WEIGHTS_PATH": str(rolling_checkpoint),
                "NOTAGEN_LOGS_PATH": str(loss_log),
                "NOTAGEN_EXP_TAG": f"goldberg_large_sft_epoch_sampling_epoch{epoch:02d}",
                "NOTAGEN_SAVE_LAST_EPOCH": "true",
                "NOTAGEN_LOAD_FROM_CHECKPOINT": "true" if epoch > 1 else "false",
            }
        )
        if args.train_prefix_mask_root is not None:
            env["NOTAGEN_PREFIX_MASK_ROOT"] = str(args.train_prefix_mask_root)
        if args.train_prefix_mask_source_root is not None:
            env["NOTAGEN_PREFIX_MASK_SOURCE_ROOT"] = str(args.train_prefix_mask_source_root)
        if args.train_prefix_mask_manifest is not None:
            env["NOTAGEN_PREFIX_MASK_MANIFEST"] = str(args.train_prefix_mask_manifest)

        try:
            run([str(args.venv_python), "train-gen.py"], cwd=args.notagen_dir / "finetune", env=env, log=train_stdout)
        except subprocess.CalledProcessError:
            # Some remote runs return a nonzero exit even though the epoch completed
            # and wrote both the loss log and rolling checkpoint. Recover in that case.
            losses = parse_loss_log(loss_log)
            if "train_loss" not in losses or "eval_loss" not in losses or not rolling_checkpoint.exists():
                raise
            print(
                f"train-gen.py exited nonzero after completing epoch {epoch}; "
                "continuing with recorded losses and checkpoint",
                flush=True,
            )
        losses = parse_loss_log(loss_log)
        epoch_checkpoint: Path | None = None
        if args.save_epoch_checkpoints:
            if not rolling_checkpoint.exists():
                raise FileNotFoundError(f"missing rolling checkpoint after epoch {epoch}: {rolling_checkpoint}")
            epoch_checkpoint = checkpoints_dir / f"epoch{epoch:02d}.pth"
            shutil.copy2(rolling_checkpoint, epoch_checkpoint)
        checkpoint_for_epoch = epoch_checkpoint or rolling_checkpoint

        if args.samples_per_epoch == 0:
            row = {
                "epoch": epoch,
                "row_type": "sft_epoch",
                "checkpoint": str(checkpoint_for_epoch),
                "checkpoint_is_rolling": epoch_checkpoint is None,
                "rolling_checkpoint": str(rolling_checkpoint),
                "epoch_checkpoint": None if epoch_checkpoint is None else str(epoch_checkpoint),
                "losses": losses,
                "generation_failures": [],
                "samples": [],
                "mean_clamp2_aria_similarity": None,
                "mean_clamp2_variation_centroid_similarity": None,
                "mean_reward": None,
                "mean_structural_total_reward": None,
                "mean_active_similarity_reward": None,
                "mean_effective_similarity_reward": None,
                "mean_aria_chroma_harmonic_hist": None,
                "mean_aria_chroma_top_hist": None,
                "mean_aria_harmony_dtw_combined": None,
                "exact_pretrained_kl": None,
                "mean_exact_kl_to_pretrained": None,
                "prefix_shuffle_seed": args.prefix_shuffle_seed,
            }
            write_summary_row(summary_path, row)
            print(
                f"epoch={epoch} train_loss={losses.get('train_loss')} "
                f"eval_loss={losses.get('eval_loss')} samples=0",
                flush=True,
            )
            continue

        row = evaluate_checkpoint_samples(
            args=args,
            row_type="sft_epoch",
            epoch=epoch,
            checkpoint=checkpoint_for_epoch,
            checkpoint_is_rolling=epoch_checkpoint is None,
            rolling_checkpoint=rolling_checkpoint,
            epoch_checkpoint=epoch_checkpoint,
            losses=losses,
            prefix_specs=prefix_specs,
            variation_refs=variation_refs,
            exact_kl_reference_checkpoint=exact_kl_reference_checkpoint,
            max_generated_patches=max_generated_patches,
            samples_dir=samples_dir,
            logs_dir=logs_dir,
            scores_dir=scores_dir,
        )
        write_summary_row(summary_path, row)
        print_summary_row(row)

    if args.delete_rolling_checkpoint_at_end:
        rolling_checkpoint.unlink(missing_ok=True)
    print(f"summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
