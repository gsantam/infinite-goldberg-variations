#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
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


def load_cached_sampler(project_dir: Path):
    if str(project_dir) not in sys.path:
        sys.path.insert(0, str(project_dir))
    from scripts.generate_notagen_cached_inference import sample_completion_cached  # type: ignore
    from grpo.notagen_wrapper import build_model, count_stream_lines, set_seed  # type: ignore

    return build_model, count_stream_lines, sample_completion_cached, set_seed


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
    precision: str,
    epoch: int,
) -> tuple[list[Path], list[dict], list[dict]]:
    build_model, count_stream_lines, sample_completion_cached, set_seed = load_cached_sampler(project_dir)
    model, model_shape = build_model(weights, precision=precision)
    candidates: list[Path] = []
    generation_failures: list[dict] = []
    sample_metadata: list[dict] = []
    rng = random.Random(epoch)
    shuffled_prefix_specs = prefix_specs[:] if prefix_specs else []
    rng.shuffle(shuffled_prefix_specs)

    for attempt_seed in range(max_generation_attempts):
        sample_idx = len(candidates)
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
        prompt = prefix_path.read_text(encoding="utf-8")
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
        if len(candidates) >= samples_per_epoch:
            break

    del model
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    return candidates, generation_failures, sample_metadata


def score_rewards(
    *,
    project_dir: Path,
    prefix: Path | None,
    candidate_prefixes: dict[str, Path],
    reward_target_json: Path,
    candidates: list[Path],
    out_path: Path,
) -> list[dict]:
    if str(project_dir) not in sys.path:
        sys.path.insert(0, str(project_dir))
    from grpo.rewards import GoldbergRewardConfig, load_structural_target, score_prompt_completion_pair  # type: ignore

    target = load_structural_target(reward_target_json)
    config = GoldbergRewardConfig()
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
        breakdown["path"] = str(candidate)
        breakdown["completion_chars"] = len(completion)
        rows.append(breakdown)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--notagen-dir", type=Path, default=Path("/home/jl_fs/music-generation/NotaGen"))
    parser.add_argument("--project-dir", type=Path, default=Path("/home/jl_fs/music-generation/infinite-goldberg-variations"))
    parser.add_argument("--venv-python", type=Path, default=Path("/home/jl_fs/music-generation/.venvs/notagen-trl/bin/python"))
    parser.add_argument("--pretrained", type=Path, default=Path("/home/jl_fs/music-generation/models/weights_notagen_pretrain-finetune_p_size_16_p_length_1024_p_layers_c_layers_6_20_h_size_1280_lr_1e-05_batch_1.pth"))
    parser.add_argument("--train-jsonl", type=Path, default=Path("/home/jl_fs/music-generation/infinite-goldberg-variations/data/processed/notagen/goldberg_aria_conditioned/augmented_train.jsonl"))
    parser.add_argument("--eval-jsonl", type=Path, default=Path("/home/jl_fs/music-generation/infinite-goldberg-variations/data/processed/notagen/goldberg_aria_conditioned/augmented_eval.jsonl"))
    parser.add_argument("--train-prefix-mask-root", type=Path, default=None)
    parser.add_argument("--train-prefix-mask-source-root", type=Path, default=None)
    parser.add_argument("--prefix", type=Path, default=Path("/home/jl_fs/music-generation/infinite-goldberg-variations/data/processed/notagen/aria_plus_variation01_setup_G.abc"))
    parser.add_argument("--prefix-manifest", type=Path, default=None)
    parser.add_argument("--aria-reference", type=Path, default=Path("/home/jl_fs/music-generation/infinite-goldberg-variations/data/processed/notagen/aria_prefix_G_streamed.abc"))
    parser.add_argument("--clamp2-dir", type=Path, default=Path("/home/jl_fs/music-generation/NotaGen/clamp2"))
    parser.add_argument("--output-dir", type=Path, default=Path("/home/jl_fs/music-generation/outputs/large_sft10_epoch_sampling_clamp2"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--start-epoch", type=int, default=1)
    parser.add_argument("--samples-per-epoch", type=int, default=4)
    parser.add_argument("--max-generation-attempts", type=int, default=16)
    parser.add_argument("--rolling-checkpoint", type=Path, default=None)
    parser.add_argument("--delete-rolling-checkpoint-at-end", action="store_true")
    parser.add_argument("--lr", type=str, default="1e-6")
    parser.add_argument("--target-stream-lines", type=int, default=32)
    parser.add_argument("--temperature", type=str, default="1.0")
    parser.add_argument("--top-k", type=str, default="8")
    parser.add_argument("--top-p", type=str, default="0.95")
    parser.add_argument("--timeout-s", type=str, default="300")
    parser.add_argument("--max-chars", type=str, default="24000")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
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
    args = parser.parse_args()

    required_paths = [args.notagen_dir, args.project_dir, args.pretrained, args.train_jsonl, args.eval_jsonl, args.aria_reference, args.clamp2_dir, args.reward_target_json]
    if args.train_prefix_mask_root is not None:
        required_paths.append(args.train_prefix_mask_root)
    if args.train_prefix_mask_source_root is not None:
        required_paths.append(args.train_prefix_mask_source_root)
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
                "NOTAGEN_BATCH_SIZE": "1",
                "NOTAGEN_LEARNING_RATE": args.lr,
                "NOTAGEN_NUM_EPOCHS": str(epoch),
                "NOTAGEN_ACCUMULATION_STEPS": "1",
                "NOTAGEN_DISABLE_KEY_AUGMENTATION": "true",
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

        run([str(args.venv_python), "train-gen.py"], cwd=args.notagen_dir / "finetune", env=env, log=train_stdout)
        losses = parse_loss_log(loss_log)

        print(f"===== epoch {epoch}/{args.epochs} sample =====", flush=True)
        epoch_samples_dir = samples_dir / f"epoch{epoch:02d}"
        epoch_samples_dir.mkdir(parents=True, exist_ok=True)
        candidates, generation_failures, sample_metadata = sample_cached_trajectories(
            project_dir=args.project_dir,
            weights=rolling_checkpoint,
            prefix=args.prefix if args.prefix_manifest is None else None,
            prefix_specs=prefix_specs,
            out_dir=epoch_samples_dir,
            samples_per_epoch=args.samples_per_epoch,
            max_generation_attempts=args.max_generation_attempts,
            target_stream_lines=args.target_stream_lines,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            timeout_s=args.timeout_s,
            max_chars=args.max_chars,
            precision=args.precision,
            epoch=epoch,
        )
        if len(candidates) < args.samples_per_epoch:
            raise RuntimeError(
                f"epoch {epoch} produced {len(candidates)} successful samples "
                f"after {args.max_generation_attempts} attempts"
            )

        print(f"===== epoch {epoch}/{args.epochs} score =====", flush=True)
        score_json = scores_dir / f"epoch{epoch:02d}_clamp2_aria_similarity.json"
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
            log=logs_dir / f"epoch{epoch:02d}_clamp2.log",
        )

        variation_score_json = scores_dir / f"epoch{epoch:02d}_clamp2_variation_centroid_similarity.json"
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
            log=logs_dir / f"epoch{epoch:02d}_clamp2_variation_centroid.log",
        )

        rewards_jsonl = scores_dir / f"epoch{epoch:02d}_rewards.jsonl"
        candidate_prefixes = {row["path"]: Path(row["prefix_path"]) for row in sample_metadata if "prefix_path" in row}
        reward_rows = score_rewards(
            project_dir=args.project_dir,
            prefix=args.prefix if args.prefix_manifest is None else None,
            candidate_prefixes=candidate_prefixes,
            reward_target_json=args.reward_target_json,
            candidates=candidates,
            out_path=rewards_jsonl,
        )

        scores = json.loads(score_json.read_text(encoding="utf-8"))
        variation_scores = json.loads(variation_score_json.read_text(encoding="utf-8"))
        aria_by_path = {row["path"]: row for row in scores["rows"]}
        variation_by_path = {row["path"]: row for row in variation_scores["rows"]}
        metadata_by_path = {row["path"]: row for row in sample_metadata}
        reward_by_path = {row["path"]: row for row in reward_rows}
        sample_rows = []
        for candidate in candidates:
            path_key = str(candidate)
            sample_rows.append(
                {
                    **metadata_by_path[path_key],
                    "clamp2_aria": aria_by_path[path_key],
                    "clamp2_variation_centroid": variation_by_path[path_key],
                    "reward_breakdown": reward_by_path[path_key],
                }
            )
        row = {
            "epoch": epoch,
            "checkpoint": str(rolling_checkpoint),
            "checkpoint_is_rolling": True,
            "losses": losses,
            "generation_failures": generation_failures,
            "samples": sample_rows,
            "mean_clamp2_aria_similarity": mean_metric(scores["rows"], "cosine_similarity_to_reference"),
            "mean_clamp2_variation_centroid_similarity": mean_metric(variation_scores["rows"], "cosine_similarity_to_reference"),
            "mean_reward": mean_metric(reward_rows, "total_reward"),
        }
        with summary_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        print(
            f"epoch={epoch} train_loss={losses.get('train_loss')} eval_loss={losses.get('eval_loss')} "
            f"mean_clamp2_aria_similarity={row['mean_clamp2_aria_similarity']:.6f} "
            f"mean_clamp2_variation_centroid_similarity={row['mean_clamp2_variation_centroid_similarity']:.6f} "
            f"mean_reward={row['mean_reward']:.6f}",
            flush=True,
        )

    if args.delete_rolling_checkpoint_at_end:
        rolling_checkpoint.unlink(missing_ok=True)
    print(f"summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
