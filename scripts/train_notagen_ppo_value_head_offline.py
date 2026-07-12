from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from scripts.custom_grpo_notagen import (
    PATCH_STREAM,
    build_model,
    build_rollout_prefix,
    disable_dropout_modules,
    infer_model_shape,
    load_prompt_rows,
    prompt_row_name,
    select_device,
    set_seed,
)
from scripts.custom_ppo_notagen import (
    PatchValueHead,
    discounted_returns,
    load_value_head_checkpoint,
    save_value_head_checkpoint,
    trajectory_patch_hidden_states,
    value_mse_loss,
    value_prediction_metrics,
)
from utils import Patchilizer


@dataclass(frozen=True)
class OfflineRolloutSample:
    source_json: str
    step: int
    prompt_index: int
    prompt_name: str
    trajectory_index: int
    rollout_seed: int | None
    prompt_flat_ids: list[int]
    generated_patches: list[list[int]]
    patch_rewards: list[float]
    reward: float | None


@dataclass
class PreparedValueSample:
    hidden_states: torch.Tensor
    targets: torch.Tensor
    meta: dict


def _select_device(name: str) -> torch.device:
    if name == "auto":
        return select_device()
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available")
    if device.type == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError("--device mps requested but MPS is not available")
    return device


def _json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _load_rollout_samples(
    *,
    ppo_json_paths: list[Path],
    prompts_jsonl: Path,
    target_stream_lines: int,
) -> list[OfflineRolloutSample]:
    prompts = load_prompt_rows(prompts_jsonl, limit=None)
    if not prompts:
        raise ValueError(f"no prompts loaded from {prompts_jsonl}")

    patchilizer = Patchilizer(stream=PATCH_STREAM)
    prompt_flat_cache: dict[int, list[int]] = {}
    samples: list[OfflineRolloutSample] = []
    missing_patch_records: list[str] = []

    for json_path in ppo_json_paths:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        for step in payload.get("steps", []):
            step_idx = int(step["step"])
            prompt_idx = int(step.get("prompt_index", 0))
            if prompt_idx < 0 or prompt_idx >= len(prompts):
                raise IndexError(
                    f"{json_path}: step {step_idx} references prompt_index={prompt_idx}, "
                    f"but {prompts_jsonl} has {len(prompts)} rows"
                )
            if prompt_idx not in prompt_flat_cache:
                prompt = prompts[prompt_idx]["prompt"]
                rollout_prefix = build_rollout_prefix(prompt, target_stream_lines)
                prompt_flat_cache[prompt_idx] = [
                    token for patch in patchilizer.encode_generate(rollout_prefix) for token in patch
                ]
            prompt_name = str(step.get("prompt_name") or prompt_row_name(prompts[prompt_idx], prompt_idx))

            for trajectory in step.get("trajectories", []):
                trajectory_index = int(trajectory["trajectory_index"])
                generated_patches = trajectory.get("generated_patches")
                if generated_patches is None:
                    missing_patch_records.append(f"{json_path}:step={step_idx}:trajectory={trajectory_index}")
                    continue
                patch_rewards = [float(item) for item in trajectory.get("patch_rewards", [])]
                if len(generated_patches) != len(patch_rewards):
                    raise RuntimeError(
                        f"{json_path}: step {step_idx} trajectory {trajectory_index} has "
                        f"{len(generated_patches)} generated patches but {len(patch_rewards)} patch rewards"
                    )
                samples.append(
                    OfflineRolloutSample(
                        source_json=str(json_path),
                        step=step_idx,
                        prompt_index=prompt_idx,
                        prompt_name=prompt_name,
                        trajectory_index=trajectory_index,
                        rollout_seed=trajectory.get("rollout_seed"),
                        prompt_flat_ids=prompt_flat_cache[prompt_idx],
                        generated_patches=[[int(token) for token in patch] for patch in generated_patches],
                        patch_rewards=patch_rewards,
                        reward=None if trajectory.get("reward") is None else float(trajectory["reward"]),
                    )
                )

    if missing_patch_records:
        preview = ", ".join(missing_patch_records[:5])
        suffix = "" if len(missing_patch_records) <= 5 else f", ... +{len(missing_patch_records) - 5} more"
        raise RuntimeError(
            "PPO rollout JSON is missing trajectories[*].generated_patches, so the frozen policy "
            f"hidden states cannot be reconstructed offline. Rerun PPO with the updated logger. Missing: {preview}{suffix}"
        )
    if not samples:
        raise ValueError("no trajectory samples found in PPO JSON inputs")
    return samples


def _prepare_hidden_state_samples(
    *,
    policy_model,
    rollout_samples: list[OfflineRolloutSample],
    precision: str,
    replay_context_patches: int | None,
    score_chunk_patches: int,
    gamma: float,
    device: torch.device,
) -> list[PreparedValueSample]:
    prepared: list[PreparedValueSample] = []
    start = time.perf_counter()
    for idx, sample in enumerate(rollout_samples, start=1):
        hidden_states = trajectory_patch_hidden_states(
            policy_model,
            sample.prompt_flat_ids,
            sample.generated_patches,
            precision,
            replay_context_patches=replay_context_patches,
            target_chunk_patches=score_chunk_patches,
            detach_policy=True,
        )
        rewards = torch.tensor(sample.patch_rewards, device=device, dtype=torch.float32)
        targets = discounted_returns(rewards, gamma=gamma).detach()
        if hidden_states.shape[0] != targets.shape[0]:
            raise RuntimeError(
                f"value target mismatch for step={sample.step} trajectory={sample.trajectory_index}: "
                f"hidden_states={tuple(hidden_states.shape)} targets={tuple(targets.shape)}"
            )
        prepared.append(
            PreparedValueSample(
                hidden_states=hidden_states.detach().float().cpu(),
                targets=targets.detach().float().cpu(),
                meta={
                    "source_json": sample.source_json,
                    "step": sample.step,
                    "prompt_index": sample.prompt_index,
                    "prompt_name": sample.prompt_name,
                    "trajectory_index": sample.trajectory_index,
                    "rollout_seed": sample.rollout_seed,
                    "reward": sample.reward,
                    "patch_count": int(targets.numel()),
                },
            )
        )
        print(
            json.dumps(
                {
                    "event": "prepared_value_sample",
                    "index": idx,
                    "total": len(rollout_samples),
                    **prepared[-1].meta,
                    "elapsed_s": time.perf_counter() - start,
                }
            ),
            flush=True,
        )
    return prepared


def _load_hidden_cache(path: Path) -> tuple[list[PreparedValueSample], dict]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise RuntimeError(f"unsupported hidden cache payload type: {type(payload)!r}")
    hidden_states = payload.get("hidden_states")
    targets = payload.get("targets")
    metadata = payload.get("metadata")
    if not isinstance(hidden_states, list) or not isinstance(targets, list) or not isinstance(metadata, list):
        raise RuntimeError(f"hidden cache {path} is missing hidden_states/targets/metadata lists")
    if not (len(hidden_states) == len(targets) == len(metadata)):
        raise RuntimeError(
            f"hidden cache {path} length mismatch: hidden={len(hidden_states)} "
            f"targets={len(targets)} metadata={len(metadata)}"
        )
    samples = [
        PreparedValueSample(
            hidden_states=hidden.float().cpu(),
            targets=target.float().cpu(),
            meta=dict(meta),
        )
        for hidden, target, meta in zip(hidden_states, targets, metadata, strict=True)
    ]
    return samples, dict(payload.get("config", {}))


def _save_hidden_cache(path: Path, samples: list[PreparedValueSample], config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": config,
            "metadata": [sample.meta for sample in samples],
            "hidden_states": [sample.hidden_states.cpu() for sample in samples],
            "targets": [sample.targets.cpu() for sample in samples],
        },
        path,
    )


def _split_samples(
    samples: list[PreparedValueSample],
    *,
    holdout_last_step: bool,
    eval_fraction: float,
    seed: int,
) -> tuple[list[PreparedValueSample], list[PreparedValueSample], dict]:
    if holdout_last_step:
        last_step = max(int(sample.meta["step"]) for sample in samples)
        train = [sample for sample in samples if int(sample.meta["step"]) != last_step]
        eval_samples = [sample for sample in samples if int(sample.meta["step"]) == last_step]
        if train and eval_samples:
            return train, eval_samples, {"mode": "holdout_last_step", "last_step": last_step}

    if eval_fraction > 0.0 and len(samples) > 1:
        rng = random.Random(seed)
        shuffled = samples[:]
        rng.shuffle(shuffled)
        eval_count = max(1, min(len(shuffled) - 1, round(len(shuffled) * eval_fraction)))
        return shuffled[eval_count:], shuffled[:eval_count], {
            "mode": "random_fraction",
            "eval_fraction": eval_fraction,
            "eval_count": eval_count,
        }

    return samples[:], [], {"mode": "train_all"}


def _iter_sample_batches(samples: list[PreparedValueSample], batch_size: int):
    if batch_size <= 0:
        batch_size = len(samples)
    for start in range(0, len(samples), batch_size):
        yield samples[start : start + batch_size]


def _cat_batch(batch: list[PreparedValueSample], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    hidden_states = torch.cat([sample.hidden_states for sample in batch], dim=0).to(device)
    targets = torch.cat([sample.targets for sample in batch], dim=0).to(device)
    return hidden_states, targets


def _evaluate_value_head(
    value_head: PatchValueHead,
    samples: list[PreparedValueSample],
    *,
    device: torch.device,
    trajectory_batch_size: int,
) -> dict | None:
    if not samples:
        return None
    value_head.eval()
    values_by_batch: list[torch.Tensor] = []
    targets_by_batch: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in _iter_sample_batches(samples, trajectory_batch_size):
            hidden_states, targets = _cat_batch(batch, device)
            values_by_batch.append(value_head(hidden_states).detach().float().cpu())
            targets_by_batch.append(targets.detach().float().cpu())
    return value_prediction_metrics(torch.cat(values_by_batch), torch.cat(targets_by_batch))


def _train_value_head(
    value_head: PatchValueHead,
    train_samples: list[PreparedValueSample],
    eval_samples: list[PreparedValueSample],
    args,
    device: torch.device,
) -> dict:
    optimizer = torch.optim.AdamW(value_head.parameters(), lr=args.value_learning_rate, weight_decay=args.weight_decay)
    logs: list[dict] = []
    initial_train_metrics = _evaluate_value_head(
        value_head,
        train_samples,
        device=device,
        trajectory_batch_size=args.trajectory_batch_size,
    )
    initial_eval_metrics = _evaluate_value_head(
        value_head,
        eval_samples,
        device=device,
        trajectory_batch_size=args.trajectory_batch_size,
    )
    print(
        json.dumps(
            {
                "event": "offline_value_initial",
                "train_metrics": initial_train_metrics,
                "eval_metrics": initial_eval_metrics,
            }
        ),
        flush=True,
    )

    rng = random.Random(args.seed)
    start = time.perf_counter()
    for epoch_idx in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        shuffled = train_samples[:]
        rng.shuffle(shuffled)
        value_head.train()
        batch_logs: list[dict] = []
        for batch_idx, batch in enumerate(_iter_sample_batches(shuffled, args.trajectory_batch_size), start=1):
            hidden_states, targets = _cat_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            values = value_head(hidden_states)
            loss, raw_loss, scale = value_mse_loss(
                values,
                targets,
                normalize_value_loss=args.normalize_value_loss,
                eps=args.value_loss_eps,
                scale_min=args.value_loss_scale_min,
            )
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(value_head.parameters(), args.max_grad_norm)
            optimizer.step()
            batch_logs.append(
                {
                    "batch": batch_idx,
                    "patches": int(targets.numel()),
                    "loss": float(loss.detach().cpu()),
                    "raw_value_loss": float(raw_loss.detach().cpu()),
                    "value_loss_scale": float(scale.detach().cpu()),
                    "grad_norm": float(grad_norm.detach().cpu() if torch.is_tensor(grad_norm) else grad_norm),
                }
            )

        train_metrics = _evaluate_value_head(
            value_head,
            train_samples,
            device=device,
            trajectory_batch_size=args.trajectory_batch_size,
        )
        eval_metrics = _evaluate_value_head(
            value_head,
            eval_samples,
            device=device,
            trajectory_batch_size=args.trajectory_batch_size,
        )
        epoch_log = {
            "epoch": epoch_idx,
            "duration_s": time.perf_counter() - epoch_start,
            "train_metrics": train_metrics,
            "eval_metrics": eval_metrics,
            "batch_logs": batch_logs,
        }
        logs.append(epoch_log)
        print(json.dumps({"event": "offline_value_epoch", **epoch_log}), flush=True)

    return {
        "initial_train_metrics": initial_train_metrics,
        "initial_eval_metrics": initial_eval_metrics,
        "epochs": logs,
        "duration_s": time.perf_counter() - start,
        "final_train_metrics": logs[-1]["train_metrics"] if logs else initial_train_metrics,
        "final_eval_metrics": logs[-1]["eval_metrics"] if logs else initial_eval_metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train only the NotaGen PPO patch value head from saved PPO rollout JSONs."
    )
    parser.add_argument("--ppo-json", nargs="+", type=Path, help="PPO output JSON files with trajectories.")
    parser.add_argument("--prompts-jsonl", type=Path, default=Path("data/processed/notagen/goldberg_ppo_prompts_e3_header_allvoices.jsonl"))
    parser.add_argument("--policy-weights", type=Path, help="Frozen SFT policy checkpoint used to replay patch states.")
    parser.add_argument("--hidden-cache-in", type=Path, help="Precomputed hidden-state cache from this script.")
    parser.add_argument("--hidden-cache-out", type=Path, help="Write precomputed hidden states for later critic-only runs.")
    parser.add_argument("--output-value-head", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--input-value-head", type=Path)
    parser.add_argument("--target-stream-lines", type=int, default=32)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--trajectory-batch-size", type=int, default=4)
    parser.add_argument("--value-learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--value-head-hidden-size", type=int, default=512)
    parser.add_argument("--value-head-dropout", type=float, default=0.0)
    parser.add_argument("--normalize-value-loss", action="store_true")
    parser.add_argument("--value-loss-eps", type=float, default=1e-6)
    parser.add_argument("--value-loss-scale-min", type=float, default=1e-6)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--replay-context-patches", type=int, default=128)
    parser.add_argument("--score-chunk-patches", type=int, default=16)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--holdout-last-step", action="store_true")
    parser.add_argument("--eval-fraction", type=float, default=0.0)
    args = parser.parse_args()

    if args.epochs < 0:
        raise ValueError(f"--epochs must be non-negative, got {args.epochs}")
    if args.trajectory_batch_size <= 0:
        raise ValueError(f"--trajectory-batch-size must be positive, got {args.trajectory_batch_size}")
    if not 0.0 <= args.eval_fraction < 1.0:
        raise ValueError(f"--eval-fraction must be in [0, 1), got {args.eval_fraction}")
    if args.value_loss_eps <= 0.0:
        raise ValueError(f"--value-loss-eps must be positive, got {args.value_loss_eps}")
    if args.value_loss_scale_min <= 0.0:
        raise ValueError(f"--value-loss-scale-min must be positive, got {args.value_loss_scale_min}")

    set_seed(args.seed)
    device = _select_device(args.device)

    cache_config: dict = {}
    if args.hidden_cache_in:
        prepared_samples, cache_config = _load_hidden_cache(args.hidden_cache_in)
    else:
        if not args.policy_weights:
            raise ValueError("--policy-weights is required unless --hidden-cache-in is provided")
        if not args.ppo_json:
            raise ValueError("--ppo-json is required unless --hidden-cache-in is provided")
        if not args.policy_weights.exists():
            raise FileNotFoundError(args.policy_weights)
        rollout_samples = _load_rollout_samples(
            ppo_json_paths=args.ppo_json,
            prompts_jsonl=args.prompts_jsonl,
            target_stream_lines=args.target_stream_lines,
        )
        policy_shape = infer_model_shape(args.policy_weights)
        policy_model = build_model(
            args.policy_weights,
            device,
            precision=args.precision,
            freeze_before_precision_cast=True,
        )
        policy_model.eval()
        for param in policy_model.parameters():
            param.requires_grad_(False)
        disabled_dropout_modules = disable_dropout_modules(policy_model)
        prepared_samples = _prepare_hidden_state_samples(
            policy_model=policy_model,
            rollout_samples=rollout_samples,
            precision=args.precision,
            replay_context_patches=args.replay_context_patches,
            score_chunk_patches=args.score_chunk_patches,
            gamma=args.gamma,
            device=device,
        )
        cache_config = {
            "policy_weights": str(args.policy_weights),
            "policy_shape": asdict(policy_shape),
            "prompts_jsonl": str(args.prompts_jsonl),
            "target_stream_lines": args.target_stream_lines,
            "gamma": args.gamma,
            "precision": args.precision,
            "replay_context_patches": args.replay_context_patches,
            "score_chunk_patches": args.score_chunk_patches,
            "policy_dropout_modules_disabled": disabled_dropout_modules,
        }
        del policy_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not prepared_samples:
        raise ValueError("no prepared value samples available")
    hidden_size = int(prepared_samples[0].hidden_states.shape[1])
    for sample in prepared_samples:
        if sample.hidden_states.ndim != 2 or int(sample.hidden_states.shape[1]) != hidden_size:
            raise RuntimeError("all prepared samples must have the same 2D hidden-state width")
        if sample.hidden_states.shape[0] != sample.targets.shape[0]:
            raise RuntimeError(f"prepared sample target mismatch: {sample.meta}")

    if args.hidden_cache_out:
        _save_hidden_cache(args.hidden_cache_out, prepared_samples, cache_config)

    value_head = PatchValueHead(
        hidden_size,
        value_hidden_size=args.value_head_hidden_size,
        dropout=args.value_head_dropout,
    ).to(device)
    loaded_value_head = None
    if args.input_value_head:
        loaded_value_head = load_value_head_checkpoint(value_head, args.input_value_head, device)

    train_samples, eval_samples, split_meta = _split_samples(
        prepared_samples,
        holdout_last_step=args.holdout_last_step,
        eval_fraction=args.eval_fraction,
        seed=args.seed,
    )
    if not train_samples:
        raise ValueError("offline value training split produced no training samples")

    train_log = _train_value_head(value_head, train_samples, eval_samples, args, device)
    save_value_head_checkpoint(value_head, args.output_value_head)

    patch_counts = [int(sample.targets.numel()) for sample in prepared_samples]
    payload = {
        "run_config": {
            "args": _json_safe(vars(args)),
            "device": str(device),
            "cache_config": cache_config,
            "loaded_value_head": loaded_value_head,
            "value_head": value_head.config(),
            "split": split_meta,
        },
        "dataset": {
            "trajectories": len(prepared_samples),
            "train_trajectories": len(train_samples),
            "eval_trajectories": len(eval_samples),
            "patches": int(sum(patch_counts)),
            "patch_count_mean": float(np.mean(patch_counts)),
            "patch_count_min": int(min(patch_counts)),
            "patch_count_max": int(max(patch_counts)),
            "samples": [sample.meta for sample in prepared_samples],
        },
        "training": train_log,
        "saved_value_head": str(args.output_value_head),
        "saved_hidden_cache": None if args.hidden_cache_out is None else str(args.hidden_cache_out),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"event": "offline_value_complete", **payload["dataset"], "output_json": str(args.output_json)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
