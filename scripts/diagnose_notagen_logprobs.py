from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.custom_grpo_notagen import (  # noqa: E402
    PATCH_SIZE,
    autocast_context,
    build_model,
    build_rollout_prefix,
    char_patch_logprobs,
    disable_dropout_modules,
    grpo_kl_term,
    normalize_patch_for_context,
    patch_logprobs,
    select_device,
    trajectory_logprob_chunks,
)
from scripts.custom_grpo_notagen import _replay_start_patch  # noqa: E402
from utils import Patchilizer  # noqa: E402


def resolve_manifest_path(path_text: str | None, manifest_dir: Path) -> Path | None:
    if not path_text:
        return None
    path = Path(path_text)
    if path.exists():
        return path
    local_path = manifest_dir / path.name
    if local_path.exists():
        return local_path
    return path


def load_record_from_jsonl(path: Path, sample_index: int) -> dict:
    manifest_dir = path.parent
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if int(record.get("sample_index", -1)) != sample_index:
                continue

            prompt_path = resolve_manifest_path(record.get("prompt_abc_path"), manifest_dir)
            full_path = resolve_manifest_path(record.get("full_abc_path"), manifest_dir)
            completion_path = resolve_manifest_path(record.get("completion_abc_path"), manifest_dir)
            prompt = prompt_path.read_text(encoding="utf-8") if prompt_path and prompt_path.exists() else None
            full_text = full_path.read_text(encoding="utf-8") if full_path and full_path.exists() else None
            completion = completion_path.read_text(encoding="utf-8") if completion_path and completion_path.exists() else None

            if prompt is None and full_text is not None and completion is not None and full_text.endswith(completion):
                prompt = full_text[: -len(completion)] if completion else full_text
            record["prompt_text"] = prompt
            record["full_text"] = full_text
            record["completion_text"] = completion
            return record
    raise ValueError(f"sample_index={sample_index} not found in {path}")


def load_record_from_run_json(path: Path, step_index: int, sample_index: int) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    step = data["steps"][step_index - 1]
    for record in step.get("trajectories", []):
        if int(record.get("sample_index", -1)) == sample_index:
            record = dict(record)
            record["prompt_text"] = record.get("prompt")
            record["full_text"] = record.get("full_text")
            record["completion_text"] = record.get("completion")
            return record
    raise ValueError(f"step={step_index} sample_index={sample_index} not found in {path}")


def load_trajectory(args) -> dict:
    if args.trajectory_jsonl:
        record = load_record_from_jsonl(Path(args.trajectory_jsonl), args.sample_index)
    elif args.run_json:
        record = load_record_from_run_json(Path(args.run_json), args.step_index, args.sample_index)
    else:
        raise ValueError("provide --trajectory-jsonl or --run-json")

    generated_patches = record.get("generated_patches")
    if not isinstance(generated_patches, list) or not generated_patches or not isinstance(generated_patches[0], list):
        raise ValueError(
            "trajectory does not contain generated patch IDs. "
            "Use a run produced with --trajectories-dir, or a run JSON whose trajectories store generated_patches."
        )
    if args.max_generated_patches > 0:
        generated_patches = generated_patches[: args.max_generated_patches]
    record["generated_patches"] = [[int(tok) for tok in patch] for patch in generated_patches]

    if not record.get("prompt_text"):
        raise ValueError("could not recover prompt text for trajectory")
    return record


def tail_logprobs_chunk_no_patch_mask(
    model,
    current_ids: list[int],
    remaining_patches: list[list[int]],
    chunk_start: int,
    chunk_end: int,
    precision: str,
    replay_context_patches: int | None = None,
) -> torch.Tensor:
    normalized_prefix = [
        normalize_patch_for_context(
            patch,
            eos_token_id=model.eos_token_id,
            special_token_id=model.special_token_id,
        )
        for patch in remaining_patches[:chunk_end]
    ]
    all_ids = current_ids[:]
    for patch in normalized_prefix:
        all_ids.extend(patch)

    if len(all_ids) % PATCH_SIZE != 0:
        raise RuntimeError("trajectory replay expected full-patch alignment before chunked tail scoring")

    total_patches = len(all_ids) // PATCH_SIZE
    context_patch_count = len(current_ids) // PATCH_SIZE
    start_patch = _replay_start_patch(total_patches, context_patch_count, replay_context_patches)
    trimmed_ids = all_ids[start_patch * PATCH_SIZE :]
    patches_tensor = torch.tensor(trimmed_ids, device=next(model.parameters()).device, dtype=torch.long).reshape(1, -1, PATCH_SIZE)
    first_target_local = context_patch_count - start_patch
    if first_target_local <= 0:
        raise RuntimeError("trajectory replay window dropped all context before generated target patches")

    with autocast_context(patches_tensor.device, precision):
        encoded = model.patch_level_decoder(patches_tensor)["last_hidden_state"][0]
    encoded_start = first_target_local + chunk_start - 1
    encoded_end = first_target_local + chunk_end - 1
    encoded_targets = encoded[encoded_start:encoded_end]
    target_tensor = torch.tensor(remaining_patches[chunk_start:chunk_end], device=patches_tensor.device, dtype=torch.long)
    return char_patch_logprobs(model, encoded_targets, target_tensor, precision)


def trajectory_logprob_chunks_no_patch_mask(
    model,
    flat_prompt_ids: list[int],
    generated_patches: list[list[int]],
    precision: str,
    replay_context_patches: int | None = None,
    target_chunk_patches: int = 0,
):
    if not generated_patches:
        return

    current_ids = list(flat_prompt_ids)
    start_idx = 0
    while start_idx < len(generated_patches) and len(current_ids) % PATCH_SIZE != 0:
        patch = generated_patches[start_idx]
        logprob_list = patch_logprobs(
            model,
            current_ids,
            [patch],
            precision,
            replay_context_patches=replay_context_patches,
        )
        if logprob_list:
            yield torch.stack(logprob_list)
        current_ids.extend(
            normalize_patch_for_context(
                patch,
                eos_token_id=model.eos_token_id,
                special_token_id=model.special_token_id,
            )
        )
        start_idx += 1

    if start_idx >= len(generated_patches):
        return

    remaining_patches = generated_patches[start_idx:]
    chunk_size = len(remaining_patches) if target_chunk_patches <= 0 else target_chunk_patches
    for chunk_start in range(0, len(remaining_patches), chunk_size):
        chunk_end = min(len(remaining_patches), chunk_start + chunk_size)
        yield tail_logprobs_chunk_no_patch_mask(
            model,
            current_ids,
            remaining_patches,
            chunk_start,
            chunk_end,
            precision,
            replay_context_patches=replay_context_patches,
        )


def collect_logprobs(
    model,
    method: str,
    prompt_flat: list[int],
    generated_patches: list[list[int]],
    precision: str,
    replay_context_patches: int,
    chunk_patches: int,
) -> torch.Tensor:
    with torch.no_grad():
        if method == "tokenwise":
            values = patch_logprobs(
                model,
                prompt_flat,
                generated_patches,
                precision,
                replay_context_patches=replay_context_patches,
            )
            return torch.stack(values).detach().cpu().float()
        if method == "chunked_current":
            chunks = trajectory_logprob_chunks(
                model,
                prompt_flat,
                generated_patches,
                precision,
                replay_context_patches=replay_context_patches,
                target_chunk_patches=chunk_patches,
            )
            return torch.cat([chunk.detach().cpu().float() for chunk in chunks])
        if method == "chunked_nomask":
            chunks = trajectory_logprob_chunks_no_patch_mask(
                model,
                prompt_flat,
                generated_patches,
                precision,
                replay_context_patches=replay_context_patches,
                target_chunk_patches=chunk_patches,
            )
            return torch.cat([chunk.detach().cpu().float() for chunk in chunks])
    raise ValueError(f"unknown method: {method}")


def tensor_stats(tensor: torch.Tensor) -> dict:
    tensor = tensor.detach().cpu().float().reshape(-1)
    if tensor.numel() == 0:
        return {"n": 0}
    finite = tensor[torch.isfinite(tensor)]
    if finite.numel() == 0:
        return {"n": int(tensor.numel()), "finite_n": 0}
    return {
        "n": int(tensor.numel()),
        "finite_n": int(finite.numel()),
        "mean": float(finite.mean()),
        "std": float(finite.std(unbiased=False)) if finite.numel() > 1 else 0.0,
        "min": float(finite.min()),
        "p50": float(torch.quantile(finite, 0.50)),
        "p95": float(torch.quantile(finite, 0.95)),
        "p99": float(torch.quantile(finite, 0.99)),
        "max": float(finite.max()),
    }


def token_location(generated_patches: list[list[int]], flat_index: int) -> dict:
    offset = 0
    for patch_idx, patch in enumerate(generated_patches):
        live_tokens = [tok for tok in patch if tok != 0]
        next_offset = offset + len(live_tokens)
        if flat_index < next_offset:
            token_idx = flat_index - offset
            token_id = live_tokens[token_idx]
            printable = chr(token_id) if 32 <= token_id < 127 else repr(chr(token_id))
            return {
                "flat_index": flat_index,
                "patch_index": patch_idx,
                "token_index": token_idx,
                "token_id": token_id,
                "token_text": printable,
            }
        offset = next_offset
    return {"flat_index": flat_index}


def compare_tensors(name: str, left: torch.Tensor, right: torch.Tensor, generated_patches: list[list[int]], threshold: float) -> dict:
    result = {
        "name": name,
        "left_n": int(left.numel()),
        "right_n": int(right.numel()),
        "same_length": bool(left.numel() == right.numel()),
    }
    if left.numel() != right.numel():
        return result
    diff = left - right
    abs_diff = diff.abs()
    result["diff"] = tensor_stats(diff)
    result["abs_diff"] = tensor_stats(abs_diff)
    over = torch.nonzero(abs_diff > threshold, as_tuple=False)
    result["first_abs_gt_threshold"] = None
    if over.numel() > 0:
        idx = int(over[0].item())
        loc = token_location(generated_patches, idx)
        loc.update(
            {
                "left": float(left[idx]),
                "right": float(right[idx]),
                "diff": float(diff[idx]),
                "abs_diff": float(abs_diff[idx]),
            }
        )
        result["first_abs_gt_threshold"] = loc
    return result


def choose_device(text: str) -> torch.device:
    if text == "auto":
        return select_device()
    return torch.device(text)


def main() -> int:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--trajectory-jsonl")
    source.add_argument("--run-json")
    parser.add_argument("--step-index", type=int, default=1)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--policy-weights", required=True)
    parser.add_argument("--reference-weights", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-stream-lines", type=int, default=32)
    parser.add_argument("--replay-context-patches", type=int, default=0)
    parser.add_argument("--chunk-patches", type=int, default=8)
    parser.add_argument("--max-generated-patches", type=int, default=0)
    parser.add_argument("--diff-threshold", type=float, default=1e-4)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    trajectory = load_trajectory(args)
    generated_patches = trajectory["generated_patches"]
    rollout_prompt = build_rollout_prefix(trajectory["prompt_text"], args.target_stream_lines)
    patchilizer = Patchilizer(stream=True)
    prompt_flat = [item for patch in patchilizer.encode_generate(rollout_prompt) for item in patch]

    device = choose_device(args.device)
    policy_model = build_model(
        Path(args.policy_weights),
        device,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        precision=args.precision,
    )
    policy_model.eval()
    disable_dropout_modules(policy_model)

    reference_weights = Path(args.reference_weights) if args.reference_weights else Path(args.policy_weights)
    ref_model = build_model(
        reference_weights,
        device,
        precision=args.precision,
        freeze_before_precision_cast=True,
    )
    ref_model.eval()

    methods = ("tokenwise", "chunked_current", "chunked_nomask")
    policy_logprobs = {
        method: collect_logprobs(
            policy_model,
            method,
            prompt_flat,
            generated_patches,
            args.precision,
            args.replay_context_patches,
            args.chunk_patches,
        )
        for method in methods
    }
    ref_logprobs = {
        method: collect_logprobs(
            ref_model,
            method,
            prompt_flat,
            generated_patches,
            args.precision,
            args.replay_context_patches,
            args.chunk_patches,
        )
        for method in methods
    }

    comparisons = []
    for method in methods:
        comparisons.append(
            compare_tensors(
                f"policy_vs_reference:{method}",
                policy_logprobs[method],
                ref_logprobs[method],
                generated_patches,
                args.diff_threshold,
            )
        )
    comparisons.extend(
        [
            compare_tensors(
                "policy:tokenwise_vs_chunked_current",
                policy_logprobs["tokenwise"],
                policy_logprobs["chunked_current"],
                generated_patches,
                args.diff_threshold,
            ),
            compare_tensors(
                "policy:tokenwise_vs_chunked_nomask",
                policy_logprobs["tokenwise"],
                policy_logprobs["chunked_nomask"],
                generated_patches,
                args.diff_threshold,
            ),
            compare_tensors(
                "policy:chunked_current_vs_chunked_nomask",
                policy_logprobs["chunked_current"],
                policy_logprobs["chunked_nomask"],
                generated_patches,
                args.diff_threshold,
            ),
        ]
    )

    kl = {}
    for method in methods:
        if policy_logprobs[method].numel() == ref_logprobs[method].numel():
            terms = grpo_kl_term(policy_logprobs[method], ref_logprobs[method])
            kl[method] = {
                "sum": float(terms.sum()),
                "mean": float(terms.mean()) if terms.numel() else math.nan,
                "terms": tensor_stats(terms),
                "delta_ref_minus_policy": tensor_stats(ref_logprobs[method] - policy_logprobs[method]),
            }

    payload = {
        "source": {
            "trajectory_jsonl": args.trajectory_jsonl,
            "run_json": args.run_json,
            "step_index": args.step_index,
            "sample_index": args.sample_index,
        },
        "config": {
            "policy_weights": args.policy_weights,
            "reference_weights": str(reference_weights),
            "device": str(device),
            "precision": args.precision,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "target_stream_lines": args.target_stream_lines,
            "replay_context_patches": args.replay_context_patches,
            "chunk_patches": args.chunk_patches,
            "max_generated_patches": args.max_generated_patches,
            "diff_threshold": args.diff_threshold,
        },
        "trajectory": {
            "generated_patch_count": len(generated_patches),
            "generated_token_slots": sum(len(patch) for patch in generated_patches),
            "prompt_flat_tokens": len(prompt_flat),
        },
        "policy_logprobs": {method: tensor_stats(values) for method, values in policy_logprobs.items()},
        "reference_logprobs": {method: tensor_stats(values) for method, values in ref_logprobs.items()},
        "comparisons": comparisons,
        "kl": kl,
    }

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
