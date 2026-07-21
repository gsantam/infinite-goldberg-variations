from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from safetensors.torch import load_file as load_safetensors
from transformers import GPT2Config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTAGEN_ROOT = PROJECT_ROOT.parent / "NotaGen"
RL_DIR = NOTAGEN_ROOT / "RL"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(RL_DIR) not in sys.path:
    sys.path.insert(0, str(RL_DIR))

from grpo import GoldbergRewardConfig, compute_group_advantages, load_structural_target  # noqa: E402
from grpo.notagen_cached_generation import CachedNotaGenPatchGenerator  # noqa: E402
from grpo.notagen_cached_generation_batch import sample_completions_cached_batch  # noqa: E402
from grpo.notagen_replay import (  # noqa: E402
    PATCH_SIZE,
    _encoded_last_patch,
    _pad_generated_patch,
    _replay_start_patch,
    _split_flat_logprobs,
    autocast_context,
    batched_tail_logprobs_chunk,
    batched_trajectory_logprobs,
    char_patch_logprobs,
    char_patch_logprob_sums,
    normalize_patch_for_context,
    patch_logprobs,
    split_tensor_by_counts,
    tail_encoded_targets,
    tail_logprobs_chunk,
    trajectory_logprob_chunks,
    trajectory_logprob_forward_count,
    trajectory_logprobs,
)
from evaluation.rewards import score_prompt_completion_pair  # noqa: E402
from evaluation.similarity_rewards import SimilarityReference, SimilarityRewardWeights, load_similarity_reference, score_similarity_reward  # noqa: E402
from evaluation.stream_tags import (
    count_stream_lines as _count_stream_lines,
    latest_stream_line as _latest_stream_line,
    latest_stream_line_closed as _latest_stream_line_closed,
    stream_target_reached,
    trim_to_stream_lines as _trim_to_stream_lines,
)  # noqa: E402
from utils import NotaGenLMHeadModel, Patchilizer  # noqa: E402


PATCH_STREAM = True


@dataclass(frozen=True)
class ModelShape:
    patch_length: int
    char_num_layers: int
    patch_num_layers: int
    hidden_size: int


@dataclass
class RolloutSample:
    prompt: str
    completion: str
    full_text: str
    generated_patches: list[list[int]]
    reward: float
    reward_breakdown: dict


def _rollout_seed(base_seed: int, step_idx: int, group_idx: int, retry_idx: int) -> int:
    return base_seed + step_idx * 1000 + group_idx * 100 + retry_idx


def infer_model_shape(weights_path: Path) -> ModelShape:
    name = weights_path.name
    if "p_length_2048" in name and "h_size_1024" in name and "p_layers_16" in name and "c_layers_3" in name:
        return ModelShape(patch_length=2048, char_num_layers=3, patch_num_layers=16, hidden_size=1024)
    if "p_length_2048" in name and "h_size_768" in name and "p_layers_12" in name and "c_layers_3" in name:
        return ModelShape(patch_length=2048, char_num_layers=3, patch_num_layers=12, hidden_size=768)
    return ModelShape(patch_length=1024, char_num_layers=6, patch_num_layers=20, hidden_size=1280)


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def apply_lora(
    model: NotaGenLMHeadModel,
    r: int,
    alpha: float,
    dropout: float,
    target_modules: tuple[str, ...] = ("attn.c_attn", "attn.c_proj"),
) -> NotaGenLMHeadModel:
    cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=list(target_modules),
    )
    model.patch_level_decoder.base = get_peft_model(model.patch_level_decoder.base, cfg)
    model.char_level_decoder.base = get_peft_model(model.char_level_decoder.base, cfg)
    return model


def count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def disable_dropout_modules(model: torch.nn.Module) -> int:
    count = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.eval()
            count += 1
    return count


def save_policy_checkpoint(model: NotaGenLMHeadModel, checkpoint_dir: Path, step_idx: int) -> dict:
    step_dir = checkpoint_dir / f"step_{step_idx:06d}"
    step_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    saved_parts: dict[str, str] = {}
    for name, module in (
        ("patch_level_decoder_base", model.patch_level_decoder.base),
        ("char_level_decoder_base", model.char_level_decoder.base),
    ):
        part_dir = step_dir / name
        if hasattr(module, "save_pretrained"):
            module.save_pretrained(part_dir)
        else:
            torch.save(module.state_dict(), part_dir.with_suffix(".pt"))
        saved_parts[name] = str(part_dir)
    elapsed_s = time.perf_counter() - start
    return {
        "step": step_idx,
        "path": str(step_dir),
        "parts": saved_parts,
        "elapsed_s": elapsed_s,
    }


def load_policy_checkpoint(model: NotaGenLMHeadModel, checkpoint_dir: Path) -> dict:
    loaded_parts: dict[str, str] = {}
    for name, module in (
        ("patch_level_decoder_base", model.patch_level_decoder.base),
        ("char_level_decoder_base", model.char_level_decoder.base),
    ):
        part_dir = checkpoint_dir / name
        adapter_path = part_dir / "adapter_model.safetensors"
        if not adapter_path.exists():
            raise FileNotFoundError(f"missing LoRA adapter checkpoint: {adapter_path}")
        state_dict = load_safetensors(str(adapter_path))
        result = set_peft_model_state_dict(module, state_dict, adapter_name="default")
        if getattr(result, "unexpected_keys", None):
            raise RuntimeError(f"unexpected keys while loading {adapter_path}: {result.unexpected_keys}")
        loaded_parts[name] = str(part_dir)
    return {"path": str(checkpoint_dir), "parts": loaded_parts}


def _to_cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _to_cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu_tree(item) for item in value)
    return value


def save_optimizer_checkpoint(
    optimizer: torch.optim.Optimizer,
    checkpoint_dir: Path,
    step_idx: int,
    archive_every_steps: int,
) -> dict:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    payload = {
        "step": step_idx,
        "optimizer": _to_cpu_tree(optimizer.state_dict()),
    }

    latest_path = checkpoint_dir / "optimizer_latest.pt"
    tmp_latest_path = latest_path.with_suffix(".tmp")
    torch.save(payload, tmp_latest_path)
    tmp_latest_path.replace(latest_path)

    archive_path = None
    if archive_every_steps > 0 and step_idx % archive_every_steps == 0:
        step_dir = checkpoint_dir / f"step_{step_idx:06d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        archive_path = step_dir / "optimizer.pt"
        tmp_archive_path = archive_path.with_suffix(".tmp")
        torch.save(payload, tmp_archive_path)
        tmp_archive_path.replace(archive_path)

    return {
        "step": step_idx,
        "latest_path": str(latest_path),
        "archive_path": str(archive_path) if archive_path else None,
        "elapsed_s": time.perf_counter() - start,
    }


def load_optimizer_checkpoint(
    optimizer: torch.optim.Optimizer,
    optimizer_checkpoint_path: Path,
    learning_rate: float | None = None,
) -> dict:
    payload = torch.load(optimizer_checkpoint_path, map_location="cpu")
    optimizer.load_state_dict(payload["optimizer"])
    loaded_lrs = [float(group.get("lr", 0.0)) for group in optimizer.param_groups]
    if learning_rate is not None:
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
    return {
        "path": str(optimizer_checkpoint_path),
        "step": int(payload.get("step", 0)),
        "loaded_lrs": loaded_lrs,
        "active_lrs": [float(group.get("lr", 0.0)) for group in optimizer.param_groups],
    }


def save_step_trajectories(samples: list[RolloutSample], trajectories_dir: Path, step_idx: int) -> dict:
    step_dir = trajectories_dir / f"step_{step_idx:06d}"
    step_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = step_dir / "trajectories.jsonl"
    records: list[dict] = []

    with manifest_path.open("w", encoding="utf-8") as f:
        for sample_idx, sample in enumerate(samples):
            sample_name = f"sample_{sample_idx:02d}"
            prompt_path = step_dir / f"{sample_name}.prompt.abc"
            full_path = step_dir / f"{sample_name}.full.abc"
            completion_path = step_dir / f"{sample_name}.completion.abc"
            prompt_path.write_text(sample.prompt, encoding="utf-8")
            full_path.write_text(sample.full_text, encoding="utf-8")
            completion_path.write_text(sample.completion, encoding="utf-8")

            record = {
                "step": step_idx,
                "sample_index": sample_idx,
                "candidate_path": sample.reward_breakdown.get("candidate_path"),
                "prompt_abc_path": str(prompt_path),
                "full_abc_path": str(full_path),
                "completion_abc_path": str(completion_path),
                "reward": sample.reward,
                "reward_breakdown": sample.reward_breakdown,
                "generated_patches": sample.generated_patches,
                "generated_patch_count": len(sample.generated_patches),
                "generated_token_slots": generated_token_slots(sample.generated_patches),
            }
            records.append(record)
            f.write(json.dumps(record) + "\n")

    return {
        "step": step_idx,
        "path": str(step_dir),
        "manifest": str(manifest_path),
        "count": len(records),
    }


def build_model(
    weights_path: Path,
    device: torch.device,
    lora_r: int = 0,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.0,
    precision: str = "fp32",
    freeze_before_precision_cast: bool = False,
) -> NotaGenLMHeadModel:
    shape = infer_model_shape(weights_path)
    patch_config = GPT2Config(
        num_hidden_layers=shape.patch_num_layers,
        max_length=shape.patch_length,
        max_position_embeddings=shape.patch_length,
        n_embd=shape.hidden_size,
        num_attention_heads=shape.hidden_size // 64,
        vocab_size=1,
    )
    byte_config = GPT2Config(
        num_hidden_layers=shape.char_num_layers,
        max_length=PATCH_SIZE + 1,
        max_position_embeddings=PATCH_SIZE + 1,
        hidden_size=shape.hidden_size,
        num_attention_heads=shape.hidden_size // 64,
        vocab_size=128,
    )
    model = NotaGenLMHeadModel(encoder_config=patch_config, decoder_config=byte_config)
    checkpoint = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    if lora_r > 0:
        model = apply_lora(model, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)
        print(f"LoRA enabled: trainable_params={count_trainable_parameters(model)}")
    if freeze_before_precision_cast:
        for param in model.parameters():
            param.requires_grad_(False)
    model = model.to(device)
    if device.type == "cuda" and precision == "bf16":
        for param in model.parameters():
            if param.requires_grad:
                param.data = param.data.float()
            else:
                param.data = param.data.to(torch.bfloat16)
    return model


def enable_gradient_checkpointing(model: NotaGenLMHeadModel) -> None:
    for module in (model.patch_level_decoder.base, model.char_level_decoder.base):
        if hasattr(module, "config"):
            module.config.use_cache = False
        if not hasattr(module, "gradient_checkpointing_enable"):
            continue
        try:
            module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            module.gradient_checkpointing_enable()


def load_prompt_rows(path: str | Path, limit: int | None = None) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if limit is not None and limit > 0:
        rows = rows[:limit]
    return rows


def prompt_row_name(row: dict, prompt_idx: int) -> str:
    for key in ("name", "id", "source", "prefix", "continuation"):
        value = row.get(key)
        if value:
            return str(value)
    return f"prompt_{prompt_idx}"


def split_metadata_and_tunebody(abc_text: str) -> tuple[list[str], list[str]]:
    lines = abc_text.splitlines(keepends=True)
    tunebody_index = None
    for i, line in enumerate(lines):
        if line.startswith("[V:") or line.startswith("[r:"):
            tunebody_index = i
            break
    if tunebody_index is None:
        return lines, []
    return lines[:tunebody_index], lines[tunebody_index:]


def sanitize_abc(abc_text: str) -> str:
    metadata_lines, tunebody_lines = split_metadata_and_tunebody(abc_text)
    clean_metadata: list[str] = []
    for line in metadata_lines:
        stripped = line.lstrip()
        if (
            stripped.startswith("%")
            or stripped.startswith("%%score")
            or stripped.startswith("L:")
            or stripped.startswith("Q:")
            or stripped.startswith("M:")
            or stripped.startswith("K:")
            or stripped.startswith("V:")
        ):
            clean_metadata.append(line)

    clean_tunebody: list[str] = []
    for line in tunebody_lines:
        stripped = line.lstrip()
        if stripped.startswith("[r:") or stripped.startswith("[V:"):
            clean_tunebody.append(line)

    return "".join(clean_metadata + clean_tunebody)


def count_stream_lines(abc_text: str) -> int:
    return _count_stream_lines(sanitize_abc(abc_text))


def latest_stream_tag(abc_text: str) -> tuple[int, int] | None:
    line = _latest_stream_line(sanitize_abc(abc_text))
    if line is None:
        return None
    return line.tag.index, line.tag.marker


def latest_countdown(abc_text: str) -> tuple[int, int] | None:
    return latest_stream_tag(abc_text)


def latest_stream_line(abc_text: str) -> str | None:
    lines = [line for line in sanitize_abc(abc_text).splitlines() if line.startswith("[r:")]
    return lines[-1] if lines else None


def latest_stream_line_closed(abc_text: str) -> bool:
    return _latest_stream_line_closed(sanitize_abc(abc_text))


def trim_to_stream_lines(abc_text: str, target_stream_lines: int) -> str:
    abc_text = sanitize_abc(abc_text)
    metadata_lines, tunebody_lines = split_metadata_and_tunebody(abc_text)
    return "".join(metadata_lines) + _trim_to_stream_lines("".join(tunebody_lines), target_stream_lines)


def build_rollout_prefix(prompt: str, target_stream_lines: int) -> str:
    if count_stream_lines(prompt) == 0:
        return prompt + f"[r:0/{target_stream_lines - 1}]"
    return prompt


def sample_completion(
    model: NotaGenLMHeadModel,
    model_shape: ModelShape,
    prompt: str,
    temperature: float,
    top_k: int,
    top_p: float,
    target_stream_lines: int,
    max_chars: int,
    max_generated_patches: int,
    timeout_s: int,
    precision: str,
    cached_rollout: bool = False,
) -> tuple[str, list[list[int]]]:
    patchilizer = Patchilizer(stream=PATCH_STREAM)
    device = next(model.parameters()).device

    prefix = build_rollout_prefix(prompt, target_stream_lines)
    input_patches = patchilizer.encode_generate(prefix)
    flat_ids = [item for sublist in input_patches for item in sublist]
    input_tensor = torch.tensor([flat_ids], device=device).reshape(1, -1) if not cached_rollout else None
    cached_generator = CachedNotaGenPatchGenerator(model, precision=precision) if cached_rollout else None
    if cached_generator is not None:
        cached_generator.reset(flat_ids)
    byte_list = list(prefix)
    generated_patches: list[list[int]] = []
    start_time = time.time()

    with torch.inference_mode():
        while True:
            predicted_patch = None
            for _ in range(8):
                if cached_generator is not None:
                    candidate_patch = cached_generator.generate_patch(
                        top_k=top_k,
                        top_p=top_p,
                        temperature=temperature,
                    )
                else:
                    if input_tensor is None:
                        raise RuntimeError("uncached rollout expected input tensor")
                    with autocast_context(device, precision):
                        candidate_patch = model.generate(
                            input_tensor.unsqueeze(0),
                            top_k=top_k,
                            top_p=top_p,
                            temperature=temperature,
                        )
                current_text = "".join(byte_list)
                eos_only = (
                    len(candidate_patch) >= 2
                    and candidate_patch[0] == patchilizer.bos_token_id
                    and candidate_patch[1] == patchilizer.eos_token_id
                )
                if eos_only:
                    allow_eos = stream_target_reached(sanitize_abc(current_text), target_stream_lines)
                    if not allow_eos:
                        continue
                predicted_patch = candidate_patch
                break

            if predicted_patch is None:
                raise RuntimeError("decoder produced only early EOS candidates before target stream line completion")

            if (
                len(predicted_patch) >= 2
                and predicted_patch[0] == patchilizer.bos_token_id
                and predicted_patch[1] == patchilizer.eos_token_id
            ):
                break

            generated_patches.append(predicted_patch[:])
            next_patch = patchilizer.decode([predicted_patch])
            byte_list.extend(next_patch)

            if cached_generator is not None:
                cached_generator.accept_patch(predicted_patch)
            else:
                if input_tensor is None:
                    raise RuntimeError("uncached rollout expected input tensor")
                normalized_patch = normalize_patch_for_context(
                    predicted_patch,
                    eos_token_id=patchilizer.eos_token_id,
                    special_token_id=patchilizer.special_token_id,
                )
                input_tensor = torch.cat(
                    [input_tensor, torch.tensor([normalized_patch], device=device)],
                    dim=1,
                )

            current_text = "".join(byte_list)
            if count_stream_lines(current_text) >= target_stream_lines and latest_stream_line_closed(current_text):
                return trim_to_stream_lines(current_text, target_stream_lines), generated_patches
            if max_generated_patches > 0 and len(generated_patches) >= max_generated_patches:
                return sanitize_abc(current_text), generated_patches
            if len(byte_list) > max_chars:
                return sanitize_abc(current_text), generated_patches
            if time.time() - start_time > timeout_s:
                raise RuntimeError(f"generation exceeded {timeout_s}s")
            current_patch_tokens = (
                len(cached_generator.state.flat_ids)
                if cached_generator is not None and cached_generator.state is not None
                else input_tensor.shape[1]
            )
            if current_patch_tokens >= model_shape.patch_length * PATCH_SIZE:
                raise RuntimeError("stream rollover not implemented in custom GRPO rollout")

    return sanitize_abc("".join(byte_list)), generated_patches


def generated_token_slots(generated_patches: list[list[int]]) -> int:
    return sum(len(patch) for patch in generated_patches)


def grpo_kl_term(policy_logprobs: torch.Tensor, ref_logprobs: torch.Tensor) -> torch.Tensor:
    delta = ref_logprobs - policy_logprobs
    return torch.exp(delta) - delta - 1.0


def run_smoke(
    policy_model: NotaGenLMHeadModel,
    policy_shape: ModelShape,
    ref_model: NotaGenLMHeadModel,
    prompts: list[dict],
    target,
    reward_config: GoldbergRewardConfig,
    args,
) -> dict:
    patchilizer = Patchilizer(stream=PATCH_STREAM)
    trainable_params = [param for param in policy_model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)
    optimizer_resume_payload = None
    if args.resume_optimizer_checkpoint:
        optimizer_resume_payload = load_optimizer_checkpoint(
            optimizer,
            Path(args.resume_optimizer_checkpoint),
            learning_rate=args.learning_rate,
        )
        print(
            "Resumed optimizer checkpoint from "
            f"{args.resume_optimizer_checkpoint}; "
            f"loaded_lrs={optimizer_resume_payload['loaded_lrs']} "
            f"active_lrs={optimizer_resume_payload['active_lrs']}"
        )
    device = next(policy_model.parameters()).device
    policy_model.eval()
    dropout_modules_disabled = disable_dropout_modules(policy_model)
    ref_model.eval()

    logs: list[dict] = []
    checkpoints: list[dict] = []
    trajectory_dumps: list[dict] = []
    similarity_weights = SimilarityRewardWeights(
        aria_chroma=args.aria_chroma_reward_weight,
        aria_harmony=args.aria_harmony_reward_weight,
    )
    aria_similarity_ref: SimilarityReference | None = None
    if similarity_weights.enabled:
        aria_similarity_ref = load_similarity_reference(
            args.aria_reference_abc,
            load_chroma=similarity_weights.aria_chroma != 0.0,
            load_harmony=similarity_weights.aria_harmony != 0.0,
            bins=args.similarity_chroma_bins,
        )

    if not prompts:
        raise ValueError("no prompt rows loaded")

    for local_step_idx in range(1, args.max_steps + 1):
        step_start = time.perf_counter()
        timings: dict[str, float] = {}
        step_idx = args.step_offset + local_step_idx
        prompt_idx = (step_idx - 1) % len(prompts)
        row = prompts[prompt_idx]
        prompt_name = prompt_row_name(row, prompt_idx)
        prompt = row["prompt"]
        group_samples: list[RolloutSample] = []
        rollout_payloads: list[tuple[str, list[list[int]], dict]] = []

        rollout_start = time.perf_counter()
        if args.rollout_batch_size > 1:
            if not args.cached_rollout:
                raise RuntimeError("--rollout-batch-size > 1 requires --cached-rollout")
            pending = list(range(args.group_size))
            last_errors: dict[int, str] = {}
            for retry_idx in range(args.rollout_retries):
                next_pending: list[int] = []
                for batch_start in range(0, len(pending), args.rollout_batch_size):
                    batch_indices = pending[batch_start : batch_start + args.rollout_batch_size]
                    batch_results = sample_completions_cached_batch(
                        model=policy_model,
                        model_shape=policy_shape,
                        prompts=[prompt] * len(batch_indices),
                        seeds=[
                            _rollout_seed(args.seed, step_idx, group_idx, retry_idx)
                            for group_idx in batch_indices
                        ],
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
                    for group_idx, result in zip(batch_indices, batch_results, strict=True):
                        if result.ok and result.full_text is not None and result.generated_patches is not None:
                            rollout_payloads.append(
                                (
                                    result.full_text,
                                    result.generated_patches,
                                    {
                                        "cached_rollout": True,
                                        "batched_rollout": True,
                                        "rollout_batch_size": args.rollout_batch_size,
                                        "group_index": group_idx,
                                        "rollout_seed": _rollout_seed(args.seed, step_idx, group_idx, retry_idx),
                                        **(result.meta or {}),
                                    },
                                )
                            )
                        else:
                            last_errors[group_idx] = result.error or "unknown batch rollout error"
                            next_pending.append(group_idx)
                if not next_pending:
                    pending = []
                    break
                pending = next_pending
            if pending:
                raise RuntimeError(f"failed to sample batched rollouts after retries: {last_errors}")
        else:
            for group_idx in range(args.group_size):
                sample_built = False
                last_error: Exception | None = None
                for retry_idx in range(args.rollout_retries):
                    rollout_seed = _rollout_seed(args.seed, step_idx, group_idx, retry_idx)
                    set_seed(rollout_seed)
                    try:
                        full_text, generated_patches = sample_completion(
                            model=policy_model,
                            model_shape=policy_shape,
                            prompt=prompt,
                            temperature=args.temperature,
                            top_k=args.top_k,
                            top_p=args.top_p,
                            target_stream_lines=args.target_stream_lines,
                            max_chars=args.max_chars,
                            max_generated_patches=args.max_generated_patches,
                            timeout_s=args.timeout_s,
                            precision=args.precision,
                            cached_rollout=args.cached_rollout,
                        )
                        rollout_payloads.append(
                            (
                                full_text,
                                generated_patches,
                                {
                                    "cached_rollout": bool(args.cached_rollout),
                                    "batched_rollout": False,
                                    "rollout_batch_size": 1,
                                    "group_index": group_idx,
                                    "rollout_seed": rollout_seed,
                                },
                            )
                        )
                        sample_built = True
                        break
                    except RuntimeError as exc:
                        last_error = exc
                        continue

                if not sample_built:
                    raise RuntimeError(f"failed to sample rollout after retries: {last_error}")

        rollout_payloads = rollout_payloads[: args.group_size]
        timings["rollout_s"] = time.perf_counter() - rollout_start
        timings["rollout_per_sample_s"] = timings["rollout_s"] / max(1, len(rollout_payloads))

        reward_start = time.perf_counter()
        structural_reward_s = 0.0
        similarity_reward_s = 0.0
        for group_idx, (full_text, generated_patches, rollout_meta) in enumerate(rollout_payloads):
            completion = full_text[len(prompt):] if full_text.startswith(prompt) else full_text
            structural_reward_start = time.perf_counter()
            breakdown = score_prompt_completion_pair(
                prompt_text=prompt,
                completion_text=completion,
                target=target,
                config=reward_config,
                candidate_name=f"step{step_idx}_sample{group_idx}",
            )
            structural_reward_s += time.perf_counter() - structural_reward_start
            reward_breakdown = breakdown.to_json()
            structural_total_reward = breakdown.total_reward
            similarity_reward_start = time.perf_counter()
            similarity_payload = score_similarity_reward(
                prompt_text=prompt,
                completion_text=completion,
                weights=similarity_weights,
                aria=aria_similarity_ref,
                variation=None,
                bins=args.similarity_chroma_bins,
                band_ratio=args.similarity_band_ratio,
                timeout_s=args.similarity_timeout_s,
            )
            similarity_reward_s += time.perf_counter() - similarity_reward_start
            raw_similarity_reward = float(similarity_payload.get("similarity_reward", 0.0))
            clipped_similarity_reward = (
                min(raw_similarity_reward, args.max_similarity_reward)
                if args.max_similarity_reward > 0
                else raw_similarity_reward
            )
            similarity_validity_gate = float(breakdown.parse_reward * breakdown.bar_count_reward)
            effective_similarity_reward = clipped_similarity_reward * similarity_validity_gate
            total_reward = structural_total_reward + effective_similarity_reward
            reward_breakdown["structural_total_reward"] = structural_total_reward
            reward_breakdown.update(similarity_payload)
            reward_breakdown["raw_similarity_reward"] = raw_similarity_reward
            reward_breakdown["clipped_similarity_reward"] = clipped_similarity_reward
            reward_breakdown["max_similarity_reward"] = args.max_similarity_reward
            reward_breakdown["similarity_validity_gate"] = similarity_validity_gate
            reward_breakdown["effective_similarity_reward"] = effective_similarity_reward
            reward_breakdown["similarity_reward"] = effective_similarity_reward
            reward_breakdown["total_reward"] = total_reward
            reward_breakdown["generated_patches"] = len(generated_patches)
            reward_breakdown["generated_token_slots"] = generated_token_slots(generated_patches)
            reward_breakdown["prompt_index"] = prompt_idx
            reward_breakdown["prompt_name"] = prompt_name
            reward_breakdown.update(rollout_meta)
            reward_breakdown["rollout_prefix_stream_lines"] = count_stream_lines(
                build_rollout_prefix(prompt, args.target_stream_lines)
            )
            group_samples.append(
                RolloutSample(
                    prompt=prompt,
                    completion=completion,
                    full_text=full_text,
                    generated_patches=generated_patches,
                    reward=total_reward,
                    reward_breakdown=reward_breakdown,
                )
            )
        timings["reward_s"] = time.perf_counter() - reward_start
        timings["structural_reward_s"] = structural_reward_s
        timings["similarity_reward_s"] = similarity_reward_s
        timings["reward_per_sample_s"] = timings["reward_s"] / max(1, len(group_samples))

        advantage_start = time.perf_counter()
        advantages_payload = compute_group_advantages(
            [type("Tmp", (), {"candidate_path": f"s{i}", "total_reward": s.reward})() for i, s in enumerate(group_samples)]
        )
        advantages = [item["advantage"] for item in advantages_payload]
        timings["advantage_s"] = time.perf_counter() - advantage_start

        # Keep policy scoring in eval mode. Gradients still flow to the LoRA
        # weights, but the initial policy/reference logprobs stay comparable.
        replay_start = time.perf_counter()
        policy_model.eval()
        disable_dropout_modules(policy_model)
        optimizer.zero_grad(set_to_none=True)
        sample_loss_values: list[float] = []
        rollout_prompt = build_rollout_prefix(prompt, args.target_stream_lines)
        prompt_flat = [item for sublist in patchilizer.encode_generate(rollout_prompt) for item in sublist]
        replay_count = len(group_samples)

        batched_ref_logprobs_batch: list[torch.Tensor | None] | None = None
        if args.batch_logprob_replay and args.beta != 0.0:
            reference_replay_start = time.perf_counter()
            generated_batch = [sample.generated_patches for sample in group_samples]
            ref_precision = args.precision if next(ref_model.parameters()).device.type == "cuda" else "fp32"
            with torch.no_grad():
                batched_ref_logprobs_batch = [
                    tensor.detach().cpu().float()
                    for tensor in batched_trajectory_logprobs(
                        ref_model,
                        prompt_flat,
                        generated_batch,
                        ref_precision,
                        replay_context_patches=args.replay_context_patches,
                        target_chunk_patches=args.score_chunk_patches,
                        replay_batch_size=args.logprob_replay_batch_size,
                    )
                ]
            timings["batched_reference_logprob_s"] = time.perf_counter() - reference_replay_start

        for sample_idx, (sample, advantage) in enumerate(zip(group_samples, advantages, strict=True)):
            if args.no_step:
                with torch.no_grad():
                    policy_logprob_list = trajectory_logprobs(
                        policy_model,
                        prompt_flat,
                        sample.generated_patches,
                        args.precision,
                        replay_context_patches=args.replay_context_patches,
                    )
                    if not policy_logprob_list:
                        continue
                    policy_logprobs = torch.stack(policy_logprob_list)
                    sample.reward_breakdown["scored_tokens"] = int(policy_logprobs.numel())
                    if args.beta != 0.0:
                        if batched_ref_logprobs_batch is not None:
                            ref_logprobs = batched_ref_logprobs_batch[sample_idx]
                        else:
                            ref_precision = args.precision if next(ref_model.parameters()).device.type == "cuda" else "fp32"
                            ref_logprobs = torch.stack(
                                trajectory_logprobs(
                                    ref_model,
                                    prompt_flat,
                                    sample.generated_patches,
                                    ref_precision,
                                    replay_context_patches=args.replay_context_patches,
                                )
                            )
                    else:
                        ref_logprobs = None
            elif args.score_chunk_patches > 0:
                total_tokens = trajectory_logprob_forward_count(
                    policy_model,
                    prompt_flat,
                    sample.generated_patches,
                    args.precision,
                    replay_context_patches=args.replay_context_patches,
                    target_chunk_patches=args.score_chunk_patches,
                )
                if total_tokens <= 0:
                    continue
                sample.reward_breakdown["scored_tokens"] = total_tokens
                adv = torch.tensor(advantage, device=device, dtype=torch.float32)
                ref_logprobs = None
                if args.beta != 0.0:
                    if batched_ref_logprobs_batch is not None:
                        ref_logprobs = batched_ref_logprobs_batch[sample_idx]
                    else:
                        ref_precision = args.precision if next(ref_model.parameters()).device.type == "cuda" else "fp32"
                        with torch.no_grad():
                            ref_chunks = list(
                                trajectory_logprob_chunks(
                                    ref_model,
                                    prompt_flat,
                                    sample.generated_patches,
                                    ref_precision,
                                    replay_context_patches=args.replay_context_patches,
                                    target_chunk_patches=args.score_chunk_patches,
                                )
                            )
                        if not ref_chunks:
                            continue
                        ref_logprobs = torch.cat([chunk.detach().cpu().float() for chunk in ref_chunks])
                    if ref_logprobs.numel() != total_tokens:
                        raise RuntimeError(f"reference logprob count mismatch: got {ref_logprobs.numel()}, expected {total_tokens}")

                offset = 0
                sample_loss_value = 0.0
                sample_kl_sum = 0.0
                for policy_logprobs in trajectory_logprob_chunks(
                    policy_model,
                    prompt_flat,
                    sample.generated_patches,
                    args.precision,
                    replay_context_patches=args.replay_context_patches,
                    target_chunk_patches=args.score_chunk_patches,
                ):
                    if policy_logprobs.numel() == 0:
                        continue
                    policy_logprobs = policy_logprobs.float()
                    if ref_logprobs is not None:
                        ref_chunk = ref_logprobs[offset : offset + policy_logprobs.numel()].to(policy_logprobs.device)
                        kl_sum = grpo_kl_term(policy_logprobs, ref_chunk).sum()
                        sample_kl_sum += float(kl_sum.detach().cpu())
                    else:
                        kl_sum = torch.zeros((), device=policy_logprobs.device, dtype=policy_logprobs.dtype)
                    offset += policy_logprobs.numel()
                    chunk_loss = (-(adv.to(policy_logprobs.dtype) * policy_logprobs.sum()) + args.beta * kl_sum) / total_tokens
                    sample_loss_value += float(chunk_loss.detach().cpu())
                    (chunk_loss / replay_count).backward()
                    del policy_logprobs, kl_sum, chunk_loss

                if offset != total_tokens:
                    raise RuntimeError(f"policy logprob count mismatch: got {offset}, expected {total_tokens}")
                sample.reward_breakdown["kl_reference_active"] = bool(ref_logprobs is not None)
                sample.reward_breakdown["kl_beta"] = float(args.beta)
                sample.reward_breakdown["kl_sum"] = sample_kl_sum
                sample.reward_breakdown["kl_mean"] = sample_kl_sum / total_tokens if total_tokens > 0 else 0.0
                sample_loss_values.append(sample_loss_value)
                del adv, ref_logprobs
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                continue
            else:
                policy_logprob_list = trajectory_logprobs(
                    policy_model,
                    prompt_flat,
                    sample.generated_patches,
                    args.precision,
                    replay_context_patches=args.replay_context_patches,
                )
                if not policy_logprob_list:
                    continue
                policy_logprobs = torch.stack(policy_logprob_list)
                sample.reward_breakdown["scored_tokens"] = int(policy_logprobs.numel())
                if args.beta != 0.0:
                    if batched_ref_logprobs_batch is not None:
                        ref_logprobs = batched_ref_logprobs_batch[sample_idx]
                    else:
                        with torch.no_grad():
                            ref_precision = args.precision if next(ref_model.parameters()).device.type == "cuda" else "fp32"
                            ref_logprobs = torch.stack(
                                trajectory_logprobs(
                                    ref_model,
                                    prompt_flat,
                                    sample.generated_patches,
                                    ref_precision,
                                    replay_context_patches=args.replay_context_patches,
                                )
                            )
                else:
                    ref_logprobs = None

            policy_logprobs = policy_logprobs.float()
            if ref_logprobs is not None:
                if ref_logprobs.numel() != policy_logprobs.numel():
                    raise RuntimeError(
                        f"reference logprob count mismatch: got {ref_logprobs.numel()}, "
                        f"expected {policy_logprobs.numel()}"
                    )
                ref_logprobs = ref_logprobs.to(policy_logprobs.device).float()
                kl = grpo_kl_term(policy_logprobs, ref_logprobs).mean()
            else:
                kl = torch.zeros((), device=policy_logprobs.device, dtype=policy_logprobs.dtype)
            sample.reward_breakdown["kl_reference_active"] = bool(ref_logprobs is not None)
            sample.reward_breakdown["kl_beta"] = float(args.beta)
            sample.reward_breakdown["kl_sum"] = float((kl * policy_logprobs.numel()).detach().cpu())
            sample.reward_breakdown["kl_mean"] = float(kl.detach().cpu())
            adv = torch.tensor(advantage, device=device, dtype=policy_logprobs.dtype)
            loss = -(adv * policy_logprobs.mean()) + args.beta * kl
            sample_loss_values.append(float(loss.detach().cpu()))

            if not args.no_step:
                # True sequential replay: keep only one trajectory graph alive at a time.
                (loss / replay_count).backward()

            del policy_logprob_list, policy_logprobs, ref_logprobs, kl, adv, loss
        timings["logprob_backward_s"] = time.perf_counter() - replay_start

        if not sample_loss_values:
            raise RuntimeError("no valid rollout samples were produced for GRPO smoke test")

        total_loss_value = sum(sample_loss_values) / len(sample_loss_values)
        total_loss = torch.tensor(total_loss_value, device=device, dtype=torch.float32)
        optimizer_start = time.perf_counter()
        if not args.no_step:
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), args.max_grad_norm)
            optimizer.step()
        timings["optimizer_s"] = time.perf_counter() - optimizer_start
        checkpoint_payload = None
        checkpoint_start = time.perf_counter()
        if (
            not args.no_step
            and args.checkpoint_dir
            and args.checkpoint_every_steps > 0
            and step_idx % args.checkpoint_every_steps == 0
        ):
            checkpoint_payload = save_policy_checkpoint(policy_model, Path(args.checkpoint_dir), step_idx)
            checkpoint_payload["optimizer"] = save_optimizer_checkpoint(
                optimizer,
                Path(args.checkpoint_dir),
                step_idx,
                args.optimizer_checkpoint_every_steps,
            )
            checkpoints.append(checkpoint_payload)
        timings["checkpoint_s"] = time.perf_counter() - checkpoint_start
        trajectory_dump_payload = None
        trajectory_dump_start = time.perf_counter()
        if args.trajectories_dir:
            trajectory_dump_payload = save_step_trajectories(group_samples, Path(args.trajectories_dir), step_idx)
            trajectory_dumps.append(trajectory_dump_payload)
        timings["trajectory_dump_s"] = time.perf_counter() - trajectory_dump_start
        timings["total_step_s"] = time.perf_counter() - step_start

        step_log = {
            "step": step_idx,
            "prompt_index": prompt_idx,
            "prompt_name": prompt_name,
            "prompt_pool_size": len(prompts),
            "loss": float(total_loss.detach().cpu()),
            "timings": timings,
            "samples": [sample.reward_breakdown for sample in group_samples],
            "trajectories": [
                {
                    "sample_index": sample_idx,
                    "prompt": sample.prompt,
                    "completion": sample.completion,
                    "full_text": sample.full_text,
                    "generated_patches": sample.reward_breakdown.get("generated_patches"),
                    "generated_token_slots": sample.reward_breakdown.get("generated_token_slots"),
                    "reward": sample.reward,
                    "reward_breakdown": sample.reward_breakdown,
                }
                for sample_idx, sample in enumerate(group_samples)
            ],
            "advantages": advantages_payload,
            "checkpoint": checkpoint_payload,
            "trajectory_dump": trajectory_dump_payload,
        }
        logs.append(step_log)
        print(
            json.dumps(
                {
                    "event": "step_complete",
                    "step": step_idx,
                    "prompt_index": prompt_idx,
                    "prompt_name": prompt_name,
                    "prompt_pool_size": len(prompts),
                    "loss": step_log["loss"],
                    "rewards": [sample.reward for sample in group_samples],
                    "advantages": [item["advantage"] for item in advantages_payload],
                    "observed_bars": [sample.reward_breakdown.get("observed_bars") for sample in group_samples],
                    "validated_bars": [sample.reward_breakdown.get("validated_bars") for sample in group_samples],
                    "scored_tokens": [sample.reward_breakdown.get("scored_tokens") for sample in group_samples],
                    "kl_mean": [sample.reward_breakdown.get("kl_mean") for sample in group_samples],
                    "raw_similarity_rewards": [sample.reward_breakdown.get("raw_similarity_reward") for sample in group_samples],
                    "clipped_similarity_rewards": [sample.reward_breakdown.get("clipped_similarity_reward") for sample in group_samples],
                    "effective_similarity_rewards": [sample.reward_breakdown.get("effective_similarity_reward") for sample in group_samples],
                    "timings": timings,
                    "checkpoint": checkpoint_payload,
                    "trajectory_dump": trajectory_dump_payload,
                }
            ),
            flush=True,
        )

    return {
        "steps": logs,
        "checkpoints": checkpoints,
        "trajectory_dumps": trajectory_dumps,
        "policy_dropout_modules_disabled": dropout_modules_disabled,
        "optimizer_resume": optimizer_resume_payload,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-weights", required=True)
    parser.add_argument("--reference-weights", default=None)
    parser.add_argument("--prompts-jsonl", default=str(PROJECT_ROOT / "data" / "processed" / "notagen" / "goldberg_grpo_prompts.jsonl"))
    parser.add_argument("--target-json", default=str(PROJECT_ROOT / "data" / "processed" / "goldberg" / "structure" / "aria_bar_skeleton.json"))
    parser.add_argument(
        "--target-structure-abc",
        required=True,
        help="Reference NotaGen ABC whose body/stream-line count is used for the bar-count reward.",
    )
    parser.add_argument("--aria-reference-abc", default=str(PROJECT_ROOT / "data" / "processed" / "goldberg" / "abc" / "aria-bwv-988.abc"))
    parser.add_argument("--aria-chroma-reward-weight", type=float, default=1.0)
    parser.add_argument("--aria-harmony-reward-weight", type=float, default=1.0)
    parser.add_argument("--max-similarity-reward", type=float, default=2.0, help="Cap raw added similarity reward before structural validity gating. Use <=0 to disable.")
    parser.add_argument("--similarity-chroma-bins", type=int, default=128)
    parser.add_argument("--similarity-band-ratio", type=float, default=0.25)
    parser.add_argument("--similarity-timeout-s", type=float, default=20.0)
    parser.add_argument("--output-json", required=True)
    parser.add_argument(
        "--prompt-limit",
        type=int,
        default=1,
        help="Number of prompt rows to keep in the rotation pool. Use 0 for all prompts.",
    )
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--target-stream-lines", type=int, default=32)
    parser.add_argument("--max-chars", type=int, default=16000)
    parser.add_argument(
        "--max-generated-patches",
        type=int,
        default=0,
        help="Stop a rollout after this many generated NotaGen patches. Use 0 for no patch cap.",
    )
    parser.add_argument("--timeout-s", type=int, default=180)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--beta", type=float, default=0.02)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--rollout-retries", type=int, default=4)
    parser.add_argument("--replay-context-patches", type=int, default=128)
    parser.add_argument("--score-chunk-patches", type=int, default=0)
    parser.add_argument(
        "--batch-logprob-replay",
        action="store_true",
        help="Batch frozen-reference logprob replay across trajectories after the prompt-alignment prefix.",
    )
    parser.add_argument(
        "--logprob-replay-batch-size",
        type=int,
        default=0,
        help="Maximum trajectories per batched reference replay call. Use 0 for all active trajectories.",
    )
    parser.add_argument("--lora-r", type=int, default=0)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--step-offset", type=int, default=0)
    parser.add_argument("--no-step", action="store_true")
    parser.add_argument("--reference-on-cpu", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--cached-rollout", action="store_true")
    parser.add_argument(
        "--rollout-batch-size",
        type=int,
        default=1,
        help="Generate rollout completions in cached batches. Values >1 require --cached-rollout.",
    )
    parser.add_argument("--resume-checkpoint-dir", default=None)
    parser.add_argument("--resume-optimizer-checkpoint", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--checkpoint-every-steps", type=int, default=1)
    parser.add_argument("--optimizer-checkpoint-every-steps", type=int, default=10)
    parser.add_argument("--trajectories-dir", default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    device = select_device()
    policy_weights = Path(args.policy_weights)
    policy_shape = infer_model_shape(policy_weights)
    policy_model = build_model(
        policy_weights,
        device,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        precision=args.precision,
    )
    resume_payload = None
    if args.resume_checkpoint_dir:
        resume_payload = load_policy_checkpoint(policy_model, Path(args.resume_checkpoint_dir))
        print(f"Resumed policy LoRA checkpoint from {args.resume_checkpoint_dir}")
    if args.gradient_checkpointing:
        enable_gradient_checkpointing(policy_model)
        print("Gradient checkpointing enabled for policy model")
    reference_weights = Path(args.reference_weights) if args.reference_weights else policy_weights
    ref_device = torch.device("cpu") if (args.no_step or args.reference_on_cpu) else device
    ref_precision = args.precision if ref_device.type == "cuda" else "fp32"
    ref_model = build_model(
        reference_weights,
        ref_device,
        precision=ref_precision,
        freeze_before_precision_cast=True,
    )
    for param in ref_model.parameters():
        param.requires_grad_(False)

    prompts = load_prompt_rows(args.prompts_jsonl, limit=args.prompt_limit)
    target = load_structural_target(args.target_json, structure_path=args.target_structure_abc)
    reward_config = GoldbergRewardConfig()

    payload = run_smoke(
        policy_model=policy_model,
        policy_shape=policy_shape,
        ref_model=ref_model,
        prompts=prompts,
        target=target,
        reward_config=reward_config,
        args=args,
    )
    payload["run_config"] = {
        "args": vars(args),
        "policy_shape": asdict(policy_shape),
        "reward_config": asdict(reward_config),
        "similarity_reward": {
            "aria_chroma_weight": args.aria_chroma_reward_weight,
            "aria_harmony_weight": args.aria_harmony_reward_weight,
            "max_similarity_reward": args.max_similarity_reward,
            "aria_reference_abc": args.aria_reference_abc,
            "chroma_bins": args.similarity_chroma_bins,
            "band_ratio": args.similarity_band_ratio,
            "timeout_s": args.similarity_timeout_s,
        },
        "policy_weights": str(policy_weights),
        "reference_weights": str(reference_weights),
        "reference_device": str(ref_device),
        "resume_checkpoint": resume_payload,
        "kl": {
            "enabled": bool(args.beta != 0.0),
            "beta": float(args.beta),
            "reference_policy": "frozen_reference_weights",
        },
        "reference_frozen_before_precision_cast": True,
        "policy_dropout_disabled_for_scoring": True,
    }
    Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
