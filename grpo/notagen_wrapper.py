from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import importlib.util
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from transformers import GPT2Config

from grpo.stream_tags import (
    count_stream_lines as _count_stream_lines,
    latest_stream_line as _latest_stream_line,
    latest_stream_line_closed,
    stream_target_reached,
    trim_to_stream_lines as _trim_to_stream_lines,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTAGEN_ROOT = PROJECT_ROOT.parent / "NotaGen"
RL_DIR = NOTAGEN_ROOT / "RL"


def _load_notagen_rl_utils():
    config_path = RL_DIR / "config.py"
    utils_path = RL_DIR / "utils.py"

    config_spec = importlib.util.spec_from_file_location("notagen_rl_config", config_path)
    if config_spec is None or config_spec.loader is None:
        raise ImportError(f"could not load NotaGen RL config from {config_path}")
    config_module = importlib.util.module_from_spec(config_spec)
    config_spec.loader.exec_module(config_module)
    sys.modules["config"] = config_module

    utils_spec = importlib.util.spec_from_file_location("notagen_rl_utils", utils_path)
    if utils_spec is None or utils_spec.loader is None:
        raise ImportError(f"could not load NotaGen RL utils from {utils_path}")
    utils_module = importlib.util.module_from_spec(utils_spec)
    utils_spec.loader.exec_module(utils_module)
    return utils_module


_RL_UTILS = _load_notagen_rl_utils()
NotaGenLMHeadModel = _RL_UTILS.NotaGenLMHeadModel
Patchilizer = _RL_UTILS.Patchilizer


PATCH_STREAM = True
PATCH_SIZE = 16


@dataclass(frozen=True)
class ModelShape:
    patch_length: int
    char_num_layers: int
    patch_num_layers: int
    hidden_size: int


@dataclass
class GenerationResult:
    prompt: str
    full_text: str
    completion: str
    generated_patches: list[list[int]]


def flatten_patch_ids(patches: list[list[int]]) -> list[int]:
    return [int(tok) for patch in patches for tok in patch]


def chunk_patch_ids(flat_ids: list[int]) -> list[list[int]]:
    return [flat_ids[i : i + PATCH_SIZE] for i in range(0, len(flat_ids), PATCH_SIZE)]


def decode_flat_patch_ids(flat_ids: list[int]) -> str:
    patchilizer = Patchilizer(stream=PATCH_STREAM)
    if not flat_ids:
        return ""
    return patchilizer.decode(chunk_patch_ids(flat_ids))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def infer_model_shape(weights_path: Path, checkpoint: dict | None = None) -> ModelShape:
    name = weights_path.name
    if "p_length_2048" in name and "h_size_1024" in name and "p_layers_16" in name and "c_layers_3" in name:
        return ModelShape(patch_length=2048, char_num_layers=3, patch_num_layers=16, hidden_size=1024)
    if "p_length_2048" in name and "h_size_768" in name and "p_layers_12" in name and "c_layers_3" in name:
        return ModelShape(patch_length=2048, char_num_layers=3, patch_num_layers=12, hidden_size=768)
    if checkpoint is not None:
        model_state = checkpoint.get("model", checkpoint)
        if isinstance(model_state, dict):
            hidden_size = int(model_state["patch_level_decoder.base.h.0.ln_1.weight"].shape[0])
            patch_indices = {
                int(parts[3])
                for key in model_state
                if key.startswith("patch_level_decoder.base.h.")
                for parts in [key.split(".")]
                if len(parts) > 4 and parts[3].isdigit()
            }
            char_indices = {
                int(parts[4])
                for key in model_state
                if key.startswith("char_level_decoder.base.transformer.h.")
                for parts in [key.split(".")]
                if len(parts) > 5 and parts[4].isdigit()
            }
            return ModelShape(
                patch_length=2048 if hidden_size in (768, 1024) else 1024,
                char_num_layers=(max(char_indices) + 1) if char_indices else 6,
                patch_num_layers=(max(patch_indices) + 1) if patch_indices else 20,
                hidden_size=hidden_size,
            )
    return ModelShape(patch_length=1024, char_num_layers=6, patch_num_layers=20, hidden_size=1280)


def autocast_context(device: torch.device, precision: str):
    if device.type == "cuda" and precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def latest_stream_tag(abc_text: str) -> tuple[int, int] | None:
    line = _latest_stream_line(abc_text)
    if line is None:
        return None
    return line.tag.index, line.tag.marker


def latest_countdown(abc_text: str) -> tuple[int, int] | None:
    return latest_stream_tag(abc_text)


def count_stream_lines(abc_text: str) -> int:
    return _count_stream_lines(abc_text)


def split_metadata_and_tunebody_lines(abc_text: str) -> tuple[list[str], list[str]]:
    lines = abc_text.splitlines(keepends=True)
    tunebody_index = None
    for i, line in enumerate(lines):
        if line.startswith("[r:") or line.startswith("[V:"):
            tunebody_index = i
            break
    if tunebody_index is None:
        return lines, []
    return lines[:tunebody_index], lines[tunebody_index:]


def trim_to_stream_lines(abc_text: str, target_lines: int) -> str:
    return _trim_to_stream_lines(abc_text, target_lines)


def normalize_patch_for_context(patch: list[int], eos_token_id: int, special_token_id: int) -> list[int]:
    normalized = [int(tok) for tok in patch if int(tok) not in (eos_token_id, special_token_id)]
    return (normalized + [0] * PATCH_SIZE)[:PATCH_SIZE]


def build_model(weights_path: str | Path, device: torch.device | None = None, precision: str = "fp32") -> tuple[NotaGenLMHeadModel, ModelShape]:
    weights_path = Path(weights_path)
    device = device or select_device()
    checkpoint = torch.load(weights_path, map_location="cpu")
    shape = infer_model_shape(weights_path, checkpoint=checkpoint)
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
    model.load_state_dict(checkpoint["model"])
    model = model.to(device)
    if device.type == "cuda" and precision == "bf16":
        for param in model.parameters():
            param.data = param.data.to(torch.bfloat16)
    model.eval()
    return model, shape


def sample_completion(
    model: NotaGenLMHeadModel,
    model_shape: ModelShape,
    prompt: str,
    temperature: float,
    top_k: int,
    top_p: float,
    target_stream_lines: int,
    max_chars: int,
    timeout_s: int,
    precision: str,
) -> tuple[str, list[list[int]]]:
    patchilizer = Patchilizer(stream=PATCH_STREAM)
    device = next(model.parameters()).device
    prefix = prompt
    if count_stream_lines(prefix) == 0:
        prefix = prefix + f"[r:0/{target_stream_lines - 1}]"
    input_patches = patchilizer.encode_generate(prefix)
    flat_ids = [item for sublist in input_patches for item in sublist]
    input_tensor = torch.tensor([flat_ids], device=device).reshape(1, -1)
    byte_list = list(prefix)
    generated_patches: list[list[int]] = []
    start_time = time.time()
    cut_index = None

    with torch.inference_mode():
        while True:
            predicted_patch = None
            for _ in range(8):
                with autocast_context(device, precision):
                    candidate_patch = model.generate(
                        input_tensor.unsqueeze(0),
                        top_k=top_k,
                        top_p=top_p,
                        temperature=temperature,
                    )
                current_text = "".join(byte_list)
                eos_only = candidate_patch[0] == patchilizer.bos_token_id and candidate_patch[1] == patchilizer.eos_token_id
                if eos_only:
                    allow_eos = stream_target_reached(current_text, target_stream_lines)
                    if not allow_eos:
                        continue
                predicted_patch = candidate_patch
                break

            if predicted_patch is None:
                raise RuntimeError("decoder produced only early EOS candidates before target stream line completion")

            if predicted_patch[0] == patchilizer.bos_token_id and predicted_patch[1] == patchilizer.eos_token_id:
                break

            generated_patches.append(predicted_patch[:])
            next_patch = patchilizer.decode([predicted_patch])
            byte_list.extend(next_patch)

            normalized_patch = normalize_patch_for_context(
                predicted_patch,
                eos_token_id=patchilizer.eos_token_id,
                special_token_id=patchilizer.special_token_id,
            )
            input_tensor = torch.cat([input_tensor, torch.tensor([normalized_patch], device=device)], dim=1)

            current_text = "".join(byte_list)
            if count_stream_lines(current_text) >= target_stream_lines and latest_stream_line_closed(current_text):
                return trim_to_stream_lines(current_text, target_stream_lines), generated_patches
            if len(byte_list) > max_chars:
                return "".join(byte_list), generated_patches
            if time.time() - start_time > timeout_s:
                raise RuntimeError(f"generation exceeded {timeout_s}s")
            if input_tensor.shape[1] >= model_shape.patch_length * PATCH_SIZE:
                current_text = "".join(byte_list)
                metadata_lines, tunebody_lines = split_metadata_and_tunebody_lines(current_text)
                if not tunebody_lines:
                    raise RuntimeError("stream rollover hit before tunebody generation")
                if cut_index is None:
                    cut_index = max(1, len(tunebody_lines) // 2)
                abc_slice = "".join(metadata_lines + tunebody_lines[-cut_index:])
                repatched = patchilizer.encode_generate(abc_slice)
                flat_ids = [item for sublist in repatched for item in sublist]
                input_tensor = torch.tensor([flat_ids], device=device).reshape(1, -1)

    return "".join(byte_list), generated_patches


def _encoded_last_patch(
    model: NotaGenLMHeadModel,
    flat_ids: list[int],
    device: torch.device,
    precision: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    rem = len(flat_ids) % PATCH_SIZE
    if rem != 0:
        leftover = flat_ids[-rem:]
        prefix = flat_ids[:-rem]
        tokens = torch.tensor([model.bos_token_id] + leftover, device=device, dtype=torch.long)
    else:
        prefix = flat_ids
        tokens = torch.tensor([model.bos_token_id], device=device, dtype=torch.long)
    if not prefix:
        raise RuntimeError("prompt prefix is too short for NotaGen patch replay")
    prefix_tensor = torch.tensor(prefix, device=device, dtype=torch.long).reshape(1, -1, PATCH_SIZE)
    with autocast_context(device, precision):
        encoded = model.patch_level_decoder(prefix_tensor)["last_hidden_state"][0, -1]
    return encoded, tokens


def score_completion_logprobs(
    model: NotaGenLMHeadModel,
    flat_prompt_ids: list[int],
    generated_patches: list[list[int]],
    precision: str = "fp32",
) -> list[torch.Tensor]:
    base_ids = list(flat_prompt_ids)
    all_logprobs: list[torch.Tensor] = []

    for patch in generated_patches:
        current_prefix = list(base_ids)
        for tok in patch:
            logits = next_token_logits(model, current_prefix, precision=precision)
            logprob = torch.log_softmax(logits, dim=-1)[tok]
            all_logprobs.append(logprob)
            current_prefix.append(int(tok))

        base_ids.extend(
            normalize_patch_for_context(
                patch,
                eos_token_id=model.eos_token_id,
                special_token_id=model.special_token_id,
            )
        )

    return all_logprobs


def next_token_logits(
    model: NotaGenLMHeadModel,
    flat_prefix_ids: list[int],
    precision: str = "fp32",
) -> torch.Tensor:
    device = next(model.parameters()).device
    encoded_patch, tokens = _encoded_last_patch(model, flat_prefix_ids, device, precision)
    with autocast_context(device, precision):
        probs = model.char_level_decoder.generate(encoded_patch, tokens)
    probs = probs.float().clamp_min(1e-12)
    return probs.log()


class NotaGenPolicyWrapper:
    def __init__(self, weights_path: str | Path, device: torch.device | None = None, precision: str = "fp32"):
        self.model, self.shape = build_model(weights_path, device=device, precision=precision)
        self.precision = precision
        self.patchilizer = Patchilizer(stream=PATCH_STREAM)

    def encode_text_ids(self, text: str) -> list[int]:
        return [int(item) for sublist in self.patchilizer.encode_generate(text) for item in sublist]

    def decode_text_ids(self, flat_ids: list[int]) -> str:
        return self.patchilizer.decode(chunk_patch_ids([int(tok) for tok in flat_ids])) if flat_ids else ""

    def forward_logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        device = next(self.model.parameters()).device
        batch_size, seq_len = input_ids.shape
        vocab_size = int(self.model.char_level_decoder.base.config.vocab_size)
        batch_logits: list[torch.Tensor] = []

        input_ids_cpu = input_ids.detach().cpu()
        attention_mask_cpu = attention_mask.detach().cpu() if attention_mask is not None else None

        for b in range(batch_size):
            sample_logits = torch.zeros((seq_len, vocab_size), device=device, dtype=torch.float32)
            ids = input_ids_cpu[b].tolist()
            if attention_mask_cpu is not None:
                valid_len = int(attention_mask_cpu[b].sum().item())
                start = seq_len - valid_len
                seq_ids = ids[start:]
            else:
                start = 0
                seq_ids = ids

            for rel_pos in range(len(seq_ids) - 1):
                prefix = [int(tok) for tok in seq_ids[: rel_pos + 1]]
                abs_pos = start + rel_pos
                try:
                    sample_logits[abs_pos] = next_token_logits(self.model, prefix, precision=self.precision)
                except RuntimeError:
                    continue

            batch_logits.append(sample_logits)

        return torch.stack(batch_logits, dim=0)

    def generate_batch(
        self,
        prompts: list[str],
        *,
        temperature: float = 1.0,
        top_k: int = 8,
        top_p: float = 0.95,
        target_stream_lines: int = 8,
        max_chars: int = 16000,
        timeout_s: int = 45,
        seed: int = 0,
    ) -> list[GenerationResult]:
        results: list[GenerationResult] = []
        for idx, prompt in enumerate(prompts):
            set_seed(seed + idx)
            full_text, generated_patches = sample_completion(
                model=self.model,
                model_shape=self.shape,
                prompt=prompt,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                target_stream_lines=target_stream_lines,
                max_chars=max_chars,
                timeout_s=timeout_s,
                precision=self.precision,
            )
            completion = full_text[len(prompt):] if full_text.startswith(prompt) else full_text
            results.append(
                GenerationResult(
                    prompt=prompt,
                    full_text=full_text,
                    completion=completion,
                    generated_patches=generated_patches,
                )
            )
        return results

    def score_batch(self, prompt_completion_pairs: list[tuple[str, str]]) -> list[list[float]]:
        scored: list[list[float]] = []
        for prompt, completion in prompt_completion_pairs:
            prompt_flat = [item for sublist in self.patchilizer.encode_generate(prompt) for item in sublist]
            generated_patches = self.patchilizer.encode_generate(completion)
            logprobs = score_completion_logprobs(self.model, prompt_flat, generated_patches, precision=self.precision)
            scored.append([float(x.detach().cpu()) for x in logprobs])
        return scored
