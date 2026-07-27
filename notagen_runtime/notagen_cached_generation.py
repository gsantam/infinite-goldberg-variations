from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np
import torch


PATCH_SIZE = 16
ASCII_VOCAB_SIZE = 128


@dataclass
class CachedPatchState:
    flat_ids: list[int]
    cached_patch_count: int
    partial_ids: list[int]
    last_patch_hidden: torch.Tensor
    past_key_values: Any


def normalize_patch_for_context(
    patch: list[int],
    *,
    eos_token_id: int,
    special_token_id: int,
) -> list[int]:
    out = [int(tok) for tok in patch]
    patch_end = False
    for i, tok in enumerate(out):
        if patch_end:
            out[i] = special_token_id
        if tok == eos_token_id:
            patch_end = True
    return out


def _safe_normalize_probs(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.where(np.isnan(probs) | (probs < 0), 0.0, probs)
    probs = probs + 1e-12
    total = probs.sum()
    if total > 0:
        return probs / total
    fallback = np.zeros_like(probs)
    fallback[0] = 1.0
    return fallback


def _top_k_filter(probs: np.ndarray, top_k: int) -> np.ndarray:
    if top_k <= 0 or top_k >= len(probs):
        return probs
    kept = np.argsort(probs)[::-1][:top_k]
    filtered = np.zeros_like(probs)
    filtered[kept] = probs[kept]
    return _safe_normalize_probs(filtered)


def _top_p_filter(probs: np.ndarray, top_p: float) -> np.ndarray:
    if top_p <= 0 or top_p >= 1:
        return probs
    sorted_tokens = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_tokens]
    remove = np.cumsum(sorted_probs) > top_p
    remove[1:] = remove[:-1]
    remove[0] = False
    filtered = probs.copy()
    filtered[sorted_tokens[remove]] = 0.0
    return _safe_normalize_probs(filtered)


def _temperature_sample(probs: np.ndarray, temperature: float) -> int:
    if temperature != 1:
        if temperature <= 0:
            return int(np.argmax(probs))
        probs = np.exp(np.log(probs) / temperature)
        probs = probs / probs.sum()
    return int(np.random.choice(range(len(probs)), p=probs))


class CachedNotaGenPatchGenerator:
    """
    Prototype patch-level KV-cache wrapper for NotaGen rollout generation.

    This is intended for no-grad sampling only. It caches the patch-level GPT2
    keys/values across accepted patches, while leaving the char-level decoder
    behavior unchanged.
    """

    def __init__(self, model: torch.nn.Module, *, precision: str = "fp32", force_eval: bool = True):
        self.model = model
        self.precision = precision
        self.force_eval = force_eval
        self.state: CachedPatchState | None = None

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def eos_token_id(self) -> int:
        return int(self.model.eos_token_id)

    @property
    def bos_token_id(self) -> int:
        return int(self.model.bos_token_id)

    @property
    def special_token_id(self) -> int:
        return int(self.model.special_token_id)

    @contextmanager
    def _model_mode(self) -> Iterator[None]:
        was_training = bool(self.model.training)
        if self.force_eval:
            self.model.eval()
        try:
            yield
        finally:
            if self.force_eval:
                self.model.train(was_training)

    def _autocast_context(self):
        if self.device.type == "cuda" and self.precision == "bf16":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    def _patch_embeddings(self, patches: torch.Tensor) -> torch.Tensor:
        patch_decoder = self.model.patch_level_decoder
        embed_dtype = patch_decoder.patch_embedding.weight.dtype
        one_hot = torch.nn.functional.one_hot(patches, num_classes=ASCII_VOCAB_SIZE).to(embed_dtype)
        one_hot = one_hot.reshape(patches.shape[0], -1, PATCH_SIZE * ASCII_VOCAB_SIZE)
        return patch_decoder.patch_embedding(one_hot.to(self.device))

    def _run_patch_base(self, patches: torch.Tensor, *, past_key_values=None):
        embeds = self._patch_embeddings(patches)
        return self.model.patch_level_decoder.base(
            inputs_embeds=embeds,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )

    def reset(self, flat_ids: list[int]) -> CachedPatchState:
        flat_ids = [int(tok) for tok in flat_ids]
        rem = len(flat_ids) % PATCH_SIZE
        partial_ids = flat_ids[-rem:] if rem else []
        prefix_ids = flat_ids[:-rem] if rem else flat_ids
        if not prefix_ids:
            raise RuntimeError("NotaGen cache reset requires at least one full patch in the prefix")

        prefix_tensor = torch.tensor(prefix_ids, device=self.device, dtype=torch.long).reshape(1, -1, PATCH_SIZE)
        with torch.inference_mode(), self._model_mode(), self._autocast_context():
            outputs = self._run_patch_base(prefix_tensor)

        self.state = CachedPatchState(
            flat_ids=flat_ids,
            cached_patch_count=len(prefix_ids) // PATCH_SIZE,
            partial_ids=partial_ids,
            last_patch_hidden=outputs.last_hidden_state[0, -1].detach(),
            past_key_values=outputs.past_key_values,
        )
        return self.state

    def generate_patch(self, *, top_k: int = 0, top_p: float = 1.0, temperature: float = 1.0) -> list[int]:
        if self.state is None:
            raise RuntimeError("call reset(flat_ids) before generate_patch()")

        generated_patch: list[int] = []
        tokens = torch.tensor(
            [self.bos_token_id] + self.state.partial_ids,
            device=self.device,
            dtype=torch.long,
        )

        with torch.inference_mode(), self._model_mode():
            while True:
                with self._autocast_context():
                    probs = self.model.char_level_decoder.generate(self.state.last_patch_hidden, tokens)
                probs_np = _safe_normalize_probs(probs.float().detach().cpu().numpy())
                probs_np = _top_k_filter(probs_np, top_k=top_k)
                probs_np = _top_p_filter(probs_np, top_p=top_p)
                token = _temperature_sample(probs_np, temperature=temperature)
                generated_patch.append(token)

                if len(tokens) >= PATCH_SIZE:
                    break
                tokens = torch.cat((tokens, torch.tensor([token], device=self.device, dtype=torch.long)), dim=0)

        return generated_patch

    def accept_patch(self, patch: list[int]) -> CachedPatchState:
        if self.state is None:
            raise RuntimeError("call reset(flat_ids) before accept_patch()")

        normalized_patch = normalize_patch_for_context(
            patch,
            eos_token_id=self.eos_token_id,
            special_token_id=self.special_token_id,
        )
        completed_patch = self.state.partial_ids + normalized_patch
        if len(completed_patch) != PATCH_SIZE:
            raise RuntimeError(
                "accepted patch must complete exactly one 16-token NotaGen patch "
                f"(partial={len(self.state.partial_ids)}, new={len(normalized_patch)})"
            )

        patch_tensor = torch.tensor(completed_patch, device=self.device, dtype=torch.long).reshape(1, 1, PATCH_SIZE)
        with torch.inference_mode(), self._model_mode(), self._autocast_context():
            outputs = self._run_patch_base(patch_tensor, past_key_values=self.state.past_key_values)

        self.state = CachedPatchState(
            flat_ids=self.state.flat_ids + normalized_patch,
            cached_patch_count=self.state.cached_patch_count + 1,
            partial_ids=[],
            last_patch_hidden=outputs.last_hidden_state[0, -1].detach(),
            past_key_values=outputs.past_key_values,
        )
        return self.state
