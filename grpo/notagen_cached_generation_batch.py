from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import numpy as np
import torch

from .notagen_cached_generation import (
    PATCH_SIZE,
    CachedPatchState,
    CachedNotaGenPatchGenerator,
    normalize_patch_for_context,
    _safe_normalize_probs,
    _top_k_filter,
    _top_p_filter,
)
from .notagen_wrapper import (
    PATCH_STREAM,
    Patchilizer,
    count_stream_lines,
    latest_stream_line_closed,
    split_metadata_and_tunebody_lines,
    trim_to_stream_lines,
)
from evaluation.stream_tags import stream_target_reached


@dataclass
class BatchedSampleResult:
    ok: bool
    full_text: str | None = None
    generated_patches: list[list[int]] | None = None
    meta: dict[str, Any] | None = None
    error: str | None = None


@dataclass
class _BatchContext:
    generator: CachedNotaGenPatchGenerator
    rng: np.random.Generator
    prompt_stream_lines: int
    target_total_stream_lines: int
    byte_list: list[str]
    generated_patches: list[list[int]]
    start_time: float
    cut_index: int | None
    resets: int


def _add_timing(timings: dict[str, float] | None, key: str, elapsed_s: float) -> None:
    if timings is not None:
        timings[key] = timings.get(key, 0.0) + float(elapsed_s)


def _inc_counter(counters: dict[str, int] | None, key: str, amount: int = 1) -> None:
    if counters is not None:
        counters[key] = counters.get(key, 0) + int(amount)


def _timing_payload(timings: dict[str, float], counters: dict[str, int], *, total_s: float) -> dict[str, Any]:
    payload: dict[str, Any] = {"batch_timing_total_s": round(float(total_s), 6)}
    for key, value in sorted(timings.items()):
        payload[f"batch_timing_{key}_s"] = round(float(value), 6)
    for key, value in sorted(counters.items()):
        payload[f"batch_counter_{key}"] = int(value)
    return payload


def _top_k_top_p_filter_logits(logits: torch.Tensor, *, top_k: int, top_p: float) -> torch.Tensor:
    logits = torch.nan_to_num(logits, nan=-1e9, posinf=1e9, neginf=-1e9)
    filtered = logits

    if 0 < top_k < filtered.shape[-1]:
        kth_values = torch.topk(filtered, top_k, dim=-1).values[..., -1, None]
        filtered = filtered.masked_fill(filtered < kth_values, -float("inf"))

    if 0 < top_p < 1:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True, dim=-1)
        sorted_probs = torch.nn.functional.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_remove = cumulative_probs > top_p
        sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
        sorted_remove[..., 0] = False
        remove = torch.zeros_like(sorted_remove, dtype=torch.bool).scatter(-1, sorted_indices, sorted_remove)
        filtered = filtered.masked_fill(remove, -float("inf"))

    return filtered


def _sample_from_logits(logits: torch.Tensor, *, top_k: int, top_p: float, temperature: float) -> torch.Tensor:
    logits = torch.nan_to_num(logits.float(), nan=-1e9, posinf=1e9, neginf=-1e9)

    if temperature <= 0:
        filtered_logits = _top_k_top_p_filter_logits(logits, top_k=top_k, top_p=top_p)
        return torch.argmax(filtered_logits, dim=-1)

    logits = logits / float(temperature)
    if 0 < top_k < logits.shape[-1]:
        top_values, top_indices = torch.topk(logits, top_k, dim=-1)
        probs = torch.nn.functional.softmax(top_values, dim=-1)
        if 0 < top_p < 1:
            cumulative_probs = torch.cumsum(probs, dim=-1)
            remove = cumulative_probs > top_p
            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = False
            probs = probs.masked_fill(remove, 0.0)
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        sampled_top = torch.multinomial(probs, num_samples=1)
        return top_indices.gather(-1, sampled_top).squeeze(-1)

    filtered_logits = _top_k_top_p_filter_logits(logits, top_k=top_k, top_p=top_p)
    probs = torch.nn.functional.softmax(filtered_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def _sample_chars_batch(
    *,
    generators: list[CachedNotaGenPatchGenerator],
    token_lists: list[list[int]],
    precision: str,
    top_k: int,
    top_p: float,
    temperature: float,
    timings: dict[str, float] | None = None,
    counters: dict[str, int] | None = None,
) -> list[int]:
    if not generators:
        return []

    t0 = time.perf_counter()
    device = generators[0].device
    hidden_batch = torch.stack([gen.state.last_patch_hidden for gen in generators], dim=0)
    max_len = max(len(tokens) for tokens in token_lists)
    token_tensor = torch.full(
        (len(token_lists), max_len),
        fill_value=generators[0].special_token_id,
        device=device,
        dtype=torch.long,
    )
    attention_mask = torch.zeros((len(token_lists), max_len), device=device, dtype=torch.long)
    last_positions: list[int] = []
    for row_idx, tokens in enumerate(token_lists):
        token_tensor[row_idx, : len(tokens)] = torch.tensor(tokens, device=device, dtype=torch.long)
        attention_mask[row_idx, : len(tokens)] = 1
        last_positions.append(len(tokens) - 1)

    token_embeds = torch.nn.functional.embedding(token_tensor, generators[0].model.char_level_decoder.base.transformer.wte.weight)
    inputs_embeds = torch.cat((hidden_batch.unsqueeze(1), token_embeds[:, 1:, :]), dim=1)
    _add_timing(timings, "char_batch_prepare", time.perf_counter() - t0)

    t0 = time.perf_counter()
    autocast = generators[0]._autocast_context()
    with torch.inference_mode(), generators[0]._model_mode(), autocast:
        outputs = generators[0].model.char_level_decoder.base(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
    _add_timing(timings, "char_forward", time.perf_counter() - t0)
    _inc_counter(counters, "char_forward_batches")
    _inc_counter(counters, "char_forward_items", len(token_lists))

    t0 = time.perf_counter()
    logits = outputs.logits
    next_logits = torch.stack([logits[row_idx, last_pos] for row_idx, last_pos in enumerate(last_positions)], dim=0).float()
    sampled = _sample_from_logits(next_logits, top_k=top_k, top_p=top_p, temperature=temperature)
    _add_timing(timings, "char_torch_filter_sample", time.perf_counter() - t0)

    t0 = time.perf_counter()
    tokens = [int(token) for token in sampled.detach().cpu().tolist()]
    _add_timing(timings, "char_tokens_to_cpu", time.perf_counter() - t0)
    return tokens


def _generate_candidate_patches_batch(
    *,
    contexts: list[_BatchContext],
    top_k: int,
    top_p: float,
    temperature: float,
    timings: dict[str, float] | None = None,
    counters: dict[str, int] | None = None,
) -> list[list[int]]:
    token_lists = [[ctx.generator.bos_token_id] + list(ctx.generator.state.partial_ids) for ctx in contexts]
    generated = [[] for _ in contexts]
    active = list(range(len(contexts)))

    while active:
        batch_gens = [contexts[i].generator for i in active]
        batch_tokens = [token_lists[i] for i in active]
        batch_tokens_sampled = _sample_chars_batch(
            generators=batch_gens,
            token_lists=batch_tokens,
            precision=batch_gens[0].precision,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            timings=timings,
            counters=counters,
        )

        next_active: list[int] = []
        t0 = time.perf_counter()
        for local_idx, token in enumerate(batch_tokens_sampled):
            ctx_idx = active[local_idx]
            generated[ctx_idx].append(token)
            _inc_counter(counters, "sampled_char_tokens")

            if len(token_lists[ctx_idx]) >= PATCH_SIZE:
                continue

            token_lists[ctx_idx].append(token)
            next_active.append(ctx_idx)
        _add_timing(timings, "char_filter_sample", time.perf_counter() - t0)
        active = next_active

    return generated


def _maybe_roll_cache(ctx: _BatchContext, *, model_shape, patchilizer: Patchilizer) -> None:
    state = ctx.generator.state
    if state is None or len(state.flat_ids) < model_shape.patch_length * PATCH_SIZE:
        return

    current_text = "".join(ctx.byte_list)
    metadata_lines, tunebody_lines = split_metadata_and_tunebody_lines(current_text)
    if not tunebody_lines:
        raise RuntimeError("stream rollover hit before tunebody generation")
    if ctx.cut_index is None:
        ctx.cut_index = max(1, len(tunebody_lines) // 2)
    abc_slice = "".join(metadata_lines + tunebody_lines[-ctx.cut_index :])
    repatched = patchilizer.encode_generate(abc_slice)
    flat_ids = [int(item) for sublist in repatched for item in sublist]
    ctx.generator.reset(flat_ids)
    ctx.resets += 1


def _past_cache_layers(past_key_values: Any) -> list[tuple[torch.Tensor, torch.Tensor]] | None:
    if hasattr(past_key_values, "to_legacy_cache"):
        past_key_values = past_key_values.to_legacy_cache()
    if hasattr(past_key_values, "layers"):
        layers = []
        for layer in past_key_values.layers:
            keys = getattr(layer, "keys", None)
            values = getattr(layer, "values", None)
            if not torch.is_tensor(keys) or not torch.is_tensor(values):
                return None
            layers.append((keys, values))
        return layers
    if isinstance(past_key_values, (tuple, list)):
        layers = []
        for layer in past_key_values:
            if not isinstance(layer, (tuple, list)) or len(layer) < 2:
                return None
            keys, values = layer[0], layer[1]
            if not torch.is_tensor(keys) or not torch.is_tensor(values):
                return None
            layers.append((keys, values))
        return layers
    return None


def _past_cache_batch_size(past_key_values: Any) -> int | None:
    layers = _past_cache_layers(past_key_values)
    if not layers:
        return None
    first_tensor = layers[0][0]
    if not torch.is_tensor(first_tensor) or first_tensor.ndim == 0:
        return None
    return int(first_tensor.shape[0])


def _stack_past_key_values(past_key_values: list[Any], *, template: Any) -> Any | None:
    if not past_key_values:
        return None
    layer_lists = [_past_cache_layers(past) for past in past_key_values]
    if any(layers is None for layers in layer_lists):
        return None
    typed_layer_lists: list[list[tuple[torch.Tensor, torch.Tensor]]] = [layers for layers in layer_lists if layers is not None]
    if not typed_layer_lists:
        return None
    layer_count = len(typed_layer_lists[0])
    if any(len(layers) != layer_count for layers in typed_layer_lists):
        return None

    stacked_layers = []
    for layer_idx in range(layer_count):
        try:
            keys = torch.cat([layers[layer_idx][0] for layers in typed_layer_lists], dim=0)
            values = torch.cat([layers[layer_idx][1] for layers in typed_layer_lists], dim=0)
        except Exception:
            return None
        stacked_layers.append((keys, values))
    if hasattr(template, "layers"):
        return _build_dynamic_cache(stacked_layers, template=template)
    return tuple(stacked_layers)


def _build_dynamic_cache(layers: list[tuple[torch.Tensor, torch.Tensor]], *, template: Any) -> Any | None:
    try:
        from transformers.cache_utils import DynamicCache
    except Exception:
        return None

    try:
        config = getattr(template, "config", None)
        cache = DynamicCache(config=config)
        for layer_idx, (keys, values) in enumerate(layers):
            cache.update(keys, values, layer_idx)
        return cache
    except Exception:
        return None


def _slice_past_key_values(past_key_values: Any, row_idx: int, *, template: Any) -> Any | None:
    layers = _past_cache_layers(past_key_values)
    if layers is None:
        return None

    sliced_layers = [(keys[row_idx : row_idx + 1].detach(), values[row_idx : row_idx + 1].detach()) for keys, values in layers]
    if hasattr(template, "layers"):
        rebuilt = _build_dynamic_cache(sliced_layers, template=template)
        if rebuilt is None:
            return None
        return rebuilt
    return tuple(sliced_layers)


def _accept_patches_one_by_one(contexts_and_patches: list[tuple[_BatchContext, list[int]]]) -> None:
    for ctx, predicted_patch in contexts_and_patches:
        ctx.generator.accept_patch(predicted_patch)


def _accept_patches_batch(
    contexts_and_patches: list[tuple[_BatchContext, list[int]]],
    *,
    precision: str,
    counters: dict[str, int] | None = None,
) -> None:
    if not contexts_and_patches:
        return

    grouped: dict[int, list[tuple[_BatchContext, list[int], list[int]]]] = {}
    fallback: list[tuple[_BatchContext, list[int]]] = []

    for ctx, predicted_patch in contexts_and_patches:
        state = ctx.generator.state
        if state is None:
            raise RuntimeError("call reset(flat_ids) before accepting patches")

        normalized_patch = normalize_patch_for_context(
            predicted_patch,
            eos_token_id=ctx.generator.eos_token_id,
            special_token_id=ctx.generator.special_token_id,
        )
        completed_patch = state.partial_ids + normalized_patch
        if len(completed_patch) != PATCH_SIZE:
            raise RuntimeError(
                "accepted patch must complete exactly one 16-token NotaGen patch "
                f"(partial={len(state.partial_ids)}, new={len(normalized_patch)})"
            )

        if _past_cache_batch_size(state.past_key_values) != 1:
            fallback.append((ctx, predicted_patch))
            continue
        grouped.setdefault(state.cached_patch_count, []).append((ctx, normalized_patch, completed_patch))

    for group in grouped.values():
        if len(group) == 1:
            ctx, normalized_patch, _ = group[0]
            fallback.append((ctx, normalized_patch))
            continue

        generators = [ctx.generator for ctx, _, _ in group]
        first_generator = generators[0]
        past = _stack_past_key_values(
            [ctx.generator.state.past_key_values for ctx, _, _ in group],
            template=group[0][0].generator.state.past_key_values,
        )
        if past is None:
            fallback.extend((ctx, normalized_patch) for ctx, normalized_patch, _ in group)
            continue

        _inc_counter(counters, "patch_accept_batched_groups")
        _inc_counter(counters, "patch_accept_batched_items", len(group))
        patch_tensor = torch.tensor(
            [completed_patch for _, _, completed_patch in group],
            device=first_generator.device,
            dtype=torch.long,
        ).reshape(len(group), 1, PATCH_SIZE)
        with torch.inference_mode(), first_generator._model_mode(), first_generator._autocast_context():
            outputs = first_generator._run_patch_base(patch_tensor, past_key_values=past)

        for row_idx, (ctx, normalized_patch, _) in enumerate(group):
            state = ctx.generator.state
            if state is None:
                raise RuntimeError("call reset(flat_ids) before accepting patches")
            sliced_past = _slice_past_key_values(outputs.past_key_values, row_idx, template=state.past_key_values)
            if sliced_past is None:
                fallback.append((ctx, normalized_patch))
                continue
            ctx.generator.state = CachedPatchState(
                flat_ids=state.flat_ids + normalized_patch,
                cached_patch_count=state.cached_patch_count + 1,
                partial_ids=[],
                last_patch_hidden=outputs.last_hidden_state[row_idx, -1].detach(),
                past_key_values=sliced_past,
            )

    if fallback:
        _inc_counter(counters, "patch_accept_fallback_items", len(fallback))
        _accept_patches_one_by_one(fallback)


def sample_completions_cached_batch(
    *,
    model,
    model_shape,
    prompts: list[str],
    seeds: list[int],
    temperature: float,
    top_k: int,
    top_p: float,
    target_stream_lines: int,
    target_new_stream_lines: bool,
    max_chars: int,
    max_generated_patches: int,
    timeout_s: int,
    precision: str,
) -> list[BatchedSampleResult]:
    if len(prompts) != len(seeds):
        raise ValueError("prompts and seeds must have the same length")

    patchilizer = Patchilizer(stream=PATCH_STREAM)
    contexts: list[_BatchContext] = []
    results: list[BatchedSampleResult | None] = [None] * len(prompts)
    batch_start_time = time.perf_counter()
    timings: dict[str, float] = {}
    counters: dict[str, int] = {}

    for prompt, seed in zip(prompts, seeds, strict=True):
        t0 = time.perf_counter()
        prefix = prompt
        if count_stream_lines(prefix) == 0:
            prefix = prefix + f"[r:0/{target_stream_lines - 1}]"
        prompt_stream_lines = count_stream_lines(prefix)
        target_total_stream_lines = (
            prompt_stream_lines + target_stream_lines
            if target_new_stream_lines
            else target_stream_lines
        )
        input_patches = patchilizer.encode_generate(prefix)
        flat_ids = [int(item) for sublist in input_patches for item in sublist]
        _add_timing(timings, "prompt_encode", time.perf_counter() - t0)

        t0 = time.perf_counter()
        generator = CachedNotaGenPatchGenerator(model, precision=precision)
        generator.reset(flat_ids)
        _add_timing(timings, "prompt_cache_reset", time.perf_counter() - t0)
        contexts.append(
            _BatchContext(
                generator=generator,
                rng=np.random.default_rng(seed),
                prompt_stream_lines=prompt_stream_lines,
                target_total_stream_lines=target_total_stream_lines,
                byte_list=list(prefix),
                generated_patches=[],
                start_time=time.time(),
                cut_index=None,
                resets=1,
            )
        )

    active = list(range(len(contexts)))
    while active:
        _inc_counter(counters, "outer_loops")
        pending = [contexts[i] for i in active]
        chosen_patches: list[list[int] | None] = [None] * len(pending)
        unresolved = list(range(len(pending)))

        for _ in range(8):
            if not unresolved:
                break
            t0 = time.perf_counter()
            candidate_patches = _generate_candidate_patches_batch(
                contexts=[pending[i] for i in unresolved],
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                timings=timings,
                counters=counters,
            )
            _add_timing(timings, "candidate_patch_generate", time.perf_counter() - t0)
            next_unresolved: list[int] = []
            t0 = time.perf_counter()
            for unresolved_pos, candidate_patch in zip(unresolved, candidate_patches, strict=True):
                ctx = pending[unresolved_pos]
                current_text = "".join(ctx.byte_list)
                eos_only = (
                    len(candidate_patch) >= 2
                    and candidate_patch[0] == patchilizer.bos_token_id
                    and candidate_patch[1] == patchilizer.eos_token_id
                )
                if eos_only and not stream_target_reached(current_text, ctx.target_total_stream_lines):
                    next_unresolved.append(unresolved_pos)
                    continue
                chosen_patches[unresolved_pos] = candidate_patch
            unresolved = next_unresolved
            _add_timing(timings, "candidate_select", time.perf_counter() - t0)
            _inc_counter(counters, "candidate_patch_attempts", len(candidate_patches))

        if unresolved:
            for unresolved_pos in unresolved:
                results[active[unresolved_pos]] = BatchedSampleResult(
                    ok=False,
                    error="decoder produced only early EOS candidates before target stream line completion",
                )
            active = [idx for pos, idx in enumerate(active) if pos not in set(unresolved)]
            continue

        next_active: list[int] = []
        accepted: list[tuple[_BatchContext, list[int]]] = []
        for local_idx, sample_idx in enumerate(active):
            ctx = contexts[sample_idx]
            predicted_patch = chosen_patches[local_idx]
            if predicted_patch is None:
                results[sample_idx] = BatchedSampleResult(ok=False, error="internal error: missing predicted patch")
                continue

            if (
                len(predicted_patch) >= 2
                and predicted_patch[0] == patchilizer.bos_token_id
                and predicted_patch[1] == patchilizer.eos_token_id
            ):
                results[sample_idx] = BatchedSampleResult(
                    ok=True,
                    full_text="".join(ctx.byte_list),
                    generated_patches=ctx.generated_patches,
                    meta={
                        "stop_reason": "terminal_eos",
                        "cache_resets": ctx.resets,
                        "prompt_stream_lines": ctx.prompt_stream_lines,
                        "target_total_stream_lines": ctx.target_total_stream_lines,
                    },
                )
                continue

            t0 = time.perf_counter()
            ctx.generated_patches.append(predicted_patch[:])
            ctx.byte_list.extend(patchilizer.decode([predicted_patch]))
            _add_timing(timings, "patch_decode_append", time.perf_counter() - t0)
            _inc_counter(counters, "accepted_patches")
            accepted.append((ctx, predicted_patch))

        t0 = time.perf_counter()
        _accept_patches_batch(accepted, precision=precision, counters=counters)
        _add_timing(timings, "patch_accept", time.perf_counter() - t0)

        for local_idx, sample_idx in enumerate(active):
            ctx = contexts[sample_idx]
            predicted_patch = chosen_patches[local_idx]
            if predicted_patch is None:
                continue
            if (
                len(predicted_patch) >= 2
                and predicted_patch[0] == patchilizer.bos_token_id
                and predicted_patch[1] == patchilizer.eos_token_id
            ):
                continue
            if results[sample_idx] is not None:
                continue

            t0 = time.perf_counter()
            current_text = "".join(ctx.byte_list)

            if count_stream_lines(current_text) >= ctx.target_total_stream_lines and latest_stream_line_closed(current_text):
                results[sample_idx] = BatchedSampleResult(
                    ok=True,
                    full_text=trim_to_stream_lines(current_text, ctx.target_total_stream_lines),
                    generated_patches=ctx.generated_patches,
                    meta={
                        "stop_reason": "target_stream_lines",
                        "cache_resets": ctx.resets,
                        "prompt_stream_lines": ctx.prompt_stream_lines,
                        "target_total_stream_lines": ctx.target_total_stream_lines,
                    },
                )
                _add_timing(timings, "stop_check", time.perf_counter() - t0)
                continue

            if max_generated_patches > 0 and len(ctx.generated_patches) >= max_generated_patches:
                results[sample_idx] = BatchedSampleResult(
                    ok=True,
                    full_text=current_text,
                    generated_patches=ctx.generated_patches,
                    meta={
                        "stop_reason": "max_generated_patches",
                        "cache_resets": ctx.resets,
                        "prompt_stream_lines": ctx.prompt_stream_lines,
                        "target_total_stream_lines": ctx.target_total_stream_lines,
                    },
                )
                _add_timing(timings, "stop_check", time.perf_counter() - t0)
                continue

            if len(ctx.byte_list) > max_chars:
                results[sample_idx] = BatchedSampleResult(
                    ok=True,
                    full_text=current_text,
                    generated_patches=ctx.generated_patches,
                    meta={
                        "stop_reason": "max_chars",
                        "cache_resets": ctx.resets,
                        "prompt_stream_lines": ctx.prompt_stream_lines,
                        "target_total_stream_lines": ctx.target_total_stream_lines,
                    },
                )
                _add_timing(timings, "stop_check", time.perf_counter() - t0)
                continue

            if time.time() - ctx.start_time > timeout_s:
                results[sample_idx] = BatchedSampleResult(ok=False, error=f"generation exceeded {timeout_s}s")
                _add_timing(timings, "stop_check", time.perf_counter() - t0)
                continue

            _add_timing(timings, "stop_check", time.perf_counter() - t0)
            t0 = time.perf_counter()
            _maybe_roll_cache(ctx, model_shape=model_shape, patchilizer=patchilizer)
            _add_timing(timings, "cache_rollover", time.perf_counter() - t0)
            next_active.append(sample_idx)

        active = next_active

    timing_meta = _timing_payload(timings, counters, total_s=time.perf_counter() - batch_start_time)
    finalized: list[BatchedSampleResult] = []
    for row in results:
        if row is None:
            finalized.append(BatchedSampleResult(ok=False, error="internal error: missing batch result"))
        else:
            if row.meta is not None:
                row.meta.update(timing_meta)
            finalized.append(row)
    return finalized
