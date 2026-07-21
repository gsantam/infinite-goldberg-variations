from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch


PATCH_SIZE = 16


def autocast_context(device: torch.device, precision: str):
    if device.type == "cuda" and precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def normalize_patch_for_context(patch: list[int], eos_token_id: int, special_token_id: int) -> list[int]:
    out = patch[:]
    patch_end = False
    for i, tok in enumerate(out):
        if patch_end:
            out[i] = special_token_id
        if tok == eos_token_id:
            patch_end = True
    return out


def _encoded_last_patch(
    model: Any,
    flat_ids: list[int],
    device: torch.device,
    precision: str,
    replay_context_patches: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if replay_context_patches is not None and replay_context_patches > 0:
        max_tokens = replay_context_patches * PATCH_SIZE
        if len(flat_ids) > max_tokens:
            flat_ids = flat_ids[-max_tokens:]
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


def _replay_start_patch(total_patches: int, context_patch_count: int, replay_context_patches: int | None) -> int:
    if replay_context_patches is None or replay_context_patches <= 0:
        return 0
    target_start_patch = context_patch_count
    min_start = max(0, target_start_patch - 1)
    start_patch = max(0, total_patches - replay_context_patches)
    return min(start_patch, min_start)


def patch_logprobs(
    model: Any,
    flat_prompt_ids: list[int],
    generated_patches: list[list[int]],
    precision: str,
    replay_context_patches: int | None = None,
) -> list[torch.Tensor]:
    device = next(model.parameters()).device
    current_ids = list(flat_prompt_ids)
    all_logprobs: list[torch.Tensor] = []

    for patch in generated_patches:
        encoded_patch, tokens = _encoded_last_patch(
            model,
            current_ids,
            device,
            precision,
            replay_context_patches=replay_context_patches,
        )
        for tok in patch:
            token_embeddings = torch.nn.functional.embedding(
                tokens.reshape(1, -1),
                model.char_level_decoder.base.transformer.wte.weight,
            )
            inputs_embeds = torch.cat((encoded_patch.reshape(1, 1, -1), token_embeddings[:, 1:, :]), dim=1)
            with autocast_context(device, precision):
                outputs = model.char_level_decoder.base(inputs_embeds=inputs_embeds)
                logits = outputs.logits[0, -1]
            logprob = torch.log_softmax(logits.float(), dim=-1)[tok]
            all_logprobs.append(logprob)
            if len(tokens) >= PATCH_SIZE:
                break
            tokens = torch.cat((tokens, torch.tensor([tok], device=device, dtype=torch.long)), dim=0)

        current_ids.extend(
            normalize_patch_for_context(
                patch,
                eos_token_id=model.eos_token_id,
                special_token_id=model.special_token_id,
            )
        )

    return all_logprobs


def char_patch_logprobs(
    model: Any,
    encoded_patches: torch.Tensor,
    target_patches: torch.Tensor,
    precision: str,
) -> torch.Tensor:
    bos = torch.ones_like(target_patches[:, 0:1]) * model.bos_token_id
    target_with_bos = torch.cat((bos, target_patches), dim=1)
    target_masks = target_with_bos == model.special_token_id
    labels = target_with_bos.clone().masked_fill_(target_masks, -100)
    target_masks = torch.ones_like(labels).masked_fill_(labels == -100, 0)

    input_embeds = torch.nn.functional.embedding(target_with_bos, model.char_level_decoder.base.transformer.wte.weight)
    input_embeds = torch.cat((encoded_patches.unsqueeze(1), input_embeds[:, 1:, :]), dim=1)
    with autocast_context(encoded_patches.device, precision):
        logits = model.char_level_decoder.base(inputs_embeds=input_embeds, attention_mask=target_masks).logits
    logits = logits[:, :-1, :].float()
    token_logps = torch.gather(logits.log_softmax(-1), dim=-1, index=target_with_bos[:, 1:].unsqueeze(-1)).squeeze(-1)
    return token_logps[target_masks[:, 1:] == 1]


def _pad_generated_patch(patch: list[int], special_token_id: int) -> list[int]:
    if len(patch) > PATCH_SIZE:
        raise RuntimeError(f"generated patch is longer than {PATCH_SIZE}: {len(patch)}")
    return patch + [special_token_id] * (PATCH_SIZE - len(patch))


def _split_flat_logprobs(flat_logprobs: torch.Tensor, token_counts: list[int]) -> list[torch.Tensor]:
    out: list[torch.Tensor] = []
    offset = 0
    for count in token_counts:
        out.append(flat_logprobs[offset : offset + count])
        offset += count
    if offset != flat_logprobs.numel():
        raise RuntimeError(f"batched logprob split mismatch: consumed {offset}, got {flat_logprobs.numel()}")
    return out


def split_tensor_by_counts(tensor: torch.Tensor, counts: list[int]) -> list[torch.Tensor]:
    output: list[torch.Tensor] = []
    offset = 0
    for count in counts:
        output.append(tensor[offset : offset + count])
        offset += count
    if offset != tensor.shape[0]:
        raise RuntimeError(f"batched replay split mismatch: consumed {offset}, got {tensor.shape[0]}")
    return output


def char_patch_logprob_sums(
    model: Any,
    encoded_patches: torch.Tensor,
    target_patches: list[list[int]],
    precision: str,
) -> torch.Tensor:
    special_token_id = model.special_token_id
    target_tensor = torch.tensor(
        [_pad_generated_patch(patch, special_token_id) for patch in target_patches],
        device=encoded_patches.device,
        dtype=torch.long,
    )
    token_counts = [sum(1 for token in patch if token != special_token_id) for patch in target_tensor.tolist()]
    flat_logprobs = char_patch_logprobs(model, encoded_patches, target_tensor, precision)
    per_patch = _split_flat_logprobs(flat_logprobs, token_counts)
    return torch.stack(
        [
            item.sum() if item.numel() > 0 else torch.zeros((), device=encoded_patches.device, dtype=flat_logprobs.dtype)
            for item in per_patch
        ]
    )


def tail_encoded_targets(
    *,
    model: Any,
    current_ids: list[int],
    remaining_patches: list[list[int]],
    chunk_start: int,
    chunk_end: int,
    precision: str,
    replay_context_patches: int | None = None,
    detach_policy: bool = False,
    error_context: str = "trajectory replay",
) -> tuple[torch.Tensor, list[list[int]]]:
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
        raise RuntimeError(f"{error_context} expected full-patch alignment before chunked tail scoring")

    total_patches = len(all_ids) // PATCH_SIZE
    context_patch_count = len(current_ids) // PATCH_SIZE
    start_patch = _replay_start_patch(total_patches, context_patch_count, replay_context_patches)

    trimmed_ids = all_ids[start_patch * PATCH_SIZE :]
    device = next(model.parameters()).device
    patches_tensor = torch.tensor(trimmed_ids, device=device, dtype=torch.long).reshape(1, -1, PATCH_SIZE)
    first_target_local = context_patch_count - start_patch
    if first_target_local <= 0:
        raise RuntimeError(f"{error_context} window dropped all context before generated target patches")

    context = torch.no_grad() if detach_policy else nullcontext()
    with context:
        with autocast_context(device, precision):
            encoded = model.patch_level_decoder(patches_tensor)["last_hidden_state"][0]
        encoded_start = first_target_local + chunk_start - 1
        encoded_end = first_target_local + chunk_end - 1
        encoded_targets = encoded[encoded_start:encoded_end]
    if detach_policy:
        encoded_targets = encoded_targets.detach()
    return encoded_targets, remaining_patches[chunk_start:chunk_end]


def batched_tail_encoded_targets(
    *,
    model: Any,
    current_ids_batch: list[list[int]],
    remaining_patches_batch: list[list[list[int]]],
    chunk_start: int,
    target_chunk_patches: int,
    precision: str,
    replay_context_patches: int | None = None,
    replay_batch_size: int = 0,
    detach_policy: bool = False,
    error_context: str = "batched replay",
) -> dict[int, tuple[torch.Tensor, list[list[int]]]]:
    device = next(model.parameters()).device
    special_token_id = model.special_token_id
    active: list[tuple[int, int, int, int, list[list[int]]]] = []

    for sample_idx, (current_ids, remaining_patches) in enumerate(zip(current_ids_batch, remaining_patches_batch, strict=True)):
        if chunk_start >= len(remaining_patches):
            continue
        chunk_end = len(remaining_patches)
        if target_chunk_patches > 0:
            chunk_end = min(chunk_end, chunk_start + target_chunk_patches)

        normalized_prefix = [
            normalize_patch_for_context(
                patch,
                eos_token_id=model.eos_token_id,
                special_token_id=special_token_id,
            )
            for patch in remaining_patches[:chunk_end]
        ]
        all_ids = current_ids[:]
        for patch in normalized_prefix:
            all_ids.extend(patch)
        if len(all_ids) % PATCH_SIZE != 0:
            raise RuntimeError(f"{error_context} expected full-patch alignment before tail scoring")

        total_patches = len(all_ids) // PATCH_SIZE
        context_patch_count = len(current_ids) // PATCH_SIZE
        start_patch = _replay_start_patch(total_patches, context_patch_count, replay_context_patches)
        trimmed_ids = all_ids[start_patch * PATCH_SIZE :]
        trimmed_patches = [
            trimmed_ids[i : i + PATCH_SIZE]
            for i in range(0, len(trimmed_ids), PATCH_SIZE)
        ]
        first_target_local = context_patch_count - start_patch
        if first_target_local <= 0:
            raise RuntimeError(f"{error_context} window dropped all context before generated target patches")

        encoded_start = first_target_local + chunk_start - 1
        encoded_end = first_target_local + chunk_end - 1
        active.append((sample_idx, encoded_start, encoded_end, chunk_end - chunk_start, trimmed_patches))

    if not active:
        return {}

    batch_size = len(active) if replay_batch_size <= 0 else replay_batch_size
    result: dict[int, tuple[torch.Tensor, list[list[int]]]] = {}
    policy_context = torch.no_grad() if detach_policy else nullcontext()

    for batch_start in range(0, len(active), batch_size):
        active_batch = active[batch_start : batch_start + batch_size]
        max_seq_patches = max(len(trimmed_patches) for *_unused, trimmed_patches in active_batch)
        patch_rows: list[list[list[int]]] = []
        for _sample_idx, _encoded_start, _encoded_end, _patch_count, trimmed_patches in active_batch:
            pad_patch_count = max_seq_patches - len(trimmed_patches)
            patch_rows.append(trimmed_patches + [[special_token_id] * PATCH_SIZE for _idx in range(pad_patch_count)])

        patches_tensor = torch.tensor(patch_rows, device=device, dtype=torch.long)
        with policy_context:
            with autocast_context(device, precision):
                encoded_batch = model.patch_level_decoder(patches_tensor)["last_hidden_state"]
        if detach_policy:
            encoded_batch = encoded_batch.detach()

        for active_idx, (sample_idx, encoded_start, encoded_end, patch_count, _trimmed_patches) in enumerate(active_batch):
            chunk_end = chunk_start + patch_count
            target_patches = remaining_patches_batch[sample_idx][chunk_start:chunk_end]
            result[sample_idx] = (encoded_batch[active_idx, encoded_start:encoded_end], target_patches)

    return result


def batched_tail_logprobs_chunk(
    model: Any,
    current_ids_batch: list[list[int]],
    remaining_patches_batch: list[list[list[int]]],
    chunk_start: int,
    target_chunk_patches: int,
    precision: str,
    replay_context_patches: int | None = None,
    replay_batch_size: int = 0,
) -> dict[int, torch.Tensor]:
    payload = batched_tail_encoded_targets(
        model=model,
        current_ids_batch=current_ids_batch,
        remaining_patches_batch=remaining_patches_batch,
        chunk_start=chunk_start,
        target_chunk_patches=target_chunk_patches,
        precision=precision,
        replay_context_patches=replay_context_patches,
        replay_batch_size=replay_batch_size,
        error_context="batched replay",
    )
    if not payload:
        return {}

    sample_indices: list[int] = []
    encoded_targets: list[torch.Tensor] = []
    target_rows: list[list[int]] = []
    token_counts: list[int] = []
    for sample_idx, (encoded, targets) in payload.items():
        sample_indices.append(sample_idx)
        encoded_targets.append(encoded)
        for patch in targets:
            target_rows.append(_pad_generated_patch(patch, model.special_token_id))
        token_counts.append(
            sum(1 for patch in targets for token in patch if token != model.special_token_id)
        )

    encoded_target_tensor = torch.cat(encoded_targets, dim=0)
    target_tensor = torch.tensor(target_rows, device=encoded_target_tensor.device, dtype=torch.long)
    flat_logprobs = char_patch_logprobs(model, encoded_target_tensor, target_tensor, precision)
    split_logprobs = _split_flat_logprobs(flat_logprobs, token_counts)
    return dict(zip(sample_indices, split_logprobs, strict=True))


def batched_trajectory_logprobs(
    model: Any,
    flat_prompt_ids: list[int],
    generated_patches_batch: list[list[list[int]]],
    precision: str,
    replay_context_patches: int | None = None,
    target_chunk_patches: int = 0,
    replay_batch_size: int = 0,
) -> list[torch.Tensor]:
    current_ids_batch = [list(flat_prompt_ids) for _ in generated_patches_batch]
    remaining_batch: list[list[list[int]]] = []
    prefix_logprobs: list[list[torch.Tensor]] = [[] for _ in generated_patches_batch]

    for sample_idx, generated_patches in enumerate(generated_patches_batch):
        current_ids = current_ids_batch[sample_idx]
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
                prefix_logprobs[sample_idx].append(torch.stack(logprob_list))
            current_ids.extend(
                normalize_patch_for_context(
                    patch,
                    eos_token_id=model.eos_token_id,
                    special_token_id=model.special_token_id,
                )
            )
            start_idx += 1
        current_ids_batch[sample_idx] = current_ids
        remaining_batch.append(generated_patches[start_idx:])

    outputs: list[list[torch.Tensor]] = [chunks[:] for chunks in prefix_logprobs]
    max_remaining = max((len(remaining) for remaining in remaining_batch), default=0)
    chunk_size = max_remaining if target_chunk_patches <= 0 else target_chunk_patches
    if chunk_size > 0:
        for chunk_start in range(0, max_remaining, chunk_size):
            chunk_payload = batched_tail_logprobs_chunk(
                model,
                current_ids_batch,
                remaining_batch,
                chunk_start,
                target_chunk_patches,
                precision,
                replay_context_patches=replay_context_patches,
                replay_batch_size=replay_batch_size,
            )
            for sample_idx, logprobs in chunk_payload.items():
                if logprobs.numel() > 0:
                    outputs[sample_idx].append(logprobs)

    result: list[torch.Tensor] = []
    device = next(model.parameters()).device
    for chunks in outputs:
        if chunks:
            result.append(torch.cat(chunks))
        else:
            result.append(torch.empty(0, device=device))
    return result


def tail_logprobs_chunk(
    model: Any,
    current_ids: list[int],
    remaining_patches: list[list[int]],
    chunk_start: int,
    chunk_end: int,
    precision: str,
    replay_context_patches: int | None = None,
) -> torch.Tensor:
    encoded_targets, target_patches = tail_encoded_targets(
        model=model,
        current_ids=current_ids,
        remaining_patches=remaining_patches,
        chunk_start=chunk_start,
        chunk_end=chunk_end,
        precision=precision,
        replay_context_patches=replay_context_patches,
        error_context="trajectory replay",
    )
    target_tensor = torch.tensor(
        [_pad_generated_patch(patch, model.special_token_id) for patch in target_patches],
        device=encoded_targets.device,
        dtype=torch.long,
    )
    return char_patch_logprobs(model, encoded_targets, target_tensor, precision)


def trajectory_logprob_chunks(
    model: Any,
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
        yield tail_logprobs_chunk(
            model,
            current_ids,
            remaining_patches,
            chunk_start,
            chunk_end,
            precision,
            replay_context_patches=replay_context_patches,
        )


def trajectory_logprob_forward_count(
    model: Any,
    flat_prompt_ids: list[int],
    generated_patches: list[list[int]],
    precision: str,
    replay_context_patches: int | None = None,
    target_chunk_patches: int = 0,
) -> int:
    device = next(model.parameters()).device
    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=cuda_devices), torch.no_grad():
        return sum(
            int(chunk.numel())
            for chunk in trajectory_logprob_chunks(
                model,
                flat_prompt_ids,
                generated_patches,
                precision,
                replay_context_patches=replay_context_patches,
                target_chunk_patches=target_chunk_patches,
            )
        )


def trajectory_logprobs(
    model: Any,
    flat_prompt_ids: list[int],
    generated_patches: list[list[int]],
    precision: str,
    replay_context_patches: int | None = None,
) -> list[torch.Tensor]:
    chunks = list(
        trajectory_logprob_chunks(
            model,
            flat_prompt_ids,
            generated_patches,
            precision,
            replay_context_patches=replay_context_patches,
            target_chunk_patches=0,
        )
    )
    if not chunks:
        return []
    return list(torch.cat(chunks).unbind())
