---
name: sample-notagen-trajectories
description: Sample ABC continuations/trajectories from NotaGen checkpoints using the optimized cached patch-level rollout path. Use when Codex needs to generate NotaGen music trajectories quickly, compare cached vs uncached rollout behavior, sample from the Goldberg prompt, or sample from a GRPO LoRA checkpoint without doing a training update.
---

# Sample NotaGen Trajectories

Use the cached rollout path by default. The optimized implementation is `CachedNotaGenPatchGenerator` in `grpo/notagen_cached_generation.py`; it caches patch-level GPT keys/values and only runs the char decoder for the next patch. Do not call `model.generate(...)` for long rollouts unless explicitly comparing against the uncached baseline.

## Entry Points

For plain `.pth` checkpoints, use:

```bash
python scripts/generate_notagen_cached_inference.py \
  --weights /path/to/notagen_model.pth \
  --prefix data/processed/notagen/aria_conditioned_training_prefix_G.abc \
  --out-dir /path/to/out \
  --seeds 0 1 2 3 \
  --target-stream-lines 32 \
  --max-chars 24000 \
  --precision bf16
```

This writes one `.abc` per seed plus `notagen_large_rerun_cached_summary.json`.

For GRPO LoRA adapter checkpoints, use the GRPO script in no-step mode so the adapter is loaded before sampling:

```bash
python scripts/custom_grpo_notagen.py \
  --policy-weights /path/to/base_notagen_model.pth \
  --reference-weights /path/to/base_notagen_model.pth \
  --resume-checkpoint-dir /path/to/checkpoints/step_000123 \
  --prompts-jsonl data/processed/notagen/goldberg_grpo_prompt_aria_continuation_300.jsonl \
  --target-json data/processed/goldberg/structure/aria_bar_skeleton.json \
  --output-json /path/to/out/sampling.json \
  --trajectories-dir /path/to/out/trajectories \
  --prompt-limit 1 \
  --max-steps 1 \
  --group-size 4 \
  --target-stream-lines 32 \
  --max-chars 24000 \
  --timeout-s 300 \
  --cached-rollout \
  --no-step \
  --lora-r 8 \
  --lora-alpha 16 \
  --lora-dropout 0.05 \
  --precision bf16 \
  --seed 0
```

Use `--group-size` for the number of trajectories. `--no-step` prevents optimizer/backward/update and only samples/scores/dumps trajectories.

## Prompt Rules

Use the canonical Goldberg continuation prompt unless the user explicitly asks otherwise:

```text
data/processed/notagen/goldberg_grpo_prompt_aria_continuation_300.jsonl
```

This prompt includes the Aria plus the variation voice setup. Avoid older variation-labeled prompt files for fixed-prompt sampling.

## Rollout Details

- `sample_completion(...)` in `scripts/custom_grpo_notagen.py` is the GRPO trajectory sampler.
- With `--cached-rollout`, it constructs `CachedNotaGenPatchGenerator`, calls `reset(flat_ids)` once, then loops over `generate_patch(...)` and `accept_patch(...)`.
- It rejects early EOS until the countdown reaches zero and the current stream line is closed.
- It trims output to `--target-stream-lines` stream lines once the last line is closed.
- It records raw generated patches in the trajectory JSONL; keep those when the user may want logprob rescoring or debugging.

## Checks

After sampling, inspect:

- `trajectories/step_*/trajectories.jsonl` for rewards, `generated_patches`, `generated_token_slots`, and completion paths.
- `*.completion.abc` when the user wants only the continuation, not prompt plus continuation.
- `parse_valid`, `validated_bars`, `meter_alignment_reward`, `bar_count_reward`, `root_similarity_reward`, and `bass_pitch_class_reward` when judging quality.
