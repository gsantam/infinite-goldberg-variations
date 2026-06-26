---
name: grpo-jarvis
description: Run the current NotaGen GRPO experiments on JarvisLabs from the infinite-goldberg-variations repo. Use when launching, resuming, monitoring, or pausing the remote H100 run for the custom GRPO loop.
---

# GRPO on Jarvis

Use this skill when you need to launch the current custom NotaGen GRPO experiment on JarvisLabs.

## Preconditions

- Local repo root: `/Users/donguille/projects/infinite-goldberg-variations`
- NotaGen repo root: `/Users/donguille/projects/NotaGen`
- Jarvis CLI authenticated
- Current remote target is the Jarvis H100 VM

## Current model and script

- Training script:
  `/home/jl_fs/music-generation/infinite-goldberg-variations/scripts/custom_grpo_notagen.py`
- Model checkpoint:
  `/home/jl_fs/music-generation/models/weights_notagen_pretrain_p_size_16_p_length_2048_p_layers_16_c_layers_3_h_size_1024_lr_0.0001_batch_4.pth`
- Prompt file:
  `/home/jl_fs/music-generation/infinite-goldberg-variations/data/processed/notagen/goldberg_grpo_prompts.jsonl`

## Resume the paused H100

```bash
jl resume 433981 --yes --json
```

Always use the returned `machine_id`, `public_ip`, and `ssh_command`. Jarvis can assign a new machine id on resume.

## Sync the GRPO script

Replace `<IP>` with the resumed VM IP:

```bash
scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  /Users/donguille/projects/infinite-goldberg-variations/scripts/custom_grpo_notagen.py \
  ubuntu@<IP>:/home/jl_fs/music-generation/infinite-goldberg-variations/scripts/custom_grpo_notagen.py
```

## Launch the current Aria-conditioned GRPO run

This is the current command shape for the Aria-conditioned target run. Use `--target-stream-lines 4` only for short smoke tests.

```bash
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null ubuntu@<IP> '
  rm -f /home/jl_fs/music-generation/outputs/grpo_run.json \
        /home/jl_fs/music-generation/outputs/grpo_run.log
  cd /home/jl_fs/music-generation/infinite-goldberg-variations
  nohup /home/jl_fs/music-generation/.venvs/notagen-trl/bin/python \
    scripts/custom_grpo_notagen.py \
    --policy-weights /home/jl_fs/music-generation/models/weights_notagen_pretrain_p_size_16_p_length_2048_p_layers_16_c_layers_3_h_size_1024_lr_0.0001_batch_4.pth \
    --reference-weights /home/jl_fs/music-generation/models/weights_notagen_pretrain_p_size_16_p_length_2048_p_layers_16_c_layers_3_h_size_1024_lr_0.0001_batch_4.pth \
    --prompts-jsonl /home/jl_fs/music-generation/infinite-goldberg-variations/data/processed/notagen/goldberg_grpo_prompts.jsonl \
    --output-json /home/jl_fs/music-generation/outputs/grpo_run.json \
    --prompt-limit 1 \
    --max-steps 1 \
    --group-size 4 \
    --temperature 1.0 \
    --beta 0.0 \
    --target-stream-lines 32 \
    --timeout-s 120 \
    --precision bf16 \
    --replay-context-patches 0 \
    --score-chunk-patches 8 \
    --gradient-checkpointing \
    --lora-r 8 \
    --lora-alpha 16 \
    --lora-dropout 0.05 \
    --reference-on-cpu \
    > /home/jl_fs/music-generation/outputs/grpo_run.log 2>&1 < /dev/null & echo $!
'
```

## Known-important flags

- `--group-size`
  Number of GRPO samples per prompt.
- `--target-stream-lines`
  Generated NotaGen stream lines, not true bars.
- `--replay-context-patches`
  Replay truncation window in patch units. Use `0` for the full available replay window.
- `--score-chunk-patches`
  Number of generated patches to score/backprop per chunk. Keeps policy graphs small while preserving full replay context.
- `--gradient-checkpointing`
  Recomputes activations during backward to reduce patch-level replay memory.
- `--reference-on-cpu`
  Keeps the reference model off GPU.
- `--precision bf16`
  Required for current memory envelope.
- `--lora-r 8`
  Current adapter rank used in the successful smoke tests.

## Monitoring

```bash
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null ubuntu@<IP> \
  'tail -n 200 /home/jl_fs/music-generation/outputs/grpo_run.log'
```

Check whether the JSON landed:

```bash
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null ubuntu@<IP> \
  'ls -l /home/jl_fs/music-generation/outputs/grpo_run.json'
```

## Pull result locally

```bash
scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  ubuntu@<IP>:/home/jl_fs/music-generation/outputs/grpo_run.json \
  /Users/donguille/projects/infinite-goldberg-variations/data/processed/grpo-smoke/grpo_run.json
```

## Pause the machine after the run

Use the resumed machine id, not the old paused one:

```bash
jl pause <MACHINE_ID> --yes --json
```

## Current frontier

- Intended target:
  - `target_stream_lines=32`
  - `replay_context_patches=0`
  - `score_chunk_patches=8`
  - `gradient_checkpointing=true`
- Works:
  - Aria prompt
  - `group_size=4`
  - `target_stream_lines=32`
  - `replay_context_patches=0`
  - `score_chunk_patches=8`
  - `gradient_checkpointing=true`
  - H100 80GB
- Smoke tested:
  - `target_stream_lines=4` smoke test
  - `replay_context_patches=128`
  - H100 80GB
- Fails at 32 stream lines:
  - `replay_context_patches=128`
  - H100 80GB OOMs during trajectory logprob replay
- Fails without replay optimization:
  - repeated per-patch patch-level replay
- Important caveat:
  - `score_chunk_patches=8` preserves replay context but adds extra scoring compute
  - the current custom loop uses `beta=0.0`, so reference logprobs are skipped
