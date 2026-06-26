---
name: reward-plot-generation
description: Regenerate Goldberg reward component plots from baseline and SFT reward JSONL files. Use when updating README reward figures, refreshing reward breakdowns after scorer changes, or explaining how normalized reward curves are computed.
---

# Reward Plot Generation

Use this skill to regenerate the reward breakdown figure:

```text
docs/assets/sft_reward_breakdown_vs_epoch.svg
docs/assets/sft_reward_breakdown_vs_epoch.json
```

## Inputs

Default inputs are:

- baseline epoch 0 rewards:
  `data/processed/notagen/base_large_metadata_only_cached10_20260626/metadata_only_rewards.jsonl`
- corrected SFT epoch rewards:
  `data/processed/notagen/reward_exports/large_sft10_cached10_rewards_refactored_20260626/epoch*_rewards.jsonl`

Each SFT epoch file contains 10 sampled trajectories. The baseline file is treated as
epoch `0`.

## Normalization

Do not min-max normalize across epochs. Each plotted component is already a
normalized reward in `[0, 1]` as emitted by `grpo/rewards.py`.

For each epoch and each component:

```text
plotted_value = mean(component_reward over sampled trajectories)
```

The plot intentionally excludes `total_reward` because it is a weighted sum of
heterogeneous components and can be dominated by easy structural signals. Use
component curves for interpretation.

## Components

Structure / format panel:

- `parse_reward`: Music21 can parse the ABC
- `countdown_reward`: stream tags are internally consistent
- `line_closure_reward`: generated stream lines close as bars
- `bar_token_reward`: generated lines emit bar tokens
- `meter_alignment_reward`: bars align with the declared meter
- `meter_duration_closeness_reward`: durations are close to the meter
- `bar_count_reward`: validated bar count is close to 32

Harmony panel:

- `root_similarity_reward`: inferred chord roots match the Aria skeleton
- `bass_pitch_class_reward`: bass pitch classes match the Aria skeleton
- `cadence_root_reward`: cadence bars 8/16/24/32 match by root
- `cadence_bass_reward`: cadence bars 8/16/24/32 match by bass pitch class

## Regenerate

From the repo root:

```bash
python3 .agents/skills/reward-plot-generation/scripts/plot_reward_components.py
```

This writes the SVG and JSON source data to `docs/assets/`.

## Refresh Rewards First

If `grpo/rewards.py` changed, recompute reward JSONL files before plotting.
The plot script only aggregates existing rewards; it does not rescore ABC files.

Important scorer detail: `_ensure_renderable_abc(...)` must add missing ABC
defaults such as `X:`, `L:`, `M:`, and `K:` before Music21 parsing. Otherwise
metadata-only baseline generations can incorrectly get `parse_reward = 0`.
