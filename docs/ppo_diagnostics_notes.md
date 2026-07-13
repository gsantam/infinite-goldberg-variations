# PPO Diagnostics Notes

## Last Captured Diagnostic

Run:

```text
ppo_e3_prompt1_t16_lr1e5_ep2_logprob_diag_step1_20260713T081758Z
```

Context:

- Prompt: `variation-01_G`
- Trajectories per PPO step: `16`
- Rollout batch size: `16`
- Learning rate: `1e-5`
- PPO epochs: `2`
- Critic: `ppo_e3_rollout_only_t200_seedfix_20260712T135300Z_combined_full_reward_value_head_e80_all200.pt`
- Note: the full remote `result.json` was not pulled because the instance was deleted while rsync was still running. These values came from monitor output captured before deletion.

Step summary:

| Metric | Value |
| --- | ---: |
| Reward mean | `8.0795` |
| Reward std | `0.4285` |
| Scored patches | `2979` |
| Post-step approx KL | `0.00451` |
| Post-step clip fraction | `0.0359` |

Post-step log ratio, defined as `post_step_logprob - old_logprob`:

| Metric | Value |
| --- | ---: |
| Mean | `-0.00145` |
| Std | `0.09495` |
| Min | `-0.83474` |
| P05 | `-0.10752` |
| P25 | `-0.02546` |
| P50 | `-0.00008` |
| P75 | `0.01557` |
| P95 | `0.10774` |
| Max | `1.56286` |

Raw advantage:

| Metric | Value |
| --- | ---: |
| Mean | `0.04830` |
| Std | `0.39038` |
| Min | `-1.78442` |
| P05 | `-0.56339` |
| P25 | `-0.18561` |
| P50 | `0.01521` |
| P75 | `0.27566` |
| P95 | `0.71126` |
| Max | `2.26269` |

Update alignment:

| Metric | Value |
| --- | ---: |
| Positive advantage patches | `1623` |
| Negative advantage patches | `1356` |
| Zero advantage patches | `0` |
| Advantage/log-ratio correlation | `0.0992` |
| Normalized advantage/log-ratio correlation | `0.0992` |
| Sign alignment fraction | `0.5569` |
| Mean log ratio, positive advantage | `0.00944` |
| Mean log ratio, negative advantage | `-0.01449` |
| Mean log ratio, top advantage decile | `0.01342` |
| Mean log ratio, bottom advantage decile | `0.00024` |

Interpretation:

- The PPO update moved in the right direction, but weakly: positive-advantage patches had a positive average log-ratio and negative-advantage patches had a negative average log-ratio.
- Clip fraction was low, so the update was not mainly blocked by clipping.
- The advantage/log-ratio correlation was positive but small, so the policy update signal was present but not strong.
- We still need the split clipping diagnostics to see whether positive-advantage patches are being capped by the upper PPO clip and negative-advantage patches by the lower PPO clip.

## Add Next Iteration

Capture these in the next full PPO diagnostic run:

- Persist the full remote `result.json` before deleting the instance.
- Save sampled trajectories and the exact local replay command.
- Log split PPO clipping:
  - overall any/upper/lower clip fraction
  - positive-advantage any/upper/lower clip fraction
  - positive-advantage active clip fraction, where active means upper-clipped
  - negative-advantage any/upper/lower clip fraction
  - negative-advantage active clip fraction, where active means lower-clipped
  - active clip fraction over nonzero-advantage patches
- Log policy movement by advantage bucket:
  - mean log-ratio for positive vs negative advantage
  - percent of positive-advantage patches with positive log-ratio
  - percent of negative-advantage patches with negative log-ratio
  - mean log-ratio for top/bottom advantage deciles
  - advantage/log-ratio correlation
  - sign-alignment fraction
- Log critic quality before warmup, after warmup, and after PPO:
  - value-target correlation
  - MSE/MAE
  - explained variance
  - residual mean/std
- Log per-trajectory outliers:
  - reward
  - patch count
  - old/post logprob sum
  - log-ratio mean/std
  - mean/std advantage
  - sign-alignment fraction
- Keep timing breakdown:
  - rollout
  - reward attribution
  - value warmup
  - old logprob/value replay
  - PPO replay/backprop
  - post-step diagnostic replay
