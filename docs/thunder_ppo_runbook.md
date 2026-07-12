# Thunder PPO Runbook

This is the deterministic path for running NotaGen PPO/GRPO checks on Thunder without re-discovering instance setup each time.

## Principles

- Prefer `tnr` directly when the helper script has auth/API issues.
- Never print or commit Thunder tokens, `~/.thunder/cli_config.json`, private keys, or raw `tnr create --json` output containing key material.
- Use the `model-sft` snapshot when available. It already has the SFT checkpoint/environment state we need.
- Pull results before deleting the instance.
- Delete the instance by numeric `id` from `tnr status --json`; deleting by UUID/name failed in our test.

## Local Variables

Run from the repo root:

```bash
cd /Users/donguille/projects/infinite-goldberg-variations
TNR=/Users/donguille/.tnr/bin/tnr
```

`tnr` is not necessarily on non-interactive `PATH`, so use the absolute path above.

## Check Login, Snapshots, And Instances

```bash
"$TNR" login
"$TNR" status --json
"$TNR" snapshot list --json
```

Expected useful snapshot:

```text
name: model-sft
id: Y4spuAYk3FZOdYH5sPSa
minimumDiskSizeGb: 100
```

If `status --json` returns instances you did not intend to keep, delete them before launching new work.

## Create Cheapest A100 From The SFT Snapshot

```bash
"$TNR" create \
  --gpu a100 \
  --num-gpus 1 \
  --vcpus 8 \
  --disk 100 \
  --snapshot model-sft \
  --yes \
  --json
```

Notes:

- `--json` create output can include private key material. Do not paste it into chat, docs, or commits.
- A100 creation required `--vcpus`; valid values seen were `8`, `12`, `16`. Use `8` for the cheapest dev run.
- A tested A100 1x, 8 vCPU, 100 GB disk estimate was about `$1.09/hr`.

Wait until the instance is `RUNNING`:

```bash
while true; do
  "$TNR" status --json
  sleep 10
done
```

Stop the loop once the target instance is running.

## Get SSH Details

```bash
"$TNR" connect <instance_uuid_or_numeric_id> --json
```

This returns the IP, SSH port, and local key file, for example:

```text
key_file: /Users/donguille/.thunder/keys/<instance_uuid>
ssh_command: ssh -i /Users/donguille/.thunder/keys/<instance_uuid> root@<ip> -p <port>
```

Use these variables for sync/run commands. The `model-sft` snapshot accepted SSH as `ubuntu`; `root` returned `Permission denied`.

```bash
HOST=ubuntu@<ip>
PORT=<port>
KEY=/Users/donguille/.thunder/keys/<instance_uuid>
SSH="ssh -i $KEY -p $PORT -o StrictHostKeyChecking=no"
RSYNC_RSH="$SSH"
REMOTE_ROOT=/home/ubuntu/music-generation
REMOTE_REPO=$REMOTE_ROOT/infinite-goldberg-variations
REMOTE_NOTAGEN=$REMOTE_ROOT/NotaGen
REMOTE_PY=$REMOTE_ROOT/.venvs/notagen-trl/bin/python
```

## Verify Remote GPU And Python

```bash
$SSH "$HOST" "nvidia-smi && $REMOTE_PY - <<'PY'
import torch
print('cuda', torch.cuda.is_available())
print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY"
```

If the snapshot environment is missing, rebuild it before running PPO:

```bash
$SSH "$HOST" "python3 -m venv --system-site-packages $REMOTE_ROOT/.venvs/notagen-trl"
$SSH "$HOST" "$REMOTE_PY -m pip install -U pip"
$SSH "$HOST" "$REMOTE_PY -m pip install -r $REMOTE_REPO/requirements.txt music21 transformers datasets accelerate tqdm abctoolkit==0.0.6 samplings==0.1.7"
```

## Build PPO Prompt JSONL Locally

`scripts/custom_ppo_notagen.py` expects rows with literal `prompt` text. The E3 manifest has prefix file paths, so convert it once before sync:

```bash
python3 - <<'PY'
import json
from pathlib import Path

root = Path("data/processed/notagen/goldberg_metadata_only_split2")
src = root / "header_prefix_manifest_G.jsonl"
out = Path("data/processed/notagen/goldberg_ppo_prompts_e3_header_allvoices.jsonl")

rows = []
for line in src.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    prefix_path = Path(row["prefix"])
    rows.append({
        "name": prefix_path.stem,
        "prompt": prefix_path.read_text(encoding="utf-8"),
        "source": row.get("source"),
        "prefix": str(prefix_path),
        "continuation": row.get("continuation"),
    })

out.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
print(out, len(rows))
PY
```

## Sync Minimal Project State

Create remote directories:

```bash
$SSH "$HOST" "mkdir -p $REMOTE_REPO $REMOTE_REPO/data/processed/notagen $REMOTE_REPO/data/processed/goldberg"
```

Sync code and the minimum data needed for the E3 PPO dry run:

```bash
rsync -az -e "$RSYNC_RSH" \
  requirements.txt README.md \
  scripts evaluation preprocessing grpo tests \
  "$HOST:$REMOTE_REPO/"

rsync -az -e "$RSYNC_RSH" \
  data/processed/goldberg/ \
  "$HOST:$REMOTE_REPO/data/processed/goldberg/"

rsync -az -e "$RSYNC_RSH" \
  data/processed/notagen/goldberg_metadata_only_split2/ \
  "$HOST:$REMOTE_REPO/data/processed/notagen/goldberg_metadata_only_split2/"

rsync -az -e "$RSYNC_RSH" \
  data/processed/notagen/goldberg_ppo_prompts_e3_header_allvoices.jsonl \
  "$HOST:$REMOTE_REPO/data/processed/notagen/"
```

Do not sync `.venv`, rendered audio, all checkpoints, or old remote runs.

The `model-sft` snapshot already contained the E3 SFT checkpoint here:

```text
data/processed/notagen/remote_runs/E3_metadata_header_allvoices_noaug_e8_s0_rebuild_20260702_142840/checkpoints/current.pth
```

Use that path for PPO before uploading the local 5.8 GB checkpoint.

## One-Step PPO Dry Run

This runs rollout, rewards, replay/logprobs, value estimates, GAE, and PPO loss, but does not update weights:

```bash
$SSH "$HOST" "cd $REMOTE_REPO && \
  export PYTHONPATH=$REMOTE_NOTAGEN:$REMOTE_REPO:\$PYTHONPATH && \
  mkdir -p data/processed/notagen/remote_runs && \
  $REMOTE_PY scripts/custom_ppo_notagen.py \
    --policy-weights data/processed/notagen/remote_runs/E3_metadata_header_allvoices_noaug_e8_s0_rebuild_20260702_142840/checkpoints/current.pth \
    --prompts-jsonl data/processed/notagen/goldberg_ppo_prompts_e3_header_allvoices.jsonl \
    --target-structure-abc data/processed/notagen/goldberg_metadata_only_split2/augmented/G/variation-01_G.abc \
    --output-json data/processed/notagen/remote_runs/ppo_e3_dryrun_\$(date -u +%Y%m%dT%H%M%SZ).json \
    --max-steps 1 \
    --prompt-limit 1 \
    --no-step \
    --cached-rollout \
    --precision bf16 \
    --max-generated-patches 512 \
    --timeout-s 900 \
    --replay-context-patches 128 \
    --score-chunk-patches 16 \
    --gae-lambda 0.95"
```

Default PPO similarity rewards are currently active:

- `--aria-chroma-reward-weight 1.0`
- `--aria-harmony-reward-weight 1.0`
- `--max-similarity-reward 2.0`

The dry run logs timings for rollout, reward scoring, replay/logprob, and PPO loss. Use these timings before launching longer training.

## Current PPO Reward Objective

The current PPO scalar reward is:

```text
total_reward = structural_total_reward + effective_similarity_reward
```

Structural terms are `parse_reward`, `countdown_reward`, `line_closure_reward`, `bar_token_reward`, `meter_alignment_reward`, `meter_duration_closeness_reward`, `bar_meter_consistency_reward`, `bar_count_reward`, `voice_declaration_reward`, and `score_voice_reward`.

The active aria similarity terms are:

```text
aria_chroma_harmonic_hist = mean(aria_chroma_full_hist, aria_chroma_bass_hist)
aria_harmony_combined = mean(aria_harmony_harmony_dtw, aria_harmony_root_dtw, aria_harmony_bass_dtw)
effective_similarity_reward = gate * min(
  aria_chroma_harmonic_hist + aria_harmony_combined,
  max_similarity_reward,
)
```

Defaults are:

```text
--aria-chroma-reward-weight 1.0
--aria-harmony-reward-weight 1.0
--max-similarity-reward 2.0
```

`top_hist` and `top_contour_dtw` are logged/candidate metrics, not active in the scalar reward. `density_dtw` is diagnostic only.

## Patch-Level PPO Reward Assignment

PPO uses one reward per generated NotaGen patch because the value head predicts patch values.

The exact prefix-delta method would score every prefix:

```python
for t in generated_patches:
    score_t = score_full_completion(prompt + patches[:t])
    reward_t = score_t - score_{t - 1}
```

That repeated full parsing is too slow. The implemented path in `scripts/custom_ppo_notagen.py` is `patch_rewards_single_pass(...)`:

1. Score the full completion once.
2. Extract local structural line/bar events.
3. Extract harmony DTW events for `harmony_dtw`, `root_dtw`, and `bass_dtw`.
4. Attribute each local event to generated patches by text-span overlap.
5. Add the terminal residual to the last patch:

```python
terminal_residual = final_score.total - sum(projected_patch_rewards)
projected_patch_rewards[-1] += terminal_residual
```

This keeps the invariant:

```text
sum(patch_rewards) == final_sequence_reward
```

Global effects such as chroma histogram similarity and parse/global validity mostly arrive through the terminal residual. Harmony DTW remains sequence-aligned: the DTW path aligns aria bars/spans to generated bars/spans, then the generated span credit is projected onto patches by overlap.

Advantages use GAE:

```text
delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
A_t = delta_t + gamma * gae_lambda * A_{t+1}
value_target_t = A_t + V(s_t)
```

Current defaults are `--gamma 1.0` and `--gae-lambda 0.95`.

## Offline Critic Training On Saved PPO Rollouts

The PPO value head can be trained independently from rollout and policy updates. This is useful when the policy step is slow but we want to tune critic epochs, learning rate, hidden size, dropout, or train/eval splits.

Important: the PPO JSON must include `trajectories[*].generated_patches`. Older PPO JSONs only stored rewards and cannot reconstruct the frozen-policy value states offline.

Collect critic data cheaply on Thunder with rollout-only mode. This samples and scores trajectories, but skips PPO replay/logprob/backward:

```bash
$SSH "$HOST" "cd $REMOTE_REPO && \
  export PYTHONPATH=$REMOTE_NOTAGEN:$REMOTE_REPO:\$PYTHONPATH && \
  $REMOTE_PY scripts/custom_ppo_notagen.py \
    --policy-weights data/processed/notagen/remote_runs/E3_metadata_header_allvoices_noaug_e8_s0_rebuild_20260702_142840/checkpoints/current.pth \
    --prompts-jsonl data/processed/notagen/goldberg_ppo_prompts_e3_header_allvoices.jsonl \
    --target-structure-abc data/processed/notagen/goldberg_metadata_only_split2/augmented/G/variation-01_G.abc \
    --output-json data/processed/notagen/remote_runs/ppo_e3_rollout_only.json \
    --max-steps 3 \
    --prompt-limit 3 \
    --trajectories-per-step 4 \
    --rollout-batch-size 4 \
    --rollout-only \
    --cached-rollout \
    --precision bf16 \
    --max-generated-patches 512 \
    --timeout-s 900 \
    --target-stream-lines 32"
```

First precompute frozen SFT patch hidden states and train a critic:

```bash
python scripts/train_notagen_ppo_value_head_offline.py \
  --policy-weights data/processed/notagen/remote_runs/E3_metadata_header_allvoices_noaug_e8_s0_rebuild_20260702_142840/checkpoints/current.pth \
  --ppo-json data/processed/notagen/remote_runs/<ppo_run_with_generated_patches>.json \
  --prompts-jsonl data/processed/notagen/goldberg_ppo_prompts_e3_header_allvoices.jsonl \
  --hidden-cache-out data/processed/notagen/remote_runs/ppo_value_hidden_cache.pt \
  --output-value-head data/processed/notagen/remote_runs/ppo_value_head_offline.pt \
  --output-json data/processed/notagen/remote_runs/ppo_value_head_offline_metrics.json \
  --epochs 50 \
  --holdout-last-step \
  --normalize-value-loss \
  --value-loss-scale-min 1.0
```

Then reuse the hidden-state cache for architecture/optimizer sweeps without loading the policy checkpoint:

```bash
python scripts/train_notagen_ppo_value_head_offline.py \
  --hidden-cache-in data/processed/notagen/remote_runs/ppo_value_hidden_cache.pt \
  --output-value-head data/processed/notagen/remote_runs/ppo_value_head_offline_h1024.pt \
  --output-json data/processed/notagen/remote_runs/ppo_value_head_offline_h1024_metrics.json \
  --epochs 100 \
  --value-head-hidden-size 1024 \
  --value-head-dropout 0.05 \
  --holdout-last-step \
  --normalize-value-loss \
  --value-loss-scale-min 1.0
```

The main diagnostics are `explained_variance`, `correlation`, `mse`, `mae`, and `bias` on both train and holdout samples. A useful critic should improve explained variance/correlation, not just reduce train MSE.

The current critic artifact was trained from 200 raw E3 SFT rollout trajectories with full active rewards for 80 epochs:

```text
data/processed/notagen/remote_runs/ppo_e3_rollout_only_t200_seedfix_20260712T135300Z_combined_full_reward_value_head_e80_all200.pt
```

The matching command shape is:

```bash
python scripts/train_notagen_ppo_value_head_offline.py \
  --ppo-json data/processed/notagen/remote_runs/ppo_e3_rollout_only_t200_seedfix_20260712T135300Z_combined.json \
  --prompts-jsonl data/processed/notagen/goldberg_ppo_prompts_e3_header_allvoices.jsonl \
  --policy-weights data/processed/notagen/remote_runs/E3_metadata_header_allvoices_noaug_e8_s0_rebuild_20260702_142840/checkpoints/current.pth \
  --output-value-head data/processed/notagen/remote_runs/ppo_e3_rollout_only_t200_seedfix_20260712T135300Z_combined_full_reward_value_head_e80_all200.pt \
  --output-json data/processed/notagen/remote_runs/ppo_e3_rollout_only_t200_seedfix_20260712T135300Z_combined_full_reward_value_head_e80_all200.json \
  --epochs 80 \
  --trajectory-batch-size 4 \
  --value-learning-rate 1e-4 \
  --normalize-value-loss \
  --value-loss-scale-min 1.0 \
  --replay-context-patches 128 \
  --score-chunk-patches 8 \
  --precision bf16 \
  --seed 0
```

## Critic / Value Head Options

PPO now supports a persistent MLP value head and optional critic warmup:

```text
--value-head-hidden-size 512
--value-head-dropout 0.0
--value-head-weights <existing_value_head.pt>
--save-value-head-weights <new_value_head.pt>
--value-warmup-epochs <n>
--ppo-epochs <n>
--normalize-value-loss
--value-loss-eps 1e-6
--value-loss-scale-min 1.0
```

`--normalize-value-loss` does not normalize rewards or change the critic target. It keeps the critic predicting raw discounted returns, but divides the MSE scale by the target standard deviation so value gradients stay comparable across reward scales. JSON logs include both `raw_value_loss` and the scaled `value_loss`.
Use `--value-loss-scale-min 1.0` for early checks so tiny target variance does not over-amplify the critic loss.

For the first real PPO checks, use one or two value warmup epochs and keep one PPO epoch:

```text
--value-warmup-epochs 1
--ppo-epochs 1
--normalize-value-loss
--value-loss-scale-min 1.0
--save-value-head-weights data/processed/notagen/remote_runs/value_head_latest.pt
```

## One-Step PPO Update

Remove `--no-step` to perform one optimizer update:

```bash
$SSH "$HOST" "cd $REMOTE_REPO && \
  export PYTHONPATH=$REMOTE_NOTAGEN:$REMOTE_REPO:\$PYTHONPATH && \
  $REMOTE_PY scripts/custom_ppo_notagen.py \
    --policy-weights data/processed/notagen/remote_runs/E3_metadata_header_allvoices_noaug_e8_s0_rebuild_20260702_142840/checkpoints/current.pth \
    --prompts-jsonl data/processed/notagen/goldberg_ppo_prompts_e3_header_allvoices.jsonl \
    --target-structure-abc data/processed/notagen/goldberg_metadata_only_split2/augmented/G/variation-01_G.abc \
    --output-json data/processed/notagen/remote_runs/ppo_e3_step1_\$(date -u +%Y%m%dT%H%M%SZ).json \
    --max-steps 1 \
    --prompt-limit 1 \
    --cached-rollout \
    --precision bf16 \
    --max-generated-patches 512 \
    --timeout-s 900 \
    --replay-context-patches 128 \
    --score-chunk-patches 16 \
    --gae-lambda 0.95 \
    --value-warmup-epochs 1 \
    --ppo-epochs 1 \
    --normalize-value-loss \
    --value-loss-scale-min 1.0 \
    --save-value-head-weights data/processed/notagen/remote_runs/value_head_latest.pt"
```

## One-Prompt Batched PPO Check

Use this to validate PPO with multiple trajectories from the same prompt before rotating prompts across steps:

```bash
$SSH "$HOST" "cd $REMOTE_REPO && \
  export PYTHONPATH=$REMOTE_NOTAGEN:$REMOTE_REPO:\$PYTHONPATH && \
  OUT=data/processed/notagen/remote_runs/ppo_e3_step1_t4_batch4_\$(date -u +%Y%m%dT%H%M%SZ).json && \
  LOG=\${OUT%.json}.log && \
  $REMOTE_PY scripts/custom_ppo_notagen.py \
    --policy-weights data/processed/notagen/remote_runs/E3_metadata_header_allvoices_noaug_e8_s0_rebuild_20260702_142840/checkpoints/current.pth \
    --prompts-jsonl data/processed/notagen/goldberg_ppo_prompts_e3_header_allvoices.jsonl \
    --target-structure-abc data/processed/notagen/goldberg_metadata_only_split2/augmented/G/variation-01_G.abc \
    --output-json \$OUT \
    --max-steps 1 \
    --prompt-limit 1 \
    --trajectories-per-step 4 \
    --rollout-batch-size 4 \
    --rollout-retries 1 \
    --cached-rollout \
    --precision bf16 \
    --max-generated-patches 512 \
    --timeout-s 900 \
    --replay-context-patches 0 \
    --score-chunk-patches 16 \
    --gae-lambda 0.95 \
    --value-warmup-epochs 1 \
    --ppo-epochs 1 \
    --normalize-value-loss \
    --value-loss-scale-min 1.0 \
    --post-step-kl-check > \$LOG 2>&1 && \
  echo \$OUT"
```

For a dry run, add `--no-step`. In the first successful A100 check, `--trajectories-per-step 4 --rollout-batch-size 4` took about `55s` for `--no-step` and `66s` for one real update with a 128-patch replay window. With full-context replay and `--post-step-kl-check`, the same one-step update took about `76s`.

The normal `approx_kl` and `clip_fraction` are computed before `optimizer.step()`, so they are expected to be near zero on a single-pass update. Use `--post-step-kl-check` to replay the same trajectories after the update and inspect `post_step_approx_kl`, `post_step_clip_fraction`, and `post_step_log_ratio_max_abs`.

## Current Recommended One-Step PPO Setup

The best full-PPO setup measured so far is 16 rollout trajectories with replay/backprop microbatched in groups of 4. This keeps rollout batched while avoiding the OOM from replaying all 16 differentiably at once.

```bash
$SSH "$HOST" "cd $REMOTE_REPO && \
  export PYTHONPATH=$REMOTE_NOTAGEN:$REMOTE_REPO:\$PYTHONPATH && \
  RUN_DIR=data/processed/notagen/remote_runs/ppo_e3_full_update_t16_mb4_\$(date -u +%Y%m%dT%H%M%SZ) && \
  mkdir -p \$RUN_DIR && \
  $REMOTE_PY scripts/custom_ppo_notagen.py \
    --policy-weights data/processed/notagen/remote_runs/E3_metadata_header_allvoices_noaug_e8_s0_rebuild_20260702_142840/checkpoints/current.pth \
    --prompts-jsonl data/processed/notagen/goldberg_ppo_prompts_e3_header_allvoices.jsonl \
    --target-structure-abc data/processed/notagen/goldberg_metadata_only_split2/augmented/G/variation-01_G.abc \
    --output-json \$RUN_DIR/result.json \
    --max-steps 1 \
    --prompt-limit 1 \
    --trajectories-per-step 16 \
    --rollout-batch-size 16 \
    --ppo-replay-microbatch-size 4 \
    --rollout-retries 8 \
    --cached-rollout \
    --precision bf16 \
    --max-generated-patches 512 \
    --timeout-s 1200 \
    --similarity-timeout-s 20 \
    --replay-context-patches 128 \
    --score-chunk-patches 8 \
    --gamma 1.0 \
    --gae-lambda 0.95 \
    --value-head-weights data/processed/notagen/remote_runs/ppo_e3_rollout_only_t200_seedfix_20260712T135300Z_combined_full_reward_value_head_e80_all200.pt \
    --save-value-head-weights \$RUN_DIR/value_head_after_step1.pt \
    --value-warmup-epochs 1 \
    --ppo-epochs 1 \
    --ppo-clip-range 0.2 \
    --value-loss-coef 0.5 \
    --normalize-value-loss \
    --value-loss-scale-min 1.0 \
    --lora-r 8 \
    --lora-alpha 16 \
    --lora-dropout 0.05 \
    --checkpoint-dir \$RUN_DIR/checkpoints \
    --checkpoint-every-steps 1 \
    --post-step-kl-check \
    --seed 0 > \$RUN_DIR/run.log 2>&1"
```

If replay/backward OOMs, lower `--ppo-replay-microbatch-size` to `2` before lowering `--rollout-batch-size`. Do not reduce `--max-generated-patches` unless incomplete pieces are acceptable.

## Current Reward Baselines

Current active reward exports:

| Source | n | total mean | total min | total max | structural mean | similarity mean | chroma hist mean | harmony mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Ground-truth variations | 30 | 9.202 | 8.896 | 9.284 | 7.567 | 1.634 | 0.854 | 0.780 |
| GT variation 01 | 1 | 9.222 | 9.222 | 9.222 | 7.589 | 1.633 | 0.850 | 0.782 |
| Raw E3 SFT rollouts | 200 | 7.795 | 4.445 | 9.138 | 6.264 | 1.531 | 0.808 | 0.762 |
| PPO step-30 rollout check | 100 | 8.020 | 5.973 | 8.677 | 6.450 | 1.570 | 0.818 | 0.766 |

Raw E3 SFT rollout patch stats:

```text
generated_patch_count mean 168.6, min 98, max 512
stop_reasons: target_stream_lines 197, max_generated_patches 3
parse_valid_count: 194 / 200
```

Rollout-only A100 batch sweep:

| rollout batch | trajectories | mean rollout | rollout / trajectory | mean reward | mean total |
|---:|---:|---:|---:|---:|---:|
| 4 | 4 | 52.0s | 13.0s | 9.4s | 61.4s |
| 8 | 8 | 47.3s | 5.9s | 17.4s | 64.7s |
| 16 | 16 | 49.8s | 3.1s | 25.7s | 75.5s |

Full PPO one-step A100 timing:

| setup | total | total / trajectory | rollout | rollout / trajectory | reward | replay/backward | patches/sec | post KL | post clip |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 trajectories, old replay | 108.4s | 27.1s | 65.5s | 16.4s | 6.8s | 36.0s | 6.0 | 0.00199 | 0.0201 |
| 16 trajectories, replay mb4 | 242.4s | 15.1s | 122.1s | 7.6s | 24.9s | 95.4s | 12.3 | 0.00301 | 0.0245 |
| 32 trajectories, replay mb4 | 584.1s | 18.3s | 206.2s | 6.4s | 57.8s | 320.1s | 9.3 | 0.00419 | 0.0300 |

Use `16` trajectories as the current full-PPO default. Batch `32` improves rollout per trajectory but is slower overall because differentiable replay/backward dominates.

## Pull Results

```bash
mkdir -p data/processed/notagen/remote_runs
rsync -az -e "$RSYNC_RSH" \
  "$HOST:$REMOTE_REPO/data/processed/notagen/remote_runs/" \
  data/processed/notagen/remote_runs/
```

## Delete Instance And Confirm

Use the numeric `id` from `status --json`, not the UUID/name:

```bash
"$TNR" status --json
"$TNR" delete <numeric_id> --yes --json
"$TNR" status --json
```

The final `status --json` should be `[]`.

## Common Failure Modes

- `thunder_notagen.py` SSL or 403 auth errors: use `/Users/donguille/.tnr/bin/tnr` directly.
- `tnr create --json` asks for template/snapshot: pass `--snapshot model-sft`.
- A100 create fails with vCPU options: add `--vcpus 8`.
- Delete by instance UUID/name fails: delete by numeric `id`.
- PPO says prompt rows are missing `prompt`: rebuild `goldberg_ppo_prompts_e3_header_allvoices.jsonl`.
- Remote import fails for `utils` or NotaGen classes: set `PYTHONPATH=$REMOTE_NOTAGEN:$REMOTE_REPO:$PYTHONPATH`.
- SSH as `root` fails on the `model-sft` snapshot: use `ubuntu`.
