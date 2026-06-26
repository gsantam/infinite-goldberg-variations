---
name: notagen-large-inference
description: Generate three Aria-conditioned samples from the latest large NotaGen Goldberg checkpoint on Jarvis, then pull them locally and render ABC to MIDI and WAV. Use when running the exact large-rerun inference workflow, including stop conditions, rollover-capable generation, and local audio rendering.
---

# NotaGen Large Inference

Use this skill for the exact inference workflow around the current large Goldberg rerun checkpoint.
The critical invariant is the prompt boundary: generation must start **after the variation voice declarations**, not immediately after the Aria.

## Preconditions

- Local repo root:
  `/Users/donguille/projects/infinite-goldberg-variations`
- Remote NotaGen repo:
  `/home/jl_fs/music-generation/NotaGen`
- Remote old inference runner:
  `/home/jl_fs/music-generation/NotaGen/run_prompt_inference.py`
- Jarvis instance must be running.

## Current checkpoint and prefix

- Large rerun checkpoint:
  `/home/jl_fs/music-generation/models/notagen_goldberg_aria_large_rerun.pth`
- Canonical prompt source:
  `/home/jl_fs/music-generation/infinite-goldberg-variations/data/processed/notagen/goldberg_grpo_prompts.jsonl`
- Correct concrete prefix for variation-01 style inference:
  `/home/jl_fs/music-generation/infinite-goldberg-variations/data/processed/notagen/aria_plus_variation01_setup_G.abc`

Do **not** use these as generation prefixes for this workflow:

- `aria_prefix_G.abc`
- `aria_prefix_G_streamed.abc`

Those stop too early, before the target variation's voice setup, and can collapse generation to one voice.

## Important generation rules

- Generate exactly `3` seeds: `0`, `1`, `2`
- Use:
  - `temperature=1.0`
  - `top_k=8`
  - `top_p=0.95`
  - `target_stream_lines=32`
  - `max_chars=24000`
  - `timeout_s=300`
- Stop condition is **stream-line based**, not validated-bar based:
  - stop when `count_stream_lines(current_text) >= 32`
  - and the latest stream line is closed
- Prompt boundary validation:
  - prefix must contain the full Aria
  - prefix must end after target variation voice declarations such as `V:1 ...`, `V:2 ...`, `V:4 ...`
  - prefix must not already contain generated `[r:...]` continuation lines
- Expected generated output:
  - generated stream lines should contain multiple voices, commonly `[V:1]` and `[V:4]`
  - all generated lines containing only `[V:1]` is a prompt-boundary failure

## Generate 3 trajectories on Jarvis

Replace `<IP>` with the current running A100/H100 IP.

```bash
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null root@<IP> '
  cd /home/jl_fs/music-generation/NotaGen &&
  mkdir -p /home/jl_fs/music-generation/outputs/large_rerun_correct_prefix &&
  for seed in 0 1 2; do
    /home/jl_fs/music-generation/.venvs/notagen-trl/bin/python run_prompt_inference.py \
      --weights /home/jl_fs/music-generation/models/notagen_goldberg_aria_large_rerun.pth \
      --prefix-file /home/jl_fs/music-generation/infinite-goldberg-variations/data/processed/notagen/aria_plus_variation01_setup_G.abc \
      --target-stream-lines 32 \
      --temperature 1.0 \
      --top-k 8 \
      --top-p 0.95 \
      --timeout-s 300 \
      --max-chars 24000 \
      --seed $seed \
      --output /home/jl_fs/music-generation/outputs/large_rerun_correct_prefix/notagen_large_rerun_correct_prefix_seed${seed}.abc
    echo wrote seed=$seed
  done'
```

If the concrete prefix file is missing, rebuild it from the canonical prompt JSONL:

```bash
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null root@<IP> '
  cd /home/jl_fs/music-generation/infinite-goldberg-variations &&
  /home/jl_fs/music-generation/.venvs/notagen-trl/bin/python - <<\"PY\"
import json
from pathlib import Path

src = Path("/home/jl_fs/music-generation/infinite-goldberg-variations/data/processed/notagen/goldberg_grpo_prompts.jsonl")
dst = Path("/home/jl_fs/music-generation/infinite-goldberg-variations/data/processed/notagen/aria_plus_variation01_setup_G.abc")
row = json.loads(src.read_text(encoding="utf-8").splitlines()[0])
dst.write_text(row["prompt"], encoding="utf-8")
print(dst)
PY'
```

## Validate voices before rendering

Run this after pulling or on the remote files. A healthy output should not have every stream line with only one voice.

```bash
python3 - <<'PY'
from pathlib import Path
import re, collections

base = Path("/Users/donguille/projects/infinite-goldberg-variations/data/processed/notagen/large_rerun_correct_prefix")
for seed in (0, 1, 2):
    p = base / f"notagen_large_rerun_correct_prefix_seed{seed}.abc"
    lines = [l for l in p.read_text(encoding="utf-8").splitlines() if l.startswith("[r:")]
    dist = collections.Counter()
    voices = collections.Counter()
    for line in lines:
        vs = set(re.findall(r"\[V:(\d+)\]", line))
        dist[len(vs)] += 1
        for v in vs:
            voices[v] += 1
    print(seed, "stream_lines", len(lines), "voices_per_line", dict(sorted(dist.items())), "voices", dict(sorted(voices.items(), key=lambda x: int(x[0]))))
PY
```

## Pull outputs locally

Local target folder:

- `/Users/donguille/projects/infinite-goldberg-variations/data/processed/notagen/large_rerun_correct_prefix`

```bash
mkdir -p /Users/donguille/projects/infinite-goldberg-variations/data/processed/notagen/large_rerun_correct_prefix

scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  root@<IP>:/home/jl_fs/music-generation/outputs/large_rerun_correct_prefix/notagen_large_rerun_correct_prefix_seed0.abc \
  root@<IP>:/home/jl_fs/music-generation/outputs/large_rerun_correct_prefix/notagen_large_rerun_correct_prefix_seed1.abc \
  root@<IP>:/home/jl_fs/music-generation/outputs/large_rerun_correct_prefix/notagen_large_rerun_correct_prefix_seed2.abc \
  /Users/donguille/projects/infinite-goldberg-variations/data/processed/notagen/large_rerun_correct_prefix/
```

## Convert ABC to renderable ABC, MIDI, and WAV locally

Prerequisites on local machine:

- `abc2midi`
- `timidity`

The raw outputs usually need an `X:1` header to be renderable.

```bash
set -e
outdir=/Users/donguille/projects/infinite-goldberg-variations/data/processed/notagen/large_rerun_correct_prefix

for seed in 0 1 2; do
  raw="$outdir/notagen_large_rerun_correct_prefix_seed${seed}.abc"
  render="$outdir/notagen_large_rerun_correct_prefix_seed${seed}_renderable.abc"
  mid="$outdir/notagen_large_rerun_correct_prefix_seed${seed}.mid"
  wav="$outdir/notagen_large_rerun_correct_prefix_seed${seed}.wav"

  { echo 'X:1'; cat "$raw"; } > "$render"
  abc2midi "$render" -o "$mid" || true
  timidity "$mid" -Ow -o "$wav"
done
```

## Listen locally

```bash
afplay "/Users/donguille/projects/infinite-goldberg-variations/data/processed/notagen/large_rerun_correct_prefix/notagen_large_rerun_correct_prefix_seed0.wav"
afplay "/Users/donguille/projects/infinite-goldberg-variations/data/processed/notagen/large_rerun_correct_prefix/notagen_large_rerun_correct_prefix_seed1.wav"
afplay "/Users/donguille/projects/infinite-goldberg-variations/data/processed/notagen/large_rerun_correct_prefix/notagen_large_rerun_correct_prefix_seed2.wav"
```

## Known caveats

- `target_stream_lines=32` is a **streamed-line** target, not guaranteed `32` validated musical bars.
- `abc2midi` may render malformed outputs with warnings instead of rejecting them.
- If generated lines contain only `[V:1]`, the prompt boundary is wrong. Regenerate using `goldberg_grpo_prompts.jsonl` / `aria_plus_variation01_setup_G.abc`.
- Very long WAVs usually mean:
  - structurally bad ABC
  - over-dense content within a stream line
  - weak bar accounting
