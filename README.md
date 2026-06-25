# Music Generation

This project explores symbolic music generation with modern post-training
methods, especially rule-based rewards for classical structure, harmony, and
voice leading.

## First Model

We start with `m-a-p/MuPT-v1-8192-190M`:

https://huggingface.co/m-a-p/MuPT-v1-8192-190M

It is the best first model for experimentation because it is open source,
small enough to run locally, available through standard Hugging Face
Transformers, and used as a baseline in the NotaGen paper. NotaGen itself is a
better conceptual match, but its released checkpoints and inference setup are
more custom, so it is a second step after we have a working generate/parse/reward
loop.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Generate a Sample

```bash
python scripts/generate_mupt.py --seed 0
```

The script prints the raw MuPT text and a best-effort decoded ABC-like version.
The next milestone is to parse that output with `music21` and compute simple
rule rewards such as parse validity, bar duration consistency, voice range, and
voice crossing.
