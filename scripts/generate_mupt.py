#!/usr/bin/env python3
"""Generate a short ABC-like sample from MuPT.

Run from the project root after installing requirements:

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    python scripts/generate_mupt.py
"""

from __future__ import annotations

import argparse
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "m-a-p/MuPT-v1-8192-190M"

SEPARATORS = ["|", "|]", "||", "[|", "|:", ":|", "::"]
SEP_DICT = {f" {sep} ": f" <{i}>" for i, sep in enumerate(SEPARATORS, start=1)}
NEWSEP = "<|>"


def choose_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sep2tok(row: str) -> str:
    for sep, tok in SEP_DICT.items():
        row = row.replace(sep, tok + "<=> ")
    return row


def tok2sep(bar: str) -> str:
    for sep, tok in SEP_DICT.items():
        bar = bar.replace(tok, sep)
    return bar


def decode_mupt(piece: str) -> str:
    """Best-effort MuPT post-processing from the model card."""
    piece = piece.replace("<n>", "\n")
    idx = piece.find(f" {NEWSEP} ")
    if idx < 0:
        return piece

    heads = piece[:idx]
    scores = piece[idx:]
    scores_lst = re.split(r" <\|>", scores)

    all_bar_lst: list[list[str]] = []
    for bar in scores_lst:
        if not bar:
            continue
        bar = sep2tok(bar)
        bar_lst = re.split("<=>", bar)
        bar_lst = list(map(tok2sep, bar_lst))
        if len(all_bar_lst) == 0:
            all_bar_lst = [[] for _ in range(len(bar_lst))]
        for i in range(min(len(all_bar_lst), len(bar_lst))):
            all_bar_lst[i].append(bar_lst[i])

    dec_piece = heads
    for i, bars in enumerate(all_bar_lst):
        if len(all_bar_lst) > 1:
            dec_piece += f"V:{i + 1}\n"
        dec_piece += "".join(bars)
        dec_piece += "\n"

    return re.sub(" {2,}", " ", dec_piece).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--prompt",
        default='X:1<n>L:1/8<n>Q:1/8=120<n>M:4/4<n>K:C<n>|:"C" CEGc',
        help="MuPT 8192 models use <n> instead of literal newlines.",
    )
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = choose_device()
    dtype = torch.float16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        use_fast=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=dtype,
    ).to(device)
    model.eval()

    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        pad_token_id = 0

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            pad_token_id=pad_token_id,
        )

    raw = tokenizer.decode(outputs[0], skip_special_tokens=False)
    decoded = decode_mupt(raw)

    print("=== RAW ===")
    print(raw)
    print("\n=== DECODED ===")
    print(decoded)


if __name__ == "__main__":
    main()
