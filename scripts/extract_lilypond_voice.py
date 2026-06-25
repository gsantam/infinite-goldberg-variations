#!/usr/bin/env python3
"""Extract a named LilyPond voice block and split it into bars.

This is intentionally a lightweight extractor, not a full LilyPond parser. It is
useful for bootstrapping Goldberg Aria structure data from the local LilyPond
source.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ASSIGNMENT_RE = re.compile(r"(?m)^\s*([A-Za-z][A-Za-z0-9_]*)\s*=")


def strip_comments(text: str) -> str:
    return "\n".join(line.split("%", 1)[0] for line in text.splitlines())


def list_assignments(text: str) -> list[str]:
    return sorted({match.group(1) for match in ASSIGNMENT_RE.finditer(text)})


def find_assigned_block(text: str, name: str) -> str:
    match = re.search(rf"(?m)^\s*{re.escape(name)}\s*=", text)
    if match is None:
        raise ValueError(f"Could not find LilyPond assignment: {name}")

    start = text.find("{", match.end())
    if start < 0:
        raise ValueError(f"Assignment has no block: {name}")

    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : index]

    raise ValueError(f"Unclosed LilyPond block: {name}")


def normalize_bar(bar: str) -> str:
    bar = bar.replace("\n", " ")
    bar = re.sub(r"\\repeat\s+volta\s+\d+", " ", bar)
    bar = re.sub(r"\\barNumberCheck\s+\d+", " ", bar)
    bar = re.sub(r"\\[A-Za-z][A-Za-z0-9-]*", " ", bar)
    bar = bar.replace("<<", " ").replace(">>", " ").replace("\\\\", " ")
    bar = bar.replace("{", " ").replace("}", " ")
    bar = bar.translate(str.maketrans({"[": " ", "]": " ", "^": " ", "_": " "}))
    bar = bar.replace("!", "").replace("?", "")
    bar = re.sub(r"\s+", " ", bar)
    return bar.strip()


def split_bars(block: str) -> list[str]:
    bars: list[str] = []
    for raw_bar in block.split("|"):
        bar = normalize_bar(raw_bar)
        if not bar:
            continue
        if not re.search(r"[a-grs](?:[,']|[a-z!?.~])*[0-9]?", bar):
            continue
        bars.append(bar)
    return bars


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--voice", default="leftHandLower")
    parser.add_argument("--list-voices", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    text = strip_comments(args.path.read_text(encoding="utf-8"))

    if args.list_voices:
        for name in list_assignments(text):
            print(name)
        return

    block = find_assigned_block(text, args.voice)
    bars = split_bars(block)
    result = {
        "source": str(args.path),
        "voice": args.voice,
        "bar_count": len(bars),
        "bars": [{"index": index + 1, "content": bar} for index, bar in enumerate(bars)],
    }

    payload = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()
