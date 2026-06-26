from __future__ import annotations

import argparse
import json
from pathlib import Path


def extract_prompt_prefix(abc_text: str) -> str:
    lines = [line for line in abc_text.splitlines() if line]
    cut = None
    for i, line in enumerate(lines):
        if "[V:" in line:
            cut = i
            break
    if cut is None:
        raise ValueError("could not find generated-body boundary '[V:'")
    return "\n".join(lines[:cut]) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fixed-prompt-abc",
        default=None,
        help="Use one canonical ABC prompt instead of extracting variation-specific prefixes.",
    )
    parser.add_argument(
        "--fixed-prompt-repeat",
        type=int,
        default=1,
        help="Number of identical rows to write when --fixed-prompt-abc is set.",
    )
    parser.add_argument(
        "--fixed-section-name",
        default="goldberg-aria-continuation",
        help="Neutral section_name for rows written from --fixed-prompt-abc.",
    )
    parser.add_argument(
        "--augmented-dir",
        default=str(
            Path(__file__).resolve().parents[1]
            / "data"
            / "processed"
            / "notagen"
            / "goldberg_aria_conditioned"
            / "augmented"
            / "G"
        ),
    )
    parser.add_argument(
        "--output-jsonl",
        default=str(
            Path(__file__).resolve().parents[1]
            / "data"
            / "processed"
            / "notagen"
            / "goldberg_grpo_prompts.jsonl"
        ),
    )
    args = parser.parse_args()

    output_jsonl = Path(args.output_jsonl)

    rows = []
    if args.fixed_prompt_abc:
        prompt_path = Path(args.fixed_prompt_abc)
        prompt = prompt_path.read_text(encoding="utf-8")
        if not prompt.endswith("\n"):
            prompt += "\n"
        if args.fixed_prompt_repeat < 1:
            raise ValueError("--fixed-prompt-repeat must be >= 1")
        rows = [
            {
                "prompt": prompt,
                "section_name": args.fixed_section_name,
                "source_path": str(prompt_path),
            }
            for _ in range(args.fixed_prompt_repeat)
        ]
    else:
        augmented_dir = Path(args.augmented_dir)
        for path in sorted(augmented_dir.glob("variation-*_G.abc")):
            prompt = extract_prompt_prefix(path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "prompt": prompt,
                    "section_name": path.stem.removesuffix("_G"),
                    "source_path": str(path),
                }
            )

    with output_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(output_jsonl)
    print(f"rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
