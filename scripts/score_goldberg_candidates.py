from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from grpo import (
    GoldbergRewardConfig,
    compute_group_advantages,
    load_structural_target,
    score_candidate_file,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidates", nargs="+", help="ABC or MusicXML candidate files to score")
    parser.add_argument(
        "--target-json",
        default=str(
            PROJECT_ROOT
            / "data"
            / "processed"
            / "goldberg"
            / "structure"
            / "aria_bar_skeleton.json"
        ),
    )
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    target = load_structural_target(args.target_json)
    config = GoldbergRewardConfig()
    rewards = [score_candidate_file(path, target, config) for path in args.candidates]
    advantages = compute_group_advantages(rewards)

    payload = {
        "target_json": args.target_json,
        "rewards": [reward.to_json() for reward in rewards],
        "group_advantages": advantages,
    }

    rendered = json.dumps(payload, indent=2)
    if args.output_json:
        Path(args.output_json).write_text(rendered, encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
