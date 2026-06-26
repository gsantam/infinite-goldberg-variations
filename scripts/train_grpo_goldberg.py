from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from grpo import GoldbergRewardConfig, load_structural_target
from grpo.rewards import make_trl_reward_func


def load_prompt_dataset(path: str | Path) -> Dataset:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return Dataset.from_list(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--prompts-jsonl",
        default=str(
            PROJECT_ROOT
            / "data"
            / "processed"
            / "notagen"
            / "goldberg_grpo_prompts.jsonl"
        ),
    )
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
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-completion-length", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    args = parser.parse_args()

    target = load_structural_target(args.target_json)
    reward_config = GoldbergRewardConfig()
    reward_func = make_trl_reward_func(target=target, config=reward_config)
    dataset = load_prompt_dataset(args.prompts_jsonl)

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        num_train_epochs=args.num_train_epochs,
        remove_unused_columns=False,
        logging_steps=1,
        bf16=False,
        fp16=False,
    )

    trainer = GRPOTrainer(
        model=args.model,
        args=training_args,
        reward_funcs=reward_func,
        train_dataset=dataset,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
