from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from grpo.notagen_hf_adapter import build_trl_ready_notagen


def load_prompts(path: str | Path, limit: int) -> list[str]:
    rows: list[str] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows.append(obj["prompt"])
            if len(rows) >= limit:
                break
    return rows


def build_rollout_func(adapter_bundle, *, target_stream_lines: int, temperature: float, top_k: int, top_p: float, timeout_s: int):
    model = adapter_bundle.model
    processing = adapter_bundle.processing_class

    def rollout_func(prompts: list[str], trainer: Any):
        outputs = model.generate_batch(
            prompts,
            generation_config=getattr(trainer, "generation_config", None),
            target_stream_lines=target_stream_lines,
            timeout_s=timeout_s,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        prompt_ids = [model.wrapper.encode_text_ids(prompt) for prompt in prompts]
        completion_ids = [outputs[i].generated_tokens for i in sorted(outputs)]
        completions = processing.batch_decode(completion_ids)

        pair_scores = model.score_completions(list(zip(prompts, completions, strict=True)))
        return {
            "prompt_ids": prompt_ids,
            "completion_ids": completion_ids,
            "logprobs": pair_scores,
        }

    return rollout_func


def dry_run(args) -> int:
    adapter_bundle = build_trl_ready_notagen(
        weights_path=args.weights,
        precision=args.precision,
        use_lora=not args.disable_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    prompts = load_prompts(args.prompts_jsonl, args.prompt_limit)
    rollout = build_rollout_func(
        adapter_bundle,
        target_stream_lines=args.target_stream_lines,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        timeout_s=args.timeout_s,
    )
    fake_trainer = type("FakeTrainer", (), {"generation_config": None})()
    payload = rollout(prompts, fake_trainer)
    print(json.dumps(
        {
            "num_prompts": len(prompts),
            "num_completions": len(payload["completion_ids"]),
            "first_completion_chars": len(adapter_bundle.processing_class.decode(payload["completion_ids"][0])) if payload["completion_ids"] else 0,
            "first_logprob_count": len(payload["logprobs"][0]) if payload["logprobs"] else 0,
        },
        indent=2,
    ))
    return 0


def trainer_smoke(args) -> int:
    try:
        from datasets import Dataset
        from trl import GRPOConfig, GRPOTrainer
    except Exception as exc:
        raise SystemExit(f"TRL smoke requires `trl` and `datasets`: {exc}")

    adapter_bundle = build_trl_ready_notagen(
        weights_path=args.weights,
        precision=args.precision,
        use_lora=not args.disable_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    prompts = load_prompts(args.prompts_jsonl, args.prompt_limit)
    dataset = Dataset.from_dict({"prompt": prompts})

    def simple_reward(prompts, completions, **kwargs):
        return [float(len(c) > 0) for c in completions]

    rollout = build_rollout_func(
        adapter_bundle,
        target_stream_lines=args.target_stream_lines,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        timeout_s=args.timeout_s,
    )

    config = GRPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=max(1, args.num_generations),
        num_generations=args.num_generations,
        generation_batch_size=args.num_generations,
        max_completion_length=256,
        beta=0.0,
        max_steps=1,
        remove_unused_columns=False,
        report_to=[],
        do_train=True,
    )

    trainer = GRPOTrainer(
        model=adapter_bundle.model,
        processing_class=adapter_bundle.processing_class,
        reward_funcs=[simple_reward],
        train_dataset=dataset,
        args=config,
        rollout_func=rollout,
    )
    trainer.train()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--prompts-jsonl", default=str(Path(__file__).resolve().parents[1] / "data/processed/notagen/goldberg_grpo_prompts_metadata_only.jsonl"))
    parser.add_argument("--prompt-limit", type=int, default=1)
    parser.add_argument("--target-stream-lines", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--timeout-s", type=int, default=45)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--disable-lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--num-generations", type=int, default=2)
    parser.add_argument("--output-dir", default="/tmp/trl-notagen-smoke")
    parser.add_argument("--trainer-smoke", action="store_true")
    args = parser.parse_args()

    if args.trainer_smoke:
        return trainer_smoke(args)
    return dry_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
