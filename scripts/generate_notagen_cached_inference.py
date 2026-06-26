from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from grpo.notagen_cached_generation import CachedNotaGenPatchGenerator
from grpo.notagen_wrapper import (
    PATCH_SIZE,
    PATCH_STREAM,
    Patchilizer,
    build_model,
    count_stream_lines,
    latest_countdown,
    latest_stream_line_closed,
    set_seed,
    split_metadata_and_tunebody_lines,
    trim_to_stream_lines,
)


def sample_completion_cached(
    *,
    model,
    model_shape,
    prompt: str,
    temperature: float,
    top_k: int,
    top_p: float,
    target_stream_lines: int,
    target_new_stream_lines: bool,
    max_chars: int,
    timeout_s: int,
    precision: str,
) -> tuple[str, list[list[int]], dict]:
    patchilizer = Patchilizer(stream=PATCH_STREAM)
    prefix = prompt
    if count_stream_lines(prefix) == 0:
        prefix = prefix + f"[r:0/{target_stream_lines - 1}]"
    prompt_stream_lines = count_stream_lines(prefix)
    target_total_stream_lines = (
        prompt_stream_lines + target_stream_lines
        if target_new_stream_lines
        else target_stream_lines
    )

    input_patches = patchilizer.encode_generate(prefix)
    flat_ids = [int(item) for sublist in input_patches for item in sublist]
    byte_list = list(prefix)
    generated_patches: list[list[int]] = []
    generator = CachedNotaGenPatchGenerator(model, precision=precision)
    generator.reset(flat_ids)
    start_time = time.time()
    cut_index = None
    resets = 1

    while True:
        predicted_patch = None
        for _ in range(8):
            candidate_patch = generator.generate_patch(
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
            )
            current_text = "".join(byte_list)
            countdown = latest_countdown(current_text)
            eos_only = (
                len(candidate_patch) >= 2
                and candidate_patch[0] == patchilizer.bos_token_id
                and candidate_patch[1] == patchilizer.eos_token_id
            )
            if eos_only:
                allow_eos = False
                if countdown is not None:
                    _, remaining = countdown
                    allow_eos = remaining == 0 and latest_stream_line_closed(current_text)
                if not allow_eos:
                    continue
            predicted_patch = candidate_patch
            break

        if predicted_patch is None:
            raise RuntimeError("decoder produced only early EOS candidates before countdown completion")

        if (
            len(predicted_patch) >= 2
            and predicted_patch[0] == patchilizer.bos_token_id
            and predicted_patch[1] == patchilizer.eos_token_id
        ):
            break

        generated_patches.append(predicted_patch[:])
        byte_list.extend(patchilizer.decode([predicted_patch]))
        generator.accept_patch(predicted_patch)

        current_text = "".join(byte_list)
        if count_stream_lines(current_text) >= target_total_stream_lines and latest_stream_line_closed(current_text):
            return (
                trim_to_stream_lines(current_text, target_total_stream_lines),
                generated_patches,
                {
                    "stop_reason": "target_stream_lines",
                    "cache_resets": resets,
                    "prompt_stream_lines": prompt_stream_lines,
                    "target_total_stream_lines": target_total_stream_lines,
                },
            )
        if len(byte_list) > max_chars:
            return current_text, generated_patches, {
                "stop_reason": "max_chars",
                "cache_resets": resets,
                "prompt_stream_lines": prompt_stream_lines,
                "target_total_stream_lines": target_total_stream_lines,
            }
        if time.time() - start_time > timeout_s:
            raise RuntimeError(f"generation exceeded {timeout_s}s")

        state = generator.state
        if state is not None and len(state.flat_ids) >= model_shape.patch_length * PATCH_SIZE:
            metadata_lines, tunebody_lines = split_metadata_and_tunebody_lines(current_text)
            if not tunebody_lines:
                raise RuntimeError("stream rollover hit before tunebody generation")
            if cut_index is None:
                cut_index = max(1, len(tunebody_lines) // 2)
            abc_slice = "".join(metadata_lines + tunebody_lines[-cut_index:])
            repatched = patchilizer.encode_generate(abc_slice)
            flat_ids = [int(item) for sublist in repatched for item in sublist]
            generator.reset(flat_ids)
            resets += 1

    return "".join(byte_list), generated_patches, {
        "stop_reason": "terminal_eos",
        "cache_resets": resets,
        "prompt_stream_lines": prompt_stream_lines,
        "target_total_stream_lines": target_total_stream_lines,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--precision", default="bf16", choices=["fp32", "bf16"])
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--target-stream-lines", type=int, default=32)
    parser.add_argument("--target-new-stream-lines", action="store_true")
    parser.add_argument("--max-chars", type=int, default=24000)
    parser.add_argument("--timeout-s", type=int, default=300)
    args = parser.parse_args()

    weights = Path(args.weights)
    prefix_text = Path(args.prefix).read_text(encoding="utf-8")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, model_shape = build_model(weights, precision=args.precision)
    summary = []
    for seed in args.seeds:
        set_seed(seed)
        t0 = time.perf_counter()
        full_text, generated_patches, meta = sample_completion_cached(
            model=model,
            model_shape=model_shape,
            prompt=prefix_text,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            target_stream_lines=args.target_stream_lines,
            target_new_stream_lines=args.target_new_stream_lines,
            max_chars=args.max_chars,
            timeout_s=args.timeout_s,
            precision=args.precision,
        )
        elapsed_s = time.perf_counter() - t0
        out_path = out_dir / f"notagen_large_rerun_cached_seed{seed}.abc"
        out_path.write_text(full_text, encoding="utf-8")
        row = {
            "seed": seed,
            "path": str(out_path),
            "generated_patches": len(generated_patches),
            "chars": len(full_text),
            "stream_lines": count_stream_lines(full_text),
            "new_stream_lines": max(0, count_stream_lines(full_text) - int(meta["prompt_stream_lines"])),
            "elapsed_s": elapsed_s,
            **meta,
        }
        summary.append(row)
        print(json.dumps(row), flush=True)

    summary_path = out_dir / "notagen_large_rerun_cached_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
