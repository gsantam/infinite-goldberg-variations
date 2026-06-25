from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path


def split_metadata_and_body(text: str) -> tuple[str, str]:
    lines = text.splitlines(keepends=True)
    body_index = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("V:") and len(stripped) > 2:
            body_index = i
            break
    if body_index is None:
        raise ValueError("could not find tune body")
    return "".join(lines[:body_index]), "".join(lines[body_index:])


def run_checked(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=str(cwd), check=True)


def write_indices(augmented_dir: Path, out_root: Path, eval_variations: set[str]) -> None:
    all_rows: list[dict[str, str]] = []
    train_rows: list[dict[str, str]] = []
    eval_rows: list[dict[str, str]] = []

    for key_dir in sorted(p for p in augmented_dir.iterdir() if p.is_dir()):
        key = key_dir.name
        for abc_file in sorted(key_dir.glob("*.abc")):
            stem = abc_file.stem
            base_name = stem[: -(len(key) + 1)]
            row = {
                "path": str(augmented_dir / base_name),
                "key": key,
            }
            all_rows.append(row)
            if base_name in eval_variations:
                eval_rows.append(row)
            else:
                train_rows.append(row)

    for name, rows in [
        ("augmented.jsonl", all_rows),
        ("augmented_train.jsonl", train_rows),
        ("augmented_eval.jsonl", eval_rows),
    ]:
        path = out_root / name
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-abc-dir", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--notagen-root", required=True)
    parser.add_argument(
        "--python-exe",
        default=sys.executable,
        help="Python interpreter with NotaGen preprocessing dependencies installed",
    )
    parser.add_argument("--eval-count", type=int, default=3)
    args = parser.parse_args()

    source_abc_dir = Path(args.source_abc_dir).resolve()
    out_root = Path(args.out_root).resolve()
    notagen_root = Path(args.notagen_root).resolve()
    python_exe = Path(args.python_exe)

    conditioned_abc_dir = out_root / "abc"
    interleaved_dir = out_root / "interleaved"
    augmented_dir = out_root / "augmented"

    for path in [conditioned_abc_dir, interleaved_dir, augmented_dir]:
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)

    aria_text = (source_abc_dir / "aria.abc").read_text(encoding="utf-8")
    aria_metadata, aria_body = split_metadata_and_body(aria_text)
    variation_names = sorted(src_file.stem for src_file in source_abc_dir.glob("variation-*.abc"))
    if args.eval_count <= 0 or args.eval_count >= len(variation_names):
        raise ValueError("eval-count must be between 1 and the number of variations - 1")
    eval_variations = set(random.sample(variation_names, args.eval_count))

    for src_file in sorted(source_abc_dir.glob("variation-*.abc")):
        variation_text = src_file.read_text(encoding="utf-8")
        _, variation_body = split_metadata_and_body(variation_text)
        conditioned_text = aria_metadata + aria_body + variation_body
        (conditioned_abc_dir / src_file.name).write_text(conditioned_text, encoding="utf-8")

    preprocess_script = notagen_root / "data" / "2_data_preprocess.py"
    preprocess_tmp = out_root / "_2_data_preprocess_tmp.py"
    template = preprocess_script.read_text(encoding="utf-8")
    patched = template
    patched = patched.replace("ORI_FOLDER = ''", f"ORI_FOLDER = {str(conditioned_abc_dir)!r}", 1)
    patched = patched.replace("INTERLEAVED_FOLDER = ''", f"INTERLEAVED_FOLDER = {str(interleaved_dir)!r}", 1)
    patched = patched.replace("AUGMENTED_FOLDER = ''", f"AUGMENTED_FOLDER = {str(augmented_dir)!r}", 1)
    patched = patched.replace("EVAL_SPLIT = 0.1", "EVAL_SPLIT = 0.0", 1)
    preprocess_tmp.write_text(patched, encoding="utf-8")
    try:
        run_checked([str(python_exe), str(preprocess_tmp)], cwd=out_root)
    finally:
        preprocess_tmp.unlink(missing_ok=True)

    write_indices(augmented_dir, out_root, eval_variations)

    split_path = out_root / "eval_variations.txt"
    split_path.write_text("\n".join(sorted(eval_variations)) + "\n", encoding="utf-8")

    print(f"Prepared aria-conditioned Goldberg dataset under {out_root}")
    print(f"Train sections: {len(variation_names) - len(eval_variations)}")
    print(f"Eval sections: {len(eval_variations)} ({', '.join(sorted(eval_variations))})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
