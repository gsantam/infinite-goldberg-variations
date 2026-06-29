#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from grpo.notagen_abc_preprocess import preprocess_notagen_abc


INDEX_NAMES = (
    "augmented.jsonl",
    "augmented_train.jsonl",
    "augmented_eval.jsonl",
)


def preprocess_augmented_abc(source_root: Path, out_root: Path) -> dict[str, int]:
    source_augmented = source_root / "augmented"
    out_augmented = out_root / "augmented"
    if not source_augmented.is_dir():
        raise FileNotFoundError(f"missing augmented directory: {source_augmented}")

    files = 0
    changed = 0
    for source_path in sorted(source_augmented.glob("*/*.abc")):
        rel_path = source_path.relative_to(source_augmented)
        out_path = out_augmented / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        source_text = source_path.read_text(encoding="utf-8")
        out_text = preprocess_notagen_abc(source_text)
        out_path.write_text(out_text, encoding="utf-8")
        files += 1
        changed += int(out_text != source_text)
    return {"abc_files": files, "changed_abc_files": changed}


def rewrite_index(source_index: Path, out_index: Path, source_root: Path, out_root: Path) -> int:
    rows = 0
    out_index.parent.mkdir(parents=True, exist_ok=True)
    with source_index.open("r", encoding="utf-8") as src, out_index.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            if "path" in row:
                source_path = Path(row["path"])
                base_name = source_path.name
                row["path"] = str(out_root / "augmented" / base_name)
            dst.write(json.dumps(row) + "\n")
            rows += 1
    return rows


def preprocess_indices(source_root: Path, out_root: Path) -> dict[str, int]:
    row_counts: dict[str, int] = {}
    for name in INDEX_NAMES:
        source_index = source_root / name
        if not source_index.exists():
            continue
        row_counts[name] = rewrite_index(
            source_index=source_index,
            out_index=out_root / name,
            source_root=source_root,
            out_root=out_root,
        )
    return row_counts


def copy_metadata_files(source_root: Path, out_root: Path) -> list[str]:
    copied = []
    for name in ("sections.txt",):
        source_path = source_root / name
        if source_path.exists():
            out_path = out_root / name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, out_path)
            copied.append(name)
    return copied


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create a NotaGen dataset root whose augmented ABC files have been "
            "preprocessed before training/evaluation. This does not modify the "
            "source dataset."
        )
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing output root.",
    )
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    out_root = args.out_root.resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"missing source root: {source_root}")
    if out_root.exists() and any(out_root.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output root is not empty: {out_root}")
    out_root.mkdir(parents=True, exist_ok=True)

    abc_stats = preprocess_augmented_abc(source_root, out_root)
    index_counts = preprocess_indices(source_root, out_root)
    copied_metadata = copy_metadata_files(source_root, out_root)

    manifest = {
        "source_root": str(source_root),
        "out_root": str(out_root),
        "preprocessing": [
            "strip_unsupported_abc_instructions",
            "expand_notagen_rest_omitted_voice_segments",
        ],
        "abc": abc_stats,
        "indices": index_counts,
        "copied_metadata": copied_metadata,
        "notes": [
            "Prefix masks are intentionally not copied because preprocessing changes text positions.",
            "Rebuild masks from this output root if a run needs prefix masking.",
        ],
    }
    (out_root / "preprocess_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
