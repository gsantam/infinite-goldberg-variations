from __future__ import annotations

import argparse
import json
from pathlib import Path


def split_header_and_body(text: str) -> tuple[str, str]:
    lines = text.splitlines(keepends=True)
    for index, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith("[V:") or stripped.startswith("[r:"):
            return "".join(lines[:index]), "".join(lines[index:])
    return text, ""


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract per-piece NotaGen ABC header prompts and note continuations."
    )
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--prefix-dir", type=Path, required=True)
    parser.add_argument("--continuation-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--include-aria", action="store_true")
    args = parser.parse_args()

    args.prefix_dir.mkdir(parents=True, exist_ok=True)
    args.continuation_dir.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str | int]] = []
    for source in sorted(args.source_dir.rglob("*.abc")):
        if not args.include_aria and source.name.startswith("aria_"):
            continue
        rel = source.relative_to(args.source_dir)
        prefix_path = args.prefix_dir / rel
        continuation_path = args.continuation_dir / rel
        prefix_path.parent.mkdir(parents=True, exist_ok=True)
        continuation_path.parent.mkdir(parents=True, exist_ok=True)

        header, body = split_header_and_body(source.read_text(encoding="utf-8"))
        if not header.endswith("\n"):
            header += "\n"
        prefix_path.write_text(header, encoding="utf-8")
        continuation_path.write_text(body, encoding="utf-8")

        rows.append(
            {
                "source": str(source),
                "prefix": str(prefix_path),
                "continuation": str(continuation_path),
                "header_lines": len(header.splitlines()),
                "continuation_lines": len(body.splitlines()),
            }
        )

    with args.manifest.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    print(f"wrote {len(rows)} header prefixes to {args.prefix_dir}")
    print(f"manifest={args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
