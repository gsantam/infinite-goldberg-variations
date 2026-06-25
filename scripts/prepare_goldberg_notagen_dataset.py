from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from music21 import converter, stream


SECTION_STARTS = [
    1,
    33,
    65,
    99,
    115,
    149,
    181,
    217,
    249,
    281,
    297,
    329,
    361,
    393,
    425,
    457,
    489,
    539,
    571,
    603,
    635,
    667,
    683,
    715,
    747,
    779,
    813,
    845,
    877,
    909,
    941,
]

PROMPT_LINES = [
    "%Baroque",
    "%Bach, Johann Sebastian",
    "%Keyboard",
]


def build_section_specs(total_measures: int) -> list[tuple[str, int, int]]:
    starts = SECTION_STARTS + [total_measures + 1]
    specs: list[tuple[str, int, int]] = [("aria", starts[0], starts[1] - 1)]
    for idx in range(1, len(SECTION_STARTS)):
        specs.append((f"variation-{idx:02d}", starts[idx], starts[idx + 1] - 1))
    return specs


def extract_measure_range(score: stream.Score, start_measure: int, end_measure: int) -> stream.Score:
    sliced = stream.Score()
    sliced.metadata = score.metadata
    for part in score.parts:
        sliced.insert(0, part.measures(start_measure, end_measure))
    return sliced


def run_checked(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=str(cwd), check=True)


def prepend_notagen_prompt(abc_path: Path) -> None:
    text = abc_path.read_text(encoding="utf-8")
    if text.startswith(PROMPT_LINES[0] + "\n"):
        return
    prompt_block = "\n".join(PROMPT_LINES) + "\n"
    abc_path.write_text(prompt_block + text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mxl", required=True, help="Path to Goldberg .mxl score")
    parser.add_argument("--out-root", required=True, help="Output root directory")
    parser.add_argument("--notagen-root", required=True, help="Path to NotaGen repo")
    parser.add_argument(
        "--python-exe",
        default=sys.executable,
        help="Python interpreter with NotaGen preprocessing dependencies installed",
    )
    args = parser.parse_args()

    mxl_path = Path(args.mxl).resolve()
    out_root = Path(args.out_root).resolve()
    notagen_root = Path(args.notagen_root).resolve()
    python_exe = Path(args.python_exe)

    musicxml_dir = out_root / "musicxml"
    abc_dir = out_root / "abc"
    interleaved_dir = out_root / "interleaved"
    augmented_dir = out_root / "augmented"

    for path in [musicxml_dir, abc_dir, interleaved_dir, augmented_dir]:
        path.mkdir(parents=True, exist_ok=True)

    score = converter.parse(str(mxl_path))
    total_measures = len(score.parts[0].getElementsByClass("Measure"))
    section_specs = build_section_specs(total_measures)

    manifest_path = out_root / "sections.txt"
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for name, start, end in section_specs:
            manifest.write(f"{name}\t{start}\t{end}\n")
            section_score = extract_measure_range(score, start, end)
            section_musicxml = musicxml_dir / f"{name}.musicxml"
            if section_musicxml.exists():
                section_musicxml.unlink()
            section_score.write("musicxml", fp=str(section_musicxml))

    xml2abc_path = notagen_root / "data" / "xml2abc.py"
    for musicxml_file in sorted(musicxml_dir.glob("*.musicxml")):
        run_checked(
            [str(python_exe), str(xml2abc_path), "-o", str(abc_dir), str(musicxml_file)],
            cwd=notagen_root / "data",
        )
    for abc_file in sorted(abc_dir.glob("*.abc")):
        prepend_notagen_prompt(abc_file)

    preprocess_script = notagen_root / "data" / "2_data_preprocess.py"
    preprocess_tmp = out_root / "_2_data_preprocess_tmp.py"
    template = preprocess_script.read_text(encoding="utf-8")
    patched = template
    patched = patched.replace("ORI_FOLDER = ''", f"ORI_FOLDER = {str(abc_dir)!r}", 1)
    patched = patched.replace("INTERLEAVED_FOLDER = ''", f"INTERLEAVED_FOLDER = {str(interleaved_dir)!r}", 1)
    patched = patched.replace("AUGMENTED_FOLDER = ''", f"AUGMENTED_FOLDER = {str(augmented_dir)!r}", 1)
    patched = patched.replace("EVAL_SPLIT = 0.1", "EVAL_SPLIT = 0.125", 1)
    preprocess_tmp.write_text(patched, encoding="utf-8")
    try:
        run_checked([str(python_exe), str(preprocess_tmp)], cwd=out_root)
    finally:
        preprocess_tmp.unlink(missing_ok=True)

    print(f"Prepared {len(section_specs)} Goldberg sections under {out_root}")
    print(f"Train index: {augmented_dir}_train.jsonl")
    print(f"Eval index:  {augmented_dir}_eval.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
