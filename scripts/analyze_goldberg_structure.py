from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from music21 import converter
from music21.pitch import Pitch


SECTION_LENGTH = 32
SECTION_NAMES = ["aria"] + [f"variation-{i:02d}" for i in range(1, 31)] + ["aria-da-capo"]
CADENCE_BARS = {8, 16, 24, 32}


@dataclass
class BarSkeleton:
    bar_index: int
    chord_root: str | None
    bass_pitch_class: str | None
    bass_midi: int | None
    cadence_bar: bool


@dataclass
class SectionSummary:
    name: str
    start_measure: int
    end_measure: int
    measure_count: int
    time_signature: str | None
    key_signature_sharps: int | None


def first_chord_root(measure) -> str | None:
    for chord in measure.recurse().getElementsByClass("Chord"):
        if chord.pitches:
            root = chord.root()
            return root.name if root else None
    return None


def lowest_pitch_info(score, measure_number: int) -> tuple[str | None, int | None]:
    pitches: list[int] = []
    for part in score.parts:
        measure = part.measure(measure_number)
        if measure is None:
            continue
        for note in measure.recurse().notes:
            note_pitches = note.pitches if hasattr(note, "pitches") else [note.pitch]
            pitches.extend(p.midi for p in note_pitches)

    if not pitches:
        return None, None

    lowest_midi = min(pitches)
    pitch_name = Pitch()
    pitch_name.midi = lowest_midi
    return pitch_name.name, lowest_midi


def section_summary(score, start_measure: int, name: str) -> SectionSummary:
    end_measure = start_measure + SECTION_LENGTH - 1
    first_measure = score.parts[0].measure(start_measure)
    ts = first_measure.timeSignature.ratioString if first_measure and first_measure.timeSignature else None
    ks = first_measure.keySignature.sharps if first_measure and first_measure.keySignature else None
    return SectionSummary(
        name=name,
        start_measure=start_measure,
        end_measure=end_measure,
        measure_count=SECTION_LENGTH,
        time_signature=ts,
        key_signature_sharps=ks,
    )


def aria_skeleton(score, start_measure: int = 1) -> list[BarSkeleton]:
    end_measure = start_measure + SECTION_LENGTH - 1
    excerpt = score.measures(start_measure, end_measure)
    chordified = excerpt.chordify()

    bars: list[BarSkeleton] = []
    for offset, measure in enumerate(chordified.recurse().getElementsByClass("Measure"), start=1):
        global_measure_number = start_measure + offset - 1
        bass_pc, bass_midi = lowest_pitch_info(score, global_measure_number)
        bars.append(
            BarSkeleton(
                bar_index=offset,
                chord_root=first_chord_root(measure),
                bass_pitch_class=bass_pc,
                bass_midi=bass_midi,
                cadence_bar=offset in CADENCE_BARS,
            )
        )
    return bars


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    source_path = project_root / "data" / "goldberg-variations.mxl"
    out_dir = project_root / "data" / "processed" / "goldberg" / "structure"
    out_dir.mkdir(parents=True, exist_ok=True)

    score = converter.parse(source_path)

    starts = [1 + SECTION_LENGTH * i for i in range(len(SECTION_NAMES))]
    sections = [section_summary(score, start, name) for start, name in zip(starts, SECTION_NAMES, strict=True)]
    aria = aria_skeleton(score, 1)

    (out_dir / "sections.json").write_text(
        json.dumps([asdict(section) for section in sections], indent=2),
        encoding="utf-8",
    )
    (out_dir / "aria_bar_skeleton.json").write_text(
        json.dumps([asdict(bar) for bar in aria], indent=2),
        encoding="utf-8",
    )

    print(f"Wrote {out_dir / 'sections.json'}")
    print(f"Wrote {out_dir / 'aria_bar_skeleton.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
