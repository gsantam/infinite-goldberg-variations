#!/usr/bin/env python3
"""Convert a simple monophonic LilyPond voice to ABC.

This is a bootstrap converter for Goldberg structure extraction. It supports the
subset needed for the Aria bass voice: relative pitches, note durations, rests,
ties, and bar lines.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

from extract_lilypond_voice import find_assigned_block, split_bars, strip_comments


TOKEN_RE = re.compile(
    r"^(?P<name>[a-grs](?:s|f)?)(?P<octave>[,']*)(?P<duration>\d+)?(?P<dots>\.*)(?P<tie>~?)$"
)

BASE_PC = {
    "c": 0,
    "d": 2,
    "e": 4,
    "f": 5,
    "g": 7,
    "a": 9,
    "b": 11,
}

PC_TO_ABC = {
    0: ("", "C"),
    1: ("^", "C"),
    2: ("", "D"),
    3: ("^", "D"),
    4: ("", "E"),
    5: ("", "F"),
    6: ("^", "F"),
    7: ("", "G"),
    8: ("^", "G"),
    9: ("", "A"),
    10: ("^", "A"),
    11: ("", "B"),
}

DURATION_UNITS = {
    "1": 16,
    "2": 8,
    "4": 4,
    "8": 2,
    "16": 1,
    "32": 0.5,
}


@dataclass
class ConversionState:
    previous_midi: int = 60
    previous_duration: str = "4"
    previous_dots: str = ""


def lilypond_name_to_pc(name: str) -> int:
    if name in {"r", "s"}:
        raise ValueError("rests do not have pitch classes")

    pc = BASE_PC[name[0]]
    suffix = name[1:]
    if suffix == "s":
        pc += 1
    elif suffix == "f":
        pc -= 1
    elif suffix:
        raise ValueError(f"unsupported accidental suffix: {name}")
    return pc % 12


def nearest_relative_pitch(pc: int, previous_midi: int) -> int:
    candidates = [pc + 12 * octave for octave in range(0, 11)]
    return min(candidates, key=lambda midi: (abs(midi - previous_midi), midi))


def resolve_midi(name: str, octave_marks: str, state: ConversionState) -> int:
    pc = lilypond_name_to_pc(name)
    midi = nearest_relative_pitch(pc, state.previous_midi)
    midi += 12 * octave_marks.count("'")
    midi -= 12 * octave_marks.count(",")
    state.previous_midi = midi
    return midi


def duration_to_units(duration: str, dots: str) -> float:
    units = DURATION_UNITS[duration]
    increment = units / 2
    for _ in dots:
        units += increment
        increment /= 2
    return units


def duration_suffix(units: float) -> str:
    if units == 1:
        return ""
    if float(units).is_integer():
        return str(int(units))
    raise ValueError(f"duration is not representable with L:1/16: {units}")


def midi_to_abc(midi: int) -> str:
    accidental, letter = PC_TO_ABC[midi % 12]
    octave = midi // 12 - 1

    if octave < 5:
        suffix = "," * max(0, 4 - octave)
        note = letter + suffix
    else:
        suffix = "'" * max(0, octave - 5)
        note = letter.lower() + suffix

    return accidental + note


def convert_token(token: str, state: ConversionState) -> tuple[str, float]:
    match = TOKEN_RE.match(token)
    if match is None:
        raise ValueError(f"unsupported LilyPond token: {token}")

    name = match.group("name")
    duration = match.group("duration") or state.previous_duration
    dots = match.group("dots") if match.group("duration") else state.previous_dots
    tie = match.group("tie")

    state.previous_duration = duration
    state.previous_dots = dots

    units = duration_to_units(duration, dots)
    suffix = duration_suffix(units)

    if name in {"r", "s"}:
        abc = "z" + suffix
    else:
        midi = resolve_midi(name, match.group("octave"), state)
        abc = midi_to_abc(midi) + suffix

    if tie:
        abc += "-"

    return abc, units


def convert_bar(bar: str, state: ConversionState) -> tuple[str, float]:
    converted: list[str] = []
    total_units = 0.0
    for token in bar.split():
        abc, units = convert_token(token, state)
        converted.append(abc)
        total_units += units
    return " ".join(converted), total_units


def convert_voice(path: Path, voice: str) -> tuple[list[str], list[float]]:
    text = strip_comments(path.read_text(encoding="utf-8"))
    block = find_assigned_block(text, voice)
    bars = split_bars(block)

    state = ConversionState()
    abc_bars: list[str] = []
    durations: list[float] = []
    for bar in bars:
        abc_bar, units = convert_bar(bar, state)
        abc_bars.append(abc_bar)
        durations.append(units)
    return abc_bars, durations


def format_abc(title: str, voice_name: str, bars: list[str]) -> str:
    lines = [
        "X:1",
        f"T:{title}",
        "C:Johann Sebastian Bach",
        "M:3/4",
        "L:1/16",
        "K:G",
        f'V:1 name="{voice_name}"',
        "[V:1] " + " | ".join(bars) + " |]",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--voice", default="leftHandLower")
    parser.add_argument("--title", default="Goldberg Variations Aria - Bass Skeleton")
    parser.add_argument("--voice-name", default="Bass skeleton")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    bars, durations = convert_voice(args.path, args.voice)
    bad_bars = [
        (index + 1, units)
        for index, units in enumerate(durations)
        if abs(units - 12.0) > 1e-9
    ]
    if bad_bars:
        details = ", ".join(f"{index}:{units}" for index, units in bad_bars)
        raise ValueError(f"bars do not sum to 3/4 in L:1/16 units: {details}")

    abc = format_abc(args.title, args.voice_name, bars)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(abc, encoding="utf-8")
    else:
        print(abc, end="")


if __name__ == "__main__":
    main()
