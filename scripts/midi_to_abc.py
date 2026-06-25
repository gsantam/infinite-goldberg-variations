#!/usr/bin/env python3
"""Convert a MIDI file into a simple multi-voice ABC score.

This is intentionally pragmatic rather than notation-perfect. It is designed to
turn the Goldberg MIDI exports into an ABC representation that preserves:
  * meter and bar structure
  * key signature when present
  * multiple independent voices via greedy voice splitting
  * ties across bar boundaries

It does not attempt to preserve dynamics, ornaments, slurs, or exact engraving.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from mido import MidiFile


ACCIDENTAL_TO_ABC = {
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


@dataclass(order=True)
class NoteEvent:
    start: int
    end: int
    pitch: int


@dataclass
class TrackData:
    name: str
    notes: list[NoteEvent]


def quantize_tick(tick: int, unit_ticks: int) -> int:
    return int(round(tick / unit_ticks))


def pitch_to_abc(pitch: int) -> str:
    accidental, letter = ACCIDENTAL_TO_ABC[pitch % 12]
    octave = pitch // 12 - 1
    if octave < 5:
        suffix = "," * max(0, 4 - octave)
        return accidental + letter + suffix

    suffix = "'" * max(0, octave - 5)
    return accidental + letter.lower() + suffix


def duration_to_abc(units: int) -> str:
    if units <= 0:
        raise ValueError(f"invalid ABC duration: {units}")
    if units == 1:
        return ""
    return str(units)


def load_midi(path: Path) -> tuple[list[TrackData], str | None, tuple[int, int] | None, int]:
    midi = MidiFile(path)
    unit_ticks = midi.ticks_per_beat // 4
    min_duration_ticks = max(1, unit_ticks // 4)
    if unit_ticks <= 0:
        raise ValueError(f"unexpected ticks_per_beat: {midi.ticks_per_beat}")

    key_signature: str | None = None
    time_signature: tuple[int, int] | None = None
    tracks: list[TrackData] = []

    for track in midi.tracks:
        abs_tick = 0
        name = ""
        active: dict[tuple[int, int], list[int]] = {}
        notes: list[NoteEvent] = []

        for msg in track:
            abs_tick += msg.time

            if msg.type == "track_name":
                name = msg.name.rstrip(":")
            elif msg.type == "key_signature" and key_signature is None:
                key_signature = msg.key
            elif msg.type == "time_signature" and time_signature is None:
                time_signature = (msg.numerator, msg.denominator)
            elif msg.type == "note_on" and msg.velocity > 0:
                active.setdefault((msg.channel, msg.note), []).append(abs_tick)
            elif msg.type in {"note_off", "note_on"} and (msg.type == "note_off" or msg.velocity == 0):
                key = (msg.channel, msg.note)
                starts = active.get(key)
                if not starts:
                    continue
                start_tick = starts.pop()
                if not starts:
                    active.pop(key, None)
                duration_ticks = abs_tick - start_tick
                if duration_ticks < min_duration_ticks:
                    continue
                start = quantize_tick(start_tick, unit_ticks)
                end = quantize_tick(abs_tick, unit_ticks)
                if end <= start:
                    continue
                notes.append(NoteEvent(start=start, end=end, pitch=msg.note))

        if notes:
            notes.sort()
            tracks.append(TrackData(name=name or f"track_{len(tracks) + 1}", notes=notes))

    return tracks, key_signature, time_signature, unit_ticks


def split_track_into_voices(notes: list[NoteEvent]) -> list[list[NoteEvent]]:
    voices: list[list[NoteEvent]] = []
    voice_ends: list[int] = []

    for note in notes:
        chosen = None
        chosen_end = None
        for index, end in enumerate(voice_ends):
            if end <= note.start and (chosen_end is None or end > chosen_end):
                chosen = index
                chosen_end = end

        if chosen is None:
            voices.append([note])
            voice_ends.append(note.end)
        else:
            voices[chosen].append(note)
            voice_ends[chosen] = note.end

    return voices


def extract_highest_melody(notes: list[NoteEvent]) -> list[NoteEvent]:
    """Collapse polyphonic notes into a monophonic top line.

    Group notes by quantized onset, keep the highest pitch at each onset,
    and truncate earlier notes when a later onset begins.
    """
    by_start: dict[int, list[NoteEvent]] = {}
    for note in notes:
        by_start.setdefault(note.start, []).append(note)

    melody: list[NoteEvent] = []
    for start in sorted(by_start):
        chosen = max(by_start[start], key=lambda n: (n.pitch, n.end - n.start))
        if melody and melody[-1].end > chosen.start:
            melody[-1] = NoteEvent(
                start=melody[-1].start,
                end=chosen.start,
                pitch=melody[-1].pitch,
            )
            if melody[-1].end <= melody[-1].start:
                melody.pop()
        melody.append(chosen)
    return melody


def split_note_to_bars(note: NoteEvent, bar_units: int) -> list[tuple[int, int, int, bool]]:
    pieces: list[tuple[int, int, int, bool]] = []
    start = note.start
    while start < note.end:
        bar_index = start // bar_units
        bar_end = (bar_index + 1) * bar_units
        end = min(note.end, bar_end)
        pieces.append((bar_index, start % bar_units, end - start, end < note.end))
        start = end
    return pieces


def build_voice_bars(notes: list[NoteEvent], bar_units: int, total_bars: int) -> list[str]:
    bars = [[] for _ in range(total_bars)]
    positions = [0 for _ in range(total_bars)]

    for note in notes:
        for bar_index, offset, length, tied_forward in split_note_to_bars(note, bar_units):
            while positions[bar_index] < offset:
                gap = offset - positions[bar_index]
                bars[bar_index].append("z" + duration_to_abc(gap))
                positions[bar_index] = offset

            token = pitch_to_abc(note.pitch) + duration_to_abc(length)
            if tied_forward:
                token += "-"
            bars[bar_index].append(token)
            positions[bar_index] += length

    for bar_index in range(total_bars):
        while positions[bar_index] < bar_units:
            gap = bar_units - positions[bar_index]
            bars[bar_index].append("z" + duration_to_abc(gap))
            positions[bar_index] = bar_units

    return [" ".join(tokens) for tokens in bars]


def abc_from_midi(
    path: Path,
    title: str | None = None,
    max_voices: int | None = None,
    melody_only: bool = False,
) -> str:
    tracks, key_signature, time_signature, _unit_ticks = load_midi(path)
    if not tracks:
        raise ValueError(f"no note data found in {path}")

    if time_signature is None:
        time_signature = (4, 4)
    numerator, denominator = time_signature
    bar_units = int(numerator * (16 / denominator))

    split_voices: list[tuple[str, list[NoteEvent]]] = []
    max_end = 0
    if melody_only:
        preferred = next((t for t in tracks if "upper" in t.name.lower()), tracks[0])
        melody = extract_highest_melody(preferred.notes)
        split_voices.append((f"{preferred.name}_melody", melody))
        max_end = max(note.end for note in melody)
    else:
        for track in tracks:
            for voice_index, voice_notes in enumerate(split_track_into_voices(track.notes), start=1):
                split_voices.append((f"{track.name}_{voice_index}", voice_notes))
                max_end = max(max_end, max(note.end for note in voice_notes))

    if max_voices is not None:
        split_voices = split_voices[:max_voices]
        if not split_voices:
            raise ValueError(f"no voices retained for {path}")
        max_end = max(max(note.end for note in notes) for _voice_name, notes in split_voices)

    total_bars = max(1, (max_end + bar_units - 1) // bar_units)
    abc_lines = [
        "X:1",
        f"T:{title or path.stem}",
        "C:Johann Sebastian Bach",
        f"M:{numerator}/{denominator}",
        "L:1/16",
        f"% original_key:{key_signature or 'unknown'}",
        "K:C",
    ]

    for index, (voice_name, _notes) in enumerate(split_voices, start=1):
        abc_lines.append(f'V:{index} name="{voice_name}"')

    for index, (_voice_name, notes) in enumerate(split_voices, start=1):
        bars = build_voice_bars(notes, bar_units, total_bars)
        abc_lines.append(f"[V:{index}] " + " | ".join(bars) + " |]")

    return "\n".join(abc_lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("midi", type=Path)
    parser.add_argument("--title", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-voices", type=int, default=None)
    parser.add_argument("--melody-only", action="store_true")
    args = parser.parse_args()

    abc = abc_from_midi(
        args.midi,
        title=args.title,
        max_voices=args.max_voices,
        melody_only=args.melody_only,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(abc, encoding="utf-8")
        return
    print(abc, end="")


if __name__ == "__main__":
    main()
