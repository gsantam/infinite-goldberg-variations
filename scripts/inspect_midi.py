#!/usr/bin/env python3
"""Inspect a MIDI file and summarize its track structure.

This is meant to answer the first question for the Goldberg source: does the
released MIDI preserve usable symbolic structure for later extraction and
rewarding?
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path

from mido import MidiFile


@dataclass
class TrackSummary:
    index: int
    name: str
    messages: int
    note_on: int
    note_off: int
    programs: list[int]
    channels: list[int]
    pitch_min: int | None
    pitch_max: int | None
    max_simultaneous_notes: int
    total_ticks: int


def summarize_track(index: int, track) -> TrackSummary:
    name = ""
    messages = 0
    note_on = 0
    note_off = 0
    programs: set[int] = set()
    channels: set[int] = set()
    active = 0
    max_active = 0
    pitches: list[int] = []
    total_ticks = 0

    for msg in track:
        messages += 1
        total_ticks += msg.time
        if msg.type == "track_name":
            name = msg.name
        if getattr(msg, "channel", None) is not None:
            channels.add(msg.channel)
        if msg.type == "program_change":
            programs.add(msg.program)
        if msg.type == "note_on" and msg.velocity > 0:
            note_on += 1
            active += 1
            max_active = max(max_active, active)
            pitches.append(msg.note)
        elif msg.type in {"note_off", "note_on"} and getattr(msg, "velocity", 0) == 0:
            note_off += 1
            active = max(0, active - 1)
        elif msg.type == "note_off":
            note_off += 1
            active = max(0, active - 1)

    return TrackSummary(
        index=index,
        name=name,
        messages=messages,
        note_on=note_on,
        note_off=note_off,
        programs=sorted(programs),
        channels=sorted(channels),
        pitch_min=min(pitches) if pitches else None,
        pitch_max=max(pitches) if pitches else None,
        max_simultaneous_notes=max_active,
        total_ticks=total_ticks,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("midi", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    mf = MidiFile(args.midi)
    summaries = [summarize_track(i, track) for i, track in enumerate(mf.tracks)]
    payload = {
        "path": str(args.midi),
        "type": mf.type,
        "ticks_per_beat": mf.ticks_per_beat,
        "track_count": len(mf.tracks),
        "tracks": [asdict(s) for s in summaries],
    }

    if args.json:
        print(json.dumps(payload, indent=2))
        return

    print(f"path: {payload['path']}")
    print(f"type: {payload['type']}")
    print(f"ticks_per_beat: {payload['ticks_per_beat']}")
    print(f"tracks: {payload['track_count']}")
    for track in summaries:
        print(
            f"[{track.index}] name={track.name!r} notes={track.note_on}"
            f" pitch={track.pitch_min}-{track.pitch_max}"
            f" channels={track.channels} programs={track.programs}"
            f" max_active={track.max_simultaneous_notes}"
        )


if __name__ == "__main__":
    main()
