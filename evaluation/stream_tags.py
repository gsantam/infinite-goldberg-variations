from __future__ import annotations

import re
from dataclasses import dataclass


STREAM_TAG_RE = re.compile(r"^\[r:(\d+)/(\d+)\](.*)$")


@dataclass(frozen=True)
class StreamTag:
    index: int
    marker: int


@dataclass(frozen=True)
class StreamLine:
    tag: StreamTag
    body: str
    raw: str


def parse_stream_line(line: str) -> StreamLine | None:
    match = STREAM_TAG_RE.match(line.strip())
    if not match:
        return None
    return StreamLine(
        tag=StreamTag(index=int(match.group(1)), marker=int(match.group(2))),
        body=match.group(3),
        raw=line.strip(),
    )


def extract_stream_lines(text: str) -> list[StreamLine]:
    out: list[StreamLine] = []
    for line in text.splitlines():
        parsed = parse_stream_line(line)
        if parsed is not None:
            out.append(parsed)
    return out


def count_stream_lines(text: str) -> int:
    return len(extract_stream_lines(text))


def latest_stream_line(text: str) -> StreamLine | None:
    lines = extract_stream_lines(text)
    return lines[-1] if lines else None


def stream_line_closed(line: str | StreamLine) -> bool:
    raw = line.raw if isinstance(line, StreamLine) else line.rstrip()
    return raw.endswith("|") or raw.endswith("|]") or raw.endswith(":|") or raw.endswith("||")


def latest_stream_line_closed(text: str) -> bool:
    latest = latest_stream_line(text)
    return latest is not None and stream_line_closed(latest)


def trim_to_stream_lines(text: str, target_lines: int) -> str:
    out_lines: list[str] = []
    seen = 0
    for line in text.splitlines(keepends=True):
        out_lines.append(line)
        if parse_stream_line(line) is not None:
            seen += 1
            if seen >= target_lines:
                break
    return "".join(out_lines)


def stream_tag_sequence_reward(lines: list[StreamLine]) -> float:
    if not lines:
        return 0.0
    return max(
        _decreasing_countdown_reward(lines),
        _index_total_reward(lines),
    )


def _decreasing_countdown_reward(lines: list[StreamLine]) -> float:
    score = 0.0
    checks = 0

    first = lines[0].tag
    checks += 1
    if first.index == 0:
        score += 1.0

    for prev_line, curr_line in zip(lines, lines[1:]):
        prev = prev_line.tag
        curr = curr_line.tag
        checks += 1
        if curr.index == prev.index + 1 and curr.marker == prev.marker - 1:
            score += 1.0

    checks += 1
    if lines[-1].tag.marker == 0:
        score += 1.0

    return score / checks


def _index_total_reward(lines: list[StreamLine]) -> float:
    score = 0.0
    checks = 0

    first = lines[0].tag
    total_or_last = first.marker
    checks += 1
    if first.index in (0, 1):
        score += 1.0

    for prev_line, curr_line in zip(lines, lines[1:]):
        prev = prev_line.tag
        curr = curr_line.tag
        checks += 2
        if curr.index == prev.index + 1:
            score += 1.0
        if curr.marker == total_or_last:
            score += 1.0

    checks += 1
    if lines[-1].tag.index in (total_or_last, total_or_last - 1):
        score += 1.0

    return score / checks if checks > 0 else 0.0


def stream_target_reached(text: str, target_stream_lines: int) -> bool:
    return count_stream_lines(text) >= target_stream_lines and latest_stream_line_closed(text)

