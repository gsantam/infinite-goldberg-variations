#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import statistics
from collections import defaultdict
from pathlib import Path


STRUCTURAL = [
    ("parse_reward", "parse"),
    ("countdown_reward", "countdown"),
    ("line_closure_reward", "line closure"),
    ("bar_token_reward", "bar token"),
    ("meter_alignment_reward", "meter alignment"),
    ("meter_duration_closeness_reward", "meter duration"),
    ("bar_count_reward", "bar count"),
]
HARMONIC = [
    ("root_similarity_reward", "root"),
    ("bass_pitch_class_reward", "bass pc"),
    ("cadence_root_reward", "cadence root"),
    ("cadence_bass_reward", "cadence bass"),
]
COMPONENTS = STRUCTURAL + HARMONIC

COLORS = {
    "mean_reward": "#4e79a7",
    "max_reward": "#f28e2b",
    "bucket_mean_reward": "#1f4e79",
    "bucket_step_max": "#b45f06",
    "mean_bars": "#59a14f",
    "max_bars": "#edc948",
    "zero_fraction": "#e15759",
    "bucket_zero_fraction": "#9d2f31",
    "parse_reward": "#4e79a7",
    "countdown_reward": "#59a14f",
    "line_closure_reward": "#9c755f",
    "bar_token_reward": "#f28e2b",
    "meter_alignment_reward": "#e15759",
    "meter_duration_closeness_reward": "#b07aa1",
    "bar_count_reward": "#edc948",
    "root_similarity_reward": "#76b7b2",
    "bass_pitch_class_reward": "#ff9da7",
    "cadence_root_reward": "#af7aa1",
    "cadence_bass_reward": "#8cd17d",
}


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def load_samples(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["samples"]


def harmony_raw(row: dict) -> float:
    return (
        2.0 * float(row.get("root_similarity_reward", 0.0))
        + 2.0 * float(row.get("bass_pitch_class_reward", 0.0))
        + 3.0 * float(row.get("cadence_root_reward", 0.0))
        + 3.0 * float(row.get("cadence_bass_reward", 0.0))
    )


def harmony_contribution(row: dict) -> float:
    progress = max(0.0, min(1.0, int(row.get("validated_bars") or 0) / 32.0))
    return progress * harmony_raw(row)


def aggregate(samples: list[dict], bucket_size: int) -> tuple[list[dict], list[dict]]:
    by_step: dict[int, list[dict]] = defaultdict(list)
    for sample in samples:
        by_step[int(sample["step"])].append(sample)

    step_points: list[dict] = []
    for step in sorted(by_step):
        rows = by_step[step]
        rewards = [float(row["reward"]) for row in rows]
        bars = [int(row.get("validated_bars") or 0) for row in rows]
        harmony_values = [harmony_contribution(row) for row in rows]
        point = {
            "step": step,
            "sample_count": len(rows),
            "mean_reward": mean(rewards),
            "max_reward": max(rewards),
            "mean_bars": mean(bars),
            "max_bars": max(bars),
            "mean_harmony_contribution": mean(harmony_values),
            "max_harmony_contribution": max(harmony_values),
            "zero_fraction": sum(bar == 0 for bar in bars) / len(bars),
            "reward_ge5": sum(value >= 5.0 for value in rewards),
            "reward_ge7": sum(value >= 7.0 for value in rewards),
            "bars_ge20": sum(bar >= 20 for bar in bars),
        }
        step_points.append(point)

    first = step_points[0]["step"]
    latest = step_points[-1]["step"]
    bucket_points: list[dict] = []
    start = first
    while start <= latest:
        end = start + bucket_size - 1
        present = [point["step"] for point in step_points if start <= point["step"] <= end]
        if present:
            rows = [row for step in present for row in by_step[step]]
            rewards = [float(row["reward"]) for row in rows]
            bars = [int(row.get("validated_bars") or 0) for row in rows]
            harmony_values = [harmony_contribution(row) for row in rows]
            step_maxes = [max(float(row["reward"]) for row in by_step[step]) for step in present]
            step_max_bars = [max(int(row.get("validated_bars") or 0) for row in by_step[step]) for step in present]
            best = max(rows, key=lambda row: float(row["reward"]))
            actual_start = min(present)
            actual_end = max(present)
            is_partial = len(present) < bucket_size
            bucket = {
                "bucket": f"{actual_start:03d}-{actual_end:03d}" + (" partial" if is_partial else ""),
                "start": actual_start,
                "end": actual_end,
                "nominal_start": start,
                "nominal_end": end,
                "partial": is_partial,
                "mid": (start + end) / 2.0,
                "steps": len(present),
                "samples": len(rows),
                "avg_reward": mean(rewards),
                "avg_step_max": mean(step_maxes),
                "best_reward": float(best["reward"]),
                "best_step": int(best["step"]),
                "best_sample": int(best.get("sample", -1)),
                "avg_bars": mean(bars),
                "avg_step_max_bars": mean(step_max_bars),
                "avg_harmony_contribution": mean(harmony_values),
                "max_harmony_contribution": max(harmony_values),
                "zero_fraction": sum(bar == 0 for bar in bars) / len(bars),
                "count_reward_ge5": sum(value >= 5.0 for value in rewards),
                "count_reward_ge7": sum(value >= 7.0 for value in rewards),
                "count_bars_ge20": sum(bar >= 20 for bar in bars),
            }
            for key, _ in COMPONENTS:
                values = [float(row[key]) for row in rows if key in row]
                bucket[key] = mean(values)
            bucket_points.append(bucket)
        start += bucket_size

    return step_points, bucket_points


def scale_points(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def render_evolution_svg(step_points: list[dict], bucket_points: list[dict]) -> str:
    width, height = 1000, 780
    ml, mr, mt, mb = 74, 190, 82, 56
    plot_w = width - ml - mr
    panel_h = 160
    gap = 60
    tops = [mt, mt + panel_h + gap, mt + 2 * (panel_h + gap)]
    axis = "#252a33"
    grid = "#d9dee7"
    text = "#20242c"
    muted = "#5c6470"
    min_step = min(point["step"] for point in step_points)
    max_step = max(point["step"] for point in step_points)
    max_reward = max(max(point["max_reward"] for point in step_points), max(point["avg_step_max"] for point in bucket_points))

    def x(step: float) -> float:
        if max_step == min_step:
            return ml + plot_w / 2
        return ml + (step - min_step) / (max_step - min_step) * plot_w

    def y(value: float, top: float, ymax: float) -> float:
        return top + (1.0 - value / ymax) * panel_h

    def line_for_steps(key: str, top: float, ymax: float) -> str:
        return scale_points([(x(point["step"]), y(point[key], top, ymax)) for point in step_points])

    def line_for_buckets(key: str, top: float, ymax: float) -> str:
        return scale_points([(x(point["mid"]), y(point[key], top, ymax)) for point in bucket_points])

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">GRPO reward evolution</title>',
        '<desc id="desc">Reward, validated bars, and zero-bar rate over GRPO steps.</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{ml}" y="28" font-family="Arial, Helvetica, sans-serif" font-size="20" font-weight="700" fill="{text}">GRPO reward evolution</text>',
        f'<text x="{ml}" y="50" font-family="Arial, Helvetica, sans-serif" font-size="12" fill="{muted}">4 trajectories per step. Thick lines show {bucket_points[0]["end"] - bucket_points[0]["start"] + 1}-step bucket aggregates.</text>',
    ]

    panels = [
        ("Reward", tops[0], max(9.0, max_reward), [0, 2, 4, 6, 8], [("mean_reward", "mean reward", "step"), ("max_reward", "step best reward", "step"), ("avg_reward", "bucket mean reward", "bucket"), ("avg_step_max", "bucket mean step-best", "bucket")]),
        ("Validated bars", tops[1], 32.0, [0, 8, 16, 24, 32], [("mean_bars", "mean bars", "step"), ("max_bars", "step max bars", "step"), ("avg_bars", "bucket mean bars", "bucket"), ("avg_step_max_bars", "bucket mean max bars", "bucket")]),
        ("Zero-bar fraction", tops[2], 1.0, [0, 0.25, 0.5, 0.75, 1.0], [("zero_fraction", "zero-bar fraction", "step"), ("zero_fraction", "bucket zero-bar fraction", "bucket")]),
    ]

    bucket_key_colors = {
        "avg_reward": COLORS["bucket_mean_reward"],
        "avg_step_max": COLORS["bucket_step_max"],
        "avg_bars": "#2f6b35",
        "avg_step_max_bars": "#b28b00",
        "zero_fraction": COLORS["bucket_zero_fraction"],
    }

    for title, top, ymax, ticks, series in panels:
        parts.append(f'<text x="{ml}" y="{top-15}" font-family="Arial, Helvetica, sans-serif" font-size="15" font-weight="700" fill="{text}">{html.escape(title)}</text>')
        for tick in ticks:
            yy = y(tick, top, ymax)
            label = f"{tick:.2g}" if ymax <= 1.0 else f"{tick:g}"
            parts.append(f'<line x1="{ml}" y1="{yy:.2f}" x2="{ml+plot_w}" y2="{yy:.2f}" stroke="{grid}" stroke-width="1"/>')
            parts.append(f'<text x="{ml-12}" y="{yy+4:.2f}" font-family="Arial, Helvetica, sans-serif" font-size="12" fill="{muted}" text-anchor="end">{label}</text>')
        parts.append(f'<line x1="{ml}" y1="{top+panel_h}" x2="{ml+plot_w}" y2="{top+panel_h}" stroke="{axis}" stroke-width="1.4"/>')
        parts.append(f'<line x1="{ml}" y1="{top}" x2="{ml}" y2="{top+panel_h}" stroke="{axis}" stroke-width="1.4"/>')
        for key, label, source in series:
            if source == "step":
                color = COLORS.get(key, "#4e79a7")
                stroke_width = "1.6"
                opacity = "0.35"
                points = line_for_steps(key, top, ymax)
            else:
                color = bucket_key_colors.get(key, COLORS.get(key, "#111827"))
                stroke_width = "3.0"
                opacity = "1.0"
                points = line_for_buckets(key, top, ymax)
            parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="{stroke_width}" opacity="{opacity}" stroke-linejoin="round" stroke-linecap="round" points="{points}"><title>{html.escape(label)}</title></polyline>')

    for point in bucket_points:
        xx = x(point["mid"])
        parts.append(f'<text x="{xx:.2f}" y="{tops[-1]+panel_h+24}" font-family="Arial, Helvetica, sans-serif" font-size="11" fill="{muted}" text-anchor="middle">{html.escape(point["bucket"])}</text>')
    parts.append(f'<text x="{ml+plot_w/2:.2f}" y="{height-16}" font-family="Arial, Helvetica, sans-serif" font-size="13" fill="{text}" text-anchor="middle">GRPO step</text>')

    legend_x, legend_y = ml + plot_w + 26, mt + 6
    legend = [
        (COLORS["mean_reward"], "mean reward", "thin"),
        (COLORS["max_reward"], "step best", "thin"),
        (COLORS["bucket_mean_reward"], "bucket mean", "thick"),
        (COLORS["bucket_step_max"], "bucket step-best", "thick"),
        (COLORS["mean_bars"], "mean bars", "thin"),
        (COLORS["zero_fraction"], "zero-bar frac.", "thin"),
    ]
    parts.append(f'<text x="{legend_x}" y="{legend_y}" font-family="Arial, Helvetica, sans-serif" font-size="13" font-weight="700" fill="{text}">Series</text>')
    legend_y += 20
    for color, label, kind in legend:
        width_s = "3.0" if kind == "thick" else "1.8"
        parts.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x+24}" y2="{legend_y}" stroke="{color}" stroke-width="{width_s}" stroke-linecap="round"/>')
        parts.append(f'<text x="{legend_x+32}" y="{legend_y+4}" font-family="Arial, Helvetica, sans-serif" font-size="11" fill="{text}">{html.escape(label)}</text>')
        legend_y += 20

    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def render_components_svg(bucket_points: list[dict]) -> str:
    width, height = 1000, 700
    ml, mr, mt, mb = 74, 270, 86, 58
    plot_w = width - ml - mr
    panel_h = 210
    gap = 80
    p1_top = mt
    p2_top = mt + panel_h + gap
    axis = "#252a33"
    grid = "#d9dee7"
    text = "#20242c"
    muted = "#5c6470"

    def x(index: int) -> float:
        if len(bucket_points) == 1:
            return ml + plot_w / 2
        return ml + index / (len(bucket_points) - 1) * plot_w

    def y(value: float, top: float) -> float:
        return top + (1.0 - value) * panel_h

    def line(key: str, top: float) -> str:
        return scale_points([(x(i), y(float(point.get(key, 0.0)), top)) for i, point in enumerate(bucket_points)])

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">GRPO reward component means</title>',
        '<desc id="desc">Normalized structural and harmonic reward components averaged by GRPO step bucket.</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{ml}" y="28" font-family="Arial, Helvetica, sans-serif" font-size="20" font-weight="700" fill="{text}">GRPO reward components</text>',
        f'<text x="{ml}" y="50" font-family="Arial, Helvetica, sans-serif" font-size="12" fill="{muted}">Component means over {bucket_points[0]["end"] - bucket_points[0]["start"] + 1}-step buckets. Components are already normalized to [0, 1].</text>',
    ]

    def panel(top: float, title: str, series: list[tuple[str, str]]) -> None:
        parts.append(f'<text x="{ml}" y="{top-16}" font-family="Arial, Helvetica, sans-serif" font-size="15" font-weight="700" fill="{text}">{html.escape(title)}</text>')
        for tick in [0, 0.25, 0.5, 0.75, 1.0]:
            yy = y(tick, top)
            parts.append(f'<line x1="{ml}" y1="{yy:.2f}" x2="{ml+plot_w}" y2="{yy:.2f}" stroke="{grid}" stroke-width="1"/>')
            parts.append(f'<text x="{ml-12}" y="{yy+4:.2f}" font-family="Arial, Helvetica, sans-serif" font-size="12" fill="{muted}" text-anchor="end">{tick:.2g}</text>')
        parts.append(f'<line x1="{ml}" y1="{top+panel_h}" x2="{ml+plot_w}" y2="{top+panel_h}" stroke="{axis}" stroke-width="1.4"/>')
        parts.append(f'<line x1="{ml}" y1="{top}" x2="{ml}" y2="{top+panel_h}" stroke="{axis}" stroke-width="1.4"/>')
        for key, label in series:
            parts.append(f'<polyline fill="none" stroke="{COLORS[key]}" stroke-width="2.2" stroke-linejoin="round" stroke-linecap="round" points="{line(key, top)}"><title>{html.escape(label)}</title></polyline>')

    panel(p1_top, "Structure and format rewards", STRUCTURAL)
    panel(p2_top, "Harmony rewards", HARMONIC)

    for i, point in enumerate(bucket_points):
        parts.append(f'<text x="{x(i):.2f}" y="{p2_top+panel_h+24}" font-family="Arial, Helvetica, sans-serif" font-size="11" fill="{muted}" text-anchor="middle">{html.escape(point["bucket"])}</text>')
    parts.append(f'<text x="{ml+plot_w/2:.2f}" y="{height-16}" font-family="Arial, Helvetica, sans-serif" font-size="13" fill="{text}" text-anchor="middle">GRPO step bucket</text>')

    legend_x, legend_y = ml + plot_w + 30, mt + 2
    parts.append(f'<text x="{legend_x}" y="{legend_y}" font-family="Arial, Helvetica, sans-serif" font-size="13" font-weight="700" fill="{text}">Components</text>')
    legend_y += 20
    for key, label in COMPONENTS:
        parts.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x+24}" y2="{legend_y}" stroke="{COLORS[key]}" stroke-width="2.5" stroke-linecap="round"/>')
        parts.append(f'<text x="{legend_x+34}" y="{legend_y+4}" font-family="Arial, Helvetica, sans-serif" font-size="11" fill="{text}">{html.escape(label)}</text>')
        legend_y += 22

    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def render_harmony_svg(bucket_points: list[dict]) -> str:
    width, height = 1000, 640
    ml, mr, mt, mb = 74, 250, 82, 56
    plot_w = width - ml - mr
    panel_h = 190
    gap = 72
    p1_top = mt
    p2_top = mt + panel_h + gap
    axis = "#252a33"
    grid = "#d9dee7"
    text = "#20242c"
    muted = "#5c6470"
    max_contrib = max(4.5, max(point["max_harmony_contribution"] for point in bucket_points))

    def x(index: int) -> float:
        if len(bucket_points) == 1:
            return ml + plot_w / 2
        return ml + index / (len(bucket_points) - 1) * plot_w

    def y(value: float, top: float, ymax: float) -> float:
        return top + (1.0 - value / ymax) * panel_h

    def line(key: str, top: float, ymax: float) -> str:
        return scale_points([(x(i), y(float(point.get(key, 0.0)), top, ymax)) for i, point in enumerate(bucket_points)])

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">GRPO harmony reward evolution</title>',
        '<desc id="desc">Harmony reward contribution and raw harmony component means averaged by GRPO step bucket.</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{ml}" y="28" font-family="Arial, Helvetica, sans-serif" font-size="20" font-weight="700" fill="{text}">GRPO harmony reward evolution</text>',
        f'<text x="{ml}" y="50" font-family="Arial, Helvetica, sans-serif" font-size="12" fill="{muted}">Top panel is the bar-gated weighted harmony contribution. Bottom panel shows raw normalized harmony components.</text>',
    ]

    parts.append(f'<text x="{ml}" y="{p1_top-16}" font-family="Arial, Helvetica, sans-serif" font-size="15" font-weight="700" fill="{text}">Weighted harmony contribution</text>')
    for tick in [0, 1, 2, 3, 4]:
        yy = y(tick, p1_top, max_contrib)
        parts.append(f'<line x1="{ml}" y1="{yy:.2f}" x2="{ml+plot_w}" y2="{yy:.2f}" stroke="{grid}" stroke-width="1"/>')
        parts.append(f'<text x="{ml-12}" y="{yy+4:.2f}" font-family="Arial, Helvetica, sans-serif" font-size="12" fill="{muted}" text-anchor="end">{tick:g}</text>')
    parts.append(f'<line x1="{ml}" y1="{p1_top+panel_h}" x2="{ml+plot_w}" y2="{p1_top+panel_h}" stroke="{axis}" stroke-width="1.4"/>')
    parts.append(f'<line x1="{ml}" y1="{p1_top}" x2="{ml}" y2="{p1_top+panel_h}" stroke="{axis}" stroke-width="1.4"/>')
    parts.append(f'<polyline fill="none" stroke="#1f4e79" stroke-width="3" stroke-linejoin="round" stroke-linecap="round" points="{line("avg_harmony_contribution", p1_top, max_contrib)}"><title>bucket mean harmony contribution</title></polyline>')
    parts.append(f'<polyline fill="none" stroke="#f28e2b" stroke-width="2.4" stroke-linejoin="round" stroke-linecap="round" points="{line("max_harmony_contribution", p1_top, max_contrib)}"><title>bucket max harmony contribution</title></polyline>')

    parts.append(f'<text x="{ml}" y="{p2_top-16}" font-family="Arial, Helvetica, sans-serif" font-size="15" font-weight="700" fill="{text}">Raw harmony component means</text>')
    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        yy = y(tick, p2_top, 1.0)
        parts.append(f'<line x1="{ml}" y1="{yy:.2f}" x2="{ml+plot_w}" y2="{yy:.2f}" stroke="{grid}" stroke-width="1"/>')
        parts.append(f'<text x="{ml-12}" y="{yy+4:.2f}" font-family="Arial, Helvetica, sans-serif" font-size="12" fill="{muted}" text-anchor="end">{tick:.2g}</text>')
    parts.append(f'<line x1="{ml}" y1="{p2_top+panel_h}" x2="{ml+plot_w}" y2="{p2_top+panel_h}" stroke="{axis}" stroke-width="1.4"/>')
    parts.append(f'<line x1="{ml}" y1="{p2_top}" x2="{ml}" y2="{p2_top+panel_h}" stroke="{axis}" stroke-width="1.4"/>')
    for key, label in HARMONIC:
        parts.append(f'<polyline fill="none" stroke="{COLORS[key]}" stroke-width="2.2" stroke-linejoin="round" stroke-linecap="round" points="{line(key, p2_top, 1.0)}"><title>{html.escape(label)}</title></polyline>')

    for i, point in enumerate(bucket_points):
        parts.append(f'<text x="{x(i):.2f}" y="{p2_top+panel_h+24}" font-family="Arial, Helvetica, sans-serif" font-size="11" fill="{muted}" text-anchor="middle">{html.escape(point["bucket"])}</text>')
    parts.append(f'<text x="{ml+plot_w/2:.2f}" y="{height-16}" font-family="Arial, Helvetica, sans-serif" font-size="13" fill="{text}" text-anchor="middle">GRPO step bucket</text>')

    legend_x, legend_y = ml + plot_w + 30, mt + 2
    legend = [
        ("#1f4e79", "mean contribution"),
        ("#f28e2b", "max contribution"),
        (COLORS["root_similarity_reward"], "root"),
        (COLORS["bass_pitch_class_reward"], "bass pc"),
        (COLORS["cadence_root_reward"], "cadence root"),
        (COLORS["cadence_bass_reward"], "cadence bass"),
    ]
    parts.append(f'<text x="{legend_x}" y="{legend_y}" font-family="Arial, Helvetica, sans-serif" font-size="13" font-weight="700" fill="{text}">Harmony</text>')
    legend_y += 20
    for color, label in legend:
        parts.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x+24}" y2="{legend_y}" stroke="{color}" stroke-width="2.5" stroke-linecap="round"/>')
        parts.append(f'<text x="{legend_x+34}" y="{legend_y+4}" font-family="Arial, Helvetica, sans-serif" font-size="11" fill="{text}">{html.escape(label)}</text>')
        legend_y += 22

    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def render_summary_svg(bucket_points: list[dict]) -> str:
    width, height = 980, 690
    ml, mr, mt, mb = 78, 178, 86, 58
    plot_w = width - ml - mr
    panel_h = 135
    gap = 66
    tops = [mt, mt + panel_h + gap, mt + 2 * (panel_h + gap)]
    axis = "#252a33"
    grid = "#e1e5ec"
    text = "#20242c"
    muted = "#5c6470"
    reward_blue = "#356f95"
    reward_orange = "#d67b2a"
    bar_green = "#3f7f4f"
    bar_gold = "#b58a20"
    zero_red = "#c84e4e"
    latest_bucket = bucket_points[-1]
    bucket_size = int(bucket_points[0].get("nominal_end", bucket_points[0]["end"])) - int(
        bucket_points[0].get("nominal_start", bucket_points[0]["start"])
    ) + 1
    if latest_bucket.get("partial"):
        latest_note = f"Latest bucket is partial: {latest_bucket['steps']}/{bucket_size} steps."
    else:
        latest_note = f"Latest plotted bucket: {latest_bucket['bucket']}."

    def x(index: int) -> float:
        if len(bucket_points) == 1:
            return ml + plot_w / 2
        return ml + index / (len(bucket_points) - 1) * plot_w

    def padded_range(keys: list[str], pad_fraction: float = 0.14) -> tuple[float, float]:
        values = [float(point[key]) for point in bucket_points for key in keys]
        lo = min(values)
        hi = max(values)
        if hi == lo:
            return lo - 0.5, hi + 0.5
        pad = (hi - lo) * pad_fraction
        return lo - pad, hi + pad

    def ticks(ymin: float, ymax: float) -> list[float]:
        return [ymin + (ymax - ymin) * i / 4 for i in range(5)]

    def tick_label(value: float) -> str:
        if abs(value) >= 10:
            return f"{value:.1f}"
        return f"{value:.2f}".rstrip("0").rstrip(".")

    def y(value: float, top: float, ymin: float, ymax: float) -> float:
        return top + (1.0 - (value - ymin) / (ymax - ymin)) * panel_h

    def line(key: str, top: float, ymin: float, ymax: float) -> str:
        return scale_points([(x(i), y(float(point[key]), top, ymin, ymax)) for i, point in enumerate(bucket_points)])

    reward_min, reward_max = padded_range(["avg_reward", "avg_step_max"])
    bars_min, bars_max = padded_range(["avg_bars", "avg_step_max_bars"])
    zero_min, zero_max = padded_range(["zero_fraction"], pad_fraction=0.18)

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">GRPO training summary</title>',
        '<desc id="desc">Clean bucket summary of reward, validated bars, and zero-bar rate during GRPO.</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{ml}" y="28" font-family="Arial, Helvetica, sans-serif" font-size="21" font-weight="700" fill="{text}">GRPO training summary</text>',
        f'<text x="{ml}" y="50" font-family="Arial, Helvetica, sans-serif" font-size="12" fill="{muted}">{bucket_size}-step buckets, 4 trajectories per step. Y-ranges are zoomed to show progression. {html.escape(latest_note)}</text>',
    ]

    def partial_band(top: float) -> None:
        if latest_bucket["steps"] < bucket_size:
            xx = x(len(bucket_points) - 1)
            parts.append(f'<rect x="{xx-30:.2f}" y="{top-22}" width="60" height="{panel_h+22:.2f}" fill="#f8f0dc" opacity="0.75"/>')

    def axes(top: float, title: str, ymin: float, ymax: float) -> None:
        partial_band(top)
        parts.append(f'<text x="{ml}" y="{top-16}" font-family="Arial, Helvetica, sans-serif" font-size="15" font-weight="700" fill="{text}">{html.escape(title)}</text>')
        for tick in ticks(ymin, ymax):
            yy = y(tick, top, ymin, ymax)
            parts.append(f'<line x1="{ml}" y1="{yy:.2f}" x2="{ml+plot_w}" y2="{yy:.2f}" stroke="{grid}" stroke-width="1"/>')
            parts.append(f'<text x="{ml-12}" y="{yy+4:.2f}" font-family="Arial, Helvetica, sans-serif" font-size="12" fill="{muted}" text-anchor="end">{tick_label(tick)}</text>')
        parts.append(f'<line x1="{ml}" y1="{top+panel_h}" x2="{ml+plot_w}" y2="{top+panel_h}" stroke="{axis}" stroke-width="1.4"/>')
        parts.append(f'<line x1="{ml}" y1="{top}" x2="{ml}" y2="{top+panel_h}" stroke="{axis}" stroke-width="1.4"/>')

    axes(tops[0], "Reward", reward_min, reward_max)
    parts.append(f'<polyline fill="none" stroke="{reward_blue}" stroke-width="3.2" stroke-linejoin="round" stroke-linecap="round" points="{line("avg_reward", tops[0], reward_min, reward_max)}"><title>bucket mean reward</title></polyline>')
    parts.append(f'<polyline fill="none" stroke="{reward_orange}" stroke-width="3.2" stroke-linejoin="round" stroke-linecap="round" points="{line("avg_step_max", tops[0], reward_min, reward_max)}"><title>mean of per-step best reward</title></polyline>')

    axes(tops[1], "Validated bars", bars_min, bars_max)
    parts.append(f'<polyline fill="none" stroke="{bar_gold}" stroke-width="3.0" stroke-linejoin="round" stroke-linecap="round" points="{line("avg_bars", tops[1], bars_min, bars_max)}"><title>bucket mean validated bars</title></polyline>')
    parts.append(f'<polyline fill="none" stroke="{bar_green}" stroke-width="3.2" stroke-linejoin="round" stroke-linecap="round" points="{line("avg_step_max_bars", tops[1], bars_min, bars_max)}"><title>mean of per-step max validated bars</title></polyline>')

    axes(tops[2], "Zero-bar rate", zero_min, zero_max)
    parts.append(f'<polyline fill="none" stroke="{zero_red}" stroke-width="3.2" stroke-linejoin="round" stroke-linecap="round" points="{line("zero_fraction", tops[2], zero_min, zero_max)}"><title>zero-bar fraction</title></polyline>')

    for i, point in enumerate(bucket_points):
        xx = x(i)
        parts.append(f'<circle cx="{xx:.2f}" cy="{y(point["avg_reward"], tops[0], reward_min, reward_max):.2f}" r="3.4" fill="#ffffff" stroke="{reward_blue}" stroke-width="1.8"><title>{html.escape(point["bucket"])} avg reward {point["avg_reward"]:.3f}</title></circle>')
        parts.append(f'<circle cx="{xx:.2f}" cy="{y(point["avg_step_max"], tops[0], reward_min, reward_max):.2f}" r="3.4" fill="#ffffff" stroke="{reward_orange}" stroke-width="1.8"><title>{html.escape(point["bucket"])} avg step-best {point["avg_step_max"]:.3f}</title></circle>')
        parts.append(f'<circle cx="{xx:.2f}" cy="{y(point["avg_step_max_bars"], tops[1], bars_min, bars_max):.2f}" r="3.4" fill="#ffffff" stroke="{bar_green}" stroke-width="1.8"><title>{html.escape(point["bucket"])} avg max bars {point["avg_step_max_bars"]:.1f}</title></circle>')
        parts.append(f'<circle cx="{xx:.2f}" cy="{y(point["zero_fraction"], tops[2], zero_min, zero_max):.2f}" r="3.4" fill="#ffffff" stroke="{zero_red}" stroke-width="1.8"><title>{html.escape(point["bucket"])} zero-bar rate {point["zero_fraction"]:.3f}</title></circle>')

    for i, point in enumerate(bucket_points):
        parts.append(f'<text x="{x(i):.2f}" y="{tops[2]+panel_h+24}" font-family="Arial, Helvetica, sans-serif" font-size="11" fill="{muted}" text-anchor="middle">{html.escape(point["bucket"])}</text>')

    legend_x, legend_y = ml + plot_w + 28, mt + 10
    legend = [
        (reward_blue, "mean reward"),
        (reward_orange, "step-best reward"),
        (bar_gold, "mean bars"),
        (bar_green, "step max bars"),
        (zero_red, "zero-bar rate"),
    ]
    parts.append(f'<text x="{legend_x}" y="{legend_y}" font-family="Arial, Helvetica, sans-serif" font-size="13" font-weight="700" fill="{text}">Series</text>')
    legend_y += 22
    for color, label in legend:
        parts.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x+24}" y2="{legend_y}" stroke="{color}" stroke-width="3" stroke-linecap="round"/>')
        parts.append(f'<text x="{legend_x+32}" y="{legend_y+4}" font-family="Arial, Helvetica, sans-serif" font-size="11" fill="{text}">{html.escape(label)}</text>')
        legend_y += 22

    parts.append(f'<text x="{ml+plot_w/2:.2f}" y="{height-16}" font-family="Arial, Helvetica, sans-serif" font-size="13" fill="{text}" text-anchor="middle">GRPO step bucket</text>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--samples-json",
        type=Path,
        default=Path("data/processed/grpo_reward_exports/h100_current_metrics/grpo_reward_samples_latest.json"),
    )
    parser.add_argument("--bucket-size", type=int, default=32)
    parser.add_argument("--output-evolution-svg", type=Path, default=Path("docs/assets/grpo_reward_evolution.svg"))
    parser.add_argument("--output-evolution-json", type=Path, default=Path("docs/assets/grpo_reward_evolution.json"))
    parser.add_argument("--output-components-svg", type=Path, default=Path("docs/assets/grpo_reward_components_32step.svg"))
    parser.add_argument("--output-components-json", type=Path, default=Path("docs/assets/grpo_reward_components_32step.json"))
    parser.add_argument("--output-harmony-svg", type=Path, default=Path("docs/assets/grpo_harmony_evolution.svg"))
    parser.add_argument("--output-harmony-json", type=Path, default=Path("docs/assets/grpo_harmony_evolution.json"))
    parser.add_argument("--output-summary-svg", type=Path, default=Path("docs/assets/grpo_training_summary.svg"))
    parser.add_argument("--output-summary-json", type=Path, default=Path("docs/assets/grpo_training_summary.json"))
    parser.add_argument(
        "--include-partial-bucket",
        action="store_true",
        help="Include incomplete trailing buckets in plot outputs.",
    )
    args = parser.parse_args()

    samples = load_samples(args.samples_json)
    step_points, bucket_points = aggregate(samples, args.bucket_size)
    if not args.include_partial_bucket:
        full_buckets = [point for point in bucket_points if not point.get("partial")]
        if full_buckets:
            bucket_points = full_buckets
    args.output_evolution_svg.parent.mkdir(parents=True, exist_ok=True)
    args.output_components_svg.parent.mkdir(parents=True, exist_ok=True)
    args.output_harmony_svg.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary_svg.parent.mkdir(parents=True, exist_ok=True)
    args.output_evolution_json.write_text(
        json.dumps({"steps": step_points, "buckets": bucket_points}, indent=2),
        encoding="utf-8",
    )
    args.output_components_json.write_text(json.dumps(bucket_points, indent=2), encoding="utf-8")
    args.output_harmony_json.write_text(
        json.dumps(
            [
                {
                    "bucket": point["bucket"],
                    "steps": point["steps"],
                    "avg_harmony_contribution": point["avg_harmony_contribution"],
                    "max_harmony_contribution": point["max_harmony_contribution"],
                    **{key: point.get(key, 0.0) for key, _ in HARMONIC},
                }
                for point in bucket_points
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    args.output_summary_json.write_text(json.dumps(bucket_points, indent=2), encoding="utf-8")
    args.output_evolution_svg.write_text(render_summary_svg(bucket_points), encoding="utf-8")
    args.output_components_svg.write_text(render_components_svg(bucket_points), encoding="utf-8")
    args.output_harmony_svg.write_text(render_harmony_svg(bucket_points), encoding="utf-8")
    args.output_summary_svg.write_text(render_summary_svg(bucket_points), encoding="utf-8")
    print(f"wrote {args.output_evolution_svg}")
    print(f"wrote {args.output_evolution_json}")
    print(f"wrote {args.output_components_svg}")
    print(f"wrote {args.output_components_json}")
    print(f"wrote {args.output_harmony_svg}")
    print(f"wrote {args.output_harmony_json}")
    print(f"wrote {args.output_summary_svg}")
    print(f"wrote {args.output_summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
