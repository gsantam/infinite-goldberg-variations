#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import re
import statistics
from pathlib import Path


STRUCTURAL = [
    ("parse_reward", "parse: valid ABC"),
    ("countdown_reward", "stream tags: consistent"),
    ("line_closure_reward", "line closure: closed bars"),
    ("bar_token_reward", "bar token: emits |"),
    ("meter_alignment_reward", "meter alignment: fits meter"),
    ("meter_duration_closeness_reward", "meter duration: close to meter"),
    ("bar_count_reward", "bar count: close to 32"),
]
HARMONIC = [
    ("root_similarity_reward", "root: chord roots vs Aria"),
    ("bass_pitch_class_reward", "bass: pitch classes vs Aria"),
    ("cadence_root_reward", "cadence root: bars 8/16/24/32"),
    ("cadence_bass_reward", "cadence bass: bars 8/16/24/32"),
]
COMPONENTS = STRUCTURAL + HARMONIC


COLORS = {
    "validated_bars": "#111827",
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


def load_baseline(path: Path) -> list[dict]:
    return [
        json.loads(line)["reward_breakdown"]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_epoch(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def mean(rows: list[dict], key: str) -> float:
    return statistics.mean(float(row[key]) for row in rows)


def collect_points(baseline_rewards: Path, sft_scores_dir: Path) -> list[dict]:
    points: list[dict] = []
    rows0 = load_baseline(baseline_rewards)
    points.append(
        {
            "epoch": 0,
            **{key: mean(rows0, key) for key, _ in COMPONENTS},
            "validated_bars": mean(rows0, "validated_bars"),
            "total_reward": mean(rows0, "total_reward"),
        }
    )
    for path in sorted(sft_scores_dir.glob("epoch*_rewards.jsonl")):
        match = re.search(r"epoch(\d+)_", path.name)
        if not match:
            continue
        rows = load_epoch(path)
        points.append(
            {
                "epoch": int(match.group(1)),
                **{key: mean(rows, key) for key, _ in COMPONENTS},
                "validated_bars": mean(rows, "validated_bars"),
                "total_reward": mean(rows, "total_reward"),
            }
        )
    points.sort(key=lambda item: item["epoch"])
    return points


def render_svg(points: list[dict]) -> str:
    width, height = 940, 860
    ml, mr, mt, mb = 74, 274, 92, 58
    plot_w = width - ml - mr
    panel_h = 176
    gap = 68
    p0_top = mt
    p1_top = mt + panel_h + gap
    p2_top = mt + 2 * (panel_h + gap)
    axis_color = "#252a33"
    grid_color = "#d9dee7"
    text_color = "#20242c"
    muted = "#5c6470"

    max_epoch = max(point["epoch"] for point in points)

    def x(epoch: float) -> float:
        return ml + epoch / max_epoch * plot_w

    def y(value: float, top: float) -> float:
        return top + (1.0 - value) * panel_h

    def y_bars(value: float, top: float) -> float:
        return top + (1.0 - max(0.0, min(32.0, value)) / 32.0) * panel_h

    def polyline(key: str, top: float) -> str:
        return " ".join(
            f"{x(point['epoch']):.2f},{y(point[key], top):.2f}" for point in points
        )

    def add_panel(parts: list[str], top: float, title: str, series: list[tuple[str, str]]) -> None:
        parts.append(
            f'<text x="{ml}" y="{top-16}" font-family="Arial, Helvetica, sans-serif" '
            f'font-size="15" font-weight="700" fill="{text_color}">{html.escape(title)}</text>'
        )
        for tick in [0, 0.25, 0.5, 0.75, 1.0]:
            yy = y(tick, top)
            parts.append(
                f'<line x1="{ml}" y1="{yy:.2f}" x2="{ml+plot_w}" y2="{yy:.2f}" '
                f'stroke="{grid_color}" stroke-width="1"/>'
            )
            parts.append(
                f'<text x="{ml-12}" y="{yy+4:.2f}" font-family="Arial, Helvetica, sans-serif" '
                f'font-size="12" fill="{muted}" text-anchor="end">{tick:.2g}</text>'
            )
        for point in points:
            xx = x(point["epoch"])
            parts.append(
                f'<line x1="{xx:.2f}" y1="{top}" x2="{xx:.2f}" y2="{top+panel_h}" '
                f'stroke="{grid_color}" stroke-width="1" opacity="0.45"/>'
            )
        parts.append(
            f'<line x1="{ml}" y1="{top+panel_h}" x2="{ml+plot_w}" y2="{top+panel_h}" '
            f'stroke="{axis_color}" stroke-width="1.4"/>'
        )
        parts.append(
            f'<line x1="{ml}" y1="{top}" x2="{ml}" y2="{top+panel_h}" '
            f'stroke="{axis_color}" stroke-width="1.4"/>'
        )
        parts.append(
            f'<line x1="{x(0):.2f}" y1="{top}" x2="{x(0):.2f}" y2="{top+panel_h}" '
            f'stroke="#111827" stroke-dasharray="4 4" opacity="0.45"/>'
        )
        for key, label in series:
            parts.append(
                f'<polyline fill="none" stroke="{COLORS[key]}" stroke-width="2.2" '
                f'stroke-linejoin="round" stroke-linecap="round" points="{polyline(key, top)}"/>'
            )
            for point in points:
                parts.append(
                    f'<circle cx="{x(point["epoch"]):.2f}" cy="{y(point[key], top):.2f}" '
                    f'r="3.2" fill="#ffffff" stroke="{COLORS[key]}" stroke-width="1.6">'
                    f"<title>{html.escape(label)} epoch {point['epoch']}: {point[key]:.3f}</title>"
                    f"</circle>"
                )

    def add_validated_bars_panel(parts: list[str], top: float) -> None:
        parts.append(
            f'<text x="{ml}" y="{top-16}" font-family="Arial, Helvetica, sans-serif" '
            f'font-size="15" font-weight="700" fill="{text_color}">'
            "Validated musical bars</text>"
        )
        for tick in [0, 8, 16, 24, 32]:
            yy = y_bars(tick, top)
            parts.append(
                f'<line x1="{ml}" y1="{yy:.2f}" x2="{ml+plot_w}" y2="{yy:.2f}" '
                f'stroke="{grid_color}" stroke-width="1"/>'
            )
            parts.append(
                f'<text x="{ml-12}" y="{yy+4:.2f}" font-family="Arial, Helvetica, sans-serif" '
                f'font-size="12" fill="{muted}" text-anchor="end">{tick}</text>'
            )
        for point in points:
            xx = x(point["epoch"])
            parts.append(
                f'<line x1="{xx:.2f}" y1="{top}" x2="{xx:.2f}" y2="{top+panel_h}" '
                f'stroke="{grid_color}" stroke-width="1" opacity="0.45"/>'
            )
        parts.append(
            f'<line x1="{ml}" y1="{top+panel_h}" x2="{ml+plot_w}" y2="{top+panel_h}" '
            f'stroke="{axis_color}" stroke-width="1.4"/>'
        )
        parts.append(
            f'<line x1="{ml}" y1="{top}" x2="{ml}" y2="{top+panel_h}" '
            f'stroke="{axis_color}" stroke-width="1.4"/>'
        )
        parts.append(
            f'<line x1="{x(0):.2f}" y1="{top}" x2="{x(0):.2f}" y2="{top+panel_h}" '
            f'stroke="#111827" stroke-dasharray="4 4" opacity="0.45"/>'
        )
        points_attr = " ".join(
            f"{x(point['epoch']):.2f},{y_bars(point['validated_bars'], top):.2f}" for point in points
        )
        parts.append(
            f'<polyline fill="none" stroke="{COLORS["validated_bars"]}" stroke-width="3.0" '
            f'stroke-linejoin="round" stroke-linecap="round" points="{points_attr}"/>'
        )
        for point in points:
            parts.append(
                f'<circle cx="{x(point["epoch"]):.2f}" cy="{y_bars(point["validated_bars"], top):.2f}" '
                f'r="3.8" fill="#ffffff" stroke="{COLORS["validated_bars"]}" stroke-width="1.8">'
                f"<title>validated bars epoch {point['epoch']}: {point['validated_bars']:.2f} / 32</title>"
                f"</circle>"
            )
        best = max(points, key=lambda point: point["validated_bars"])
        parts.append(
            f'<text x="{x(best["epoch"])+8:.2f}" y="{y_bars(best["validated_bars"], top)-10:.2f}" '
            f'font-family="Arial, Helvetica, sans-serif" font-size="12" font-weight="700" '
            f'fill="{COLORS["validated_bars"]}">best SFT: epoch {best["epoch"]}, '
            f'{best["validated_bars"]:.1f} bars</text>'
        )

    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">'
    )
    parts.append(
        "<title id=\"title\">Reward component means from metadata-only baseline through SFT epochs</title>"
    )
    parts.append(
        '<desc id="desc">Two-panel line chart showing normalized structural and harmonic '
        "reward components and a validated-bar panel. Epoch zero is the base model "
        "with metadata-only prompting.</desc>"
    )
    parts.append('<rect width="100%" height="100%" fill="#ffffff"/>')
    parts.append(
        f'<text x="{ml}" y="28" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="20" font-weight="700" fill="{text_color}">Reward components during SFT</text>'
    )
    parts.append(
        f'<text x="{ml}" y="50" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="12" fill="{muted}">Means over 10 generated samples per epoch. '
        "Corrected stream-tag/bar validator; epoch 0 is the base model.</text>"
    )
    add_validated_bars_panel(parts, p0_top)
    add_panel(parts, p1_top, "Structure and format rewards", STRUCTURAL)
    add_panel(parts, p2_top, "Harmony rewards", HARMONIC)

    for point in points:
        xx = x(point["epoch"])
        parts.append(
            f'<text x="{xx:.2f}" y="{p2_top+panel_h+24}" '
            f'font-family="Arial, Helvetica, sans-serif" font-size="12" '
            f'fill="{muted}" text-anchor="middle">{point["epoch"]}</text>'
        )
    parts.append(
        f'<text x="{ml+plot_w/2:.2f}" y="{height-18}" '
        f'font-family="Arial, Helvetica, sans-serif" font-size="13" '
        f'fill="{text_color}" text-anchor="middle">Epoch</text>'
    )
    parts.append(
        f'<text x="18" y="{(p1_top+p2_top+panel_h)/2:.2f}" '
        f'transform="rotate(-90 18 {(p1_top+p2_top+panel_h)/2:.2f})" '
        f'font-family="Arial, Helvetica, sans-serif" font-size="13" '
        f'fill="{text_color}" text-anchor="middle">Mean component reward / bars</text>'
    )
    parts.append(
        f'<text x="{x(0)+8:.2f}" y="{p2_top+panel_h-10}" '
        f'font-family="Arial, Helvetica, sans-serif" font-size="12" fill="{muted}">base</text>'
    )

    legend_x = ml + plot_w + 30
    legend_y = mt + 2
    parts.append(
        f'<text x="{legend_x+119}" y="{legend_y}" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="13" font-weight="700" fill="{text_color}" '
        f'text-anchor="middle">Components</text>'
    )
    legend_y += 20
    for key, label in COMPONENTS:
        parts.append(
            f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x+24}" y2="{legend_y}" '
            f'stroke="{COLORS[key]}" stroke-width="2.5" stroke-linecap="round"/>'
        )
        parts.append(
            f'<text x="{legend_x+34}" y="{legend_y+4}" '
            f'font-family="Arial, Helvetica, sans-serif" font-size="11" '
            f'fill="{text_color}">{html.escape(label)}</text>'
        )
        legend_y += 22

    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-rewards",
        type=Path,
        default=Path(
            "data/processed/notagen/base_large_metadata_only_cached10_20260626/metadata_only_rewards.jsonl"
        ),
    )
    parser.add_argument(
        "--sft-scores-dir",
        type=Path,
        default=Path(
            "data/processed/notagen/reward_exports/large_sft10_cached10_rewards_refactored_20260626"
        ),
    )
    parser.add_argument(
        "--output-svg",
        type=Path,
        default=Path("docs/assets/sft_reward_breakdown_vs_epoch.svg"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("docs/assets/sft_reward_breakdown_vs_epoch.json"),
    )
    args = parser.parse_args()

    points = collect_points(args.baseline_rewards, args.sft_scores_dir)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_svg.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(points, indent=2), encoding="utf-8")
    args.output_svg.write_text(render_svg(points), encoding="utf-8")
    print(f"wrote {args.output_svg}")
    print(f"wrote {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
