#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 修改时间：2026-07-27
# 原始实现：输入、代码和结果位于同一Figure 3目录。
# 存在问题：三层发布目录会使原默认路径失效。
# 修改内容：统一使用仓库根目录下的supplementary_materials、results和logs。

from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_DIR))
STYLE = importlib.import_module("05_figure_style")
DEFAULT_DATA = REPO_ROOT / "supplementary_materials/figure3"
DEFAULT_OUTPUT = REPO_ROOT / "results/figure3/Fig3_reaction_analysis"
SUBSTRATES = ("co2", "methanol", "formate", "formaldehyde")
SUBSTRATE_TITLES = {
    "co2": r"CO$_2$ assimilation reactions",
    "methanol": "Methanol assimilation reactions",
    "formate": "Formate assimilation reactions",
    "formaldehyde": "Formaldehyde assimilation reactions",
}
CARBON_CLASSES = ("C2", "C3", "C4")
CLASS_COLORS = {"C2": "#3B73B9", "C3": "#8A8F98", "C4": "#C6505C"}
CLASS_OFFSETS = {"C2": -0.23, "C3": 0.0, "C4": 0.23}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--log-file", type=Path, default=REPO_ROOT / "logs/figure3_landscape.log")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def configure_logging(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler()],
    )
    logging.getLogger("fontTools").setLevel(logging.WARNING)


def load_data(directory: Path) -> dict[str, pd.DataFrame]:
    names = {
        "steps": "fig3_step_distribution.tsv",
        "summary": "fig3_reaction_carbon_summary.tsv",
        "mapping": "fig3_assimilation_reaction_mapping.tsv",
        "sizes": "fig3_group_sample_sizes.tsv",
    }
    output = {}
    for key, name in names.items():
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(path)
        output[key] = pd.read_csv(path, sep="\t")
    return output


def style_boxplot(axis: plt.Axes, values: np.ndarray, position: float, color: str) -> None:
    box = axis.boxplot(
        [values], positions=[position], orientation="horizontal", widths=0.17,
        patch_artist=True, showfliers=False, showmeans=True,
        boxprops={"facecolor": color, "edgecolor": color, "linewidth": 0.65, "alpha": 0.72},
        medianprops={"color": "white", "linewidth": 0.9},
        meanprops={"marker": "*", "markerfacecolor": "#20252A", "markeredgecolor": "#20252A", "markersize": 3.2},
        whiskerprops={"color": STYLE.NEUTRAL, "linewidth": 0.55},
        capprops={"color": STYLE.NEUTRAL, "linewidth": 0.55},
    )
    for artist in box["boxes"]:
        artist.set_zorder(2)


def module_axes(container):
    grid = container.subgridspec(1, 2, width_ratios=[1.72, 1.0], wspace=0.07)
    return plt.subplot(grid[0, 0]), plt.subplot(grid[0, 1])


def draw_module(
    step_axis: plt.Axes,
    frequency_axis: plt.Axes,
    substrate: str,
    label: str,
    data: dict[str, pd.DataFrame],
) -> None:
    mapping = data["mapping"][data["mapping"]["substrate"] == substrate].sort_values("candidate_number")
    steps = data["steps"][data["steps"]["substrate"] == substrate]
    summary = data["summary"][data["summary"]["substrate"] == substrate]
    sizes = data["sizes"][data["sizes"]["substrate"] == substrate].set_index("carbon_class")
    codes = mapping["reaction_code"].tolist()
    y = np.arange(len(codes), dtype=float)

    for index, reaction in mapping.reset_index(drop=True).iterrows():
        for carbon_class in CARBON_CLASSES:
            position = index + CLASS_OFFSETS[carbon_class]
            values = steps.loc[
                (steps["candidate_number"] == reaction["candidate_number"])
                & (steps["carbon_class"] == carbon_class),
                "unified_step_count",
            ].to_numpy(dtype=float)
            if len(values):
                style_boxplot(step_axis, values, position, CLASS_COLORS[carbon_class])
            single = summary.loc[
                (summary["candidate_number"] == reaction["candidate_number"])
                & (summary["carbon_class"] == carbon_class),
                "single_reaction_step_median",
            ]
            if len(single) and pd.notna(single.iloc[0]):
                step_axis.scatter(
                    float(single.iloc[0]), position, marker="D", s=14,
                    facecolor="white", edgecolor="#20252A", linewidth=0.65, zorder=4,
                )
            frequency = summary.loc[
                (summary["candidate_number"] == reaction["candidate_number"])
                & (summary["carbon_class"] == carbon_class),
                "le20_occurrence_percent",
            ]
            value = float(frequency.iloc[0]) if len(frequency) else 0.0
            frequency_axis.barh(
                position, value, height=0.17, color=CLASS_COLORS[carbon_class],
                alpha=0.82, edgecolor="none", zorder=2,
            )

    for axis in (step_axis, frequency_axis):
        axis.set_ylim(len(codes) - 0.55, -0.55)
        axis.set_yticks(y)
        axis.grid(axis="x", color=STYLE.GRID, linewidth=0.6, zorder=0)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    step_axis.set_yticklabels(codes, fontfamily="DejaVu Sans Mono", fontsize=6.6)
    step_axis.tick_params(axis="y", length=0, pad=3)
    frequency_axis.set_yticklabels([])
    frequency_axis.tick_params(axis="y", length=0)

    step_axis.set_xlim(0, 90)
    step_axis.set_xticks((0, 20, 40, 60, 80))
    step_axis.axvline(20, color="#30343B", linewidth=0.8, linestyle=(0, (4, 3)), zorder=1)
    step_axis.set_xlabel("Pathway steps (all feasible routes)", fontsize=7.8)
    frequency_axis.set_xlim(0, 100)
    frequency_axis.set_xticks((0, 25, 50, 75, 100))
    frequency_axis.set_xlabel("Occurrence in ≤20-step routes (%)", fontsize=7.8)

    step_axis.text(-0.13, 1.105, label, transform=step_axis.transAxes, fontsize=11, fontweight="bold", va="bottom")
    step_axis.text(0.0, 1.105, SUBSTRATE_TITLES[substrate], transform=step_axis.transAxes, fontsize=9.5, fontweight="bold", va="bottom")
    step_axis.text(0.0, 1.025, "Step distribution", transform=step_axis.transAxes, fontsize=7.2, color=STYLE.NEUTRAL, va="bottom")
    frequency_axis.text(0.0, 1.025, "Short-pathway frequency", transform=frequency_axis.transAxes, fontsize=7.2, color=STYLE.NEUTRAL, va="bottom")
    sample_text = "/".join(str(int(sizes.loc[item, "le20_pathways"])) for item in CARBON_CLASSES)
    frequency_axis.text(
        1.0, 1.025, f"n(C2/C3/C4)={sample_text}", transform=frequency_axis.transAxes,
        ha="right", va="bottom", fontsize=5.8, color=STYLE.NEUTRAL,
    )


def audit_layout(figure: plt.Figure, step_axes: list[plt.Axes]) -> dict:
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    overlap_count = 0
    for axis in step_axes:
        boxes = [label.get_window_extent(renderer) for label in axis.get_yticklabels() if label.get_text()]
        for first, second in zip(boxes, boxes[1:]):
            overlap_count += int(first.overlaps(second))
    figure_box = figure.bbox
    legend = figure.legends[0].get_window_extent(renderer) if figure.legends else None
    legend_inside = bool(
        legend is not None
        and legend.x0 >= figure_box.x0
        and legend.y0 >= figure_box.y0
        and legend.x1 <= figure_box.x1
        and legend.y1 <= figure_box.y1
    )
    result = {
        "visible_axes": len(figure.axes),
        "reaction_tick_label_overlap_count": overlap_count,
        "legend_inside_figure": legend_inside,
        "minimum_reaction_tick_font_size_pt": min(
            label.get_fontsize() for axis in step_axes for label in axis.get_yticklabels() if label.get_text()
        ),
    }
    result["passed"] = (
        result["visible_axes"] == 8
        and overlap_count == 0
        and legend_inside
        and result["minimum_reaction_tick_font_size_pt"] >= 6.5
    )
    return result


def build_figure(data: dict[str, pd.DataFrame]) -> tuple[plt.Figure, dict]:
    figure = plt.figure(figsize=(16.0, 12.6))
    outer = figure.add_gridspec(2, 2, height_ratios=[1.34, 1.0], hspace=0.25, wspace=0.22)
    step_axes = []
    for container, substrate, label in zip(outer, SUBSTRATES, "ABCD"):
        step_axis, frequency_axis = module_axes(container)
        step_axes.append(step_axis)
        draw_module(step_axis, frequency_axis, substrate, label, data)
    handles = [Patch(facecolor=CLASS_COLORS[item], edgecolor="none", label=f"{item} products") for item in CARBON_CLASSES]
    handles.extend(
        [
            Line2D([0], [0], marker="D", linestyle="", markerfacecolor="white", markeredgecolor="#20252A", markersize=4.2, label="Median of single-reaction routes"),
            Line2D([0], [0], color="#30343B", linestyle=(0, (4, 3)), linewidth=0.8, label="20-step threshold"),
        ]
    )
    figure.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.997), ncol=5,
        frameon=False, handlelength=1.5, columnspacing=1.25, handletextpad=0.45,
    )
    figure.subplots_adjust(left=0.085, right=0.985, top=0.945, bottom=0.065)
    return figure, audit_layout(figure, step_axes)


def main() -> None:
    args = parse_args()
    configure_logging(args.log_file)
    STYLE.set_publication_style()
    data = load_data(args.data_dir)
    logging.info("Plot the Figure 3 assimilation-reaction panels; smoke=%s", args.smoke)
    figure, layout = build_figure(data)
    if not layout["passed"]:
        raise ValueError(f"Figure 3 layout audit failed: {layout}")
    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        figure.savefig(args.output_stem.with_suffix(".png"), dpi=180, bbox_inches="tight", facecolor="white")
    else:
        STYLE.save_figure(figure, args.output_stem)
    (args.output_stem.parent / "fig3_layout_qc.json").write_text(
        json.dumps(layout, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    plt.close(figure)
    logging.info("Figure 3 plotting completed: %s; layout=%s", args.output_stem, layout)


if __name__ == "__main__":
    main()
