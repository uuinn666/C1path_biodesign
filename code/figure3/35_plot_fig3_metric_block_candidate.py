#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 修改时间：2026-07-27
# 原始实现：Figure 3正式图默认写入代码相邻目录。
# 存在问题：发布仓库的输入和结果已经分开归档。
# 修改内容：从supplementary_materials/figure3读取数据并写入results/figure3。

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
DATA_HELPER = importlib.import_module("32_plot_fig3_assimilation_reaction_landscape")
LAYOUT_HELPER = importlib.import_module("34_plot_fig3_six_panel_candidate")
DEFAULT_DATA = REPO_ROOT / "supplementary_materials/figure3"
DEFAULT_OUTPUT = REPO_ROOT / "results/figure3/Fig3_reaction_analysis"
SUBSTRATES = LAYOUT_HELPER.SUBSTRATES
SUBSTRATE_LABELS = LAYOUT_HELPER.SUBSTRATE_LABELS
CARBON_CLASSES = LAYOUT_HELPER.CARBON_CLASSES
SUBSTRATE_COLORS = {
    "co2": "#08306B",
    "methanol": "#CB181D",
    "formate": "#FC9272",
    "formaldehyde": "#4292C6",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--log-file", type=Path,
        default=REPO_ROOT / "logs/figure3.log",
    )
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


def draw_row_guides(axis: plt.Axes, rows: pd.DataFrame) -> None:
    for y in rows["y"]:
        axis.axhline(y, color="#EDF0F3", linewidth=0.26, zorder=-2)


def prepare_class_data(
    carbon_class: str,
    data: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, set[str]]]:
    summary = data["summary"][data["summary"]["carbon_class"] == carbon_class].copy()
    steps = data["steps"][data["steps"]["carbon_class"] == carbon_class].copy()
    pareto = {
        substrate: LAYOUT_HELPER.pareto_codes(group)
        for substrate, group in summary.groupby("substrate", sort=False)
    }
    return summary, steps, pareto


def set_reaction_labels(axis: plt.Axes, rows: pd.DataFrame, visible: bool) -> None:
    if visible:
        axis.set_yticklabels(
            rows["reaction_code"], fontfamily="DejaVu Sans Mono", fontsize=6.1,
        )
        axis.tick_params(axis="y", length=0, pad=2.2)
    else:
        axis.set_yticklabels([])
        axis.tick_params(axis="y", length=0)


def add_panel_header(axis: plt.Axes, label: str, carbon_class: str, label_x: float) -> None:
    axis.text(
        label_x, 1.025, label, transform=axis.transAxes,
        fontsize=10.5, fontweight="bold", va="bottom",
    )
    axis.text(
        0.0, 1.025, f"{carbon_class} products", transform=axis.transAxes,
        fontsize=7.4, fontweight="bold", va="bottom",
    )


def draw_step_panel(
    axis: plt.Axes,
    carbon_class: str,
    label: str,
    data: dict[str, pd.DataFrame],
    rows: pd.DataFrame,
    blocks: dict[str, dict[str, float]],
    y_max: float,
    show_labels: bool,
) -> None:
    summary, steps, pareto = prepare_class_data(carbon_class, data)
    indexed = summary.set_index(["substrate", "candidate_number"])
    LAYOUT_HELPER.add_block_guides(axis, blocks, show_names=show_labels)
    draw_row_guides(axis, rows)
    for reaction in rows.itertuples(index=False):
        statistics = indexed.loc[(reaction.substrate, reaction.candidate_number)]
        highlight = reaction.reaction_code in pareto.get(reaction.substrate, set())
        values = steps.loc[
            (steps["substrate"] == reaction.substrate)
            & (steps["candidate_number"] == reaction.candidate_number),
            "unified_step_count",
        ].to_numpy(dtype=float)
        if len(values):
            LAYOUT_HELPER.draw_step_box(
                axis, values, reaction.y, SUBSTRATE_COLORS[reaction.substrate], highlight,
            )
        single = statistics["single_reaction_step_median"]
        if pd.notna(single):
            axis.scatter(
                float(single), reaction.y, marker="D", s=10,
                facecolor="white", edgecolor="#20252A", linewidth=0.55, zorder=4,
            )
    LAYOUT_HELPER.style_common_axis(axis, rows, y_max)
    set_reaction_labels(axis, rows, show_labels)
    axis.set_xlim(0, 90)
    axis.set_xticks((0, 20, 40, 60, 80))
    axis.axvline(20, color="#30343B", linewidth=0.75, linestyle=(0, (4, 3)), zorder=1)
    axis.set_xlabel("Pathway steps", fontsize=7.0, labelpad=3)
    add_panel_header(axis, label, carbon_class, -0.31 if show_labels else -0.10)


def draw_frequency_panel(
    axis: plt.Axes,
    carbon_class: str,
    label: str,
    data: dict[str, pd.DataFrame],
    rows: pd.DataFrame,
    blocks: dict[str, dict[str, float]],
    y_max: float,
    show_labels: bool,
) -> None:
    summary, _, pareto = prepare_class_data(carbon_class, data)
    indexed = summary.set_index(["substrate", "candidate_number"])
    LAYOUT_HELPER.add_block_guides(
        axis, blocks, show_names=show_labels,
        sample_sizes=data["sizes"], carbon_class=carbon_class,
    )
    draw_row_guides(axis, rows)
    for reaction in rows.itertuples(index=False):
        statistics = indexed.loc[(reaction.substrate, reaction.candidate_number)]
        highlight = reaction.reaction_code in pareto.get(reaction.substrate, set())
        axis.barh(
            reaction.y, float(statistics["le20_occurrence_percent"]), height=0.50,
            color=SUBSTRATE_COLORS[reaction.substrate], alpha=0.84,
            edgecolor="#20252A" if highlight else "none",
            linewidth=0.65 if highlight else 0.0, zorder=2,
        )
    LAYOUT_HELPER.style_common_axis(axis, rows, y_max)
    set_reaction_labels(axis, rows, show_labels)
    axis.set_xlim(0, 80)
    axis.set_xticks((0, 20, 40, 60, 80))
    axis.set_xlabel("Occurrence in ≤20-step routes (%)", fontsize=7.0, labelpad=3)
    add_panel_header(axis, label, carbon_class, -0.31 if show_labels else -0.10)


def audit_layout(
    figure: plt.Figure,
    label_axes: list[plt.Axes],
    left_last_axis: plt.Axes,
    right_first_axis: plt.Axes,
) -> dict:
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    overlap_count = 0
    for axis in label_axes:
        boxes = [label.get_window_extent(renderer) for label in axis.get_yticklabels() if label.get_text()]
        overlap_count += sum(first.overlaps(second) for first, second in zip(boxes, boxes[1:]))
    right_label_left = min(
        label.get_window_extent(renderer).x0
        for label in right_first_axis.get_yticklabels() if label.get_text()
    )
    metric_block_intrusion = int(right_label_left < left_last_axis.bbox.x1 + 4)
    legend = figure.legends[0].get_window_extent(renderer) if figure.legends else None
    figure_box = figure.bbox
    legend_inside = bool(
        legend is not None
        and legend.x0 >= figure_box.x0 and legend.y0 >= figure_box.y0
        and legend.x1 <= figure_box.x1 and legend.y1 <= figure_box.y1
    )
    result = {
        "visible_axes": len(figure.axes),
        "reaction_tick_label_overlap_count": int(overlap_count),
        "metric_block_label_intrusion_count": metric_block_intrusion,
        "legend_inside_figure": legend_inside,
        "minimum_reaction_tick_font_size_pt": min(
            label.get_fontsize() for axis in label_axes
            for label in axis.get_yticklabels() if label.get_text()
        ),
    }
    result["passed"] = (
        result["visible_axes"] == 6
        and overlap_count == 0
        and metric_block_intrusion == 0
        and legend_inside
        and result["minimum_reaction_tick_font_size_pt"] >= 6.0
    )
    return result


def build_figure(data: dict[str, pd.DataFrame]) -> tuple[plt.Figure, dict]:
    rows, blocks, y_max = LAYOUT_HELPER.build_row_layout(data["mapping"])
    figure = plt.figure(figsize=(18.0, 12.0))
    outer = figure.add_gridspec(1, 2, width_ratios=[1.12, 1.0], wspace=0.27)
    step_grid = outer[0, 0].subgridspec(1, 3, wspace=0.12)
    frequency_grid = outer[0, 1].subgridspec(1, 3, wspace=0.12)
    step_axes = [figure.add_subplot(step_grid[0, index]) for index in range(3)]
    frequency_axes = [figure.add_subplot(frequency_grid[0, index]) for index in range(3)]
    for index, (axis, carbon_class, label) in enumerate(zip(step_axes, CARBON_CLASSES, "ABC")):
        draw_step_panel(
            axis, carbon_class, label, data, rows, blocks, y_max,
            show_labels=index == 0,
        )
    for index, (axis, carbon_class, label) in enumerate(zip(frequency_axes, CARBON_CLASSES, "DEF")):
        draw_frequency_panel(
            axis, carbon_class, label, data, rows, blocks, y_max,
            show_labels=index == 0,
        )

    handles = [
        Patch(facecolor=SUBSTRATE_COLORS[substrate], edgecolor="none", label=SUBSTRATE_LABELS[substrate])
        for substrate in SUBSTRATES
    ]
    handles.extend(
        [
            Line2D([0], [0], marker="*", linestyle="", color="#20252A", markersize=5, label="Mean"),
            Line2D(
                [0], [0], marker="D", linestyle="", markerfacecolor="white",
                markeredgecolor="#20252A", markersize=4, label="Single-reaction median",
            ),
            Patch(facecolor="#D9DEE7", edgecolor="#20252A", linewidth=0.7, label="Pareto-efficient"),
            Line2D([0], [0], color="#30343B", linestyle=(0, (4, 3)), linewidth=0.75, label="20-step threshold"),
        ]
    )
    figure.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=8,
        frameon=False, handlelength=1.35, columnspacing=1.0, handletextpad=0.38,
    )
    figure.subplots_adjust(left=0.055, right=0.992, top=0.905, bottom=0.065)
    left_center = (step_axes[0].get_position().x0 + step_axes[-1].get_position().x1) / 2
    right_center = (frequency_axes[0].get_position().x0 + frequency_axes[-1].get_position().x1) / 2
    figure.text(
        left_center, 0.953, "Pathway-length distributions", ha="center", va="center",
        fontsize=10.2, fontweight="bold", color="#30343B",
    )
    figure.text(
        right_center, 0.953, "Short-pathway reaction frequency", ha="center", va="center",
        fontsize=10.2, fontweight="bold", color="#30343B",
    )
    layout = audit_layout(figure, [step_axes[0], frequency_axes[0]], step_axes[-1], frequency_axes[0])
    return figure, layout


def main() -> None:
    args = parse_args()
    configure_logging(args.log_file)
    STYLE.set_publication_style()
    data = DATA_HELPER.load_data(args.data_dir)
    logging.info("Plot the final four-substrate figure with pathway steps in A-C and frequencies in D-F; smoke=%s", args.smoke)
    figure, layout = build_figure(data)
    if not layout["passed"]:
        raise ValueError(f"Metric-block final figure layout audit failed: {layout}")
    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        figure.savefig(args.output_stem.with_suffix(".png"), dpi=180, bbox_inches="tight", facecolor="white")
    else:
        STYLE.save_figure(figure, args.output_stem)
    (args.output_stem.parent / "fig3_metric_block_layout_qc.json").write_text(
        json.dumps(layout, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    plt.close(figure)
    logging.info("Metric-block final figure completed: %s; layout=%s", args.output_stem, layout)


if __name__ == "__main__":
    main()
