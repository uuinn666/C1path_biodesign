
"""Plot the six-panel assimilation-reaction layout with A/B, C/D, and E/F representing C2, C3, and C4 products."""

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
BASE = importlib.import_module("32_plot_fig3_assimilation_reaction_landscape")
DEFAULT_DATA = REPO_ROOT / "supplementary_materials/figure3"
DEFAULT_OUTPUT = REPO_ROOT / "validation/figure3/Fig3_six_panel_candidate"
SUBSTRATES = ("co2", "methanol", "formate", "formaldehyde")
SUBSTRATE_LABELS = {
    "co2": r"CO$_2$",
    "methanol": "Methanol",
    "formate": "Formate",
    "formaldehyde": "Formaldehyde",
}
CARBON_CLASSES = ("C2", "C3", "C4")
SUBSTRATE_COLORS = {
    "co2": "#08306B",
    "methanol": "#CB181D",
    "formate": "#FC9272",
    "formaldehyde": "#4292C6",
}
PAIR_LABELS = {"C2": ("A", "B"), "C3": ("C", "D"), "C4": ("E", "F")}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--log-file", type=Path,
        default=REPO_ROOT / "logs/figure3_six_panel.log",
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


def build_row_layout(mapping: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, float]], float]:
    rows = []
    blocks: dict[str, dict[str, float]] = {}
    cursor = 0.0
    sort_column = "display_rank" if "display_rank" in mapping.columns else "candidate_number"
    for block_index, substrate in enumerate(SUBSTRATES):
        group = mapping[mapping["substrate"] == substrate].sort_values(sort_column)
        first = cursor
        header = first - 0.82
        for _, reaction in group.iterrows():
            row = reaction.to_dict()
            row["y"] = cursor
            rows.append(row)
            cursor += 1.0
        last = cursor - 1.0
        blocks[substrate] = {
            "first": first,
            "last": last,
            "header": header,
            "shade": block_index % 2,
        }
        cursor += 1.15
    return pd.DataFrame(rows), blocks, cursor - 0.65


def add_block_guides(
    axis: plt.Axes,
    blocks: dict[str, dict[str, float]],
    *,
    show_names: bool,
    sample_sizes: pd.DataFrame | None = None,
    carbon_class: str | None = None,
) -> None:
    for substrate, block in blocks.items():
        if block["shade"]:
            axis.axhspan(
                block["header"] - 0.30, block["last"] + 0.48,
                color="#F3F5F7", zorder=-3,
            )
        if show_names:
            axis.text(
                0.01, block["header"], SUBSTRATE_LABELS[substrate],
                transform=axis.get_yaxis_transform(), ha="left", va="center",
                fontsize=7.0, fontweight="bold", color="#30343B",
            )
        if sample_sizes is not None and carbon_class is not None:
            row = sample_sizes[
                (sample_sizes["substrate"] == substrate)
                & (sample_sizes["carbon_class"] == carbon_class)
            ]
            n = int(row.iloc[0]["le20_pathways"]) if len(row) else 0
            axis.text(
                0.98, block["header"], f"n={n}",
                transform=axis.get_yaxis_transform(), ha="right", va="center",
                fontsize=6.2, color=STYLE.NEUTRAL,
            )


def pareto_codes(group: pd.DataFrame) -> set[str]:
    codes = set()
    for _, reaction in group.iterrows():
        dominated = (
            (group["step_median"] <= reaction["step_median"])
            & (group["le20_occurrence_percent"] >= reaction["le20_occurrence_percent"])
            & (
                (group["step_median"] < reaction["step_median"])
                | (group["le20_occurrence_percent"] > reaction["le20_occurrence_percent"])
            )
        ).any()
        if not dominated:
            codes.add(reaction["reaction_code"])
    return codes


def draw_step_box(
    axis: plt.Axes,
    values: np.ndarray,
    y: float,
    color: str,
    highlight: bool,
) -> None:
    edge = "#20252A" if highlight else color
    axis.boxplot(
        [values], positions=[y], vert=False, widths=0.50,
        patch_artist=True, showfliers=False, showmeans=True, manage_ticks=False,
        boxprops={"facecolor": color, "edgecolor": edge, "linewidth": 0.75, "alpha": 0.72},
        medianprops={"color": "white", "linewidth": 0.85},
        meanprops={
            "marker": "*", "markerfacecolor": "#20252A",
            "markeredgecolor": "#20252A", "markersize": 2.3,
        },
        whiskerprops={"color": STYLE.NEUTRAL, "linewidth": 0.50},
        capprops={"color": STYLE.NEUTRAL, "linewidth": 0.50},
    )


def style_common_axis(axis: plt.Axes, rows: pd.DataFrame, y_max: float) -> None:
    axis.set_ylim(y_max, -1.35)
    axis.set_yticks(rows["y"])
    axis.grid(axis="x", color=STYLE.GRID, linewidth=0.55, zorder=-2)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(axis="x", labelsize=7.0)


def draw_pair(
    step_axis: plt.Axes,
    frequency_axis: plt.Axes,
    carbon_class: str,
    data: dict[str, pd.DataFrame],
    rows: pd.DataFrame,
    blocks: dict[str, dict[str, float]],
    y_max: float,
) -> None:
    summary = data["summary"][data["summary"]["carbon_class"] == carbon_class].copy()
    steps = data["steps"][data["steps"]["carbon_class"] == carbon_class]
    pareto = {
        substrate: pareto_codes(group)
        for substrate, group in summary.groupby("substrate", sort=False)
    }
    summary_indexed = summary.set_index(["substrate", "candidate_number"])

    add_block_guides(step_axis, blocks, show_names=True)
    add_block_guides(
        frequency_axis, blocks, show_names=False,
        sample_sizes=data["sizes"], carbon_class=carbon_class,
    )
    for reaction in rows.itertuples(index=False):
        key = (reaction.substrate, reaction.candidate_number)
        statistics = summary_indexed.loc[key]
        highlight = reaction.reaction_code in pareto.get(reaction.substrate, set())
        reaction_color = SUBSTRATE_COLORS[reaction.substrate]
        values = steps.loc[
            (steps["substrate"] == reaction.substrate)
            & (steps["candidate_number"] == reaction.candidate_number),
            "unified_step_count",
        ].to_numpy(dtype=float)
        if len(values):
            draw_step_box(step_axis, values, reaction.y, reaction_color, highlight)
        single = statistics["single_reaction_step_median"]
        if pd.notna(single):
            step_axis.scatter(
                float(single), reaction.y, marker="D", s=10,
                facecolor="white", edgecolor="#20252A", linewidth=0.55, zorder=4,
            )
        frequency_axis.barh(
            reaction.y, float(statistics["le20_occurrence_percent"]), height=0.50,
            color=reaction_color, alpha=0.82,
            edgecolor="#20252A" if highlight else "none",
            linewidth=0.65 if highlight else 0.0, zorder=2,
        )

    for axis in (step_axis, frequency_axis):
        style_common_axis(axis, rows, y_max)
    step_axis.set_yticklabels(
        rows["reaction_code"], fontfamily="DejaVu Sans Mono", fontsize=6.1,
    )
    step_axis.tick_params(axis="y", length=0, pad=2.2)
    frequency_axis.set_yticklabels([])
    frequency_axis.tick_params(axis="y", length=0)

    step_axis.set_xlim(0, 90)
    step_axis.set_xticks((0, 20, 40, 60, 80))
    step_axis.axvline(20, color="#30343B", linewidth=0.75, linestyle=(0, (4, 3)), zorder=1)
    step_axis.set_xlabel("Pathway steps", fontsize=7.0, labelpad=3)
    frequency_axis.set_xlim(0, 80)
    frequency_axis.set_xticks((0, 20, 40, 60, 80))
    frequency_axis.set_xlabel("Occurrence in ≤20-step routes (%)", fontsize=7.0, labelpad=3)

    step_label, frequency_label = PAIR_LABELS[carbon_class]
    step_axis.text(
        -0.31, 1.025, step_label, transform=step_axis.transAxes,
        fontsize=10.5, fontweight="bold", va="bottom",
    )
    frequency_axis.text(
        -0.08, 1.025, frequency_label, transform=frequency_axis.transAxes,
        fontsize=10.5, fontweight="bold", va="bottom",
    )
    step_axis.text(
        0.0, 1.025, "Step distribution", transform=step_axis.transAxes,
        fontsize=7.2, fontweight="bold", va="bottom",
    )
    frequency_axis.text(
        0.12, 1.025, "Short-path frequency", transform=frequency_axis.transAxes,
        fontsize=7.2, fontweight="bold", va="bottom",
    )


def audit_layout(
    figure: plt.Figure,
    step_axes: list[plt.Axes],
    frequency_axes: list[plt.Axes],
) -> dict:
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    overlap_count = 0
    for axis in step_axes:
        boxes = [label.get_window_extent(renderer) for label in axis.get_yticklabels() if label.get_text()]
        overlap_count += sum(first.overlaps(second) for first, second in zip(boxes, boxes[1:]))
    cross_pair_intrusion = 0
    for previous, following in zip(frequency_axes[:-1], step_axes[1:]):
        previous_right = previous.bbox.x1
        label_left = min(
            label.get_window_extent(renderer).x0
            for label in following.get_yticklabels() if label.get_text()
        )
        cross_pair_intrusion += int(label_left < previous_right + 3)
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
        "cross_pair_label_intrusion_count": int(cross_pair_intrusion),
        "legend_inside_figure": legend_inside,
        "minimum_reaction_tick_font_size_pt": min(
            label.get_fontsize() for axis in step_axes
            for label in axis.get_yticklabels() if label.get_text()
        ),
    }
    result["passed"] = (
        result["visible_axes"] == 6
        and overlap_count == 0
        and cross_pair_intrusion == 0
        and legend_inside
        and result["minimum_reaction_tick_font_size_pt"] >= 6.0
    )
    return result


def build_figure(data: dict[str, pd.DataFrame]) -> tuple[plt.Figure, dict]:
    rows, blocks, y_max = build_row_layout(data["mapping"])
    figure = plt.figure(figsize=(18.0, 12.0))
    outer = figure.add_gridspec(1, 3, wspace=0.25)
    step_axes: list[plt.Axes] = []
    frequency_axes: list[plt.Axes] = []
    for container, carbon_class in zip(outer, CARBON_CLASSES):
        pair = container.subgridspec(1, 2, width_ratios=[1.45, 1.0], wspace=0.09)
        step_axis = figure.add_subplot(pair[0, 0])
        frequency_axis = figure.add_subplot(pair[0, 1])
        step_axes.append(step_axis)
        frequency_axes.append(frequency_axis)
        draw_pair(step_axis, frequency_axis, carbon_class, data, rows, blocks, y_max)

    handles = [
        Patch(facecolor=SUBSTRATE_COLORS[substrate], edgecolor="none", label=SUBSTRATE_LABELS[substrate])
        for substrate in SUBSTRATES
    ]
    handles.extend([
        Line2D([0], [0], marker="*", linestyle="", color="#20252A", markersize=5, label="Mean"),
        Line2D(
            [0], [0], marker="D", linestyle="", markerfacecolor="white",
            markeredgecolor="#20252A", markersize=4, label="Single-reaction median",
        ),
        Patch(facecolor="#D9DEE7", edgecolor="#20252A", linewidth=0.7, label="Pareto-efficient"),
        Line2D([0], [0], color="#30343B", linestyle=(0, (4, 3)), linewidth=0.75, label="20-step threshold"),
    ])
    figure.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=8,
        frameon=False, handlelength=1.35, columnspacing=1.0, handletextpad=0.38,
    )
    figure.subplots_adjust(left=0.055, right=0.992, top=0.905, bottom=0.065)
    for carbon_class, step_axis, frequency_axis in zip(CARBON_CLASSES, step_axes, frequency_axes):
        center = (step_axis.get_position().x0 + frequency_axis.get_position().x1) / 2
        figure.text(
            center, 0.955, f"{carbon_class} products", ha="center", va="center",
            fontsize=10.5, fontweight="bold", color="#30343B",
        )
    return figure, audit_layout(figure, step_axes, frequency_axes)


def main() -> None:
    args = parse_args()
    configure_logging(args.log_file)
    STYLE.set_publication_style()
    data = BASE.load_data(args.data_dir)
    logging.info("Plot the horizontal A-F Figure 3 candidate; smoke=%s", args.smoke)
    figure, layout = build_figure(data)
    if not layout["passed"]:
        raise ValueError(f"Six-panel candidate layout audit failed: {layout}")
    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        figure.savefig(args.output_stem.with_suffix(".png"), dpi=180, bbox_inches="tight", facecolor="white")
    else:
        STYLE.save_figure(figure, args.output_stem)
    (args.output_stem.parent / "fig3_six_panel_layout_qc.json").write_text(
        json.dumps(layout, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    plt.close(figure)
    logging.info("Six-panel candidate completed: %s; layout=%s", args.output_stem, layout)


if __name__ == "__main__":
    main()
