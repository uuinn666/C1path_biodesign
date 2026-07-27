#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Plot Figure 2 from the validated unified pathway dataset."""

# 修改时间：2026-07-27
# 原始实现：代码、源数据和结果位于同一个Figure 2目录。
# 存在问题：公开仓库按代码、补充材料和结果分类后，原相对路径无法定位输入数据。
# 修改内容：从supplementary_materials/figure2读取数据，并将正式输出写入results/figure2。

from __future__ import annotations

import argparse
import importlib
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d


CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_DIR))
STYLE = importlib.import_module("05_figure_style")
DEFAULT_DATA = REPO_ROOT / "supplementary_materials/figure2"
DEFAULT_OUTPUT = REPO_ROOT / "results/figure2/Fig2_pathway_overview"

FIG2_SUBSTRATE_COLORS = {
    "co2": "#08306B",
    "formaldehyde": "#2171B5",
    "formate": "#FC9272",
    "methanol": "#CB181D",
}
FIG2_STAGE_COLORS = {
    "raw_feasible_pathways": "#08306B",
    "le20_pathways": "#FC9272",
    "retained_pathways": "#CB181D",
}
FIG15_SUBSTRATE_COLORS = FIG2_SUBSTRATE_COLORS

DRIVE_EDGES = [-np.inf, 0, 50, 100, 150, 250, np.inf]
DRIVE_LABELS = ["≤0", "0–50", "50–100", "100–150", "150–250", ">250"]
RIDGE_COLORS = ["#DCEAF4", "#B8D5E8", "#86BBD8", "#4F9BC6", "#2576A8", "#0B4F6C"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--log-file", type=Path, default=REPO_ROOT / "logs/figure2.log")
    parser.add_argument("--smoke", action="store_true", help="Write only a 180 dpi PNG for visual smoke testing")
    return parser.parse_args()


def configure_logging(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler()],
    )


def read_data(directory: Path, panel: str) -> pd.DataFrame:
    return pd.read_csv(directory / f"{panel}_source.tsv", sep="\t")


def ordered(table: pd.DataFrame) -> pd.DataFrame:
    return table.assign(
        substrate=pd.Categorical(table["substrate"], STYLE.SUBSTRATES, ordered=True)
    ).sort_values("substrate")


def positions() -> tuple[np.ndarray, list[str]]:
    x = np.arange(len(STYLE.SUBSTRATES))
    return x, [STYLE.SUBSTRATE_LABELS[item] for item in STYLE.SUBSTRATES]


def violin_box(axis: plt.Axes, table: pd.DataFrame, metric: str, maximum_points: int = 360) -> None:
    rng = np.random.default_rng(20260714)
    x, labels = positions()
    groups = [table.loc[table["substrate"] == substrate, metric].dropna().to_numpy() for substrate in STYLE.SUBSTRATES]
    violins = axis.violinplot(groups, positions=x, widths=0.78, showextrema=False)
    for body, substrate in zip(violins["bodies"], STYLE.SUBSTRATES):
        body.set_facecolor(FIG15_SUBSTRATE_COLORS[substrate])
        body.set_edgecolor("none")
        body.set_alpha(0.16)
    boxes = axis.boxplot(
        groups,
        positions=x,
        widths=0.25,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "white", "linewidth": 1.4},
    )
    for patch, substrate in zip(boxes["boxes"], STYLE.SUBSTRATES):
        patch.set_facecolor(FIG15_SUBSTRATE_COLORS[substrate])
        patch.set_edgecolor(FIG15_SUBSTRATE_COLORS[substrate])
        patch.set_alpha(0.70)
    for item in boxes["whiskers"] + boxes["caps"]:
        item.set_color(STYLE.NEUTRAL)
        item.set_linewidth(0.65)
    for index, (substrate, values) in enumerate(zip(STYLE.SUBSTRATES, groups)):
        sampled = STYLE.deterministic_sample(np.sort(values), maximum_points)
        jitter = rng.uniform(-0.17, 0.17, len(sampled))
        axis.scatter(
            index + jitter,
            sampled,
            s=4,
            color=FIG15_SUBSTRATE_COLORS[substrate],
            alpha=0.15,
            edgecolors="none",
            rasterized=True,
        )
        axis.scatter(index, np.mean(values), marker="D", s=18, facecolor="white", edgecolor=STYLE.NEUTRAL, linewidth=0.7, zorder=5)
    axis.set_xticks(x, labels, rotation=18, ha="right")
    STYLE.clean_axis(axis)


def panel_a(axis: plt.Axes, table: pd.DataFrame) -> None:
    table = ordered(table).set_index("substrate").loc[STYLE.SUBSTRATES]
    x, labels = positions()
    bars = axis.bar(
        x,
        table["candidate_assimilation_reaction_count"],
        width=0.68,
        color=[FIG2_SUBSTRATE_COLORS[item] for item in STYLE.SUBSTRATES],
        alpha=0.75,
    )
    axis.bar_label(bars, padding=3, fontsize=8, fontweight="bold")
    axis.set_xticks(x, labels, rotation=18, ha="right")
    axis.set_ylabel("Candidate assimilation reactions")
    axis.set_ylim(0, table["candidate_assimilation_reaction_count"].max() * 1.20)
    STYLE.clean_axis(axis)
    STYLE.panel_label(axis, "A", "Substrate-specific candidate sets")


def panel_b(axis: plt.Axes, table: pd.DataFrame) -> None:
    violin_box(axis, table, "unified_step_count")
    axis.axhline(20, color=STYLE.REDS[4], linewidth=1.0, linestyle=(0, (4, 3)))
    axis.text(3.45, 21.5, "≤20-step screen", color=STYLE.REDS[5], fontsize=7, ha="right")
    axis.set_ylabel("Unified pathway steps")
    STYLE.panel_label(axis, "B", "Pathway-length distributions before screening")


def reducing_equivalent_boxes(axis: plt.Axes, table: pd.DataFrame) -> None:
    x, labels = positions()
    width = 0.24
    for index, substrate in enumerate(STYLE.SUBSTRATES):
        subset = table[table["substrate"] == substrate]
        for offset, metric, alpha in (
            (-width / 1.5, "corrected_NADH_auxiliary_equivalents_per_product_Cmol", 0.70),
            (width / 1.5, "corrected_NADPH_auxiliary_equivalents_per_product_Cmol", 0.34),
        ):
            values = subset[metric].dropna().to_numpy()
            box = axis.boxplot(
                [values], positions=[index + offset], widths=width,
                patch_artist=True, showfliers=False,
                medianprops={"color": "white", "linewidth": 1.1},
            )
            box["boxes"][0].set_facecolor(FIG15_SUBSTRATE_COLORS[substrate])
            box["boxes"][0].set_edgecolor(FIG15_SUBSTRATE_COLORS[substrate])
            box["boxes"][0].set_alpha(alpha)
            for item in box["whiskers"] + box["caps"]:
                item.set_color(STYLE.NEUTRAL)
                item.set_linewidth(0.6)
            axis.scatter(index + offset, np.mean(values), marker="D", s=13, facecolor="white", edgecolor=STYLE.NEUTRAL, linewidth=0.6, zorder=5)
    axis.axhline(0, color=STYLE.NEUTRAL, linewidth=0.8)
    axis.set_xticks(x, labels, rotation=18, ha="right")
    handles = [
        Line2D([0], [0], color=STYLE.NEUTRAL, linewidth=7, alpha=0.70, label="NADH"),
        Line2D([0], [0], color=STYLE.NEUTRAL, linewidth=7, alpha=0.34, label="NADPH"),
    ]
    axis.legend(handles=handles, frameon=False, loc="upper right", ncol=2, handlelength=1.0)
    STYLE.clean_axis(axis)


def panel_c(container, table: pd.DataFrame) -> None:
    grid = container.subgridspec(1, 2, wspace=0.34)
    atp = plt.subplot(grid[0, 0])
    redox = plt.subplot(grid[0, 1])
    violin_box(atp, table, "corrected_ATP_auxiliary_equivalents_per_product_Cmol", maximum_points=280)
    atp.axhline(0, color=STYLE.NEUTRAL, linewidth=0.8)
    atp.set_ylabel(r"Auxiliary equivalents (mol mol-product-C$^{-1}$)")
    atp.set_title("ATP", fontsize=8.5, fontweight="bold", pad=4)
    reducing_equivalent_boxes(redox, table)
    redox.set_ylabel("")
    redox.set_title("Reducing equivalents", fontsize=8.5, fontweight="bold", pad=4)
    atp.text(-0.16, 1.14, "D", transform=atp.transAxes, fontsize=11, fontweight="bold", va="top")
    atp.text(0.00, 1.14, "Net auxiliary cofactor requirements", transform=atp.transAxes, fontsize=10, fontweight="bold", va="top")


def panel_d(axis: plt.Axes, table: pd.DataFrame) -> None:
    table = ordered(table).set_index("substrate").loc[STYLE.SUBSTRATES]
    x, labels = positions()
    width = 0.24
    specifications = (
        ("raw_feasible_pathways", -width, FIG2_STAGE_COLORS["raw_feasible_pathways"], "Feasible"),
        ("le20_pathways", 0.0, FIG2_STAGE_COLORS["le20_pathways"], "≤20 steps"),
        ("retained_pathways", width, FIG2_STAGE_COLORS["retained_pathways"], "Retained"),
    )
    for metric, offset, color, label in specifications:
        bars = axis.bar(x + offset, table[metric], width, color=color, alpha=0.75, label=label)
        for bar in bars:
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.12,
                f"{int(bar.get_height()):,}",
                ha="center", va="bottom", fontsize=6.3, rotation=90,
            )
    axis.set_yscale("log")
    axis.set_ylim(1, max(table["raw_feasible_pathways"]) * 4.0)
    axis.set_xticks(x, labels, rotation=18, ha="right")
    axis.set_ylabel("Pathway count (log scale)")
    axis.legend(frameon=False, loc="upper right", ncol=1)
    STYLE.clean_axis(axis)
    STYLE.panel_label(axis, "C", "Contraction of the feasible route space")


def prepare_ridgeline_data(thermo: pd.DataFrame) -> pd.DataFrame:
    """Compute cumulative driving force and fixed bins without changing the original delta G or MDF values."""

    table = thermo.copy()
    table["drive"] = -table["flux_weighted_total_optimized_dg_kJ_per_product_Cmol"]
    table["drive_bin"] = pd.cut(
        table["drive"], DRIVE_EDGES, labels=DRIVE_LABELS,
        include_lowest=True, right=True,
    )
    return table


def smoothed_density(values: np.ndarray, x_edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply the validated histogram Gaussian smoothing and normalize each ridge independently."""

    histogram, edges = np.histogram(values, bins=x_edges, density=False)
    density = gaussian_filter1d(histogram.astype(float), sigma=2.0)
    if density.max() > 0:
        density /= density.max()
    centres = (edges[:-1] + edges[1:]) / 2.0
    return centres, density


def panel_e(container, thermo: pd.DataFrame) -> None:
    table = prepare_ridgeline_data(thermo)
    grid = container.subgridspec(2, 1, height_ratios=[0.16, 1.0], hspace=0.04)
    header = plt.subplot(grid[0, 0])
    header.axis("off")
    header.text(-0.11, 0.96, "E", transform=header.transAxes, fontsize=11, fontweight="bold", va="top")
    header.text(
        0.0, 0.96, "MDF distributions across cumulative driving-force tiers",
        transform=header.transAxes, fontsize=10, fontweight="bold", va="top",
    )
    header.text(
        0.99, 0.06,
        "ridge: within-tier distribution   dot: median   red line: MDF=0",
        transform=header.transAxes, ha="right", va="bottom", fontsize=5.6, color=STYLE.NEUTRAL,
    )
    axis = plt.subplot(grid[1, 0])
    x_min = table["mdf_kJ_per_mol"].min() - 3
    x_max = table["mdf_kJ_per_mol"].max() + 3
    x_edges = np.linspace(x_min, x_max, 180)
    y_positions = np.arange(len(DRIVE_LABELS), dtype=float)
    y_labels = []
    for index, (drive_bin, color) in enumerate(zip(DRIVE_LABELS, RIDGE_COLORS)):
        values = table.loc[table["drive_bin"] == drive_bin, "mdf_kJ_per_mol"].to_numpy()
        centres, density = smoothed_density(values, x_edges)
        baseline = y_positions[index]
        ridge = baseline + 0.78 * density
        axis.fill_between(centres, baseline, ridge, color=color, alpha=0.92, linewidth=0)
        axis.plot(centres, ridge, color=color, linewidth=1.0)
        median = float(np.median(values))
        axis.scatter(median, baseline + 0.03, s=18, color=color, edgecolor="white", linewidth=0.6, zorder=4)
        y_labels.append(f"{drive_bin}   n={len(values):,}")
    axis.axvline(0, color="#D73027", linewidth=1.0, linestyle=(0, (4, 3)), zorder=5)
    axis.set_yticks(y_positions + 0.18, y_labels)
    axis.set_ylim(-0.12, len(DRIVE_LABELS) - 0.05 + 0.84)
    axis.set_xlim(x_min, x_max)
    axis.set_xlabel(r"MDF (kJ mol$^{-1}$)", fontsize=6.8, labelpad=3)
    axis.set_ylabel(r"Cumulative driving force −ΔG′$_{path}$ tiers", fontsize=6.3, labelpad=4)
    axis.tick_params(labelsize=5.8, length=0)
    STYLE.clean_axis(axis)
    axis.spines["left"].set_visible(False)
    axis.grid(axis="y", visible=False)


def build_figure(data_dir: Path) -> plt.Figure:
    data = {panel: read_data(data_dir, panel) for panel in ("fig2A", "fig2B", "fig2C", "fig2D", "fig2E")}
    figure = plt.figure(figsize=(15.6, 9.5))
    grid = figure.add_gridspec(2, 6, height_ratios=[1.0, 1.25], hspace=0.48, wspace=0.95)
    panel_a(figure.add_subplot(grid[0, 0:2]), data["fig2A"])
    panel_b(figure.add_subplot(grid[0, 2:4]), data["fig2B"])
    panel_d(figure.add_subplot(grid[0, 4:6]), data["fig2D"])
    panel_c(grid[1, 0:3], data["fig2C"])
    panel_e(grid[1, 3:6], data["fig2E"])
    figure.subplots_adjust(left=0.055, right=0.985, top=0.95, bottom=0.08)
    return figure


def main() -> None:
    args = parse_args()
    configure_logging(args.log_file)
    STYLE.set_publication_style()
    logging.info("Start Figure 2 plotting; data=%s; smoke=%s", args.data_dir, args.smoke)
    figure = build_figure(args.data_dir)
    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        figure.savefig(args.output_stem.with_suffix(".png"), dpi=180, bbox_inches="tight", facecolor="white")
    else:
        STYLE.save_figure(figure, args.output_stem)
    plt.close(figure)
    logging.info("Figure 2 plotting completed: %s", args.output_stem)


if __name__ == "__main__":
    main()
