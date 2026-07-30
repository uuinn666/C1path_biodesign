
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = REPO_ROOT / "supplementary_materials/figure2"
DEFAULT_OUTPUT = REPO_ROOT / "results/figure2/Fig2_pathway_overview"

SUBSTRATES = ["co2", "methanol", "formate", "formaldehyde"]
SUBSTRATE_LABELS = {
    "co2": r"CO$_2$",
    "methanol": "Methanol",
    "formate": "Formate",
    "formaldehyde": "Formaldehyde",
}
SUBSTRATE_COLORS = {
    "co2": "#08306B",
    "methanol": "#CB181D",
    "formate": "#FC9272",
    "formaldehyde": "#2171B5",
}
STAGE_COLORS = {
    "raw_feasible_pathways": "#08306B",
    "le20_pathways": "#FC9272",
    "retained_pathways": "#CB181D",
}
REDS = ["#FEE0D2", "#FCBBA1", "#FC9272", "#FB6A4A", "#EF3B2C", "#CB181D"]
NEUTRAL = "#5B616B"
GRID = "#E5EAF1"
DRIVE_EDGES = [-np.inf, 0, 50, 100, 150, 250, np.inf]
DRIVE_LABELS = ["≤0", "0–50", "50–100", "100–150", "150–250", ">250"]
RIDGE_COLORS = ["#DCEAF4", "#B8D5E8", "#86BBD8", "#4F9BC6", "#2576A8", "#0B4F6C"]


def set_publication_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.8,
            "axes.titlesize": 9.6,
            "axes.labelsize": 8.8,
            "axes.linewidth": 0.85,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "legend.fontsize": 7.4,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )


def clean_axis(axis: plt.Axes, grid_axis: str = "y") -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis=grid_axis, color=GRID, linewidth=0.7, zorder=0)
    axis.set_axisbelow(True)


def deterministic_sample(values: np.ndarray, maximum: int) -> np.ndarray:
    if len(values) <= maximum:
        return values
    idx = np.linspace(0, len(values) - 1, maximum, dtype=int)
    return values[idx]


def read_data(directory: Path, panel: str) -> pd.DataFrame:
    return pd.read_csv(directory / f"{panel}_source.tsv", sep="\t", encoding="utf-8-sig")


def ordered(table: pd.DataFrame) -> pd.DataFrame:
    return table.assign(substrate=pd.Categorical(table["substrate"], SUBSTRATES, ordered=True)).sort_values("substrate")


def positions() -> tuple[np.ndarray, list[str]]:
    x = np.arange(len(SUBSTRATES))
    return x, [SUBSTRATE_LABELS[item] for item in SUBSTRATES]


def violin_box(axis: plt.Axes, table: pd.DataFrame, metric: str, maximum_points: int = 360) -> None:
    rng = np.random.default_rng(20260714)
    x, labels = positions()
    groups = [table.loc[table["substrate"] == substrate, metric].dropna().to_numpy() for substrate in SUBSTRATES]
    violins = axis.violinplot(groups, positions=x, widths=0.78, showextrema=False)
    for body, substrate in zip(violins["bodies"], SUBSTRATES):
        body.set_facecolor(SUBSTRATE_COLORS[substrate])
        body.set_edgecolor("none")
        body.set_alpha(0.16)
    boxes = axis.boxplot(groups, positions=x, widths=0.25, patch_artist=True, showfliers=False,
                         medianprops={"color": "white", "linewidth": 1.4})
    for patch, substrate in zip(boxes["boxes"], SUBSTRATES):
        patch.set_facecolor(SUBSTRATE_COLORS[substrate])
        patch.set_edgecolor(SUBSTRATE_COLORS[substrate])
        patch.set_alpha(0.70)
    for item in boxes["whiskers"] + boxes["caps"]:
        item.set_color(NEUTRAL)
        item.set_linewidth(0.65)
    for index, (substrate, values) in enumerate(zip(SUBSTRATES, groups)):
        sampled = deterministic_sample(np.sort(values), maximum_points)
        jitter = rng.uniform(-0.17, 0.17, len(sampled))
        axis.scatter(index + jitter, sampled, s=4.0, color=SUBSTRATE_COLORS[substrate], alpha=0.15,
                     edgecolors="none", rasterized=True)
        axis.scatter(index, np.mean(values), marker="D", s=18, facecolor="white", edgecolor=NEUTRAL,
                     linewidth=0.7, zorder=5)
    axis.set_xticks(x, labels, rotation=18, ha="right")
    clean_axis(axis)


def panel_a(axis: plt.Axes, table: pd.DataFrame) -> None:
    table = ordered(table).set_index("substrate").loc[SUBSTRATES]
    x, labels = positions()
    bars = axis.bar(x, table["candidate_assimilation_reaction_count"], width=0.68,
                    color=[SUBSTRATE_COLORS[item] for item in SUBSTRATES], alpha=0.75)
    axis.bar_label(bars, padding=3, fontsize=8.2, fontweight="bold")
    axis.set_xticks(x, labels, rotation=18, ha="right")
    axis.set_ylabel("Candidate assimilation reactions")
    axis.set_ylim(0, table["candidate_assimilation_reaction_count"].max() * 1.20)
    clean_axis(axis)


def panel_b(axis: plt.Axes, table: pd.DataFrame) -> None:
    violin_box(axis, table, "unified_step_count")
    axis.axhline(20, color=REDS[4], linewidth=1.0, linestyle=(0, (4, 3)))
    axis.set_ylabel("Unified pathway steps")


def panel_c(axis: plt.Axes, table: pd.DataFrame) -> None:
    table = ordered(table).set_index("substrate").loc[SUBSTRATES]
    x, labels = positions()
    width = 0.24
    specs = (
        ("raw_feasible_pathways", -width, STAGE_COLORS["raw_feasible_pathways"], "Feasible"),
        ("le20_pathways", 0.0, STAGE_COLORS["le20_pathways"], "≤20 steps"),
        ("retained_pathways", width, STAGE_COLORS["retained_pathways"], "Retained"),
    )
    for metric, offset, color, label in specs:
        bars = axis.bar(x + offset, table[metric], width, color=color, alpha=0.75, label=label)
        for bar in bars:
            axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.10, f"{int(bar.get_height()):,}",
                      ha="center", va="bottom", fontsize=6.4, rotation=90)
    axis.set_yscale("log")
    axis.set_ylim(1, max(table["raw_feasible_pathways"]) * 4.0)
    axis.set_xticks(x, labels, rotation=18, ha="right")
    axis.set_ylabel("Pathway count (log scale)")
    axis.legend(frameon=False, loc="upper right", ncol=1)
    clean_axis(axis)


def reducing_equivalent_boxes(axis: plt.Axes, table: pd.DataFrame) -> None:
    x, labels = positions()
    width = 0.24
    for index, substrate in enumerate(SUBSTRATES):
        subset = table[table["substrate"] == substrate]
        for offset, metric, alpha in ((-width / 1.5, "corrected_NADH_auxiliary_equivalents_per_product_Cmol", 0.70),
                                      (width / 1.5, "corrected_NADPH_auxiliary_equivalents_per_product_Cmol", 0.34)):
            values = subset[metric].dropna().to_numpy()
            box = axis.boxplot([values], positions=[index + offset], widths=width, patch_artist=True, showfliers=False,
                               medianprops={"color": "white", "linewidth": 1.1})
            box["boxes"][0].set_facecolor(SUBSTRATE_COLORS[substrate])
            box["boxes"][0].set_edgecolor(SUBSTRATE_COLORS[substrate])
            box["boxes"][0].set_alpha(alpha)
            for item in box["whiskers"] + box["caps"]:
                item.set_color(NEUTRAL)
                item.set_linewidth(0.6)
            axis.scatter(index + offset, np.mean(values), marker="D", s=15, facecolor="white", edgecolor=NEUTRAL,
                         linewidth=0.6, zorder=5)
    axis.axhline(0, color=NEUTRAL, linewidth=0.8)
    axis.set_xticks(x, labels, rotation=18, ha="right")
    handles = [Line2D([0], [0], color=NEUTRAL, linewidth=7, alpha=0.70, label="NADH"),
               Line2D([0], [0], color=NEUTRAL, linewidth=7, alpha=0.34, label="NADPH")]
    axis.legend(handles=handles, frameon=False, loc="upper right", ncol=2, handlelength=1.0)
    clean_axis(axis)


def panel_d(axis: plt.Axes, thermo: pd.DataFrame) -> None:
    table = thermo.copy()
    table["drive"] = -table["flux_weighted_total_optimized_dg_kJ_per_product_Cmol"]
    table["drive_bin"] = pd.cut(table["drive"], DRIVE_EDGES, labels=DRIVE_LABELS, include_lowest=True, right=True)
    x_min = table["mdf_kJ_per_mol"].min() - 3
    x_max = table["mdf_kJ_per_mol"].max() + 3
    x_edges = np.linspace(x_min, x_max, 180)
    y_positions = np.arange(len(DRIVE_LABELS), dtype=float)
    sample_sizes = []
    for index, (drive_bin, color) in enumerate(zip(DRIVE_LABELS, RIDGE_COLORS)):
        values = table.loc[table["drive_bin"] == drive_bin, "mdf_kJ_per_mol"].dropna().to_numpy()
        hist, edges = np.histogram(values, bins=x_edges, density=False)
        density = gaussian_filter1d(hist.astype(float), sigma=2.0)
        if density.max() > 0:
            density /= density.max()
        centres = (edges[:-1] + edges[1:]) / 2.0
        baseline = y_positions[index]
        ridge = baseline + 0.78 * density
        axis.fill_between(centres, baseline, ridge, color=color, alpha=0.92, linewidth=0)
        axis.plot(centres, ridge, color=color, linewidth=1.0)
        axis.scatter(float(np.median(values)), baseline + 0.03, s=16, color=color, edgecolor="white", linewidth=0.6, zorder=4)
        sample_sizes.append(len(values))
    axis.axvline(0, color="#D73027", linewidth=1.0, linestyle=(0, (4, 3)), zorder=5)
    axis.set_yticks(y_positions + 0.18, DRIVE_LABELS)
    for y_position, sample_size in zip(y_positions + 0.18, sample_sizes):
        axis.text(
            0.025, y_position, f"n={sample_size:,}",
            transform=axis.get_yaxis_transform(), ha="left", va="center", fontsize=6.4, color=NEUTRAL,
        )
    axis.set_ylim(-0.12, len(DRIVE_LABELS) - 0.05 + 0.84)
    axis.set_xlim(x_min, x_max)
    axis.set_xlabel(r"MDF (kJ mol$^{-1}$)", fontsize=7.6, labelpad=3)
    axis.set_ylabel("Pathway driving force")
    axis.tick_params(axis="y", labelsize=7.8, length=3, width=0.75, pad=3)
    axis.tick_params(axis="x", labelsize=6.6, length=0)
    clean_axis(axis)
    axis.grid(axis="y", visible=False)


def panel_e(container, table: pd.DataFrame):
    grid = container.subgridspec(1, 2, wspace=0.28)
    atp = plt.subplot(grid[0, 0])
    redox = plt.subplot(grid[0, 1])
    violin_box(atp, table, "corrected_ATP_auxiliary_equivalents_per_product_Cmol", maximum_points=280)
    atp.axhline(0, color=NEUTRAL, linewidth=0.8)
    atp.set_ylabel(r"Auxiliary equivalents (mol mol-product-C$^{-1}$)")
    atp.set_title("ATP", fontsize=9.0, fontweight="bold", pad=4)
    reducing_equivalent_boxes(redox, table)
    redox.set_ylabel("")
    redox.set_title("Reducing equivalents", fontsize=9.0, fontweight="bold", pad=4)
    return atp, redox


def place_header(fig: plt.Figure, renderer, header_ax: plt.Axes, ref_ax: plt.Axes, label: str, title: str,
                 note: str | None = None, label_fs: float = 10.6, title_fs: float = 8.8, note_fs: float = 5.5,
                 x_shift: float = 0.0, y_shift: float = 0.0, label_gap: float = 0.030,
                 x_left_override: float | None = None):
    hb = header_ax.get_position()
    bb = ref_ax.get_tightbbox(renderer).transformed(fig.transFigure.inverted())
    x_left = bb.x0 + x_shift if x_left_override is None else x_left_override
    y_top = hb.y1 - 0.004 + y_shift
    fig.text(x_left, y_top, label, ha="left", va="top", fontsize=label_fs, fontweight="bold")
    fig.text(x_left + label_gap, y_top, title, ha="left", va="top", fontsize=title_fs, fontweight="bold")
    if note:
        fig.text(hb.x1, hb.y0 + 0.002 + y_shift, note, ha="right", va="bottom", fontsize=note_fs, color=NEUTRAL)


def build_figure(data_dir: Path) -> plt.Figure:
    data = {panel: read_data(data_dir, panel) for panel in ("fig2A", "fig2B", "fig2C", "fig2D", "fig2E")}
    fig = plt.figure(figsize=(7.6, 10.2))
    outer = fig.add_gridspec(3, 2, height_ratios=[0.93, 1.10, 1.18], hspace=0.34, wspace=0.52)

    headers = {}
    refs = {}

    gs_a = outer[0, 0].subgridspec(2, 1, height_ratios=[0.11, 1.0], hspace=0.03)
    headers['A'] = fig.add_subplot(gs_a[0, 0]); headers['A'].axis('off')
    ax_a = fig.add_subplot(gs_a[1, 0]); panel_a(ax_a, data['fig2A']); refs['A']=ax_a

    gs_b = outer[0, 1].subgridspec(2, 1, height_ratios=[0.11, 1.0], hspace=0.03)
    headers['B'] = fig.add_subplot(gs_b[0, 0]); headers['B'].axis('off')
    ax_b = fig.add_subplot(gs_b[1, 0]); panel_b(ax_b, data['fig2B']); refs['B']=ax_b

    gs_c = outer[1, 0].subgridspec(2, 1, height_ratios=[0.11, 1.0], hspace=0.03)
    headers['C'] = fig.add_subplot(gs_c[0, 0]); headers['C'].axis('off')
    ax_c = fig.add_subplot(gs_c[1, 0]); panel_c(ax_c, data['fig2C']); refs['C']=ax_c

    gs_d = outer[1, 1].subgridspec(2, 1, height_ratios=[0.11, 1.0], hspace=0.03)
    headers['D'] = fig.add_subplot(gs_d[0, 0]); headers['D'].axis('off')
    ax_d = fig.add_subplot(gs_d[1, 0]); panel_d(ax_d, data['fig2D']); refs['D']=ax_d

    gs_e = outer[2, :].subgridspec(2, 1, height_ratios=[0.14, 1.0], hspace=0.05)
    headers['E'] = fig.add_subplot(gs_e[0, 0]); headers['E'].axis('off')
    atp_ax, redox_ax = panel_e(gs_e[1, 0], data['fig2E']); refs['E']=atp_ax

    fig.subplots_adjust(left=0.082, right=0.988, top=0.986, bottom=0.055)
    b_position = ax_b.get_position()
    d_position = ax_d.get_position()
    ax_d.set_position([b_position.x0, d_position.y0, b_position.width, d_position.height])
    ax_b.yaxis.set_label_coords(-0.24, 0.5)
    ax_d.yaxis.set_label_coords(-0.24, 0.5)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    b_header_left = b_position.x0 - 0.098

    place_header(fig, renderer, headers['A'], refs['A'], 'A', 'Substrate-specific candidate sets')
    place_header(fig, renderer, headers['B'], refs['B'], 'B', 'Pathway-length distributions before screening',
                 x_left_override=b_header_left)
    place_header(fig, renderer, headers['C'], refs['C'], 'C', 'Contraction of the feasible route space')
    place_header(
        fig, renderer, headers['D'], refs['D'], 'D', 'MDF density by driving-force range',
        x_left_override=b_header_left,
    )
    place_header(fig, renderer, headers['E'], refs['E'], 'E', 'Net auxiliary cofactor requirements', y_shift=0.008)
    return fig


def save_figure(figure: plt.Figure, output_stem: Path, smoke: bool) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    if smoke:
        figure.savefig(output_stem.with_suffix(".png"), dpi=180, bbox_inches="tight", facecolor="white")
        return
    figure.savefig(output_stem.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--log-file", type=Path, default=REPO_ROOT / "logs/figure2.log")
    return parser.parse_args()


def configure_logging(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler()],
    )


def main() -> None:
    args = parse_args()
    configure_logging(args.log_file)
    set_publication_style()
    figure = build_figure(args.data_dir)
    save_figure(figure, args.output_stem, args.smoke)
    plt.close(figure)
    logging.info("Figure 2 completed: %s; smoke=%s", args.output_stem, args.smoke)


if __name__ == '__main__':
    main()
