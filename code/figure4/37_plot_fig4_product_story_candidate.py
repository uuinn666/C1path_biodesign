from __future__ import annotations

import argparse
import importlib
import logging
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
import numpy as np
import pandas as pd


CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_DIR))
STYLE = importlib.import_module("05_figure_style")
DEFAULT_DATA = REPO_ROOT / "supplementary_materials/figure4"
DEFAULT_OUTPUT = REPO_ROOT / "results/figure4/Fig4_product_landscape"

CLASS_ORDER = [
    "Central metabolic hubs",
    "TCA/glyoxylate intermediates",
    "Sugar-phosphate precursors",
    "Organic/hydroxy-acid products",
    "Reduced/platform chemicals",
    "Biosynthetic building blocks",
]
CLASS_LABELS = {name: name for name in CLASS_ORDER}
CLASS_COLORS = {
    "Central metabolic hubs": "#08306B",
    "TCA/glyoxylate intermediates": "#2171B5",
    "Sugar-phosphate precursors": "#6BAED6",
    "Organic/hydroxy-acid products": "#FCBBA1",
    "Reduced/platform chemicals": "#FC9272",
    "Biosynthetic building blocks": "#CB181D",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke", action="store_true", help="Save a low-resolution preview only.")
    parser.add_argument("--log-file", type=Path, default=REPO_ROOT / "logs/figure4.log")
    return parser.parse_args()


def configure_logging(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler()],
    )


def set_readable_style() -> None:
    STYLE.set_publication_style()
    mpl.rcParams.update(
        {
            "font.size": 9.5,
            "axes.titlesize": 11.5,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 7.2,
            "axes.linewidth": 0.85,
            "savefig.dpi": 600,
        }
    )


def curved_edge(axis: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str) -> None:
    x0, y0 = start
    x1, y1 = end
    vertices = [start, (x0 + 0.28 * (x1 - x0), y0), (x1 - 0.24 * (x1 - x0), y1), end]
    patch = PathPatch(
        MplPath(vertices, [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4]),
        facecolor="none",
        edgecolor=color,
        linewidth=0.78,
        alpha=0.25,
        capstyle="round",
        zorder=1,
    )
    axis.add_patch(patch)


def product_order(network: pd.DataFrame) -> list[str]:
    meta = network.drop_duplicates("product_label").sort_values(["class_order", "product_order"])
    return meta["product_label"].tolist()


def panel_a(axis: plt.Axes, network: pd.DataFrame) -> None:
    products = product_order(network)
    y_product = dict(zip(products, np.linspace(0.905, 0.115, len(products))))
    y_substrate = dict(zip(STYLE.SUBSTRATES, [0.80, 0.60, 0.40, 0.20]))
    substrate_x, product_x = 0.105, 0.655
    reachable = network[network["reachable"].astype(str).str.lower().eq("true")]

    for _, row in reachable.iterrows():
        curved_edge(
            axis,
            (substrate_x + 0.027, y_substrate[row["substrate"]]),
            (product_x - 0.014, y_product[row["product_label"]]),
            STYLE.SUBSTRATE_COLORS[row["substrate"]],
        )

    for substrate in STYLE.SUBSTRATES:
        y = y_substrate[substrate]
        count = reachable.loc[reachable["substrate"] == substrate, "product_label"].nunique()
        axis.scatter(
            substrate_x,
            y,
            s=760,
            color=STYLE.SUBSTRATE_COLORS[substrate],
            edgecolor="white",
            linewidth=1.1,
            zorder=4,
        )
        short = {"co2": "CO₂", "methanol": "MeOH", "formate": "HCOO⁻", "formaldehyde": "HCHO"}[substrate]
        axis.text(
            substrate_x,
            y,
            short,
            ha="center",
            va="center",
            fontsize=7.8,
            color="white",
            fontweight="bold",
            zorder=5,
        )
        axis.text(
            substrate_x,
            y - 0.050,
            f"{count}/25 products",
            ha="center",
            va="top",
            fontsize=7.0,
            color="#343A40",
        )

    meta = network.drop_duplicates("product_label").set_index("product_label")
    degrees = reachable.groupby("product_label")["substrate"].nunique()
    for product in products:
        row = meta.loc[product]
        y = y_product[product]
        color = CLASS_COLORS[row["product_class"]]
        axis.scatter(
            product_x,
            y,
            s=38 + 13 * degrees[product],
            color=color,
            edgecolor="#24282D",
            linewidth=0.60,
            zorder=4,
        )
        axis.text(
            product_x + 0.021,
            y,
            product,
            ha="left",
            va="center",
            fontsize=7.35,
            color="#24282D",
        )

    axis.text(substrate_x, 0.962, "C1 substrates", ha="center", va="bottom", fontsize=8.8, fontweight="bold")
    axis.text(product_x + 0.060, 0.950, "Products retained\nafter all screens", ha="center", va="bottom", fontsize=8.2, fontweight="bold", linespacing=0.94)

    handles = [
        mpl.lines.Line2D(
            [0], [0], marker="o", linestyle="none", markerfacecolor=CLASS_COLORS[c],
            markeredgecolor="none", markersize=6.0, label=CLASS_LABELS[c],
        )
        for c in CLASS_ORDER
    ]
    axis.legend(
        handles=handles,
        frameon=False,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.57, -0.025),
        fontsize=6.8,
        columnspacing=1.10,
        handletextpad=0.42,
        labelspacing=0.45,
    )
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    axis.text(-0.020, 1.018, "A", transform=axis.transAxes, fontsize=12.6, fontweight="bold", va="top")
    axis.text(
        0.045,
        1.018,
        "Final substrate–product reachability",
        transform=axis.transAxes,
        fontsize=11.8,
        fontweight="bold",
        va="top",
    )


def panel_b(axis: plt.Axes, coverage: pd.DataFrame) -> None:
    table = coverage.set_index("product_class").loc[
        [c for c in CLASS_ORDER if c in set(coverage["product_class"])]
    ].reset_index()
    y = np.arange(len(table))[::-1]
    axis.barh(y, 100, color="#EDF0F4", height=0.64, edgecolor="none")
    values = 100 * table["pair_coverage_fraction"].to_numpy()
    colors = [CLASS_COLORS[c] for c in table["product_class"]]
    axis.barh(y, values, color=colors, height=0.64, edgecolor="none")

    for pos, (_, row) in zip(y, table.iterrows()):
        value = 100 * row["pair_coverage_fraction"]
        inside = value >= 58
        axis.text(
            value - 2.0 if inside else value + 2.0,
            pos,
            f"{value:.0f}%",
            ha="right" if inside else "left",
            va="center",
            fontsize=8.0,
            color="white" if inside else "#24282D",
            fontweight="bold",
        )
        axis.text(
            102.0,
            pos,
            f"{int(row['reachable_pairs'])}/{int(row['possible_pairs'])} pairs; "
            f"{int(row['analyzed_products'])} products",
            ha="left",
            va="center",
            fontsize=7.0,
            color="#59616A",
        )

    axis.set_yticks(y, [CLASS_LABELS[c] for c in table["product_class"]])
    axis.tick_params(axis="y", labelsize=8.0, length=0, pad=5)
    axis.set_xlim(0, 145)
    axis.set_xticks([0, 25, 50, 75, 100])
    axis.xaxis.set_major_formatter(mpl.ticker.PercentFormatter())
    axis.set_xlabel("Reachable substrate–product pairs / all possible pairs", fontsize=8.3, labelpad=5)
    axis.tick_params(axis="x", labelsize=7.5)
    STYLE.clean_axis(axis, grid_axis="x")
    axis.spines["left"].set_visible(False)
    axis.text(-0.105, 1.045, "B", transform=axis.transAxes, fontsize=13.0, fontweight="bold", ha="left", va="bottom")
    axis.text(
        0.00,
        1.045,
        "Fair comparison of product-class coverage",
        transform=axis.transAxes,
        fontsize=12.0,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def distribution_axis(
    axis: plt.Axes,
    summary: pd.DataFrame,
    y: np.ndarray,
    metric: str,
    title: str,
    xlabel: str,
    xlim: tuple[float, float],
    ticks: list[float],
    preferred: str,
) -> None:
    for pos in y:
        axis.axhline(pos, color="#F1F3F6", linewidth=0.60, zorder=0)
    for pos, (_, row) in zip(y, summary.iterrows()):
        color = CLASS_COLORS[row["product_class"]]
        axis.plot(
            [row[f"{metric}_min"], row[f"{metric}_max"]],
            [pos, pos],
            color=color,
            alpha=0.40,
            linewidth=1.25,
            solid_capstyle="round",
            zorder=1,
        )
        axis.plot(
            [row[f"{metric}_q25"], row[f"{metric}_q75"]],
            [pos, pos],
            color=color,
            linewidth=5.2,
            solid_capstyle="round",
            zorder=2,
        )
        axis.scatter(
            row[f"{metric}_median"],
            pos,
            s=24,
            color=color,
            edgecolor="white",
            linewidth=0.60,
            zorder=3,
        )

    axis.set_xlim(*xlim)
    axis.set_xticks(ticks)
    title_lines = title.split("\n")
    title_main = title_lines[0]
    title_sub = "\n".join(title_lines[1:]) if len(title_lines) > 1 else ""
    axis.text(0.5, 1.035, title_main, transform=axis.transAxes, ha="center", va="bottom", fontsize=8.3, fontweight="bold")
    if title_sub:
        axis.text(0.5, 1.007, title_sub, transform=axis.transAxes, ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    axis.set_xlabel(f"{xlabel}   {preferred}", fontsize=8.0, labelpad=5)
    axis.tick_params(axis="x", labelsize=7.4)
    axis.grid(axis="x", color=STYLE.GRID, linewidth=0.70)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_visible(False)
    axis.tick_params(axis="y", length=0)


def summarize_distributions(distributions: pd.DataFrame) -> pd.DataFrame:
    metrics = {"steps": "unified_step_count", "mdf": "mdf_kJ_per_mol"}
    rows: list[dict[str, object]] = []
    for _, meta in distributions.drop_duplicates("product_label").sort_values(["class_order", "product_order"]).iterrows():
        subset = distributions[distributions["product_label"] == meta["product_label"]]
        row = meta[["product_label", "product_class", "class_order", "product_order"]].to_dict()
        row["n_paths"] = len(subset)
        for prefix, column in metrics.items():
            values = subset[column].astype(float)
            row.update(
                {
                    f"{prefix}_min": values.min(),
                    f"{prefix}_q25": values.quantile(0.25),
                    f"{prefix}_median": values.median(),
                    f"{prefix}_q75": values.quantile(0.75),
                    f"{prefix}_max": values.max(),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def panel_c(container: mpl.gridspec.SubplotSpec, distributions: pd.DataFrame, figure: plt.Figure) -> None:
    summary = summarize_distributions(distributions)
    y = np.arange(len(summary))[::-1]
    subgrid = container.subgridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.18)
    axes = [figure.add_subplot(subgrid[0, index]) for index in range(2)]

    distribution_axis(
        axes[0],
        summary,
        y,
        "steps",
        "Pathway-step distribution\n(all retained pathways)",
        "steps",
        (3, 21),
        [5, 10, 15, 20],
        r"shorter $\leftarrow$",
    )
    mdf_high = np.ceil(distributions["mdf_kJ_per_mol"].max() / 5) * 5
    distribution_axis(
        axes[1],
        summary,
        y,
        "mdf",
        "MDF distribution\n(all retained pathways)",
        r"kJ mol$^{-1}$",
        (0, mdf_high),
        list(np.arange(0, mdf_high + 1, 10)),
        r"higher $\rightarrow$",
    )

    axes[0].set_yticks(y, summary["product_label"])
    axes[0].tick_params(axis="y", labelsize=7.2, pad=3)
    for label in axes[0].get_yticklabels():
        label.set_color("#24282D")
    axes[1].set_yticks(y, [])
    for axis in axes:
        axis.set_ylim(-0.8, len(summary) - 0.2)

    starts = summary.groupby("product_class", sort=False).head(1).index.tolist()[1:]
    for index in starts:
        boundary = len(summary) - index - 0.5
        for axis in axes:
            axis.axhline(boundary, color="#AEB6C1", linewidth=0.75, zorder=2)

    legend_handles = [
        mpl.lines.Line2D([0], [0], color="#59616A", linewidth=1.25, alpha=0.55, label="min–max"),
        mpl.lines.Line2D([0], [0], color="#59616A", linewidth=5.2, label="IQR"),
        mpl.lines.Line2D(
            [0], [0], marker="o", linestyle="none", markerfacecolor="#59616A",
            markeredgecolor="white", markersize=5.0, label="median",
        ),
    ]
    axes[1].legend(
        handles=legend_handles,
        frameon=False,
        ncol=3,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.125),
        fontsize=6.9,
        columnspacing=0.85,
        handlelength=1.6,
        borderaxespad=0.0,
    )
    axes[0].text(-0.255, 1.165, "C", transform=axes[0].transAxes, fontsize=12.6, fontweight="bold", ha="left", va="bottom")
    axes[0].text(
        0.00,
        1.165,
        "Pathway length and thermodynamic distributions by product",
        transform=axes[0].transAxes,
        fontsize=11.8,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def build_figure(data_dir: Path) -> plt.Figure:
    network = pd.read_csv(data_dir / "fig4A_final_reachability.csv", encoding="utf-8-sig")
    coverage = pd.read_csv(data_dir / "fig4B_class_coverage.csv", encoding="utf-8-sig")
    distributions = pd.read_csv(data_dir / "fig4C_product_path_distributions.csv", encoding="utf-8-sig")

    figure = plt.figure(figsize=(12.8, 8.4))
    outer = figure.add_gridspec(1, 2, width_ratios=[0.90, 1.10], wspace=0.29)
    right = outer[0, 1].subgridspec(2, 1, height_ratios=[0.64, 1.36], hspace=0.48)

    panel_a(figure.add_subplot(outer[0, 0]), network)
    panel_b(figure.add_subplot(right[0, 0]), coverage)
    panel_c(right[1, 0], distributions, figure)
    figure.subplots_adjust(left=0.060, right=0.988, top=0.940, bottom=0.085)
    return figure


def save_outputs(figure: plt.Figure, output_stem: Path, smoke: bool = False) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    if smoke:
        figure.savefig(output_stem.with_suffix(".png"), dpi=180, bbox_inches="tight", facecolor="white")
        return
    figure.savefig(output_stem.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")


def main() -> None:
    args = parse_args()
    configure_logging(args.log_file)
    set_readable_style()
    figure = build_figure(args.data_dir)
    save_outputs(figure, args.output_stem, smoke=args.smoke)
    plt.close(figure)
    logging.info("Readable Figure 4 completed: %s; smoke=%s", args.output_stem, args.smoke)


if __name__ == "__main__":
    main()
