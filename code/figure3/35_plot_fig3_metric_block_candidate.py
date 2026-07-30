
"""Plot the final A-F four-substrate assimilation-reaction figure grouped by metric."""

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
DEFAULT_SELECTION_OUTPUT = None
DEFAULT_CANDIDATE_OUTPUT = DEFAULT_OUTPUT
DEFAULT_CANDIDATE_SELECTION_OUTPUT = None
DEFAULT_MANUSCRIPT_OUTPUT = DEFAULT_OUTPUT
DEFAULT_MANUSCRIPT_SELECTION_OUTPUT = None
DEFAULT_TOP_PER_SUBSTRATE = 5
SUBSTRATES = LAYOUT_HELPER.SUBSTRATES
SUBSTRATE_LABELS = LAYOUT_HELPER.SUBSTRATE_LABELS
CARBON_CLASSES = LAYOUT_HELPER.CARBON_CLASSES
SUBSTRATE_COLORS = {
    "co2": "#08306B",
    "methanol": "#CB181D",
    "formate": "#FC9272",
    "formaldehyde": "#4292C6",
}
MANUSCRIPT_REACTION_CODES = {
    "co2": ("01_CCR", "02_CCR", "03_ACC", "08_PYC", "09_PNO"),
    "methanol": ("01_MTA", "02_MTB", "04_ADH", "05_MOX", "06_MDH"),
    "formate": ("01_FDR", "02_GAR", "03_AIC", "06_PFL", "07_FTL"),
    "formaldehyde": ("01_FLS", "02_RMP", "03_FTK", "08_FC1", "09_FC2"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--log-file", type=Path,
        default=REPO_ROOT / "logs/figure3.log",
    )
    parser.add_argument(
        "--top-per-substrate", type=int, default=DEFAULT_TOP_PER_SUBSTRATE,
        help="Number of reactions shown per substrate; use 0 to show all reactions.",
    )
    parser.add_argument(
        "--selection-output", type=Path, default=DEFAULT_SELECTION_OUTPUT,
        help="Optional output path for the selected-reaction manifest.",
    )
    parser.add_argument(
        "--selection-mode",
        choices=("pooled-occurrence", "candidate-order", "manuscript-selected"),
        default="manuscript-selected",
        help="Select reactions by pooled occurrence, candidate order, or the manuscript list.",
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


def rank_reactions(summary: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    ranking = (
        summary.groupby(["substrate", "candidate_number", "reaction_code"], as_index=False)
        .agg(
            pooled_occurrence_count=("le20_occurrence_count", "sum"),
            pooled_pathway_denominator=("le20_pathway_denominator", "sum"),
            mean_step_median=("step_median", "mean"),
        )
    )
    if (ranking["pooled_pathway_denominator"] <= 0).any():
        raise ValueError("Pathway denominators used for reaction ranking must be positive.")
    ranking["pooled_occurrence_percent"] = (
        100.0 * ranking["pooled_occurrence_count"] / ranking["pooled_pathway_denominator"]
    )
    ranking["substrate_order"] = pd.Categorical(
        ranking["substrate"], categories=SUBSTRATES, ordered=True,
    )
    ranking = ranking.sort_values(
        ["substrate_order", "pooled_occurrence_percent", "mean_step_median", "candidate_number"],
        ascending=[True, False, True, True], kind="mergesort",
    ).reset_index(drop=True)
    ranking["display_rank"] = ranking.groupby("substrate", observed=True).cumcount() + 1
    details = mapping[
        ["substrate", "candidate_number", "reaction_id", "reaction_name_en",
         "reaction_equation_id", "mapping_basis"]
    ]
    return ranking.merge(
        details, on=["substrate", "candidate_number"], how="left", validate="one_to_one",
    )


def select_mapping(
    mapping: pd.DataFrame,
    ranking: pd.DataFrame,
    top_n: int,
    selection_mode: str,
) -> pd.DataFrame:
    if top_n < 0:
        raise ValueError("--top-per-substrate cannot be negative.")
    if top_n == 0:
        return mapping.copy()
    if selection_mode == "pooled-occurrence":
        selected = ranking[ranking["display_rank"] <= top_n][
            ["substrate", "candidate_number", "display_rank"]
        ].copy()
    elif selection_mode == "candidate-order":
        substrate_order = pd.Categorical(
            mapping["substrate"], categories=SUBSTRATES, ordered=True,
        )
        selected = mapping.assign(substrate_order=substrate_order).sort_values(
            ["substrate_order", "candidate_number"], kind="mergesort",
        ).groupby("substrate", observed=True, sort=False).head(top_n).copy()
        selected["display_rank"] = selected.groupby(
            "substrate", observed=True, sort=False,
        ).cumcount() + 1
        selected = selected[["substrate", "candidate_number", "display_rank"]]
    else:
        if top_n != 5:
            raise ValueError("manuscript-selected mode requires five reactions per substrate.")
        records = []
        for substrate, codes in MANUSCRIPT_REACTION_CODES.items():
            for display_rank, reaction_code in enumerate(codes, start=1):
                row = mapping[
                    (mapping["substrate"] == substrate)
                    & (mapping["reaction_code"] == reaction_code)
                ]
                if len(row) != 1:
                    raise ValueError(f"Invalid manuscript reaction mapping: {substrate}/{reaction_code}")
                records.append({
                    "substrate": substrate,
                    "candidate_number": int(row.iloc[0]["candidate_number"]),
                    "display_rank": display_rank,
                })
        selected = pd.DataFrame(records)
    counts = selected.groupby("substrate").size().reindex(SUBSTRATES, fill_value=0)
    if not (counts == top_n).all():
        raise ValueError(f"Unable to select {top_n} reactions per substrate: {counts.to_dict()}")
    return mapping.merge(
        selected, on=["substrate", "candidate_number"],
        how="inner", validate="one_to_one",
    )


def write_selection_manifest(
    ranking: pd.DataFrame,
    selected_mapping: pd.DataFrame,
    top_n: int,
    path: Path,
    selection_mode: str,
) -> None:
    if top_n == 0:
        return
    metrics = ranking.drop(columns="display_rank")
    selected = selected_mapping[
        ["substrate", "candidate_number", "display_rank"]
    ].merge(metrics, on=["substrate", "candidate_number"], validate="one_to_one")
    if selection_mode == "pooled-occurrence":
        selected["selection_rule"] = (
            "pooled <=20-step occurrence desc; mean step median asc; candidate number asc"
        )
    elif selection_mode == "candidate-order":
        selected["selection_rule"] = (
            "first available reactions by candidate number ascending within each substrate"
        )
    else:
        selected["selection_rule"] = "user-confirmed manuscript reaction list and order"
    columns = [
        "substrate", "display_rank", "candidate_number", "reaction_code", "reaction_id",
        "reaction_name_en", "pooled_occurrence_count", "pooled_pathway_denominator",
        "pooled_occurrence_percent", "mean_step_median", "selection_rule",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    selected = selected.sort_values(["substrate", "display_rank"], kind="mergesort")
    selected[columns].to_csv(path, sep="\t", index=False, float_format="%.6f")



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


def set_reaction_labels(
    axis: plt.Axes,
    rows: pd.DataFrame,
    visible: bool,
    font_size: float,
) -> None:
    if visible:
        axis.set_yticklabels(
            rows["reaction_code"], fontfamily="DejaVu Sans Mono", fontsize=font_size,
        )
        axis.tick_params(axis="y", length=0, pad=3.0)
    else:
        axis.set_yticklabels([])
        axis.tick_params(axis="y", length=0)


def add_panel_header(axis: plt.Axes, label: str, carbon_class: str) -> None:
    axis.text(
        0.0, 1.025, label, transform=axis.transAxes,
        fontsize=10.5, fontweight="bold", va="bottom",
    )
    axis.text(
        0.13, 1.025, f"{carbon_class} products", transform=axis.transAxes,
        fontsize=8.2, fontweight="bold", va="bottom",
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
    show_xlabel: bool,
    reaction_font_size: float,
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
                float(single), reaction.y, marker="D", s=7,
                facecolor="white", edgecolor="#20252A", linewidth=0.50, zorder=4,
            )
    LAYOUT_HELPER.style_common_axis(axis, rows, y_max)
    set_reaction_labels(axis, rows, show_labels, reaction_font_size)
    axis.set_xlim(0, 90)
    axis.set_xticks((0, 20, 40, 60, 80))
    axis.axvline(20, color="#30343B", linewidth=0.75, linestyle=(0, (4, 3)), zorder=1)
    axis.set_xlabel("Pathway steps" if show_xlabel else "", fontsize=7.6, labelpad=2.5)
    add_panel_header(axis, label, carbon_class)


def draw_frequency_panel(
    axis: plt.Axes,
    carbon_class: str,
    label: str,
    data: dict[str, pd.DataFrame],
    rows: pd.DataFrame,
    blocks: dict[str, dict[str, float]],
    y_max: float,
    show_labels: bool,
    show_xlabel: bool,
    reaction_font_size: float,
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
    set_reaction_labels(axis, rows, show_labels, reaction_font_size)
    axis.set_xlim(0, 80)
    axis.set_xticks((0, 20, 40, 60, 80))
    axis.set_xlabel("≤20-step occurrence (%)" if show_xlabel else "", fontsize=7.6, labelpad=2.5)
    add_panel_header(axis, label, carbon_class)


def audit_layout(
    figure: plt.Figure,
    label_axes: list[plt.Axes],
    left_last_axis: plt.Axes,
    right_first_axis: plt.Axes,
    *,
    stacked_main: bool,
) -> dict:
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    overlap_count = 0
    for axis in label_axes:
        boxes = [label.get_window_extent(renderer) for label in axis.get_yticklabels() if label.get_text()]
        overlap_count += sum(first.overlaps(second) for first, second in zip(boxes, boxes[1:]))
    if stacked_main:
        metric_block_intrusion = 0
    else:
        right_label_left = min(
            label.get_window_extent(renderer).x0
            for label in right_first_axis.get_yticklabels() if label.get_text()
        )
        metric_block_intrusion = int(right_label_left < left_last_axis.bbox.x1 + 4)

    panel_header_boxes = []
    panel_header_overlap = 0
    panel_header_position_consistent = True
    for axis in figure.axes:
        header_items = [
            item for item in axis.texts
            if item.get_text()
            and item.get_transform() == axis.transAxes
            and item.get_position()[1] > 1.0
        ]
        positions = sorted(round(item.get_position()[0], 3) for item in header_items)
        panel_header_position_consistent &= positions == [0.0, 0.13]
        boxes = [item.get_window_extent(renderer) for item in header_items]
        panel_header_boxes.extend(boxes)
        panel_header_overlap += sum(
            first.overlaps(second)
            for index, first in enumerate(boxes)
            for second in boxes[index + 1:]
        )
    visible_x_axis_title_count = sum(bool(axis.get_xlabel()) for axis in figure.axes)
    x_label_overlap = 0
    if stacked_main:
        for axes in (figure.axes[:3], figure.axes[3:]):
            boxes = [axis.xaxis.label.get_window_extent(renderer) for axis in axes]
            x_label_overlap += sum(first.overlaps(second) for first, second in zip(boxes, boxes[1:]))
    group_boxes = [item.get_window_extent(renderer) for item in figure.texts if item.get_text()]
    group_header_overlap = sum(
        group.overlaps(header) for group in group_boxes for header in panel_header_boxes
    )
    legend = figure.legends[0].get_window_extent(renderer) if figure.legends else None
    figure_box = figure.bbox
    legend_inside = bool(
        legend is not None
        and legend.x0 >= figure_box.x0 and legend.y0 >= figure_box.y0
        and legend.x1 <= figure_box.x1 and legend.y1 <= figure_box.y1
    )
    legend_text_overlap = sum(legend.overlaps(box) for box in group_boxes) if legend else 0
    result = {
        "visible_axes": len(figure.axes),
        "reaction_tick_label_overlap_count": int(overlap_count),
        "panel_header_overlap_count": int(panel_header_overlap),
        "panel_header_position_consistent": bool(panel_header_position_consistent),
        "visible_x_axis_title_count": int(visible_x_axis_title_count),
        "x_label_overlap_count": int(x_label_overlap),
        "group_header_overlap_count": int(group_header_overlap),
        "legend_text_overlap_count": int(legend_text_overlap),
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
        and panel_header_overlap == 0
        and panel_header_position_consistent
        and visible_x_axis_title_count == 2
        and x_label_overlap == 0
        and group_header_overlap == 0
        and legend_text_overlap == 0
        and metric_block_intrusion == 0
        and legend_inside
        and result["minimum_reaction_tick_font_size_pt"] >= (6.6 if stacked_main else 7.0)
    )
    return result


def build_figure(
    data: dict[str, pd.DataFrame],
    *,
    compact_main: bool,
) -> tuple[plt.Figure, dict]:
    rows, blocks, y_max = LAYOUT_HELPER.build_row_layout(data["mapping"])
    if compact_main:
        figure = plt.figure(figsize=(7.2, 11.0))
        outer = figure.add_gridspec(2, 1, hspace=0.30)
        step_grid = outer[0, 0].subgridspec(1, 3, wspace=0.23)
        frequency_grid = outer[1, 0].subgridspec(1, 3, wspace=0.23)
    else:
        figure = plt.figure(figsize=(18.0, 12.0))
        outer = figure.add_gridspec(1, 2, width_ratios=[1.12, 1.0], wspace=0.27)
        step_grid = outer[0, 0].subgridspec(1, 3, wspace=0.12)
        frequency_grid = outer[0, 1].subgridspec(1, 3, wspace=0.12)
    step_axes = [figure.add_subplot(step_grid[0, index]) for index in range(3)]
    frequency_axes = [figure.add_subplot(frequency_grid[0, index]) for index in range(3)]
    for index, (axis, carbon_class, label) in enumerate(zip(step_axes, CARBON_CLASSES, "ABC")):
        draw_step_panel(
            axis, carbon_class, label, data, rows, blocks, y_max,
            show_labels=index == 0, show_xlabel=index == 1,
            reaction_font_size=6.6 if compact_main else 7.0,
        )
    for index, (axis, carbon_class, label) in enumerate(zip(frequency_axes, CARBON_CLASSES, "DEF")):
        draw_frequency_panel(
            axis, carbon_class, label, data, rows, blocks, y_max,
            show_labels=index == 0, show_xlabel=index == 1,
            reaction_font_size=6.6 if compact_main else 7.0,
        )

    handles = [
        Patch(facecolor=SUBSTRATE_COLORS[substrate], edgecolor="none", label=SUBSTRATE_LABELS[substrate])
        for substrate in SUBSTRATES
    ]
    handles.extend(
        [
            Line2D([0], [0], marker="*", linestyle="", color="#20252A", markersize=4.2, label="Mean"),
            Line2D(
                [0], [0], marker="D", linestyle="", markerfacecolor="white",
                markeredgecolor="#20252A", markersize=3.4, label="Single-reaction median",
            ),
            Patch(facecolor="#D9DEE7", edgecolor="#20252A", linewidth=0.7, label="Pareto-efficient"),
            Line2D([0], [0], color="#30343B", linestyle=(0, (4, 3)), linewidth=0.75, label="20-step threshold"),
        ]
    )
    figure.legend(
        handles=handles, loc="upper center",
        bbox_to_anchor=(0.5, 0.975 if compact_main else 0.995),
        ncol=4 if compact_main else 8, frameon=False,
        fontsize=7.3 if compact_main else None,
        handlelength=1.30, columnspacing=0.90, handletextpad=0.34,
    )
    if compact_main:
        figure.subplots_adjust(left=0.115, right=0.990, top=0.840, bottom=0.065)
        step_center = (step_axes[0].get_position().x0 + step_axes[-1].get_position().x1) / 2
        frequency_center = (
            frequency_axes[0].get_position().x0 + frequency_axes[-1].get_position().x1
        ) / 2
        figure.text(
            step_center, step_axes[0].get_position().y1 + 0.055,
            "Pathway-length distributions", ha="center", va="center",
            fontsize=10.0, fontweight="bold", color="#30343B",
        )
        figure.text(
            frequency_center, frequency_axes[0].get_position().y1 + 0.055,
            "Short-pathway reaction frequency", ha="center", va="center",
            fontsize=10.0, fontweight="bold", color="#30343B",
        )
    else:
        figure.subplots_adjust(left=0.055, right=0.992, top=0.905, bottom=0.065)
        left_center = (step_axes[0].get_position().x0 + step_axes[-1].get_position().x1) / 2
        right_center = (
            frequency_axes[0].get_position().x0 + frequency_axes[-1].get_position().x1
        ) / 2
        figure.text(
            left_center, 0.953, "Pathway-length distributions", ha="center", va="center",
            fontsize=10.2, fontweight="bold", color="#30343B",
        )
        figure.text(
            right_center, 0.953, "Short-pathway reaction frequency", ha="center", va="center",
            fontsize=10.2, fontweight="bold", color="#30343B",
        )
    layout = audit_layout(
        figure, [step_axes[0], frequency_axes[0]], step_axes[-1], frequency_axes[0],
        stacked_main=compact_main,
    )
    layout["displayed_reactions"] = int(len(rows))
    layout["reactions_per_substrate"] = {
        substrate: int((data["mapping"]["substrate"] == substrate).sum())
        for substrate in SUBSTRATES
    }
    layout["layout"] = "2x3-stacked" if compact_main else "1x6-metric-block"
    return figure, layout

def main() -> None:
    args = parse_args()
    configure_logging(args.log_file)
    STYLE.set_publication_style()
    data = DATA_HELPER.load_data(args.data_dir)
    ranking = rank_reactions(data["summary"], data["mapping"])
    plot_data = dict(data)
    plot_data["mapping"] = select_mapping(
        data["mapping"], ranking, args.top_per_substrate, args.selection_mode,
    )
    output_stem = args.output_stem
    selection_output = args.selection_output
    if args.selection_mode == "candidate-order":
        if output_stem == DEFAULT_OUTPUT:
            output_stem = DEFAULT_CANDIDATE_OUTPUT
        if selection_output == DEFAULT_SELECTION_OUTPUT:
            selection_output = DEFAULT_CANDIDATE_SELECTION_OUTPUT
    elif args.selection_mode == "manuscript-selected":
        if output_stem == DEFAULT_OUTPUT:
            output_stem = DEFAULT_MANUSCRIPT_OUTPUT
        if selection_output == DEFAULT_SELECTION_OUTPUT:
            selection_output = DEFAULT_MANUSCRIPT_SELECTION_OUTPUT
    if selection_output is not None:
        write_selection_manifest(
            ranking, plot_data["mapping"], args.top_per_substrate,
            selection_output, args.selection_mode,
        )
    substrate_order = pd.Categorical(
        plot_data["mapping"]["substrate"], categories=SUBSTRATES, ordered=True,
    )
    sort_column = "display_rank" if "display_rank" in plot_data["mapping"].columns else "candidate_number"
    selected_codes = (
        plot_data["mapping"].assign(substrate_order=substrate_order)
        .sort_values(["substrate_order", sort_column], kind="mergesort")
        .groupby("substrate", observed=True, sort=False)["reaction_code"].apply(list).to_dict()
    )
    logging.info(
        "Plot Figure 3: selection_mode=%s; reactions_per_substrate=%s; smoke=%s; selected=%s",
        args.selection_mode, args.top_per_substrate or "all", args.smoke, selected_codes,
    )
    figure, layout = build_figure(
        plot_data, compact_main=args.top_per_substrate > 0,
    )
    layout["selection_mode"] = args.selection_mode if args.top_per_substrate > 0 else "all"
    if not layout["passed"]:
        raise ValueError(f"Metric-block final figure layout audit failed: {layout}")
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        figure.savefig(
            output_stem.with_suffix(".png"), dpi=180,
            bbox_inches="tight", facecolor="white",
        )
    else:
        STYLE.save_figure(figure, output_stem)
    if args.smoke:
        (output_stem.parent / "fig3_metric_block_layout_qc.json").write_text(
            json.dumps(layout, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
        )
    plt.close(figure)
    logging.info("Figure 3 completed: %s; layout=%s", output_stem, layout)


if __name__ == "__main__":
    main()
