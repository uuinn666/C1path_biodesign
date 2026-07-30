
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


SUBSTRATES = ["co2", "methanol", "formate", "formaldehyde"]
SUBSTRATE_LABELS = {
    "co2": r"CO$_2$",
    "formaldehyde": "Formaldehyde",
    "formate": "Formate",
    "methanol": "Methanol",
}
SUBSTRATE_COLORS = {
    "co2": "#08306B",
    "formaldehyde": "#4292C6",
    "formate": "#FC9272",
    "methanol": "#CB181D",
}
BLUES = ["#DEEBF7", "#C6DBEF", "#9ECAE1", "#6BAED6", "#4292C6", "#2171B5", "#08519C", "#08306B"]
REDS = ["#FEE0D2", "#FCBBA1", "#FC9272", "#FB6A4A", "#EF3B2C", "#CB181D", "#A50F15", "#67000D"]
NEUTRAL = "#5B616B"
LIGHT_NEUTRAL = "#D9DEE7"
GRID = "#E5EAF1"
PANEL_LABEL_X = -0.12
PANEL_TITLE_X = -0.075
PANEL_HEADER_Y = 1.08
PANEL_LABEL_SIZE = 10.6
PANEL_TITLE_SIZE = 8.8
CATEGORY_COLORS = {
    "Central carbon precursors": "#08306B",
    "Sugar-phosphate precursors": "#4292C6",
    "Organic and hydroxy acids": "#6BAED6",
    "Amino acids": "#FCBBA1",
    "Alcohols, ketones and platform chemicals": "#EF3B2C",
    "Specialized biosynthetic precursors": "#67000D",
}
CATEGORY_LABELS = {
    "Central carbon precursors": "Central carbon precursors",
    "Sugar-phosphate precursors": "Sugar-phosphate precursors",
    "Organic and hydroxy acids": "Organic and hydroxy acids",
    "Amino acids": "Amino acids",
    "Alcohols, ketones and platform chemicals": "Alcohols, ketones and platform chemicals",
    "Specialized biosynthetic precursors": "Specialized biosynthetic precursors",
}


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


def panel_label(axis: plt.Axes, label: str, title: str) -> None:
    axis.text(
        PANEL_LABEL_X, PANEL_HEADER_Y, label,
        transform=axis.transAxes, fontsize=PANEL_LABEL_SIZE, fontweight="bold", va="top",
    )
    axis.text(
        PANEL_TITLE_X, PANEL_HEADER_Y, title,
        transform=axis.transAxes, fontsize=PANEL_TITLE_SIZE,
        fontweight="bold", va="top",
    )


def deterministic_sample(values: np.ndarray, maximum: int) -> np.ndarray:
    if len(values) <= maximum:
        return values
    indices = np.linspace(0, len(values) - 1, maximum, dtype=int)
    return values[indices]


def save_figure(figure: plt.Figure, output_stem: Path) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_stem.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
