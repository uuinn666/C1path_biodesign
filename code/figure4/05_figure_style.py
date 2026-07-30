
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
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "legend.fontsize": 7.5,
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
    axis.text(-0.12, 1.08, label, transform=axis.transAxes, fontsize=11, fontweight="bold", va="top")
    axis.set_title(title, loc="left", fontweight="bold", pad=8)


def deterministic_sample(values: np.ndarray, maximum: int) -> np.ndarray:
    if len(values) <= maximum:
        return values
    indices = np.linspace(0, len(values) - 1, maximum, dtype=int)
    return values[indices]


def save_figure(figure: plt.Figure, output_stem: Path) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_stem.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
