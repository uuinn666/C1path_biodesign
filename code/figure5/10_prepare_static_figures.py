#!/usr/bin/env python
# -*- coding: utf-8 -*-
# --------------------------------------------------
# Created: 2026-07-27
# Purpose: Reproduce Figure 5 from the six colocated English web screenshots.
# Usage: Run with --smoke first, then rerun without --smoke for PNG/PDF output.
# Source: English-only Figure 5 adaptation of writing3/code/10_prepare_static_figures.py.
# 修改时间：2026-07-27
# 原始实现：Figure 5面板B读取旧文件名image.png。
# 存在问题：面板B源图已重命名为image2.png，导致脚本无法找到输入文件。
# 修改内容：仅将面板B输入路径同步为image2.png，不改变拼图尺寸、顺序或绘图逻辑。
# 修改时间：2026-07-27
# 原始实现：六张截图与脚本位于同一目录。
# 存在问题：公开仓库将源截图归档到supplementary_materials。
# 修改内容：从补充材料目录读取截图，并将正式拼图写入results/figure5。
# --------------------------------------------------

"""Compose the six-panel static web-platform figure."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[2]
PANEL_DIR = REPO_ROOT / "supplementary_materials/figure5/source_panels"
DEFAULT_OUTPUT = REPO_ROOT / "results/figure5/Fig5_web_platform"
FIG5_SOURCE = REPO_ROOT / "results/figure5/Fig5_web_platform_source.png"
FIG5_PANELS = [
    PANEL_DIR / "image1.png",
    PANEL_DIR / "image2.png",
    PANEL_DIR / "image3.png",
    PANEL_DIR / "image4.png",
    PANEL_DIR / "image5.png",
    PANEL_DIR / "image6.png",
]
FIG5_TITLES = [
    "Home",
    "Task submission",
    "Calculation results",
    "Thermodynamics and MDF",
    "Pathway network",
    "Reaction details",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Write only a small validation image")
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output path without a file extension",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=REPO_ROOT / "logs/figure5.log",
        help="Path to the run log",
    )
    return parser.parse_args()


def configure_logging(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler()],
    )


def read_rgb(path: Path) -> Image.Image:
    if not path.is_file():
        raise FileNotFoundError(f"Missing static figure source: {path}")
    return Image.open(path).convert("RGB")


def load_font(size: int) -> ImageFont.FreeTypeFont:
    """Locate Droid Sans Bold without relying on one HPC-specific absolute path."""
    font_path = font_manager.findfont(
        font_manager.FontProperties(family="Droid Sans", weight="bold"),
        fallback_to_default=False,
    )
    return ImageFont.truetype(font_path, size)


def compose_fig5(column_width: int = 2500) -> Image.Image:
    """Compose six screenshots in a three-by-two layout while preserving aspect ratios and row frames."""
    panels = [read_rgb(path) for path in FIG5_PANELS]
    resized = []
    for panel in panels:
        height = round(panel.height * column_width / panel.width)
        resized.append(panel.resize((column_width, height), Image.Resampling.LANCZOS))

    scale = column_width / 2500
    margin = round(65 * scale)
    gap_x = round(70 * scale)
    gap_y = round(75 * scale)
    title_h = round(105 * scale)
    border = max(2, round(3 * scale))
    row_heights = [max(im.height for im in resized[:3]), max(im.height for im in resized[3:])]
    width = 2 * margin + 3 * column_width + 2 * gap_x
    height = 2 * margin + 2 * title_h + sum(row_heights) + gap_y
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    label_font = load_font(max(15, round(58 * scale)))
    title_font = load_font(max(14, round(46 * scale)))

    for index, panel in enumerate(resized):
        row, col = divmod(index, 3)
        x = margin + col * (column_width + gap_x)
        y = margin + row * (title_h + row_heights[0] + gap_y)
        draw.text((x, y), chr(65 + index), fill="#111111", font=label_font)
        draw.text((x + round(78 * scale), y + round(7 * scale)), FIG5_TITLES[index], fill="#111111", font=title_font)
        image_y = y + title_h
        canvas.paste(panel, (x, image_y))
        draw.rectangle(
            (x, image_y, x + column_width - 1, image_y + row_heights[row] - 1),
            outline="#9AA7B5",
            width=border,
        )

    return canvas


def save_fig5(stem: Path) -> None:
    image = compose_fig5()
    stem.parent.mkdir(parents=True, exist_ok=True)
    if stem.resolve() == DEFAULT_OUTPUT.resolve():
        image.save(FIG5_SOURCE, dpi=(600, 600), optimize=True)
    image.save(stem.with_suffix(".png"), dpi=(600, 600), optimize=True)
    image.save(stem.with_suffix(".pdf"), resolution=600.0)
    logging.info("Figure 5: %s; size=%s", stem, image.size)


def main() -> None:
    args = parse_args()
    configure_logging(args.log_file)
    if args.smoke:
        smoke = REPO_ROOT / "validation/figure5"
        smoke.mkdir(parents=True, exist_ok=True)
        fig5_smoke = compose_fig5(column_width=600)
        fig5_smoke.save(smoke / "Fig5_web_platform_smoke.png", dpi=(180, 180), optimize=True)
        logging.info(
            "Figure 5 smoke image: %s; size=%s",
            smoke / "Fig5_web_platform_smoke.png",
            fig5_smoke.size,
        )
        return
    save_fig5(args.output_stem)


if __name__ == "__main__":
    main()
