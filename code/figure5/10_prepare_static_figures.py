"""Compose six web screenshots as a publication-oriented 3-row x 2-column figure."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

REPO_ROOT = Path(__file__).resolve().parents[2]
PANEL_DIR = REPO_ROOT / "supplementary_materials/figure5/source_panels"
DEFAULT_OUTPUT = REPO_ROOT / "results/figure5/Fig5_web_platform"

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
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target-width-mm", type=float, default=180.0)
    parser.add_argument("--panel-width", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--log-file", type=Path, default=REPO_ROOT / "logs/figure5.log")
    return parser.parse_args()


def configure_logging(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler()],
    )


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/google-droid/DroidSans-Bold.ttf",
    ]
    for font_path in candidates:
        if Path(font_path).is_file():
            return ImageFont.truetype(font_path, size)
    return ImageFont.load_default()


def read_rgb(path: Path) -> Image.Image:
    if not path.is_file():
        raise FileNotFoundError(f"Missing screenshot: {path}")
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def compose_fig5(panel_width: int = 0) -> Image.Image:
    """Create a 3 x 2 montage while keeping screenshots at native resolution."""
    panels = [read_rgb(path) for path in FIG5_PANELS]

    native_common_width = min(panel.width for panel in panels)
    common_width = panel_width if 0 < panel_width < native_common_width else native_common_width

    resized: list[Image.Image] = []
    for panel in panels:
        if panel.width == common_width:
            resized.append(panel)
            continue
        height = round(panel.height * common_width / panel.width)
        resized.append(panel.resize((common_width, height), Image.Resampling.LANCZOS))

    ncols, nrows = 2, 3
    margin = 32
    gap_x = 34
    gap_y = 42
    title_h = 66
    border = 2

    row_heights = [
        max(resized[row * ncols + col].height for col in range(ncols))
        for row in range(nrows)
    ]

    width = 2 * margin + ncols * common_width + (ncols - 1) * gap_x
    height = (
        2 * margin
        + nrows * title_h
        + sum(row_heights)
        + (nrows - 1) * gap_y
    )

    output = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(output)
    label_font = load_font(52)
    title_font = load_font(38)

    row_y = margin
    for index, (panel, title) in enumerate(zip(resized, FIG5_TITLES)):
        row, col = divmod(index, ncols)

        if col == 0 and row > 0:
            row_y += title_h + row_heights[row - 1] + gap_y

        x = margin + col * (common_width + gap_x)
        y = row_y
        image_y = y + title_h

        draw.text((x, y - 2), chr(65 + index), fill="#111111", font=label_font)
        draw.text((x + 58, y + 7), title, fill="#111111", font=title_font)

        output.paste(panel, (x, image_y))
        draw.rectangle(
            (x, image_y, x + common_width - 1, image_y + row_heights[row] - 1),
            outline="#9AA7B5",
            width=border,
        )

    return output


def save_outputs(image: Image.Image, stem: Path, target_width_mm: float) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)

    dpi = image.width / (target_width_mm / 25.4)
    png_path = stem.with_suffix(".png")
    image.save(png_path, dpi=(dpi, dpi), optimize=True)

    image.save(stem.with_suffix(".pdf"), "PDF", resolution=dpi)

    logging.info("Figure written: %s; pixels=%s; effective_dpi=%.1f", stem, image.size, dpi)


def main() -> None:
    args = parse_args()
    configure_logging(args.log_file)
    if args.smoke:
        output = REPO_ROOT / "validation/figure5/Fig5_web_platform_smoke.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        figure = compose_fig5(panel_width=600)
        figure.save(output, dpi=(180, 180), optimize=True)
        logging.info("Figure 5 smoke image: %s; pixels=%s", output, figure.size)
        return
    figure = compose_fig5(panel_width=args.panel_width)
    save_outputs(figure, args.output_stem, args.target_width_mm)


if __name__ == "__main__":
    main()
