
"""Forward the Figure 3 entry point to the approved metric-block implementation."""

from pathlib import Path
import runpy


TARGET = Path(__file__).with_name("35_plot_fig3_metric_block_candidate.py")


if __name__ == "__main__":
    runpy.run_path(str(TARGET), run_name="__main__")
