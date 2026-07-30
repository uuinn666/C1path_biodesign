
from pathlib import Path
import runpy


TARGET = Path(__file__).with_name("37_plot_fig4_product_story_candidate.py")


if __name__ == "__main__":
    runpy.run_path(str(TARGET), run_name="__main__")
