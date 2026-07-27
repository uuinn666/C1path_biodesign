# C1path_biodesign

This repository provides the final figures, structured supplementary materials, and reproducible plotting code for a multi-substrate one-carbon pathway study.

## Repository Structure

```text
C1path_biodesign/
├── code/                       # Plotting code for Figures 2–5
├── supplementary_materials/    # Structured source data and Figure 5 web screenshots
├── results/                    # Final PNG and PDF files for Figures 1–5
└── requirements.txt            # Validated Python dependency versions
```

## Environment

The validated environment uses Python 3.9.21. Install the required packages with:

```bash
python -m pip install -r requirements.txt
```

## Smoke Tests

```bash
python code/figure2/06_plot_fig2_pathway_overview.py --smoke --output-stem validation/figure2/Fig2_pathway_overview
python code/figure3/07_plot_fig3_reaction_analysis.py --smoke --output-stem validation/figure3/Fig3_reaction_analysis
python code/figure4/08_plot_fig4_product_landscape.py --smoke --output-stem validation/figure4/Fig4_product_landscape
python code/figure5/10_prepare_static_figures.py --smoke
```

## Reproducing the Final Figures

```bash
python code/figure2/06_plot_fig2_pathway_overview.py
python code/figure3/07_plot_fig3_reaction_analysis.py
python code/figure4/08_plot_fig4_product_landscape.py
python code/figure5/10_prepare_static_figures.py
```

Final outputs are written to the corresponding `results/figure*` directories, and runtime logs are written to `logs/`. The Figure 3 workflow also generates a JSON layout quality-control report.

## Data Description

- `supplementary_materials/figure2`: structured TSV source data for the five panels in Figure 2.
- `supplementary_materials/figure3`: pathway-step distributions, reaction frequencies, reaction mappings, group sample sizes, and the source-data validation report.
- `supplementary_materials/figure4`: substrate–product reachability, product-class coverage, and pathway distribution data.
- `supplementary_materials/figure5/source_panels`: the six web screenshots used to assemble Figure 5.
