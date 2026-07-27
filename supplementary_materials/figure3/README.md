# Figure 3 source data

## Statistical scope

- Analysis covers all computable C2, C3, and C4 product pathways for the four substrates.
- Pathway-step distributions include all feasible pathways and are expanded by whether a pathway contains each candidate assimilation reaction.
- Short-pathway occurrence is the number of pathways containing a reaction divided by all pathways with at most 20 steps in the same substrate-carbon-class group.
- One pathway may contain multiple assimilation reactions and therefore may contribute to multiple rows. Reaction occurrence percentages may sum to more than 100%.

## Files

- `fig3_step_distribution.tsv`: Long pathway-reaction table for the left-side box plots; 50,665 rows.
- `fig3_reaction_carbon_summary.tsv`: Summary of 39 substrate-reaction entries across three carbon classes; 117 rows containing pathway-length quantiles, single-reaction medians, and short-pathway counts and percentages.
- `fig3_assimilation_reaction_mapping.tsv`: Mapping from 39 short reaction codes to model reaction identifiers, English names, full model equations, and candidate-pool metadata.
- `fig3_group_sample_sizes.tsv`: Feasible-pathway, short-pathway, and reachable-product counts for 12 substrate-carbon-class groups.
- `fig3_source_validation.json`: Row counts, candidate-reaction sets, and consistency checks.

## Provenance

The tables were reconstructed by `writing3/code/31_build_assimilation_reaction_analysis.py` from `writing3/data/processed/pathway_master.tsv` and the current model candidate-reaction definitions. The final analysis excludes C1 products and products without computable pathways.
