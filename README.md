# Language-model use in the astronomy literature

Code and derived data for **"More than half of recent astronomy papers show
language-model use"** (Saad & Ting).

We measure how much of the astronomy literature is written with the help of
large language models. From the full text of astro-ph papers spanning 2015 to
mid-2026 we count the vocabulary that language models favor, calibrate it on
the papers whose authors declared model use, and infer the assisted fraction
with a hierarchical Bayesian model.

## Corpus

The corpus is the public **AstroMLab 5** catalog (Ting et al. 2025). This
repository does not redistribute it. The pipeline reads it and writes the
derived tables in `data/`.

## Layout

```
paper/  the manuscript (main.tex), bibliography, and figures
src/    the analysis pipeline, in stage order
data/   derived tables, frozen baskets, and fitted posteriors
```

### Pipeline

| Stage | Script | Produces |
|---|---|---|
| 0 | `s0_abstract_features.py` | `abstract_features.parquet` |
| 1 | `s1_fulltext_features.py` | `ft_shard_*.parquet`, `fulltext_features.parquet` |
| 1b | `s1b_rescan_disclosure.py` | `ft_disclosure_*.json`, `declarations.csv` |
| 2 | `s2_aggregate.py`, `s2b_length_diagnostic.py`, `s2c_ladder.py` | `ladder.json`, `length_diagnostic.json` |
| 3 | `s3_fit.py`, `s3b_laplace.py` | `laplace_*.json`, `pi_*.csv`, `idata_*.nc` |
| 4 | `s4_marker_trajectories.py` | `marker_trajectories_fulltext.csv` |
| 5 | `s5_discovery.py`, `s5b_topic_spread.py`, `s5c_expanded_counts.py` | `discovered_markers.csv`, `expanded_basket.json`, `expanded_features.parquet` |
| 6 | `s6_figures.py`, `s6b_composites.py` | the paper figures |
| 7 | `s7_calibration.py` | `calibration.json` |
| 8 | `s8_fill_numbers.py`, `s9_basket_table.py` | the number macros and basket table used by the manuscript |

`model.py` holds the hierarchical model, `common.py` the shared IO and the
frozen word lists.

### Data

- **Per-paper feature tables.** `fulltext_features.parquet` (214,883 papers),
  `abstract_features.parquet` (227,438), `expanded_features.parquet`. Columns
  are per-paper word counts, section lengths, and the declaration flag. They
  contain no per-paper assisted probability.
- **Frozen baskets.** `expanded_basket.json`, `discovered_markers.csv`,
  `discovered_with_spread.csv`.
- **Fitted posteriors.** `laplace_*.json` for every specification in the
  variant grid, with the corresponding yearly assisted fractions in `pi_*.csv`
  and the MCMC cross-check in `idata_abstracts_primary_smoke.nc`.
- **Supporting tables.** `declarations.csv`, `calibration.json`, `ladder.json`,
  `cohort_summary.json`, `aggregate_reproduction.json`.

The naming follows the specifications in the paper, so `laplace_fulltext_primary`
is the primary specification and `..._frozen_drift`, `..._tracked_drift`,
`..._gamma_low`, `..._gamma_high`, `..._expanded`, `..._control_placebo`,
`..._no_disclosure`, `..._late_boundary`, and `..._unconstrained` are the
variants reported in the robustness section.

## What is not released

We do not release a per-paper assisted-probability list. The model is
calibrated for population inference and is not reliable for classifying an
individual paper, and publishing such a list would invite exactly the
document-level misuse the paper argues against.

## Reproducing

```bash
pip install -r requirements.txt
python src/s0_abstract_features.py     # then s1, s2, s3, ... in stage order
```

The Laplace fits run in about a minute each, which is what makes the full
variant grid affordable. The `.sh` drivers in `src/` reproduce the grid.
