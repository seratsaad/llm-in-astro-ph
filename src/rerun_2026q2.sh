#!/bin/zsh
# Full downstream rerun after extending the corpus to 2026Q2.
# Order matters: the discovery screen feeds the expanded basket, which two of
# the fit variants read, and s8 collects every number the manuscript quotes.
set -e
cd ~/astro_ph_llm_use_hierarchical
PY=.venv/bin/python

echo "=== aggregate"
$PY src/s2_aggregate.py

echo "=== discovery screen and expanded basket"
$PY src/s5_discovery.py
$PY src/s5b_topic_spread.py
$PY src/s5c_expanded_counts.py

echo "=== fulltext fits"
for v in primary unconstrained frozen_drift tracked_drift control_tracked \
         late_boundary no_disclosure gamma_low gamma_high control_placebo \
         expanded expanded_tracked; do
  $PY src/s3b_laplace.py --phase fulltext --variant $v 2>&1 \
    | grep -E "optimum|WARNING" | sed "s/^/[$v] /"
done

echo "=== whole-body robustness variant"
$PY src/s3b_laplace.py --phase fulltext --variant primary --section wholebody 2>&1 \
  | grep -E "optimum|WARNING" | sed "s/^/[wholebody] /"

echo "=== abstract fits"
for v in primary unconstrained; do
  $PY src/s3b_laplace.py --phase abstracts --variant $v 2>&1 \
    | grep -E "optimum|WARNING" | sed "s/^/[abs-$v] /"
done

echo "=== trajectories, calibration, numbers"
$PY src/s4_marker_trajectories.py fulltext fulltext_primary
$PY src/s7_calibration.py
$PY src/s8_fill_numbers.py

echo "=== figures"
$PY src/s6_figures.py
$PY src/s6b_composites.py

echo "RERUN-DONE"
