#!/bin/zsh
# Fits + downstream numbers after the declaration rescan. The baskets and the
# calibration grid do not read the declared column, so they are not rerun.
set -e
cd ~/astro_ph_llm_use_hierarchical
PY=.venv/bin/python

for v in primary unconstrained frozen_drift tracked_drift control_tracked \
         late_boundary no_disclosure gamma_low gamma_high control_placebo \
         expanded expanded_tracked; do
  $PY src/s3b_laplace.py --phase fulltext --variant $v 2>&1 \
    | grep -E "optimum|WARNING" | sed "s/^/[$v] /"
done
$PY src/s3b_laplace.py --phase fulltext --variant primary --section wholebody 2>&1 \
  | grep -E "optimum|WARNING" | sed "s/^/[wholebody] /"
for v in primary unconstrained; do
  $PY src/s3b_laplace.py --phase abstracts --variant $v 2>&1 \
    | grep -E "optimum|WARNING" | sed "s/^/[abs-$v] /"
done
$PY src/s3b_laplace.py --phase abstracts --variant primary --abs-cohort 2>&1 \
  | grep -E "optimum|WARNING" | sed "s/^/[abs-cohort] /"

$PY src/s4_marker_trajectories.py fulltext fulltext_primary
$PY src/s8_fill_numbers.py
echo "REFIT-DONE"
