#!/bin/zsh
cd ~/astro_ph_llm_use_hierarchical
for v in unconstrained expanded expanded_tracked late_boundary no_disclosure gamma_low gamma_high control_placebo; do
  .venv/bin/python src/s3b_laplace.py --phase fulltext --variant $v 2>&1 | grep -E "^===|optimum|WARNING" | sed "s/^/[$v] /"
done
for v in primary unconstrained; do
  .venv/bin/python src/s3b_laplace.py --phase abstracts --variant $v 2>&1 | grep -E "optimum|WARNING" | sed "s/^/[abs-$v] /"
done
echo GRID-DONE
