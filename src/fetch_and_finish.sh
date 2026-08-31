#!/bin/zsh
# Pull fit results from pitzer and finish the paper pipeline locally.
set -e
cd ~/astro_ph_llm_use_hierarchical
echo "--- pulling results from pitzer"
rsync -az --timeout=90 "pitzer:llm_hier/data/pi_*.csv" data/
rsync -az --timeout=90 "pitzer:llm_hier/data/idata_fulltext_primary.nc" data/ || true
rsync -az --timeout=90 "pitzer:llm_hier/logs/llm_*.out" logs/pitzer/ 2>/dev/null || \
  (mkdir -p logs/pitzer && rsync -az --timeout=90 "pitzer:llm_hier/logs/llm_*.out" logs/pitzer/)
echo "--- post-processing"
python3 src/s4_marker_trajectories.py fulltext fulltext_primary
python3 src/s8_fill_numbers.py
python3 src/s6_figures.py
echo "--- compiling"
cd paper
pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1 || true
pdflatex -interaction=nonstopmode main.tex 2>&1 | grep -E "^!|Output written" || true
echo "--- convergence summary"
grep -h "convergence failures" ../logs/pitzer/llm_*.out 2>/dev/null || true
echo "ALL DONE"
