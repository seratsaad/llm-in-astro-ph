#!/bin/zsh
# Waits for queues A and B, runs the robustness variants, then post-processing.
cd ~/astro_ph_llm_use_hierarchical
while [ "$(grep -c 'sampling took' logs/queueA.log)" -lt 3 ] || \
      [ "$(grep -c 'sampling took' logs/queueB.log)" -lt 3 ]; do
  sleep 180
done
echo "=== queues done $(date) ==="

# robustness + expanded variants, two at a time, reduced draws
run() { python3 src/s3_fit.py --phase fulltext --variant "$1" \
        --draws 600 --tune 800 --chains 4; }
run expanded
run late_boundary
run no_disclosure
run gamma_low
run gamma_high
echo "=== variants done $(date) ==="

python3 src/s4_marker_trajectories.py fulltext fulltext_primary
python3 src/s8_fill_numbers.py
python3 src/s6_figures.py
cd paper && pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1
pdflatex -interaction=nonstopmode main.tex 2>&1 | grep -E "^!|Output written"
echo "=== ALL DONE $(date) ==="
