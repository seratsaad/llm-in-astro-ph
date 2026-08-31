#!/bin/zsh
cd ~/astro_ph_llm_use_hierarchical
for v in primary frozen_drift tracked_drift; do
  echo "=== fulltext $v $(date) ==="
  python3 src/s3_fit.py --phase fulltext --variant $v --chains 4
done
