#!/bin/zsh
cd ~/astro_ph_llm_use_hierarchical
echo "=== fulltext control_placebo $(date) ==="
python3 src/s3_fit.py --phase fulltext --variant control_placebo --chains 4
echo "=== fulltext unconstrained $(date) ==="
python3 src/s3_fit.py --phase fulltext --variant unconstrained --chains 4
echo "=== abstracts primary $(date) ==="
python3 src/s3_fit.py --phase abstracts --variant primary --chains 4
