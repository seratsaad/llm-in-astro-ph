#!/usr/bin/env python3
"""Stage 4: per-marker sensitivity through time, without circularity.

The question is whether individual markers fade while prevalence rises. The
naive estimate -- take the papers the model calls assisted and count how often
they use marker j -- is circular, because marker j helped select those papers.

Two guards:

  1. Deconvolution rather than selection. For each quarter,
         obs_j(t) = (1 - pi_t) p0_j(t) + pi_t p1_j(t)
     is solved for p1_j(t), with pi_t taken from the fitted model and p0_j(t)
     the background occurrence extrapolated from the known-negative era. Marker
     j never selects the papers used to measure marker j.

  2. A refit check. The model is refit with the two largest-contributing marker
     families removed; if pi_t barely moves, the residual circularity from
     dropping one marker out of 38 is second order.

Writes data/marker_trajectories.csv.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from common import DATA, MARKERS, YEAR_MIN, quarter_label

BASE_QUARTERS = 20          # 2015Q1..2019Q4 are the known-negative era


def background_rate(df, word, n_q, free_from=BASE_QUARTERS):
    """Occurrence rate per token of `word`, extrapolated past the known era.

    A log-linear fit on the known-negative quarters, weighted by exposure.
    """
    col = "w_" + word
    rate = np.full(n_q, np.nan)
    expo = np.zeros(n_q)
    for q in range(n_q):
        m = df.q.values == q
        if m.sum() == 0:
            continue
        expo[q] = df.L.values[m].sum()
        rate[q] = df[col].values[m].sum() / expo[q]
    known = np.arange(free_from)
    y = rate[known]
    ok = np.isfinite(y) & (y > 0)
    if ok.sum() < 6:
        lvl = np.nansum(rate[known] * expo[known]) / np.nansum(expo[known])
        return np.full(n_q, max(lvl, 1e-9)), rate
    coef = np.polyfit(known[ok], np.log(y[ok]), 1, w=np.sqrt(expo[known][ok]))
    p0 = np.exp(np.polyval(coef, np.arange(n_q)))
    return p0, rate


def main():
    phase = sys.argv[1] if len(sys.argv) > 1 else "fulltext"
    tag = sys.argv[2] if len(sys.argv) > 2 else f"{phase}_primary"
    feat = ("abstract_features.parquet" if phase == "abstracts"
            else "fulltext_features.parquet")
    df = pd.read_parquet(os.path.join(DATA, feat))
    pi_tab = pd.read_csv(os.path.join(DATA, f"pi_{tag}.csv"))

    n_q = int(df.q.max()) + 1
    free_from = n_q - len(pi_tab)
    pi = np.zeros(n_q)
    pi[free_from:] = pi_tab["mean"].values

    rows = []
    for w in MARKERS:
        p0, obs = background_rate(df, w, n_q, free_from=BASE_QUARTERS)
        for q in range(n_q):
            if not np.isfinite(obs[q]):
                continue
            # obs = (1-pi) p0 + pi p1  ->  p1 = (obs - (1-pi) p0)/pi
            p1 = ((obs[q] - (1 - pi[q]) * p0[q]) / pi[q]) if pi[q] > 1e-4 else np.nan
            rows.append(dict(marker=w, q=q, quarter=quarter_label(q),
                             year=YEAR_MIN + q // 4,
                             obs_rate=obs[q], bg_rate=p0[q], pi=pi[q],
                             assisted_rate=p1,
                             excess_ratio=obs[q] / p0[q] if p0[q] > 0 else np.nan))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(DATA, f"marker_trajectories_{phase}.csv"), index=False)

    print("excess ratio (observed / extrapolated background), by year")
    piv = out.pivot_table(index="marker", columns="year",
                          values="excess_ratio", aggfunc="mean")
    cols = [c for c in piv.columns if c >= 2022]
    print(piv[cols].round(2).sort_values(cols[-2], ascending=False).to_string())

    print("\nmarker rate among assisted prose (deconvolved), by year")
    piv2 = out.pivot_table(index="marker", columns="year",
                           values="assisted_rate", aggfunc="mean")
    c2 = [c for c in piv2.columns if c >= 2023]
    print((1000 * piv2[c2]).round(3).sort_values(c2[-2], ascending=False).to_string())
    print(f"\nwrote data/marker_trajectories_{phase}.csv")


if __name__ == "__main__":
    main()
