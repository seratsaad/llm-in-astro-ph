#!/usr/bin/env python3
"""Stage 2c: the restriction ladder.

The paper's spine is: (1) the Kobak estimator reproduced exactly, (2) the same
estimator failing the placebo test, (3) the hierarchical model obtained by
relaxing its restrictions one at a time, (4) the assumption bracket as the
headline. This stage computes the middle rungs so the relaxation is a table of
numbers rather than an argument:

  rung 0  document frequency, linear extrapolation      (= Kobak, Eq. beta)
  rung 2  token counts per unit length, flat pre-2020
          background, assisted rate anchored to the
          declared papers (method of moments)
  rung 3  rung 2 with the background drifting linearly
          on the per-token scale
  rung 4  full hierarchy (MCMC, from Stage 3)           [read from pi_*.csv]

Rungs 2-3 are moment estimators, pi = (r_obs - r_bg)/(r_declared - r_bg): the
rate-scale analogue of the Kobak denominator, with "assisted papers show the
basket with probability one" replaced by "assisted papers show the basket at
the rate the declared papers do". An unpenalized mixture MLE was tried first
and slides to the pi boundary whenever the excess is small: the
prevalence-times-excess ridge in its rawest form, worth reporting but not
tabulating.

Each rung is evaluated for the marker basket AND the control basket; the
control column is the placebo test at every rung.

Writes data/ladder.json.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from common import DATA, MARKERS, CONTROL, PLACEBO

YEARS_EVAL = [2024, 2025]
FIT_YEARS = (2015, 2022)          # pre-adoption window for trends
PRE_YEARS = (2015, 2019)          # known-negative era for rates


def beta_docfreq(df, K, years=YEARS_EVAL):
    """Rung 0: document-frequency estimator with linear extrapolation."""
    has = (K > 0).astype(float)
    rates = df.assign(_h=has).groupby("year")._h.mean()
    m = (rates.index >= FIT_YEARS[0]) & (rates.index <= FIT_YEARS[1])
    coef = np.polyfit(rates.index[m].astype(float), rates[m].values, 1)
    out = {}
    for y in years:
        q = float(rates.loc[y])
        q0 = float(np.polyval(coef, y))
        out[y] = (q - q0) / (1 - q0)
    return out


def bootstrap_rate(K, L, n=400, rng=None):
    """Bootstrap SE of a per-token rate over papers."""
    rng = rng or np.random.default_rng(7)
    idx = np.arange(len(K))
    vals = []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        vals.append(K[b].sum() / L[b].sum())
    return float(np.std(vals))


def moment_pi(r_obs, r_bg, r_decl):
    """pi = (r_obs - r_bg) / (r_decl - r_bg); the declared-anchored rung."""
    den = r_decl - r_bg
    if den <= 0:
        return float("nan")
    return (r_obs - r_bg) / den


def run_phase(name, df, baskets):
    res = {}
    pre = df[(df.year >= PRE_YEARS[0]) & (df.year <= PRE_YEARS[1])]
    declared = df[df.get("declared", pd.Series(0, index=df.index)) == 1]
    for bname, cols in baskets.items():
        K = df[cols].sum(axis=1).values.astype("int64")
        entry = {"rung0_docfreq": beta_docfreq(df, K)}

        # anchor: marker rate of the declared (known-positive) papers, 2023+
        dsub = declared[declared.year >= 2023]
        if len(dsub) >= 30:
            r_decl = dsub[cols].sum(axis=1).sum() / dsub.L.sum()
            se_decl = bootstrap_rate(dsub[cols].sum(axis=1).values.astype("int64"),
                                     dsub.L.values.astype("float64"))
        else:
            r_decl, se_decl = float("nan"), float("nan")
        entry["declared_rate_per1k"] = 1000 * r_decl
        entry["declared_rate_se"] = 1000 * se_decl
        entry["n_declared_anchor"] = int(len(dsub))

        # rung 2: flat pre-2020 background rate
        r_flat = pre[cols].sum(axis=1).sum() / pre.L.sum()
        # rung 3: background rate drifting linearly on the per-token scale
        yr = np.arange(FIT_YEARS[0], FIT_YEARS[1] + 1)
        rate_y = np.array([
            df.loc[df.year == y, cols].to_numpy().sum() /
            df.loc[df.year == y, "L"].sum() for y in yr])
        coef = np.polyfit(yr.astype(float), rate_y, 1)

        for rung, bg in (("rung2_flatbg", lambda y: r_flat),
                         ("rung3_lineardrift",
                          lambda y: float(np.polyval(coef, y)))):
            entry[rung] = {}
            for y in YEARS_EVAL:
                sub = df[df.year == y]
                r_obs = sub[cols].sum(axis=1).sum() / sub.L.sum()
                entry[rung][y] = round(moment_pi(r_obs, bg(y), r_decl), 4)

        res[bname] = entry
        print(f"[{name}:{bname}] r0={ {y: round(v,3) for y,v in entry['rung0_docfreq'].items()} } "
              f"r_decl/1k={1000*r_decl:.3f}  r2={entry['rung2_flatbg']}  "
              f"r3={entry['rung3_lineardrift']}")
    return res


def main():
    out = {}

    ab = pd.read_parquet(os.path.join(DATA, "abstract_features.parquet"))
    decl = pd.read_csv(os.path.join(DATA, "declarations.csv"),
                       dtype={"arxiv_id": str})
    pos = set(decl.loc[decl.declared == 1, "arxiv_id"])
    ab["declared"] = ab.arxiv_id.isin(pos).astype(int)
    baskets_ab = {"markers": ["w_" + w for w in MARKERS],
                  "control": ["w_" + w for w in CONTROL],
                  "placebo": ["w_" + w for w in PLACEBO]}
    out["abstracts"] = run_phase("abstracts", ab, baskets_ab)

    ft = pd.read_parquet(os.path.join(DATA, "fulltext_features.parquet"))
    secs = ["intro", "methods", "results", "discussion", "conclusions"]
    ft["L"] = ft[[f"L_{s}" for s in secs]].sum(axis=1)
    ft = ft[ft.L >= 300].reset_index(drop=True)
    baskets_ft = {"markers": [f"K_{s}" for s in secs],
                  "control": [f"C_{s}" for s in secs],
                  "placebo": [f"P_{s}" for s in secs]}
    out["fulltext"] = run_phase("fulltext", ft, baskets_ft)

    with open(os.path.join(DATA, "ladder.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("\nwrote data/ladder.json")


if __name__ == "__main__":
    main()
