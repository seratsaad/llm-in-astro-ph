#!/usr/bin/env python3
"""Stage 2: the nested aggregate calculation (validation gate 1).

Reduces every paper to any-marker / no-marker, drops length and topic, and
fixes one background rate and one sensitivity. This reproduces both

  * the published lower bound  alpha = f_t - f_0  (Saad 2026), and
  * the biomedical-style estimator  beta = (f_t - f0_hat)/(1 - f0_hat)
    with f0_hat from linear extrapolation of the pre-adoption trend
    (Kobak et al. 2026),

so that the hierarchical model can later be checked against a calculation whose
arithmetic is transparent. Nothing here is a headline result.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from common import DATA, MARKERS, MARKERS_STRONG, CONTROL, PLACEBO

BASELINE_YEARS = (2018, 2021)      # pre-adoption window used for f0
FIT_YEARS = (2015, 2022)           # linear extrapolation window


def any_rate(df, cols, by="year"):
    K = df[cols].sum(axis=1)
    return df.assign(_any=(K > 0).astype(float)).groupby(by)._any.agg(["mean", "size"])


def linear_extrapolate(rates, fit_years, target_years):
    """OLS on the pre-adoption years, extrapolated, with prediction variance."""
    m = (rates.index >= fit_years[0]) & (rates.index <= fit_years[1])
    x = rates.index[m].values.astype(float)
    y = rates["mean"][m].values
    n = len(x)
    X = np.column_stack([np.ones(n), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    s2 = resid @ resid / (n - 2)
    XtXi = np.linalg.inv(X.T @ X)
    out = {}
    for t in target_years:
        x0 = np.array([1.0, float(t)])
        pred = float(x0 @ coef)
        var = float(s2 * (x0 @ XtXi @ x0))
        out[t] = (pred, np.sqrt(var))
    return out, coef


def analyse(df, cols, name, results):
    rates = any_rate(df, cols)
    ext, coef = linear_extrapolate(rates, FIT_YEARS, [2023, 2024, 2025, 2026])
    b0, b1 = BASELINE_YEARS
    msk = (df.year >= b0) & (df.year <= b1)
    Kb = df.loc[msk, cols].sum(axis=1)
    f0_flat = float((Kb > 0).mean())

    rows = []
    for year in [2023, 2024, 2025, 2026]:
        f = float(rates["mean"].loc[year])
        n = int(rates["size"].loc[year])
        se_f = np.sqrt(f * (1 - f) / n)
        f0_lin, se_f0 = ext[year]
        alpha_flat = f - f0_flat
        beta_lin = (f - f0_lin) / (1 - f0_lin)
        # delta method on beta = (f - f0)/(1 - f0)
        d_df = 1.0 / (1 - f0_lin)
        d_df0 = (f - 1.0) / (1 - f0_lin) ** 2
        se_beta = np.sqrt((d_df * se_f) ** 2 + (d_df0 * se_f0) ** 2)
        rows.append(dict(year=year, n=n, f_obs=round(f, 5),
                         f0_flat=round(f0_flat, 5),
                         f0_linear=round(f0_lin, 5), se_f0=round(se_f0, 5),
                         alpha_floor=round(alpha_flat, 5),
                         beta_linear=round(beta_lin, 5),
                         se_beta=round(se_beta, 5)))
    results[name] = {"slope_per_year": round(float(coef[1]), 6),
                     "rates_by_year": {int(k): round(float(v), 5)
                                       for k, v in rates["mean"].items()},
                     "estimates": rows}
    print(f"\n=== {name} ===")
    print(pd.DataFrame(rows).to_string(index=False))


def main():
    df = pd.read_parquet(os.path.join(DATA, "abstract_features.parquet"))
    results = {"cohort_n": int(len(df))}
    analyse(df, ["w_" + w for w in MARKERS], "markers_38", results)
    analyse(df, ["w_" + w for w in MARKERS_STRONG], "markers_strong_18", results)
    analyse(df, ["w_" + w for w in CONTROL], "control", results)
    analyse(df, ["w_" + w for w in PLACEBO], "placebo", results)

    with open(os.path.join(DATA, "aggregate_reproduction.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\nwrote data/aggregate_reproduction.json")


if __name__ == "__main__":
    main()
