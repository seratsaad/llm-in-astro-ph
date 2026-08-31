#!/usr/bin/env python3
"""Stage 3: fit the hierarchical model.

  python s3_fit.py --phase abstracts --variant primary [--smoke]

Variants differ only in the assumption being stressed; the data and the
likelihood are identical.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import pymc as pm

sys.path.insert(0, os.path.dirname(__file__))
from common import DATA, LOGS, MARKERS, MARKERS_STRONG, CONTROL, quarter_label
import model as M

VARIANTS = {
    # name              drift      monotone  free_from  basket         disclosure
    "primary":         dict(drift="linear",  monotone=True,  basket="markers_38"),
    "unconstrained":   dict(drift="linear",  monotone=False, basket="markers_38"),
    "frozen_drift":    dict(drift="frozen",  monotone=True,  basket="markers_38"),
    "tracked_drift":   dict(drift="tracked", monotone=True,  basket="markers_38"),
    "strong_basket":   dict(drift="linear",  monotone=True,  basket="markers_strong_18"),
    "control_placebo": dict(drift="linear",  monotone=True,  basket="control"),
    "late_boundary":   dict(drift="linear",  monotone=True,  basket="markers_38",
                            free_from=M.ADOPTION),
    "no_disclosure":   dict(drift="linear",  monotone=True,  basket="markers_38",
                            use_disclosure=False),
    "gamma_low":       dict(drift="linear",  monotone=True,  basket="markers_38",
                            gamma=np.log(0.5)),
    "gamma_high":      dict(drift="linear",  monotone=True,  basket="markers_38",
                            gamma=np.log(2.0)),
    "expanded":        dict(drift="linear",  monotone=True,  basket="expanded"),
    "expanded_tracked": dict(drift="tracked", monotone=True,  basket="expanded"),
    "control_tracked": dict(drift="tracked", monotone=True,  basket="control"),
}

BASKETS = {"markers_38": MARKERS, "markers_strong_18": MARKERS_STRONG,
           "control": CONTROL}


SECTIONS = ["intro", "methods", "results", "discussion", "conclusions"]


def load(phase, basket, section=None):
    if phase == "abstracts":
        df = pd.read_parquet(os.path.join(DATA, "abstract_features.parquet"))
        decl_path = os.path.join(DATA, "declarations.csv")
        if os.path.exists(decl_path):
            d = pd.read_csv(decl_path, dtype={"arxiv_id": str})
            df = df.merge(d[["arxiv_id", "declared"]], on="arxiv_id", how="left")
            df["declared"] = df["declared"].fillna(0).astype(int)
        else:
            df["declared"] = 0
    else:
        df = pd.read_parquet(os.path.join(DATA, "fulltext_features.parquet"))
        topics = pd.read_parquet(os.path.join(DATA, "abstract_features.parquet"),
                                 columns=["arxiv_id", "topic"])
        df = df.merge(topics, on="arxiv_id", how="left")
        df["topic"] = df["topic"].fillna(-1).astype(int)
    df = df[df.topic >= 0].reset_index(drop=True)
    if phase == "fulltext":
        # Restrict to the five recognised sections. The unlabelled "other"
        # bucket shrinks from 50% to 39% of the body over the decade, so
        # including it would mix a composition shift into the drift term.
        secs = SECTIONS if section is None else [section]
        if basket == "expanded":
            exp = pd.read_parquet(os.path.join(DATA, "expanded_features.parquet"))
            df = df.merge(exp, on="arxiv_id", how="inner")
            pre = "E"
            df["L"] = df[[f"L_{s}" for s in secs]].sum(axis=1)
            K = df[[f"{pre}_{s}" for s in secs]].sum(axis=1).values
            keep = df.L.values >= 300
            return df[keep].reset_index(drop=True), K[keep]
        pre = {"markers_38": "K", "markers_strong_18": "K",
               "control": "C"}[basket]
        if basket == "markers_strong_18":
            raise SystemExit("strong-basket variant is abstract-only "
                             "(per-section counts cover the full 38 only)")
        df["L"] = df[[f"L_{s}" for s in secs]].sum(axis=1)
        K = df[[f"{pre}_{s}" for s in secs]].sum(axis=1).values
        keep = df.L.values >= 300
        df, K = df[keep].reset_index(drop=True), K[keep]
        return df, K
    K = df[["w_" + w for w in BASKETS[basket]]].sum(axis=1).values
    return df, K


def neutral_drift_offset(df, n_q, free_from, phase="abstracts", section=None):
    """Per-quarter log drift of the neutral control basket, for 'tracked'."""
    if phase == "fulltext":
        secs = SECTIONS if section is None else [section]
        Kc = df[[f"C_{s}" for s in secs]].sum(axis=1).values
    else:
        Kc = df[["w_" + w for w in CONTROL]].sum(axis=1).values
    rate = np.zeros(n_q)
    base = None
    for q in range(n_q):
        m = df.q.values == q
        rate[q] = Kc[m].sum() / df.L.values[m].sum() if m.sum() else np.nan
    ref = np.nanmean(rate[max(0, free_from - 8):free_from])
    return np.log(np.where(np.isnan(rate), ref, rate) / ref)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="abstracts", choices=["abstracts", "fulltext"])
    ap.add_argument("--variant", default="primary")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--tune", type=int, default=1000)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--section", default=None, choices=SECTIONS)
    ap.add_argument("--nuts", default="pymc", choices=["pymc", "numpyro"],
                    help="numpyro uses the JAX NUTS sampler (much faster)")
    args = ap.parse_args()

    spec = dict(VARIANTS[args.variant])
    basket = spec.pop("basket")
    free_from = spec.pop("free_from", M.FREE_FROM)
    df, K = load(args.phase, basket, section=args.section)

    n_q = int(df.q.max()) + 1
    if args.smoke:
        df = df.sample(25000, random_state=0)
        K = K[df.index.values]
        df = df.reset_index(drop=True)

    data = {"K": K, "L": df.L.values, "t": df.q.values, "c": df.topic.values,
            "declared": df.get("declared", pd.Series(np.zeros(len(df)))).values,
            "n_quarters": n_q, "n_topics": int(df.topic.max()) + 1,
            "drift_offset": neutral_drift_offset(df, n_q, free_from,
                                                phase=args.phase,
                                                section=args.section)}

    print(f"phase={args.phase} variant={args.variant} basket={basket} "
          f"n={len(df)} free_from={quarter_label(free_from)}", flush=True)

    m = M.build(data, free_from=free_from, **spec)
    t0 = time.time()
    kw = dict(draws=200 if args.smoke else args.draws,
              tune=200 if args.smoke else args.tune,
              chains=2 if args.smoke else args.chains,
              cores=min(4, args.chains),
              target_accept=0.9, random_seed=20260826,
              progressbar=True)
    if args.nuts == "numpyro":
        kw["nuts_sampler"] = "numpyro"
    with m:
        idata = pm.sample(**kw)
    print(f"sampling took {time.time()-t0:.0f}s", flush=True)

    tag = f"{args.phase}_{args.variant}"
    if args.section:
        tag += f"_{args.section}"
    tag += "_smoke" if args.smoke else ""
    out = os.path.join(DATA, f"idata_{tag}.nc")
    idata.to_netcdf(out)

    summ = pm.summary(idata, var_names=["beta0", "g0_slope", "g0_curve", "phi0",
                                        "phi1", "m0", "sd_m", "eta0", "sd_eta"])
    print(summ.to_string())
    bad = summ[(summ.r_hat > 1.01) | (summ.ess_bulk < 400)]
    print(f"\nconvergence failures: {len(bad)}")
    if len(bad):
        print(bad.to_string())

    pi = idata.posterior["pi"].values.reshape(-1, data["n_quarters"] - free_from)
    qs = [quarter_label(free_from + i) for i in range(pi.shape[1])]
    tab = pd.DataFrame({"quarter": qs, "mean": pi.mean(0),
                        "lo": np.percentile(pi, 2.5, axis=0),
                        "hi": np.percentile(pi, 97.5, axis=0)})
    print("\nposterior prevalence by quarter:")
    print(tab.round(4).to_string(index=False))
    tab.to_csv(os.path.join(DATA, f"pi_{tag}.csv"), index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
