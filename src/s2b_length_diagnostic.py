#!/usr/bin/env python3
"""Stage 2b: is the document-frequency estimator confounded by document length?

Stage 2 found that the linear-extrapolation estimator returns a large positive
"prevalence" for neutral control vocabulary. If that is driven by abstracts
getting longer, then truncating every abstract to a common length must remove
it. This runs the same estimator three ways:

  full        every token of the abstract        (as published)
  truncated   the first TRUNC tokens only        (length held fixed)
  per-token   occurrences per 1000 tokens        (length normalised)

Writes data/length_diagnostic.json.
"""
import gzip
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from common import (KG, DATA, MARKERS, MARKERS_STRONG, CONTROL, PLACEBO,
                    TOKEN_RE, YEAR_MIN, id_to_ym, quarter_index, QUARTER_MAX)
from s2_aggregate import linear_extrapolate, FIT_YEARS

TRUNC = 150


def main():
    baskets = {"markers_38": set(MARKERS), "markers_strong_18": set(MARKERS_STRONG),
               "control": set(CONTROL), "placebo": set(PLACEBO)}
    qmax = quarter_index(*QUARTER_MAX)
    recs = []
    with gzip.open(os.path.join(KG, "abstracts_all.jsonl.gz"), "rt") as f:
        for line in f:
            r = json.loads(line)
            ym = id_to_ym(r["arxiv_id"])
            if ym is None:
                continue
            year, month = ym
            if year < YEAR_MIN:
                continue
            qi = quarter_index(year, month)
            if qi < 0 or qi > qmax:
                continue
            toks = TOKEN_RE.findall((r.get("abstract") or "").lower())
            if len(toks) < TRUNC:            # need a common length for all rows
                continue
            head = toks[:TRUNC]
            row = {"year": year, "L": len(toks)}
            for name, bs in baskets.items():
                row[name + "_full"] = int(any(t in bs for t in toks))
                row[name + "_trunc"] = int(any(t in bs for t in head))
                row[name + "_occ"] = sum(1 for t in toks if t in bs)
            recs.append(row)
    df = pd.DataFrame(recs)
    print(f"n = {len(df)} abstracts with >= {TRUNC} tokens")

    out = {"n": int(len(df)), "trunc": TRUNC,
           "median_length_by_year": df.groupby("year").L.median().to_dict()}
    for name in baskets:
        block = {}
        for mode in ("full", "trunc"):
            rates = df.groupby("year")[f"{name}_{mode}"].agg(["mean", "size"])
            ext, coef = linear_extrapolate(rates, FIT_YEARS, [2024, 2025])
            est = {}
            for y in (2024, 2025):
                f = float(rates["mean"].loc[y])
                f0 = ext[y][0]
                est[y] = round((f - f0) / (1 - f0), 5)
            block[mode] = {"rates": {int(k): round(float(v), 5)
                                     for k, v in rates["mean"].items()},
                           "beta": est}
        # length-normalised occurrence rate per 1000 tokens
        occ = df.groupby("year").apply(
            lambda g: 1000 * g[f"{name}_occ"].sum() / g.L.sum(), include_groups=False)
        block["per_1000_tokens"] = {int(k): round(float(v), 4) for k, v in occ.items()}
        out[name] = block

    with open(os.path.join(DATA, "length_diagnostic.json"), "w") as f:
        json.dump(out, f, indent=2)

    print("\nbeta estimator, full-length vs length-truncated abstracts")
    print(f"{'basket':20s} {'2024 full':>10s} {'2024 trunc':>11s} "
          f"{'2025 full':>10s} {'2025 trunc':>11s}")
    for name in baskets:
        b = out[name]
        print(f"{name:20s} {b['full']['beta'][2024]:10.4f} "
              f"{b['trunc']['beta'][2024]:11.4f} {b['full']['beta'][2025]:10.4f} "
              f"{b['trunc']['beta'][2025]:11.4f}")
    print("\noccurrences per 1000 tokens (length normalised)")
    for name in baskets:
        s = out[name]["per_1000_tokens"]
        print(f"{name:20s} " + " ".join(f"{s[y]:6.3f}" for y in sorted(s)))


if __name__ == "__main__":
    main()
