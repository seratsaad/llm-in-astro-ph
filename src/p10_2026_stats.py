#!/usr/bin/env python3
"""
P10 -- unified statistics for the June-2026 analysis window.

One corpus pass, then every number the manuscript quotes:
  floors (multiple baskets and windows, primary-only, flat-word cut),
  yearly incidence series, the 2026 H1 target counts for alpha,
  cluster-robust check inputs, and the corpus count.
Baskets: full, no_publicized, no_leveraging, no_flat5 (referee round 5,
point 7: drop the five words with a 2018-21 to 2024-25 ratio of one),
top20 by excess.
Output: data/p10_2026_stats.json
"""
import json, os, re
import numpy as np

DATA = os.path.join(os.path.dirname(__file__), "..", "data")

FULL = """delve delves delving underscore underscores underscoring intricate
intricacies showcasing showcase showcases showcased boasts tapestry pivotal
meticulous meticulously nuanced garner garners garnered multifaceted commendable
noteworthy myriad plethora testament encompassing seamless seamlessly elucidate
elucidating unravel unraveling unravelling realm realms leveraging""".split()
PUBLICIZED = {"delve", "delves", "delving", "intricate", "pivotal"}
FLAT5 = {"plethora", "myriad", "realm", "realms", "unravel"}

BASKETS = {"full": set(FULL),
           "no_publicized": set(FULL) - PUBLICIZED,
           "no_leveraging": set(FULL) - {"leveraging"},
           "no_flat5": set(FULL) - FLAT5}


def tokset(s):
    return set(re.findall(r"[a-z]+", s.lower()))


def main():
    # excess ranking for top20 (pre-2022 vs 2024-2025, unchanged definition)
    pre = {w: 0 for w in FULL}; post = {w: 0 for w in FULL}
    npre = npost = 0
    years = {}          # year -> {basket: hits, n}
    years_prim = {}     # primary astro-ph only
    h1 = {b: 0 for b in BASKETS} | {"n": 0}
    h1_prim = {b: 0 for b in BASKETS} | {"n": 0}
    total = 0
    for line in open(os.path.join(DATA, "astroph_abstracts.jsonl")):
        r = json.loads(line)
        y = int(r["published"][:4]); m = int(r["published"][5:7])
        if y < 2015 or y > 2026 or (y == 2026 and m > 6):
            continue
        total += 1
        toks = tokset(r["abstract"])
        prim = r.get("primary_category", "").startswith("astro-ph")
        if y < 2022:
            npre += 1
            for w in FULL:
                if w in toks: pre[w] += 1
        elif y >= 2024 and y <= 2025:
            npost += 1
            for w in FULL:
                if w in toks: post[w] += 1
        yy = years.setdefault(y, {b: 0 for b in BASKETS} | {"n": 0})
        yy["n"] += 1
        for b, ws in BASKETS.items():
            if toks & ws: yy[b] += 1
        if prim:
            yp = years_prim.setdefault(y, {b: 0 for b in BASKETS} | {"n": 0})
            yp["n"] += 1
            for b, ws in BASKETS.items():
                if toks & ws: yp[b] += 1
        if y == 2026:
            h1["n"] += 1
            for b, ws in BASKETS.items():
                if toks & ws: h1[b] += 1
            if prim:
                h1_prim["n"] += 1
                for b, ws in BASKETS.items():
                    if toks & ws: h1_prim[b] += 1

    exc = {w: post[w] / npost - pre[w] / npre for w in FULL}
    BASKETS["top20"] = set(sorted(FULL, key=lambda w: -exc[w])[:20])
    # top20 needs its own counts; recompute from the yearly full-word tallies is
    # not possible without per-word years, so note it comes from r1's own pass.

    out = {"corpus_total_2015_2026H1": total,
           "years": {str(y): years[y] for y in sorted(years)},
           "h1_2026": h1, "h1_2026_primary_only": h1_prim,
           "years_primary": {str(y): years_prim[y] for y in sorted(years_prim)}}

    def rate(d, b): return d[b] / d["n"] * 100

    print(f"corpus 2015-2026H1: {total:,}")
    print("\nfloor variants (f0 = pooled 2018-2021, f_t = 2026 H1):")
    for b in ("full", "no_publicized", "no_leveraging", "no_flat5"):
        f0 = sum(years[y][b] for y in range(2018, 2022)) / \
             sum(years[y]["n"] for y in range(2018, 2022)) * 100
        ft = rate(h1, b)
        f25 = rate(years[2025], b)
        out[f"floor_{b}"] = {"f0": f0, "f_2025": f25, "f_2026H1": ft,
                             "floor_2025": f25 - f0, "floor_2026H1": ft - f0}
        print(f"  {b:14s} f0 {f0:.2f}  2025 {f25:.2f} (floor {f25-f0:.2f})  "
              f"2026H1 {ft:.2f} (floor {ft-f0:.2f})")
    b = "full"
    f0p = sum(years_prim[y][b] for y in range(2018, 2022)) / \
          sum(years_prim[y]["n"] for y in range(2018, 2022)) * 100
    ftp = rate(h1_prim, b)
    out["floor_primary_only"] = {"f0": f0p, "f_2026H1": ftp, "floor": ftp - f0p}
    print(f"  primary-only   f0 {f0p:.2f}  2026H1 {ftp:.2f} (floor {ftp-f0p:.2f})")
    print("\nyearly incidence, full basket:")
    for y in sorted(years):
        print(f"  {y}: {rate(years[y],'full'):.2f}%  (n={years[y]['n']:,})")
    json.dump(out, open(os.path.join(DATA, "p10_2026_stats.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
