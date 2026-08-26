#!/usr/bin/env python3
"""
FT5 -- the word-level avoidance tests, run on the BODY of the papers.

From the ft4 word-by-quarter matrix:
  (1) per-word quarterly excess over that word's own pre-2022 trend
  (2) peak-timing test against the April-June 2024 publicity quarter,
      permuting the named/unnamed labels, plain and frequency-stratified
  (3) the pseudo-naming placebo: each unnamed word is assigned the first
      quarter its excess reached the qualification level, and its change over
      the following two quarters is compared with the named words' change
      after naming
Output: data/ft_word_tests.json
"""
import json, os, collections
import numpy as np

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
NAMED = set("""delve delves delving intricate pivotal showcasing realm
underscores intricacies meticulously""".split())
PUBQ = 2024.25


def main():
    d = json.load(open(os.path.join(DATA, "ft_word_quarters.json")))
    counts, words_tot = d["counts"], d["body_words"]
    basket = d["basket"]

    def qval(k):
        if "." in k:
            y, q = k.split("."); return int(y) + int(q) / 100
        return int(k)

    quarters = sorted((k for k in counts if "." in k), key=qval)
    years = sorted((k for k in counts if "." not in k), key=int)

    # per-word density series (tokens per 10k body words)
    dens = {}
    for w in basket:
        s = {}
        for k in years + quarters:
            s[qval(k)] = counts[k].get(w, 0) / words_tot[k] * 1e4
        dens[w] = s

    # pre-2022 linear trend per word (2012..2021 yearly+quarterly points)
    exc = {}
    pre_freq = {}
    for w in basket:
        pts = [(x, v) for x, v in dens[w].items() if x < 2022]
        coef = np.polyfit([p[0] for p in pts], [p[1] for p in pts], 1)
        pre_freq[w] = np.mean([v for x, v in pts if x >= 2018])
        exc[w] = {x: dens[w][x] - np.polyval(coef, x)
                  for x in dens[w] if x >= 2022 and "." in str(x) or x >= 2022}
        exc[w] = {qval(k): dens[w][qval(k)] - np.polyval(coef, qval(k))
                  for k in quarters if qval(k) >= 2022}

    # ---- reliability floor: words with enough excess to time ----
    rows = []
    for w in basket:
        e = exc[w]
        pk = max((q for q in e if q <= 2025.75), key=lambda q: e[q])
        if e[pk] < 0.008:            # ~ tokens/10k floor for quarterly timing
            continue
        rows.append({"w": w, "named": w in NAMED, "peak": pk,
                     "lf": np.log10(max(pre_freq[w], 1e-5))})
    named = [r for r in rows if r["named"]]
    unnamed = [r for r in rows if not r["named"]]
    print(f"timeable words: {len(rows)} ({len(named)} named, {len(unnamed)} unnamed)")

    out = {"timeable": {r["w"]: r["peak"] for r in rows}}

    # ---- (2) peak timing permutation ----
    lab = [r["named"] for r in rows]
    rng = np.random.default_rng(5)
    order = np.argsort([r["lf"] for r in rows])
    strata = np.array_split(order, 4)

    def stat(labels, win):
        a = [r for r, L in zip(rows, labels) if L]
        return sum(1 for r in a if win[0] <= r["peak"] <= win[1]) / max(len(a), 1)

    for wlab, win in (("exact_quarter", (PUBQ, PUBQ)),
                      ("quarter_or_before", (PUBQ - 0.25, PUBQ))):
        obs = stat(lab, win)
        ge1 = ge2 = 0; N = 20000
        for _ in range(N):
            if stat(rng.permutation(lab), win) >= obs - 1e-12: ge1 += 1
            L2 = np.array(lab).copy()
            for s in strata: L2[s] = rng.permutation(L2[s])
            if stat(L2, win) >= obs - 1e-12: ge2 += 1
        k = int(round(obs * len(named)))
        ku = sum(1 for r in unnamed if win[0] <= r["peak"] <= win[1])
        out[f"timing_{wlab}"] = {"named": [k, len(named)],
                                 "unnamed": [ku, len(unnamed)],
                                 "p": ge1 / N, "p_strat": ge2 / N}
        print(f"timing {wlab}: named {k}/{len(named)} vs unnamed {ku}/{len(unnamed)} "
              f"p={ge1/N:.3f} strat={ge2/N:.3f}")

    # ---- (3) pseudo-naming placebo ----
    def chg2q(w, q0):
        later = [q for q in sorted(exc[w]) if q0 < q <= q0 + 0.5]
        if not later or exc[w][q0] <= 0: return None
        return (exc[w][max(later)] - exc[w][q0]) / exc[w][q0] * 100

    named_ch = {r["w"]: chg2q(r["w"], PUBQ) for r in named}
    named_ch = {w: c for w, c in named_ch.items() if c is not None}
    qual = min(exc[w][PUBQ] for w in named_ch)      # smallest named excess at naming
    sens = {}
    for thr in (qual * 0.5, qual, qual * 1.5):
        un_ch = {}
        for r in unnamed:
            w = r["w"]
            crossed = [q for q in sorted(exc[w]) if exc[w][q] >= thr and q <= 2025.0]
            if not crossed: continue
            c = chg2q(w, crossed[0])
            if c is not None: un_ch[w] = c
        a = list(named_ch.values()); b = list(un_ch.values())
        if not b: continue
        obs = np.mean(a) - np.mean(b)
        allv = a + b; nn = len(a); ge = 0
        for _ in range(20000):
            pm = rng.permutation(allv)
            if np.mean(pm[:nn]) - np.mean(pm[nn:]) <= obs + 1e-12: ge += 1
        sens[f"{thr:.4f}"] = {"named_mean": np.mean(a), "unnamed_mean": np.mean(b),
                              "n_unnamed": len(b), "p": ge / 20000,
                              "unnamed_words": un_ch}
        print(f"placebo thr {thr:.4f}: named {np.mean(a):+.0f}% (n={nn}) vs "
              f"crossers {np.mean(b):+.0f}% (n={len(b)}) p={ge/20000:.4f}")
    out["placebo"] = {"named_changes": named_ch, "qual_threshold": qual,
                      "sensitivity": sens}

    json.dump(out, open(os.path.join(DATA, "ft_word_tests.json"), "w"),
              indent=1, default=float)
    print("saved data/ft_word_tests.json")


if __name__ == "__main__":
    main()
