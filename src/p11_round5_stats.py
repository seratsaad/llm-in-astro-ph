#!/usr/bin/env python3
"""
P11 -- round-5 statistics.

M3: prior sensitivity of alpha (flat, Jeffreys, profile likelihood).
M4: peak-timing test with the binning convention explicit, per-word peaks,
    and the stratified permutation, on the June-2026 series.
M5: pseudo-naming placebo with a permutation p-value and threshold
    sensitivity.
M6: omnibus heterogeneity (Cochran's Q) for the subfield decay spread.
Output: data/p11_round5.json
"""
import json, os, re, collections
from math import comb
import numpy as np
import importlib.util

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data")
spec = importlib.util.spec_from_file_location("n2b", os.path.join(HERE, "n2b_publicity.py"))
n2b = importlib.util.module_from_spec(spec); spec.loader.exec_module(n2b)

NAMED_Q = 2024.25          # publicity quarter, bins labelled by start month
out = {}


def series():
    qn = collections.Counter()
    wq = {w: collections.Counter() for w in n2b.WORDS}
    pre = collections.Counter(); npre = 0
    for line in open(os.path.join(DATA, "astroph_abstracts.jsonl")):
        r = json.loads(line)
        y = int(r["published"][:4]); m = int(r["published"][5:7])
        if y < 2015 or y > 2026 or (y == 2026 and m > 6):
            continue
        toks = set(re.findall(r"[a-z]+", r["abstract"].lower()))
        if y < 2022:
            npre += 1
            for w in n2b.WORDS:
                if w in toks: pre[w] += 1
        qk = y + ((m - 1) // 3) * 0.25
        qn[qk] += 1
        for w in n2b.WORDS:
            if w in toks: wq[w][qk] += 1
    return qn, wq, pre, npre


def main():
    qn, wq, pre, npre = series()
    quarters = sorted(qn)
    per = json.load(open(os.path.join(DATA, "n2f_publicity_sources.json")))["per_word"]

    exc_all = {}
    for w in n2b.WORDS:
        ser = {q: wq[w][q] / qn[q] for q in quarters}
        p = [(q, ser[q]) for q in quarters if q < 2022]
        coef = np.polyfit([x[0] for x in p], [x[1] for x in p], 1)
        exc_all[w] = {q: (ser[q] - np.polyval(coef, q)) * 100
                      for q in quarters if q >= 2022}

    # ---- M4: peaks with the convention stated -------------------------------
    rows = []
    for w in n2b.WORDS:
        exc = exc_all[w]
        pk = max((q for q in exc if q <= 2025.75), key=lambda q: exc[q])
        if exc[pk] < 0.05:
            continue
        rows.append({"word": w, "named": len(per.get(w, [])) > 0,
                     "lf": np.log10(max(pre[w] / npre * 100, 1e-4)),
                     "peak_q": pk})
    named = [r for r in rows if r["named"]]
    lab = [r["named"] for r in rows]
    out["M4_convention"] = ("bins labelled by start month, so 2024.25 is "
                            "April to June 2024, the publicity quarter, and a "
                            "peak at 2024.00 means the decline began inside it")
    out["M4_named_peaks"] = {r["word"]: r["peak_q"] for r in named}

    def stat(labels, win):
        a = [r for r, L in zip(rows, labels) if L]
        return sum(1 for r in a if win[0] <= r["peak_q"] <= win[1]) / len(a)

    rng = np.random.default_rng(3)
    order = np.argsort([r["lf"] for r in rows])
    strata = np.array_split(order, 4)
    for wlab, win in (("exact_publicity_quarter", (2024.25, 2024.25)),
                      ("publicity_or_quarter_before", (2024.00, 2024.25))):
        obs = stat(lab, win)
        ge1 = ge2 = 0
        N = 20000
        for _ in range(N):
            L = rng.permutation(lab)
            if stat(L, win) >= obs - 1e-12: ge1 += 1
            L2 = np.array(lab).copy()
            for s in strata:
                L2[s] = rng.permutation(L2[s])
            if stat(L2, win) >= obs - 1e-12: ge2 += 1
        k = int(round(obs * len(named)))
        un = [r for r in rows if not r["named"]]
        ku = sum(1 for r in un if win[0] <= r["peak_q"] <= win[1])
        out[f"M4_{wlab}"] = {"named": [k, len(named)], "unnamed": [ku, len(un)],
                             "p_perm": ge1 / N, "p_perm_freq_stratified": ge2 / N}
        print(f"M4 {wlab}: named {k}/{len(named)}, unnamed {ku}/{len(un)}, "
              f"p={ge1/N:.3f}, stratified p={ge2/N:.3f}")

    # ---- M5: placebo with a test statistic and threshold sensitivity --------
    named_at = {w: exc_all[w][NAMED_Q] for w, q in
                {"delve": 1, "delves": 1, "delving": 1, "intricate": 1,
                 "pivotal": 1, "showcasing": 1, "realm": 1, "underscores": 1,
                 "intricacies": 1, "meticulously": 1}.items() if w in exc_all}

    def chg2q(w, q0):
        later = [q for q in sorted(exc_all[w]) if q0 < q <= q0 + 0.5]
        if not later or exc_all[w][q0] <= 0:
            return None
        return (exc_all[w][max(later)] - exc_all[w][q0]) / exc_all[w][q0] * 100

    named_points = {w: chg2q(w, NAMED_Q) for w in named_at
                    if chg2q(w, NAMED_Q) is not None}
    named_ch = list(named_points.values())
    sens = {}
    unnamed_points_headline = {}
    for thr in (0.10, 0.15, 0.19, 0.25):
        un_ch = []
        un_pts = {}
        for r in rows:
            if r["named"]:
                continue
            w = r["word"]
            crossed = [q for q in sorted(exc_all[w])
                       if exc_all[w][q] >= thr / 100 * 100 and q <= 2025.0]
            crossed = [q for q in sorted(exc_all[w]) if exc_all[w][q] >= thr and q <= 2025.0]
            if not crossed:
                continue
            c = chg2q(w, crossed[0])
            if c is not None:
                un_ch.append(c); un_pts[w] = c
        if thr == 0.19:
            unnamed_points_headline = un_pts
        # permutation test on group means
        allv = named_ch + un_ch
        nn = len(named_ch)
        obs = np.mean(named_ch) - np.mean(un_ch)
        ge = tot = 0
        rng2 = np.random.default_rng(11)
        for _ in range(20000):
            pm = rng2.permutation(allv)
            if np.mean(pm[:nn]) - np.mean(pm[nn:]) <= obs + 1e-12:
                ge += 1
        sens[str(thr)] = {"named_mean": float(np.mean(named_ch)),
                          "unnamed_mean": float(np.mean(un_ch)),
                          "n_unnamed": len(un_ch), "p_perm_one_sided": ge / 20000}
        print(f"M5 thr {thr:.2f}pp: named {np.mean(named_ch):+.0f}% (n={nn}), "
              f"crossers {np.mean(un_ch):+.0f}% (n={len(un_ch)}), p={ge/20000:.4f}")
    out["M5_placebo"] = sens
    out["M5_placebo_points"] = {"named": named_points,
                                "unnamed": unnamed_points_headline,
                                "threshold": 0.19}

    # ---- M6: omnibus heterogeneity for the subfield decay -------------------
    boot = json.load(open(os.path.join(DATA, "m5_subfield_boot.json")))
    d = np.array([boot[g]["decay_pct"] for g in boot])
    se = np.array([(boot[g]["ci68"][1] - boot[g]["ci68"][0]) / 2 for g in boot])
    wgt = 1 / se ** 2
    dbar = np.sum(wgt * d) / np.sum(wgt)
    Q = float(np.sum(wgt * (d - dbar) ** 2))
    dof = len(d) - 1
    from math import lgamma
    # chi2 survival by series is overkill; use numpy
    try:
        from scipy.stats import chi2
        pQ = float(chi2.sf(Q, dof))
    except ImportError:
        # Wilson-Hilferty approximation
        z = ((Q / dof) ** (1 / 3) - (1 - 2 / (9 * dof))) / np.sqrt(2 / (9 * dof))
        from math import erfc, sqrt
        pQ = erfc(z / sqrt(2)) / 2
    out["M6_cochran_Q"] = {"Q": Q, "dof": dof, "p": pQ,
                           "decays": {g: boot[g]["decay_pct"] for g in boot}}
    print(f"M6 Cochran Q = {Q:.1f} (dof {dof}), p = {pQ:.3g}")

    json.dump(out, open(os.path.join(DATA, "p11_round5.json"), "w"), indent=1,
              default=float)


if __name__ == "__main__":
    main()
