#!/usr/bin/env python3
"""
Local referee checks: R2 (q sensitivity), R4 (detrended hype correlation),
R5 (within-subfield permutation null for co-occurrence).
Writes data/referee_checks.json.
"""
import json, os, re, collections, math
import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__); DATA = os.path.join(HERE, "..", "data")

# ----------------------------------------------------------------- R2
def q_sensitivity():
    al = json.load(open(os.path.join(DATA, "alpha_summary.json")))
    f0 = al["base_rate"]
    ft = al["by_year"]["2025"]["rate"]
    rows = []
    for q in [1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2]:
        rows.append({"q": q, "alpha": (ft - f0) / (q - f0)})
    return {"f0": f0, "ft": ft, "curve": rows}

# ----------------------------------------------------------------- R4
def detrended_hype():
    d = pd.read_csv(os.path.join(DATA, "c7_hype_hedge.csv")).sort_values("year")
    al = json.load(open(os.path.join(DATA, "alpha_summary.json")))
    d["basket"] = d.year.map({int(y): v["rate"] * 100 for y, v in al["by_year"].items()})
    d = d.dropna(subset=["basket"])
    def pear(a, b):
        a = np.asarray(a, float); b = np.asarray(b, float)
        return float(np.corrcoef(a, b)[0, 1])
    # levels
    r_lvl = pear(d.hype_per1k, d.basket)
    rh_lvl = pear(d.hedge_per1k, d.basket)
    # first differences
    r_diff = pear(np.diff(d.hype_per1k), np.diff(d.basket))
    rh_diff = pear(np.diff(d.hedge_per1k), np.diff(d.basket))
    # linear-detrended residuals (remove year trend from each, then correlate)
    yr = d.year.values.astype(float)
    def resid(v):
        v = np.asarray(v, float); c = np.polyfit(yr, v, 1); return v - np.polyval(c, yr)
    r_det = pear(resid(d.hype_per1k), resid(d.basket))
    return {"n_years": int(len(d)), "hype_levels_r": r_lvl, "hedge_levels_r": rh_lvl,
            "hype_firstdiff_r": r_diff, "hedge_firstdiff_r": rh_diff,
            "hype_detrended_r": r_det}

# ----------------------------------------------------------------- R5
MARKERS = ["delve", "delves", "underscore", "underscores", "intricate", "pivotal",
           "showcasing", "leveraging", "meticulous", "nuanced", "realm", "tapestry",
           "garner", "multifaceted"]

def pois_tail(x, mu):
    if x == 0: return 1.0
    s, term = 0.0, math.exp(-mu)
    for k in range(x):
        s += term; term *= mu / (k + 1)
    return max(0.0, 1 - s)

def subfield_permutation(n_perm=300, seed=0):
    mset = {m: i for i, m in enumerate(MARKERS)}
    rows_sub = []
    present = []
    for line in open(os.path.join(DATA, "astroph_abstracts.jsonl")):
        r = json.loads(line)
        y = int(r["published"][:4])
        if y not in (2024, 2025, 2026):
            continue
        toks = set(re.findall(r"[a-z]+", r["abstract"].lower()))
        vec = np.zeros(len(MARKERS), bool)
        for m in (toks & mset.keys()):
            vec[mset[m]] = True
        present.append(vec)
        rows_sub.append(r["primary_category"])
    X = np.array(present)                       # (N, 14) bool
    N = len(X)
    sub = np.array(rows_sub)
    single = X.sum(0)                           # marker doc counts
    # observed significant pairs (global-independence Poisson test, as in the paper)
    def count_sig(M):
        pair = M.T.astype(int) @ M.astype(int)  # (14,14) joint counts
        sig = 0
        for i in range(len(MARKERS)):
            for j in range(i + 1, len(MARKERS)):
                mu = N * (single[i] / N) * (single[j] / N)
                if pois_tail(int(pair[i, j]), mu) < 0.01:
                    sig += 1
        return sig
    obs = count_sig(X)
    # permutation: shuffle each marker column within subfield blocks (preserves per-subfield marginal,
    # destroys within-abstract co-occurrence beyond subfield structure)
    rng = np.random.default_rng(seed)
    blocks = [np.where(sub == s)[0] for s in np.unique(sub)]
    null = []
    for _ in range(n_perm):
        Xp = X.copy()
        for col in range(len(MARKERS)):
            for idx in blocks:
                Xp[idx, col] = rng.permutation(Xp[idx, col])
        null.append(count_sig(Xp))
    null = np.array(null)
    pval = float((null >= obs).mean())
    return {"N": int(N), "n_subfields": int(len(blocks)), "n_perm": n_perm,
            "observed_sig_pairs": int(obs), "null_mean": float(null.mean()),
            "null_p95": float(np.percentile(null, 95)), "perm_pvalue": pval}

def main():
    out = {"R2_q_sensitivity": q_sensitivity(),
           "R4_hype_detrend": detrended_hype(),
           "R5_subfield_permutation": subfield_permutation()}
    json.dump(out, open(os.path.join(DATA, "referee_checks.json"), "w"), indent=2)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
