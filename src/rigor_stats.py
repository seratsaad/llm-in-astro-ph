#!/usr/bin/env python3
"""
Uncertainty and robustness numbers for the paper. Everything the text asserts
gets computed here and written to data/rigor.json:

  - Wilson 95% intervals on the yearly marker-basket rate
  - two-proportion z test, 2025 vs the 2018-2021 baseline
  - alpha_lb under alternative baseline windows
  - leave-one-word-out range of alpha_lb (needs a corpus pass)
  - control-word excess (should be ~0)
  - Pearson r and p for the hype-vs-basket correlation
  - Wilson interval on the source-leak rate
  - Wilson intervals on the per-country marker rates (for figure error bars)
"""
import json, os, re, math, collections
import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__); DATA = os.path.join(HERE, "..", "data")
TOK = re.compile(r"[a-z]+")

BASKET = {  # keep in sync with alpha_estimate.py
    "delve", "delves", "delving", "underscore", "underscores", "underscoring",
    "intricate", "intricacies", "showcasing", "showcase", "showcases", "showcased",
    "boasts", "tapestry", "pivotal", "meticulous", "meticulously", "nuanced",
    "garner", "garners", "garnered", "multifaceted", "commendable", "noteworthy",
    "myriad", "plethora", "testament", "encompassing", "seamless", "seamlessly",
    "elucidate", "elucidating", "unravel", "unraveling", "unravelling",
    "realm", "realms", "leveraging",
}
CONTROL = ["observed", "measured", "obtained", "presented", "galaxy", "stellar",
           "sample", "temperature", "redshift", "spectra"]
BASE_YEARS = (2018, 2019, 2020, 2021)
REC_YEARS = (2024, 2025, 2026)

def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k / n
    d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (c - h, c + h)

def main():
    out = {}
    al = json.load(open(os.path.join(DATA, "alpha_summary.json")))
    by = {int(y): v for y, v in al["by_year"].items()}

    # --- yearly rates with Wilson intervals + z test ---
    yearly = {}
    for y, v in sorted(by.items()):
        k = v["hit"]; n = v["n"]
        lo, hi = wilson(k, n)
        yearly[y] = {"rate": v["rate"], "n": n, "lo": lo, "hi": hi}
    out["yearly"] = yearly
    k0 = sum(by[y]["hit"] for y in BASE_YEARS); n0 = sum(by[y]["n"] for y in BASE_YEARS)
    k1 = by[2025]["hit"]; n1 = by[2025]["n"]
    p0, p1 = k0/n0, k1/n1
    pp = (k0 + k1) / (n0 + n1)
    z = (p1 - p0) / math.sqrt(pp*(1-pp)*(1/n0 + 1/n1))
    out["ztest_2025_vs_base"] = {"p0": p0, "p1": p1, "z": z}

    # --- alpha under alternative baseline windows ---
    def alpha_for(base_years):
        kb = sum(by[y]["hit"] for y in base_years); nb = sum(by[y]["n"] for y in base_years)
        kr = sum(by[y]["hit"] for y in REC_YEARS); nr = sum(by[y]["n"] for y in REC_YEARS)
        a = kr/nr - kb/nb
        # binomial error propagation
        se = math.sqrt((kr/nr)*(1-kr/nr)/nr + (kb/nb)*(1-kb/nb)/nb)
        return a, se
    out["alpha_windows"] = {}
    for w in [(2018,2019,2020,2021), (2017,2018,2019,2020), (2016,2017,2018,2019),
              (2015,2016,2017,2018)]:
        a, se = alpha_for(w)
        out["alpha_windows"]["-".join(map(str, (w[0], w[-1])))] = {"alpha": a, "se": se}
    a25 = by[2025]["rate"] - p0
    out["alpha_2025"] = {"alpha": a25,
                         "se": math.sqrt(p1*(1-p1)/n1 + p0*(1-p0)/n0)}

    # --- corpus pass: leave-one-out basket + control excess ---
    pat_base = collections.Counter(); pat_rec = collections.Counter()
    nb = nr = 0
    ctrl_base = collections.Counter(); ctrl_rec = collections.Counter()
    seen = set()
    blist = sorted(BASKET); bidx = {w: i for i, w in enumerate(blist)}
    for line in open(os.path.join(DATA, "astroph_abstracts.jsonl")):
        r = json.loads(line); b = r["id"].split("v")[0]
        if b in seen: continue
        seen.add(b)
        pub = r["published"]
        if not pub or len(pub) < 7: continue
        y = int(pub[:4])
        in_base = y in BASE_YEARS; in_rec = y in REC_YEARS
        if not (in_base or in_rec): continue
        toks = set(TOK.findall((r.get("abstract") or "").lower()))
        mask = 0
        for w in toks & BASKET:
            mask |= 1 << bidx[w]
        if in_base:
            nb += 1
            if mask: pat_base[mask] += 1
            for w in CONTROL:
                if w in toks: ctrl_base[w] += 1
        else:
            nr += 1
            if mask: pat_rec[mask] += 1
            for w in CONTROL:
                if w in toks: ctrl_rec[w] += 1

    f0_all = sum(pat_base.values()) / nb
    ft_all = sum(pat_rec.values()) / nr
    loo = {}
    for w, i in bidx.items():
        bit = 1 << i
        f0 = sum(c for m, c in pat_base.items() if m & ~bit) / nb
        ft = sum(c for m, c in pat_rec.items() if m & ~bit) / nr
        loo[w] = ft - f0
    vals = sorted(loo.values())
    out["loo"] = {"alpha_full": ft_all - f0_all, "min": vals[0], "max": vals[-1],
                  "min_word": min(loo, key=loo.get), "max_word": max(loo, key=loo.get)}

    out["control_excess"] = {w: ctrl_rec[w]/nr - ctrl_base[w]/nb for w in CONTROL}
    out["control_max_abs"] = max(abs(v) for v in out["control_excess"].values())

    # --- hype vs basket correlation with p value ---
    d7 = pd.read_csv(os.path.join(DATA, "c7_hype_hedge.csv"))
    basket_rate = {y: by[y]["rate"]*100 for y in by}
    d7 = d7[d7.year.isin(basket_rate)]
    x = d7.year.map(basket_rate).values; yv = d7.hype_per1k.values
    r = np.corrcoef(x, yv)[0, 1]
    n = len(x)
    t = r*math.sqrt(n-2)/math.sqrt(1-r*r)
    from math import erf
    # two-sided p from t with n-2 dof, normal approx is fine at this size; use survival via betainc-free approx
    try:
        from scipy import stats
        p_r = 2*stats.t.sf(abs(t), n-2)
    except Exception:
        p_r = 2*(1-0.5*(1+erf(abs(t)/math.sqrt(2))))
    rh = np.corrcoef(x, d7.hedge_per1k.values)[0, 1]
    out["hype_corr"] = {"r": float(r), "n": int(n), "t": float(t), "p": float(p_r),
                        "r_hedge": float(rh)}

    # --- source-leak Wilson interval (2024-26) ---
    sl = pd.read_csv(os.path.join(DATA, "source_leak_by_year.csv"))
    rec = sl[sl.year.between(2024, 2026)]
    k = int(rec.tier1.sum() + rec.tier2.sum()); n = int(rec.scanned.sum())
    lo, hi = wilson(k, n)
    pre = sl[sl.year <= 2022]
    kp, np_ = int(pre.tier1.sum() + pre.tier2.sum()), int(pre.scanned.sum())
    out["source_leak"] = {"k": k, "n": n, "rate": k/n, "lo": lo, "hi": hi,
                          "pre_k": kp, "pre_n": np_, "pre_hi": wilson(kp, np_)[1]}

    # --- per-country Wilson intervals (2025) ---
    g = json.load(open(os.path.join(DATA, "c4_geography.json")))
    cc = {}
    for c, recd in g.items():
        v = recd["2025"]
        lo, hi = wilson(v["marker"], v["total"])
        cc[c] = {"rate": v["marker_pct"]/100, "lo": lo, "hi": hi, "n": v["total"]}
    out["country"] = cc

    json.dump(out, open(os.path.join(DATA, "rigor.json"), "w"), indent=2)

    print(f"baseline f0={p0*100:.2f}%  2025 f1={p1*100:.2f}%  z={z:.1f}")
    print(f"alpha windows: " + ", ".join(f"{k}: {v['alpha']*100:.2f}%" for k, v in out['alpha_windows'].items()))
    print(f"alpha 2025 = {a25*100:.2f}% +- {out['alpha_2025']['se']*100:.2f}% (stat)")
    print(f"LOO range: {vals[0]*100:.2f}% ({out['loo']['min_word']}) .. {vals[-1]*100:.2f}% ({out['loo']['max_word']}); full={out['loo']['alpha_full']*100:.2f}%")
    print(f"control excess max |.| = {out['control_max_abs']*100:.2f} pp; per-word:",
          {w: f"{v*100:+.2f}" for w, v in out['control_excess'].items()})
    print(f"hype: r={r:.2f} (n={n}, p={p_r:.1e}); hedge r={rh:.2f}")
    print(f"source leak: {k}/{n} = {100*k/n:.2f}% (95% {100*lo:.2f}-{100*hi:.2f}); pre-2023 {kp}/{np_} (UL {100*out['source_leak']['pre_hi']:.2f}%)")
    print("country (2025) e.g. Iran:", cc.get("Iran"))

if __name__ == "__main__":
    main()
