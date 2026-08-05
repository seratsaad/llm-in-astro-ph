#!/usr/bin/env python3
"""
R1 (referee point 1) -- likelihood-based alpha with full error budget.

Posterior for alpha under the mixture f_t = (1 - alpha) f0 + alpha q with
alpha constrained to [0, 1]:
  k0 ~ Bin(n0, f0)   baseline abstracts with a basket word
  kq ~ Bin(nq, q)    disclosed (calibration) abstracts with a basket word
  kt ~ Bin(nt, ft)   2025 abstracts with a basket word
Flat prior on alpha; conjugate Beta posteriors for f0 and q are integrated by
Monte Carlo. Systematics are the spread of the posterior median over
  baseline choice: pooled 2015-2019, pooled 2017-2021, linear extrapolation
                   of 2015-2021 yearly rates to 2025
  basket choice:   full 38-word basket, basket without the five publicized
                   words, top-20 subset by excess rank
  calibration set: all classified writing-disclosure papers, 2025-only subset
Reads data/n1b_classified.json (per-paper purpose labels) if present, else
falls back to the family-matched list.
Output: data/r1_alpha_stats.json
"""
import json, os, re, sys
import numpy as np

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
rng = np.random.default_rng(20260805)

FULL = """delve delves delving underscore underscores underscoring intricate
intricacies showcasing showcase showcases showcased boasts tapestry pivotal
meticulous meticulously nuanced garner garners garnered multifaceted commendable
noteworthy myriad plethora testament encompassing seamless seamlessly elucidate
elucidating unravel unraveling unravelling realm realms leveraging""".split()
PUBLICIZED = {"delve", "delves", "delving", "intricate", "pivotal"}

def rank_basket_words():
    """Rank basket words by 2024-2025 excess over their pre-2022 rate."""
    pre = {w: 0 for w in FULL}; post = {w: 0 for w in FULL}
    npre = npost = 0
    for line in open(os.path.join(DATA, "astroph_abstracts.jsonl")):
        r = json.loads(line)
        y = int(r["published"][:4])
        if y < 2015 or y > 2025 or (2022 <= y <= 2023):
            continue
        toks = tokset(r["abstract"])
        if y < 2022:
            npre += 1
            for w in FULL:
                if w in toks: pre[w] += 1
        else:
            npost += 1
            for w in FULL:
                if w in toks: post[w] += 1
    exc = {w: post[w] / npost - pre[w] / npre for w in FULL}
    return sorted(FULL, key=lambda w: -exc[w])

def tokset(s):
    return set(re.findall(r"[a-z]+", s.lower()))

BASKETS = {"full": set(FULL), "no_publicized": set(FULL) - PUBLICIZED}

def main():
    ranked = rank_basket_words()
    BASKETS["top20"] = set(ranked[:20])
    # corpus pass: per-year per-basket counts + per-id hit flags for calibration ids
    cls_fn = os.path.join(DATA, "n1b_classified.json")
    cls = json.load(open(cls_fn)) if os.path.exists(cls_fn) else None
    writing_ids = set(cls["writing_ids"]) if cls else set(json.load(
        open(os.path.join(DATA, "n1b_matched.json"))).keys())
    writing_2025 = set(cls["writing_ids_2025"]) if cls else set()

    years = {}
    calib_hits = {}       # id -> {basket: 0/1}
    calib_year = {}
    for line in open(os.path.join(DATA, "astroph_abstracts.jsonl")):
        r = json.loads(line)
        y = int(r["published"][:4])
        if y < 2015 or y > 2025:
            continue
        toks = tokset(r["abstract"])
        yy = years.setdefault(y, {b: 0 for b in BASKETS} | {"n": 0})
        yy["n"] += 1
        for b, ws in BASKETS.items():
            if toks & ws:
                yy[b] += 1
        base = re.sub(r"v\d+$", "", r["id"])
        if base in writing_ids:
            calib_hits[base] = {b: int(bool(toks & ws)) for b, ws in BASKETS.items()}
            calib_year[base] = y

    baselines = {}
    for name, yrs in (("pool_2015_2019", range(2015, 2020)),
                      ("pool_2017_2021", range(2017, 2022))):
        for b in BASKETS:
            k = sum(years[y][b] for y in yrs)
            n = sum(years[y]["n"] for y in yrs)
            baselines.setdefault(name, {})[b] = (k, n)
    # linear extrapolation baseline: fit yearly rates 2015-2021, predict 2025
    extrap = {}
    for b in BASKETS:
        ys = np.arange(2015, 2022)
        rates = np.array([years[y][b] / years[y]["n"] for y in ys])
        coef = np.polyfit(ys, rates, 1)
        pred = float(np.polyval(coef, 2025))
        resid = rates - np.polyval(coef, ys)
        extrap[b] = (max(pred, 1e-4), float(np.std(resid)))

    nt = years[2025]["n"]
    results = {"config": {}, "grid": {}}
    NA, NMC = 501, 4000
    agrid = np.linspace(0, 1, NA)

    for calib_name, ids in (("all_writing", set(calib_hits)),
                            ("writing_2025", {i for i in calib_hits
                                              if (i in writing_2025) or
                                                 (not cls and calib_year[i] == 2025)})):
        for b in BASKETS:
            kq = sum(calib_hits[i][b] for i in ids)
            nq = len(ids)
            kt = years[2025][b]
            for base_name in ("pool_2015_2019", "pool_2017_2021", "extrap_2025"):
                if base_name == "extrap_2025":
                    mu, sd = extrap[b]
                    f0s = rng.normal(mu, max(sd, 1e-4), NMC).clip(1e-5, 0.5)
                else:
                    k0, n0 = baselines[base_name][b]
                    f0s = rng.beta(k0 + 1, n0 - k0 + 1, NMC)
                qs = rng.beta(kq + 1, nq - kq + 1, NMC)
                # log posterior on the alpha grid, MC-averaged over (f0, q)
                post = np.zeros(NA)
                for i, a in enumerate(agrid):
                    ft = (1 - a) * f0s + a * qs
                    ft = ft.clip(1e-6, 1 - 1e-6)
                    ll = kt * np.log(ft) + (nt - kt) * np.log1p(-ft)
                    m = ll.max()
                    post[i] = m + np.log(np.mean(np.exp(ll - m)))
                post = np.exp(post - post.max())
                post /= post.sum()
                cdf = np.cumsum(post)
                med = float(agrid[np.searchsorted(cdf, 0.5)])
                lo68 = float(agrid[np.searchsorted(cdf, 0.16)])
                hi68 = float(agrid[np.searchsorted(cdf, 0.84)])
                lo95 = float(agrid[np.searchsorted(cdf, 0.025)])
                hi95 = float(agrid[np.searchsorted(cdf, 0.975)])
                key = f"{calib_name}|{b}|{base_name}"
                results["config"][key] = {
                    "nq": nq, "kq": kq, "q_hat": kq / nq if nq else None,
                    "kt": kt, "nt": nt,
                    "alpha_median": med, "alpha_68": [lo68, hi68],
                    "alpha_95": [lo95, hi95]}
                print(f"{key:44s} q={kq}/{nq}  alpha={med:.2f} "
                      f"[{lo68:.2f},{hi68:.2f}]68 [{lo95:.2f},{hi95:.2f}]95", flush=True)

    meds = [c["alpha_median"] for c in results["config"].values()]
    central = results["config"]["all_writing|full|pool_2017_2021"]
    results["summary"] = {
        "central": central,
        "systematic_range_medians": [min(meds), max(meds)],
        "n_configs": len(meds)}
    json.dump(results, open(os.path.join(DATA, "r1_alpha_stats.json"), "w"), indent=2)
    print(f"\ncentral alpha {central['alpha_median']:.2f} "
          f"68% {central['alpha_68']}  95% {central['alpha_95']}")
    print(f"systematic spread of medians across {len(meds)} configs: "
          f"{min(meds):.2f} - {max(meds):.2f}")

if __name__ == "__main__":
    main()
