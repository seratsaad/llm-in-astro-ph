#!/usr/bin/env python3
"""
P12 (round 5, point 3) -- prior sensitivity of the calibrated estimate.

Central configuration (full basket, pooled 2017-2021 baseline, all writing
papers) under three treatments:
  flat      uniform prior on alpha over [0, 1] (the paper's default)
  jeffreys  Beta(1/2, 1/2) prior on alpha
  profile   profile likelihood, maximizing over f0 and q at each alpha, with
            the point estimate and the delta-lnL = 0.5 interval
Targets both 2025 and the first half of 2026.
Output: data/p12_priors.json
"""
import json, os, re
import numpy as np

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
BASKET = set("""delve delves delving underscore underscores underscoring intricate
intricacies showcasing showcase showcases showcased boasts tapestry pivotal
meticulous meticulously nuanced garner garners garnered multifaceted commendable
noteworthy myriad plethora testament encompassing seamless seamlessly elucidate
elucidating unravel unraveling unravelling realm realms leveraging""".split())


def main():
    cls = json.load(open(os.path.join(DATA, "n1b_classified.json")))
    writing = set(cls["writing_ids"])
    # counts
    k0 = n0 = 0
    tgt = {"2025": [0, 0], "2026H1": [0, 0]}
    calib = {"2025": [0, 0], "2026H1": [0, 0], "all": [0, 0]}
    wy = {}
    for line in open(os.path.join(DATA, "astroph_abstracts.jsonl")):
        r = json.loads(line)
        y = int(r["published"][:4]); m = int(r["published"][5:7])
        if y < 2015 or y > 2026 or (y == 2026 and m > 6):
            continue
        hit = int(bool(set(re.findall(r"[a-z]+", r["abstract"].lower())) & BASKET))
        if 2017 <= y <= 2021:
            n0 += 1; k0 += hit
        if y == 2025:
            tgt["2025"][1] += 1; tgt["2025"][0] += hit
        if y == 2026:
            tgt["2026H1"][1] += 1; tgt["2026H1"][0] += hit
        b = re.sub(r"v\d+$", "", r["id"])
        if b in writing:
            calib["all"][1] += 1; calib["all"][0] += hit
            key = "2025" if y == 2025 else ("2026H1" if y == 2026 else None)
            if key:
                calib[key][1] += 1; calib[key][0] += hit

    out = {"counts": {"baseline": [k0, n0], "targets": tgt, "calib": calib}}
    agrid = np.linspace(0, 1, 1001)
    rng = np.random.default_rng(9)
    NMC = 8000

    for period in ("2025", "2026H1"):
        kt, nt = tgt[period]
        kq, nq = calib["all"]          # pooled writing calibration
        res = {}
        # Bayesian, two priors on alpha
        f0s = rng.beta(k0 + 1, n0 - k0 + 1, NMC)
        qs = rng.beta(kq + 1, nq - kq + 1, NMC)
        ll = np.zeros(len(agrid))
        for i, a in enumerate(agrid):
            ft = ((1 - a) * f0s + a * qs).clip(1e-6, 1 - 1e-6)
            l = kt * np.log(ft) + (nt - kt) * np.log1p(-ft)
            mx = l.max()
            ll[i] = mx + np.log(np.mean(np.exp(l - mx)))
        for prior, w in (("flat", np.ones_like(agrid)),
                         ("jeffreys", 1 / np.sqrt(np.clip(agrid * (1 - agrid), 1e-6, None)))):
            post = np.exp(ll - ll.max()) * w
            post /= post.sum()
            cdf = np.cumsum(post)
            res[prior] = {"median": float(agrid[np.searchsorted(cdf, 0.5)]),
                          "ci68": [float(agrid[np.searchsorted(cdf, 0.16)]),
                                   float(agrid[np.searchsorted(cdf, 0.84)])]}
        # profile likelihood over (f0, q)
        f0g = np.linspace(max(k0 / n0 - 5e-4, 1e-4), k0 / n0 + 5e-4, 21)
        qg = np.linspace(max(kq / nq - 0.12, 1e-3), min(kq / nq + 0.12, 0.9), 121)
        F0, Qg = np.meshgrid(f0g, qg, indexing="ij")
        prof = np.zeros(len(agrid))
        for i, a in enumerate(agrid):
            ft = ((1 - a) * F0 + a * Qg).clip(1e-6, 1 - 1e-6)
            l = (k0 * np.log(F0) + (n0 - k0) * np.log1p(-F0)
                 + kq * np.log(Qg) + (nq - kq) * np.log1p(-Qg)
                 + kt * np.log(ft) + (nt - kt) * np.log1p(-ft))
            prof[i] = l.max()
        mle = float(agrid[int(np.argmax(prof))])
        keep = prof >= prof.max() - 0.5
        res["profile"] = {"mle": mle,
                          "ci68": [float(agrid[keep].min()), float(agrid[keep].max())]}
        out[period] = res
        print(f"{period}: kt/nt = {kt}/{nt}, calib {kq}/{nq}")
        for k, v in res.items():
            c = v.get("ci68")
            point = v.get("median", v.get("mle"))
            print(f"  {k:9s} {point:.3f}  68% [{c[0]:.3f}, {c[1]:.3f}]")

    json.dump(out, open(os.path.join(DATA, "p12_priors.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
