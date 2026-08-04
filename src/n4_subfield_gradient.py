#!/usr/bin/env python3
"""
N4 -- subfield gradient of the publicized-word decay.
Author avoidance predicts the decay of publicized markers is deepest in the most
AI-aware subfield (Instrumentation and Methods) and shallower elsewhere.
Provider-side suppression predicts a uniform decay, since the same models serve
every subfield.

For each subfield group we build the quarterly fraction of abstracts containing
any publicized marker (delve, delves, delving, intricate, pivotal), subtract the
subfield's own 2015-2021 linear trend, and measure the decay from the 2024 peak
to the mean of the last two quarters of 2025, as a fraction of the peak.
The unpublicized composite (leveraging, offering, highlighting, underscore,
underscores) is the within-subfield control.
Output: data/n4_subfield_gradient.json
"""
import json, os, re, collections
import numpy as np

DATA = os.path.join(os.path.dirname(__file__), "..", "data")

GROUPS = {
    "astro-ph.IM": "Instrum./Methods", "astro-ph.GA": "Galaxies",
    "astro-ph.CO": "Cosmology", "astro-ph.EP": "Earth/Planetary",
    "astro-ph.SR": "Solar/Stellar", "astro-ph.HE": "High-Energy",
}
PUB = {"delve", "delves", "delving", "intricate", "pivotal"}
UNPUB = {"leveraging", "offering", "highlighting", "underscore", "underscores"}

def main():
    qn = collections.defaultdict(collections.Counter)
    qp = collections.defaultdict(collections.Counter)
    qu = collections.defaultdict(collections.Counter)
    for line in open(os.path.join(DATA, "astroph_abstracts.jsonl")):
        r = json.loads(line)
        g = GROUPS.get(r["primary_category"])
        if g is None: continue
        y = int(r["published"][:4]); m = int(r["published"][5:7])
        if y < 2015 or y > 2025: continue
        qk = y + ((m - 1) // 3) * 0.25
        toks = set(re.findall(r"[a-z]+", r["abstract"].lower()))
        qn[g][qk] += 1
        if toks & PUB: qp[g][qk] += 1
        if toks & UNPUB: qu[g][qk] += 1

    out = {}
    for g in GROUPS.values():
        quarters = sorted(qn[g])
        def series(cnt):
            return {qk: cnt[g][qk] / qn[g][qk] for qk in quarters}
        res = {}
        for name, cnt in (("publicized", qp), ("unpublicized", qu)):
            ser = series(cnt)
            pre = [(qk, v) for qk, v in ser.items() if qk < 2022]
            coef = np.polyfit([p[0] for p in pre], [p[1] for p in pre], 1)
            exc = {qk: ser[qk] - np.polyval(coef, qk) for qk in quarters if qk >= 2022}
            peak_q = max((q for q in exc if q <= 2025.0), key=lambda q: exc[q])
            peak = exc[peak_q]
            end = np.mean([exc[2025.5], exc[2025.75]])
            n_peak = qn[g][peak_q]
            res[name] = {"peak_q": peak_q, "peak_pct": peak * 100,
                         "end_pct": float(end) * 100,
                         "decay_frac": float((peak - end) / peak) if peak > 0 else None,
                         "n_per_quarter": n_peak}
        out[g] = res
        p, u = res["publicized"], res["unpublicized"]
        print(f"{g:18s} pub: peak {p['peak_pct']:.2f}pp @{p['peak_q']:.2f} -> end {p['end_pct']:.2f}pp "
              f"(decay {p['decay_frac']*100:.0f}%)   unpub decay {u['decay_frac']*100 if u['decay_frac'] else float('nan'):.0f}%  (n/q~{p['n_per_quarter']})")
    json.dump(out, open(os.path.join(DATA, "n4_subfield_gradient.json"), "w"), indent=2)

if __name__ == "__main__":
    main()
