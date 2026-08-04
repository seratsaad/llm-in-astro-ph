#!/usr/bin/env python3
"""
N2 -- avoidance metrics and tell decay.
(a) Quarterly P(>=1 basket word) and conditional density P(>=2 | >=1) with Wilson
    intervals, full 38-word basket, 2018-2025. The conditional density is the
    avoidance observable: it collapses when authors purge multiple tells while
    incidence persists.
(b) Per-word quarterly series for the ten highest-frequency markers, and a
    peak-decay measurement: for words whose frequency peaked by 2025Q1, the time
    for the excess over the pre-2022 trend to fall to half its peak value.
Outputs: data/n2_avoidance.json, data/n2_density.csv, data/n2_words.csv
"""
import json, os, re, math, collections
import numpy as np

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
BASKET = set("""delve delves delving underscore underscores underscoring intricate
intricacies showcasing showcase showcases showcased boasts tapestry pivotal
meticulous meticulously nuanced garner garners garnered multifaceted commendable
noteworthy myriad plethora testament encompassing seamless seamlessly elucidate
elucidating unravel unraveling unravelling realm realms leveraging""".split())
TOPWORDS = ["leveraging", "offering", "highlighting", "underscore", "pivotal",
            "intricate", "nuanced", "showcasing", "delve", "meticulous"]

def wilson(k, n, z=1.96):
    if n == 0: return (0, 0)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)

def main():
    qn = collections.Counter(); q1 = collections.Counter(); q2 = collections.Counter()
    wq = {w: collections.Counter() for w in TOPWORDS}
    for line in open(os.path.join(DATA, "astroph_abstracts.jsonl")):
        r = json.loads(line)
        y = int(r["published"][:4]); m = int(r["published"][5:7])
        if y < 2015 or y > 2025: continue
        qk = y + ((m - 1) // 3) * 0.25
        toks = set(re.findall(r"[a-z]+", r["abstract"].lower()))
        hits = toks & BASKET
        qn[qk] += 1
        if len(hits) >= 1: q1[qk] += 1
        if len(hits) >= 2: q2[qk] += 1
        for w in TOPWORDS:
            if w in toks: wq[w][qk] += 1

    quarters = sorted(qn)
    rows = []
    for qk in quarters:
        n = qn[qk]; k1 = q1[qk]; k2 = q2[qk]
        lo1, hi1 = wilson(k1, n)
        loc, hic = wilson(k2, k1) if k1 else (0, 0)
        rows.append({"q": qk, "n": n, "p1": k1 / n, "p1_lo": lo1, "p1_hi": hi1,
                     "cond": (k2 / k1 if k1 else 0), "cond_lo": loc, "cond_hi": hic,
                     "k1": k1, "k2": k2})
    import csv
    with open(os.path.join(DATA, "n2_density.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

    # per-word series + decay
    wrows = []
    decay = {}
    for w in TOPWORDS:
        ser = {qk: wq[w][qk] / qn[qk] for qk in quarters}
        for qk in quarters:
            wrows.append({"word": w, "q": qk, "freq": ser[qk]})
        # pre-2022 linear trend
        pre = [(qk, ser[qk]) for qk in quarters if qk < 2022]
        coef = np.polyfit([p[0] for p in pre], [p[1] for p in pre], 1)
        exc = {qk: ser[qk] - np.polyval(coef, qk) for qk in quarters if qk >= 2022}
        pk = max((qk for qk in exc if qk <= 2025.0), key=lambda qk: exc[qk])
        peak_val = exc[pk]
        if peak_val <= 0: continue
        half = None
        for qk in [q for q in sorted(exc) if q > pk]:
            if exc[qk] <= peak_val / 2:
                half = qk - pk; break
        decay[w] = {"peak_q": pk, "peak_excess_pct": peak_val * 100,
                    "half_life_quarters": half,
                    "end_excess_pct": exc[max(exc)] * 100}
    with open(os.path.join(DATA, "n2_words.csv"), "w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=["word", "q", "freq"]); wcsv.writeheader(); wcsv.writerows(wrows)

    out = {"density_last8": rows[-8:], "decay": decay}
    json.dump(out, open(os.path.join(DATA, "n2_avoidance.json"), "w"), indent=2, default=float)

    print("quarter   n     P(>=1)%   P(>=2|>=1)%")
    for r in rows:
        if r["q"] >= 2023:
            print(f"{r['q']:7.2f} {r['n']:6d} {r['p1']*100:8.2f} {r['cond']*100:10.1f}")
    print("\nword decay (peak excess -> half):")
    for w, d in sorted(decay.items(), key=lambda x: (x[1]["half_life_quarters"] is None, x[1]["half_life_quarters"] or 99)):
        hl = f"{d['half_life_quarters']:.2f} q" if d["half_life_quarters"] else "not reached"
        print(f"  {w:14s} peak {d['peak_q']:.2f} ({d['peak_excess_pct']:.2f}pp)  half-life {hl}  end {d['end_excess_pct']:.2f}pp")

if __name__ == "__main__":
    main()
