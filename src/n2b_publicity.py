#!/usr/bin/env python3
"""
N2b (referee point 4) -- objective per-word publicity and the half-life scatter.

Publicity is measured externally, identically for every word, with no manual
labeling: the GDELT DOC 2.0 news archive volume of articles containing the word
together with "ChatGPT" or "artificial intelligence", 2023-2025. For each word
we also fetch the word's overall news volume as a base rate.

Part A extends the abstract decay measurement to every basket word (plus the
non-basket controls offering/highlighting) with peak excess >= 0.03 pp, the
floor below which quarterly noise dominates.
Part B queries GDELT (cached, 5.5 s between requests per their limit).
Part C correlates abstract half-life with publicity.

Outputs: data/n2b_publicity.json (summary), data/n2b_gdelt_cache/ (raw series).
"""
import json, os, re, math, time, collections, urllib.parse, urllib.request
import numpy as np

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
CACHE = os.path.join(DATA, "n2b_gdelt_cache")
os.makedirs(CACHE, exist_ok=True)

BASKET = """delve delves delving underscore underscores underscoring intricate
intricacies showcasing showcase showcases showcased boasts tapestry pivotal
meticulous meticulously nuanced garner garners garnered multifaceted commendable
noteworthy myriad plethora testament encompassing seamless seamlessly elucidate
elucidating unravel unraveling unravelling realm realms leveraging""".split()
CONTROLS = ["offering", "highlighting"]
WORDS = BASKET + CONTROLS
FLOOR = 0.0003          # 0.03 pp peak excess floor for a usable decay curve
GDELT = "https://api.gdeltproject.org/api/v2/doc/doc"
UA = {"User-Agent": "academic-corpus-study/1.0"}
SLEEP = 8.0

def decay_all():
    qn = collections.Counter()
    wq = {w: collections.Counter() for w in WORDS}
    for line in open(os.path.join(DATA, "astroph_abstracts.jsonl")):
        r = json.loads(line)
        y = int(r["published"][:4]); m = int(r["published"][5:7])
        if y < 2015 or y > 2025:
            continue
        qk = y + ((m - 1) // 3) * 0.25
        toks = set(re.findall(r"[a-z]+", r["abstract"].lower()))
        qn[qk] += 1
        for w in WORDS:
            if w in toks:
                wq[w][qk] += 1
    quarters = sorted(qn)
    out = {}
    for w in WORDS:
        ser = {qk: wq[w][qk] / qn[qk] for qk in quarters}
        pre = [(qk, ser[qk]) for qk in quarters if qk < 2022]
        coef = np.polyfit([p[0] for p in pre], [p[1] for p in pre], 1)
        exc = {qk: ser[qk] - np.polyval(coef, qk) for qk in quarters if qk >= 2022}
        pk = max((qk for qk in exc if qk <= 2025.0), key=lambda qk: exc[qk])
        peak_val = exc[pk]
        if peak_val < FLOOR:
            continue
        half = None
        for qk in [q for q in sorted(exc) if q > pk]:
            if exc[qk] <= peak_val / 2:
                half = qk - pk
                break
        out[w] = {"peak_q": pk, "peak_excess_pct": peak_val * 100,
                  "half_life_quarters": half,
                  "end_excess_pct": exc[max(exc)] * 100,
                  "peak_docs": wq[w][pk]}
    return out

def gdelt_series(query, tag):
    fn = os.path.join(CACHE, tag + ".json")
    if os.path.exists(fn):
        return json.load(open(fn))
    url = GDELT + "?" + urllib.parse.urlencode(
        {"query": query, "mode": "timelinevol", "format": "json",
         "startdatetime": "20230101000000", "enddatetime": "20251231235959"})
    for a in range(4):
        try:
            time.sleep(SLEEP)
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r:
                txt = r.read().decode("utf-8", "ignore")
            d = json.loads(txt)
            json.dump(d, open(fn, "w"))
            return d
        except Exception as e:
            if a == 3:
                print(f"  GDELT FAIL {tag}: {e}", flush=True)
                return None
            time.sleep(30.0 * (a + 1))

def series_stats(d):
    if not d or "timeline" not in d or not d["timeline"]:
        return None
    pts = d["timeline"][0]["data"]
    vals = [p["value"] for p in pts]
    if not vals:
        return None
    imax = int(np.argmax(vals))
    return {"mean": float(np.mean(vals)), "peak": float(vals[imax]),
            "peak_date": pts[imax]["date"][:8]}

def spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])

def main():
    print("part A: decay for all words", flush=True)
    decay = decay_all()
    for w, d in sorted(decay.items(), key=lambda x: -x[1]["peak_excess_pct"]):
        hl = f"{d['half_life_quarters']:.2f}" if d["half_life_quarters"] else "none"
        print(f"  {w:14s} peak {d['peak_excess_pct']:.3f}pp @{d['peak_q']:.2f} "
              f"half-life {hl}", flush=True)

    print("part B: GDELT publicity", flush=True)
    pub = {}
    for w in decay:
        co = series_stats(gdelt_series(f'"{w}" ("ChatGPT" OR "artificial intelligence")', w + "_co"))
        base = series_stats(gdelt_series(f'"{w}"', w + "_base"))
        if co is None:
            continue
        pub[w] = {"co_mean": co["mean"], "co_peak": co["peak"],
                  "co_peak_date": co["peak_date"],
                  "base_mean": base["mean"] if base else None,
                  "ratio": (co["mean"] / base["mean"]) if base and base["mean"] > 0 else None}
        print(f"  {w:14s} co-vol {co['mean']:.4f} (peak {co['peak']:.3f} @{co['peak_date']}) "
              f"ratio {pub[w]['ratio'] if pub[w]['ratio'] is None else round(pub[w]['ratio'],3)}", flush=True)

    # part C: stats
    words = [w for w in decay if w in pub]
    measured = [w for w in words if decay[w]["half_life_quarters"] is not None]
    censored = [w for w in words if decay[w]["half_life_quarters"] is None]
    stats = {}
    if len(measured) >= 4:
        hl = [decay[w]["half_life_quarters"] for w in measured]
        for metric in ("co_mean", "ratio"):
            pv = [pub[w][metric] for w in measured]
            if any(v is None for v in pv):
                continue
            rho = spearman(pv, hl)
            stats[f"spearman_halflife_vs_{metric}"] = rho
    # decayed vs censored: rank-sum on publicity
    for metric in ("co_mean", "ratio"):
        a = sorted(pub[w][metric] for w in measured if pub[w][metric] is not None)
        b = sorted(pub[w][metric] for w in censored if pub[w][metric] is not None)
        if a and b:
            # Mann-Whitney U by direct count
            u = sum(1 for x in a for y in b if x > y) + 0.5 * sum(1 for x in a for y in b if x == y)
            stats[f"mannwhitney_{metric}"] = {"U": u, "n_dec": len(a), "n_cen": len(b),
                                              "frac_pairs_dec_gt_cen": u / (len(a) * len(b))}
    out = {"decay": decay, "publicity": pub, "stats": stats,
           "measured_words": measured, "censored_words": censored}
    json.dump(out, open(os.path.join(DATA, "n2b_publicity.json"), "w"), indent=2, default=float)
    print(json.dumps(stats, indent=1))

if __name__ == "__main__":
    main()
