#!/usr/bin/env python3
"""
M1 (independent referee) -- pseudo-naming placebo against selection-at-peak.
Concern: the 2024 sources named words whose excess was large in early 2024, so
post-naming decline could be regression to the mean, with the "publicity
window" being the selection window by construction. Placebo: assign each
UNNAMED well-measured word a pseudo-naming quarter, the first quarter at which
its excess reached the smallest excess any named word had at its naming
quarter (i.e., when it would have qualified for such a list), then measure its
change over the following two quarters. Under selection/regression the
unnamed words should also decline after crossing; under author avoidance they
should not.
Output: data/m1_pseudonaming.json
"""
import json, os, collections, re
import numpy as np, importlib.util

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data")
spec = importlib.util.spec_from_file_location("n2b", os.path.join(HERE, "n2b_publicity.py"))
n2b = importlib.util.module_from_spec(spec); spec.loader.exec_module(n2b)

NAMED = {"delve": 2024.25, "delves": 2024.25, "delving": 2024.25,
         "intricate": 2024.25, "pivotal": 2024.25, "showcasing": 2024.25,
         "realm": 2024.25, "underscores": 2024.25, "intricacies": 2024.25,
         "meticulously": 2024.25}

def main():
    # quarterly excess series for every word (same machinery as the decay fits)
    qn = collections.Counter()
    words = n2b.WORDS
    wq = {w: collections.Counter() for w in words}
    for line in open(os.path.join(DATA, "astroph_abstracts.jsonl")):
        r = json.loads(line)
        y = int(r["published"][:4]); m = int(r["published"][5:7])
        if y < 2015 or y > 2025:
            continue
        qk = y + ((m - 1) // 3) * 0.25
        toks = set(re.findall(r"[a-z]+", r["abstract"].lower()))
        qn[qk] += 1
        for w in words:
            if w in toks:
                wq[w][qk] += 1
    quarters = sorted(qn)
    exc = {}
    for w in words:
        ser = {qk: wq[w][qk] / qn[qk] for qk in quarters}
        pre = [(qk, ser[qk]) for qk in quarters if qk < 2022]
        coef = np.polyfit([p[0] for p in pre], [p[1] for p in pre], 1)
        exc[w] = {qk: (ser[qk] - np.polyval(coef, qk)) * 100 for qk in quarters if qk >= 2022}

    tab = json.load(open(os.path.join(DATA, "n2g_word_table.json")))
    wm = {r["word"]: r for r in tab if r["well_measured"]}

    # qualification threshold: smallest excess a named well-measured word had
    # at its naming quarter
    named_at = {w: exc[w][q] for w, q in NAMED.items() if w in wm}
    thresh = min(named_at.values())
    out = {"named_excess_at_naming_pp": named_at, "threshold_pp": thresh,
           "named": {}, "unnamed_placebo": {}}
    for w, q in NAMED.items():
        if w not in wm:
            continue
        later = [k for k in sorted(exc[w]) if q < k <= q + 0.5]
        out["named"][w] = {"naming_q": q, "excess_at": exc[w][q],
                           "change_2q_pct": (exc[w][max(later)] - exc[w][q]) / exc[w][q] * 100
                           if later and exc[w][q] > 0 else None}
    for w, r in wm.items():
        if r["n_sources"] > 0:
            continue
        crossed = [k for k in sorted(exc[w]) if exc[w][k] >= thresh and k <= 2025.0]
        if not crossed:
            out["unnamed_placebo"][w] = {"crossed": False}
            continue
        q0 = crossed[0]
        later = [k for k in sorted(exc[w]) if q0 < k <= q0 + 0.5]
        out["unnamed_placebo"][w] = {"crossed": True, "pseudo_naming_q": q0,
            "excess_at": exc[w][q0],
            "change_2q_pct": (exc[w][max(later)] - exc[w][q0]) / exc[w][q0] * 100
            if later and exc[w][q0] > 0 else None}
    json.dump(out, open(os.path.join(DATA, "m1_pseudonaming.json"), "w"), indent=1)
    print(f"threshold {thresh:.3f} pp")
    for grp in ("named", "unnamed_placebo"):
        ch = [v["change_2q_pct"] for v in out[grp].values()
              if isinstance(v, dict) and v.get("change_2q_pct") is not None]
        print(f"{grp}: n={len(ch)}, mean 2-quarter change after (pseudo)naming "
              f"{np.mean(ch):+.0f}%  values {[round(c) for c in ch]}")

if __name__ == "__main__":
    main()
