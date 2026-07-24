#!/usr/bin/env python3
"""
Referee check R1 -- composition control.
The aggregate marker rise could reflect a shift in the author population toward
countries where English is not a first language, rather than per-author adoption.
Here we recompute the marker rate over time restricted to native-English
affiliations (USA, UK, Australia, Canada) using NASA ADS, and compare it with the
all-affiliation rate. If the rise holds within native-English papers, an author
composition shift cannot explain it.

Same ADS aff: x abs:(basket) approach as c4_geography.py (noisy affiliation
strings, aff: matches any affiliation on a paper). Output: data/c9_composition.json
"""
import json, os, time, urllib.parse, urllib.request

TOKEN = open(os.path.expanduser("~/.ads/dev_key")).read().strip()
BASE = "https://api.adsabs.harvard.edu/v1/search/query"
DATA = os.path.join(os.path.dirname(__file__), "..", "data")
SLEEP = 0.3

NATIVE = ["USA", "United Kingdom", "Australia", "Canada"]
MARKERS = ["delve", "delves", "delving", "underscore", "underscores", "underscoring",
           "intricate", "showcasing", "pivotal", "meticulous", "nuanced", "realm",
           "leveraging", "tapestry", "boasts", "garnered", "multifaceted"]
YEARS = list(range(2018, 2026))

def q(query, fq):
    url = BASE + "?" + urllib.parse.urlencode({"q": query, "fq": fq, "rows": "0"})
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {TOKEN}"})
    last = None
    for a in range(5):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)["response"]["numFound"]
        except Exception as e:
            last = e; time.sleep(SLEEP * (a + 2))
    raise RuntimeError(last)

def basket(field, terms):
    return f'{field}:(' + " OR ".join(f'"{t}"' for t in terms) + ')'

def series(aff_clause):
    rec = {}
    for y in YEARS:
        fq = f"year:{y}"
        tot = q(f'{aff_clause} database:astronomy', fq) if aff_clause else q("database:astronomy", fq)
        pre = f'{aff_clause} ' if aff_clause else ''
        mk = q(f'{pre}{basket("abs", MARKERS)} database:astronomy', fq)
        rec[y] = {"total": tot, "marker": mk, "marker_pct": 100 * mk / tot if tot else 0}
        print(f"  {('native' if aff_clause else 'all'):7s} {y}: n={tot:>7}  marker={rec[y]['marker_pct']:5.2f}%", flush=True)
        time.sleep(SLEEP)
    return rec

def main():
    native_clause = "(" + " OR ".join(f'aff:"{c}"' for c in NATIVE) + ")"
    out = {"native_affil": series(native_clause), "all_affil": series("")}
    json.dump(out, open(os.path.join(DATA, "c9_composition.json"), "w"), indent=2)
    # summary
    na, al = out["native_affil"], out["all_affil"]
    def rise(d):
        base = sum(d[y]["marker"] for y in (2018, 2019, 2020, 2021)) / sum(d[y]["total"] for y in (2018, 2019, 2020, 2021)) * 100
        return base, d[2025]["marker_pct"]
    nb, n25 = rise(na); ab, a25 = rise(al)
    print(f"\nnative-English: {nb:.2f}% (2018-21) -> {n25:.2f}% (2025)")
    print(f"all affiliations: {ab:.2f}% (2018-21) -> {a25:.2f}% (2025)")
    print("wrote c9_composition.json")

if __name__ == "__main__":
    main()
