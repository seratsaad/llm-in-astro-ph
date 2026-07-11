#!/usr/bin/env python3
"""
C4 -- Equity map: LLM marker-word incidence and disclosure rate by author country.
Uses NASA ADS: aff: (affiliation string) x abs:(marker basket) and ack:(LLM terms).
Tests the cross-field finding (Kobak, Cong&Zhu) that non-native-English authors adopt
LLMs more -- never done for astronomy.

Caveats (report them): aff: matches ANY affiliation on a paper, so multi-country collabs
are double counted; affiliation strings are noisy; abstract-only marker signal.
"""
import json, os, time, urllib.parse, urllib.request

TOKEN = open(os.path.expanduser("~/.ads/dev_key")).read().strip()
BASE = "https://api.adsabs.harvard.edu/v1/search/query"
DATA = os.path.join(os.path.dirname(__file__), "..", "data")
SLEEP = 0.3

# native-English vs non-native, spanning regions
COUNTRIES = ["USA", "United Kingdom", "Australia", "Canada",         # native English
             "Germany", "France", "Italy", "Spain", "Netherlands",    # Europe (non-native)
             "China", "Japan", "South Korea", "India", "Iran",        # Asia
             "Brazil", "Russia", "Poland", "Turkey"]

MARKERS = ["delve", "delves", "delving", "underscore", "underscores", "underscoring",
           "intricate", "showcasing", "pivotal", "meticulous", "nuanced", "realm",
           "leveraging", "tapestry", "boasts", "garnered", "multifaceted"]
LLM = ["ChatGPT", "GPT-4", "GPT-4o", "large language model", "large language models",
       "GitHub Copilot", "generative pre-trained transformer"]

def q(query, fq):
    url = BASE + "?" + urllib.parse.urlencode({"q": query, "fq": fq, "rows": "0"})
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {TOKEN}"})
    for a in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)["response"]["numFound"]
        except Exception as e:
            time.sleep(SLEEP*(a+2)); last=e
    raise RuntimeError(last)

def basket(field, terms):
    return f'{field}:(' + " OR ".join(f'"{t}"' for t in terms) + ')'

def main():
    years = [2024, 2025]
    out = {}
    for c in COUNTRIES:
        cc = f'aff:"{c}"'
        rec = {}
        for y in years:
            fq = f"year:{y}"
            tot = q(f'{cc} database:astronomy', fq)
            mk = q(f'{cc} {basket("abs", MARKERS)} database:astronomy', fq)
            dc = q(f'{cc} {basket("ack", LLM)} database:astronomy', fq)
            rec[y] = {"total": tot, "marker": mk, "disc": dc,
                      "marker_pct": 100*mk/tot if tot else 0,
                      "disc_pct": 100*dc/tot if tot else 0}
            time.sleep(SLEEP)
        out[c] = rec
        r25 = rec[2025]
        print(f"{c:16s} 2025: n={r25['total']:>6}  marker={r25['marker_pct']:5.2f}%  disc={r25['disc_pct']:5.3f}%", flush=True)
    json.dump(out, open(os.path.join(DATA, "c4_geography.json"), "w"), indent=2)
    print("wrote c4_geography.json")

if __name__ == "__main__":
    main()
