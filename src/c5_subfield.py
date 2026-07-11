#!/usr/bin/env python3
"""
C5 -- Which astro-ph SUBFIELD adopted LLM phrasing first?
Compute the strong LLM-marker basket rate per primary arXiv subfield over time.
Hypothesis: astro-ph.IM (instrumentation & methods, the CS-adjacent subfield) leads.
"""
import json, os, re, collections
import pandas as pd

HERE = os.path.dirname(__file__); DATA = os.path.join(HERE, "..", "data")
INP = os.path.join(DATA, "astroph_abstracts.jsonl")
TOK = re.compile(r"[a-z]+")

BASKET = {"delve","delves","delving","underscore","underscores","underscoring",
    "intricate","intricacies","showcasing","showcase","showcases","showcased",
    "boasts","tapestry","pivotal","meticulous","meticulously","nuanced","garner",
    "garners","garnered","multifaceted","commendable","noteworthy","myriad","plethora",
    "testament","encompassing","seamless","seamlessly","elucidate","elucidating",
    "unravel","unraveling","unravelling","realm","realms","leveraging"}

SUBFIELDS = {"astro-ph.GA":"Galaxies", "astro-ph.CO":"Cosmology",
             "astro-ph.SR":"Solar/Stellar", "astro-ph.HE":"High-Energy",
             "astro-ph.EP":"Earth/Planetary", "astro-ph.IM":"Instrum./Methods"}

def main():
    tot = collections.defaultdict(collections.Counter)   # cat -> year -> n
    hit = collections.defaultdict(collections.Counter)
    seen = set()
    for line in open(INP):
        r = json.loads(line); b = r["id"].split("v")[0]
        if b in seen: continue
        seen.add(b)
        cat = r.get("primary_category","")
        if cat not in SUBFIELDS: continue
        pub = r["published"]
        if not pub or len(pub) < 7: continue
        y = int(pub[:4])
        if y < 2018 or y > 2026: continue
        toks = set(TOK.findall((r.get("abstract") or "").lower()))
        tot[cat][y] += 1
        if toks & BASKET: hit[cat][y] += 1

    years = list(range(2018, 2027))
    rows = []
    for cat, name in SUBFIELDS.items():
        for y in years:
            n = tot[cat][y]
            rows.append({"cat":cat, "subfield":name, "year":y, "n":n,
                         "rate": 100*hit[cat][y]/n if n else 0})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(DATA,"c5_subfield.csv"), index=False)

    # baseline (2018-21) and recent (2025) per subfield -> excess + ranking
    print("subfield        base18-21  2023   2025   excess(2025-base)")
    summ = []
    for cat, name in SUBFIELDS.items():
        base = 100*sum(hit[cat][y] for y in (2018,2019,2020,2021))/max(1,sum(tot[cat][y] for y in (2018,2019,2020,2021)))
        r23 = 100*hit[cat][2023]/max(1,tot[cat][2023])
        r25 = 100*hit[cat][2025]/max(1,tot[cat][2025])
        summ.append((name, base, r23, r25, r25-base))
        print(f"{name:16s} {base:6.2f}   {r23:5.2f}  {r25:5.2f}   {r25-base:+.2f}")
    summ.sort(key=lambda x:-x[3])
    print("\nRanked by 2025 excess (who adopted most):")
    for name,base,r23,r25,exc in summ:
        print(f"  {name:16s} excess={exc:+.2f}pp  (2023 already {r23:.2f}%)")
    # who led EARLIEST: rank by 2023 rate
    print("\nRanked by 2023 rate (who moved FIRST):")
    for name,base,r23,r25,exc in sorted(summ,key=lambda x:-x[2]):
        print(f"  {name:16s} 2023={r23:.2f}%")

if __name__ == "__main__":
    main()
