#!/usr/bin/env python3
"""
Kobak-style lower-bound estimate of the LLM-touched fraction of astro-ph abstracts.

Logic (Kobak et al. 2024): pick a basket of "excess marker" words that (a) are
stylistic, not topical, and (b) had a near-zero pre-LLM baseline. Almost every
post-2022 occurrence of such a word is LLM-attributable. Then

    alpha_lower(t) = P(abstract in period t contains >=1 basket word)
                     - P(same) in the pre-LLM baseline

is a LOWER BOUND on the fraction of abstracts whose text an LLM touched, because
(i) an LLM-polished abstract need not contain any basket word (so we undercount),
and (ii) we subtract the human baseline rate. The union (per-document) is used so
overlapping markers are not double counted.

We compute this per quarter, plus a "delve"-specific series for the cat-and-mouse
collapse, and print the headline number for 2024-2026.
"""
import json, os, re, collections

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data")
INP = os.path.join(DATA, "astroph_abstracts.jsonl")
TOKEN_RE = re.compile(r"[a-z]+")

# Strong, stylistic, low-baseline markers (canonical LLM lexicon; astronomy-neutral).
# Deliberately EXCLUDES instrument/topic words (jwst, desi, nircam...) and common
# words (offering, highlighting, aligns) to keep this a conservative lower bound.
BASKET = {
    "delve", "delves", "delving", "underscore", "underscores", "underscoring",
    "intricate", "intricacies", "showcasing", "showcase", "showcases", "showcased",
    "boasts", "tapestry", "pivotal", "meticulous", "meticulously", "nuanced",
    "garner", "garners", "garnered", "multifaceted", "commendable", "noteworthy",
    "myriad", "plethora", "testament", "encompassing", "seamless", "seamlessly",
    "elucidate", "elucidating", "unravel", "unraveling", "unravelling",
    "realm", "realms", "leveraging",
}

def qkey(pub):
    y = int(pub[:4]); m = int(pub[5:7]); return y, (m - 1) // 3 + 1

def main():
    q_total = collections.Counter()
    q_hit = collections.Counter()          # >=1 basket word
    q_delve = collections.Counter()        # delve* specifically
    y_total = collections.Counter()
    y_hit = collections.Counter()
    seen = set()
    DELVE = {"delve", "delves", "delving"}
    with open(INP) as f:
        for line in f:
            r = json.loads(line)
            base = r["id"].split("v")[0]
            if base in seen:
                continue
            seen.add(base)
            pub = r["published"]
            if not pub or len(pub) < 7:
                continue
            y = int(pub[:4])
            if y < 2015 or y > 2026:
                continue
            yq = qkey(pub)
            toks = set(TOKEN_RE.findall((r.get("abstract") or "").lower()))
            q_total[yq] += 1; y_total[y] += 1
            if toks & BASKET:
                q_hit[yq] += 1; y_hit[y] += 1
            if toks & DELVE:
                q_delve[yq] += 1

    # baseline = 2018-2021
    base_years = [2018, 2019, 2020, 2021]
    base_hit = sum(y_hit[y] for y in base_years)
    base_tot = sum(y_total[y] for y in base_years)
    base_rate = base_hit / base_tot

    rows = []
    for yq in sorted(q_total):
        y, qn = yq
        rate = q_hit[yq] / q_total[yq]
        rows.append({"year": y, "quarter": qn, "yq": y + (qn - 1) / 4.0,
                     "total": q_total[yq], "hit": q_hit[yq], "rate": rate,
                     "excess": max(0.0, rate - base_rate),
                     "delve_rate": q_delve[yq] / q_total[yq]})
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(DATA, "alpha_series.csv"), index=False)

    print(f"baseline (2018-21) basket rate = {base_rate*100:.2f}%")
    summary = {"base_rate": base_rate, "by_year": {}}
    for y in range(2015, 2027):
        if y_total[y]:
            rate = y_hit[y] / y_total[y]
            excess = max(0.0, rate - base_rate)
            summary["by_year"][y] = {"rate": rate, "excess": excess,
                                     "n": y_total[y], "hit": y_hit[y]}
            print(f"{y}: basket rate={rate*100:5.2f}%  excess(lowerbound alpha)={excess*100:5.2f}%  (n={y_total[y]})")
    # headline: 2024-2026 pooled
    rec_years = [2024, 2025, 2026]
    rec_hit = sum(y_hit[y] for y in rec_years); rec_tot = sum(y_total[y] for y in rec_years)
    rec_rate = rec_hit / rec_tot
    headline = rec_rate - base_rate
    summary["headline_2024_26"] = {"rate": rec_rate, "base_rate": base_rate,
                                   "excess_lowerbound": headline}
    print(f"\nHEADLINE 2024-26 pooled: basket rate={rec_rate*100:.2f}%, "
          f"baseline={base_rate*100:.2f}%, LOWER-BOUND alpha={headline*100:.2f}%")
    # 2025 full year (cleanest full year)
    r25 = y_hit[2025]/y_total[2025]
    print(f"2025 full year: rate={r25*100:.2f}%, lower-bound alpha={(r25-base_rate)*100:.2f}%")
    json.dump(summary, open(os.path.join(DATA, "alpha_summary.json"), "w"), indent=2)
    print("wrote alpha_series.csv, alpha_summary.json")

if __name__ == "__main__":
    main()
