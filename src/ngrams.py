#!/usr/bin/env python3
"""
N-gram and co-occurrence analysis of astro-ph abstracts (extends the single-word study).

Produces:
  data/ngram_watch.csv       -- per-year doc-frequency for a set of LLM-favoured bigrams
                                 (an astro-ph "n-gram viewer")
  data/bigram_discovery.csv  -- data-driven top-rising bigrams, 2024-26 vs 2018-21
  data/cooccur.npz           -- pairwise co-occurrence lift among marker words (2024-26)

Bigram = two consecutive lowercase word tokens. "Doc-frequency" = fraction of abstracts
containing the bigram at least once, matching the single-word convention.
"""
import json, os, re, collections
import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__); DATA = os.path.join(HERE, "..", "data")
INP = os.path.join(DATA, "astroph_abstracts.jsonl")
TOK = re.compile(r"[a-z]+")

# LLM-favoured phrases to track over time (the "n-gram viewer" panel).
WATCH = ["delve into", "sheds light", "shed light", "pivotal role", "rich tapestry",
         "wide range", "growing body", "the intricate", "a myriad", "in the realm",
         "plays a", "leveraging the", "underscores the", "highlighting the"]

# Marker unigrams for the co-occurrence / lift matrix.
MARKERS = ["delve", "delves", "underscore", "underscores", "intricate", "pivotal",
           "showcasing", "leveraging", "meticulous", "nuanced", "realm", "tapestry",
           "garner", "multifaceted"]

BASE_YEARS = {2018, 2019, 2020, 2021}
REC_YEARS = {2024, 2025, 2026}
MINC = 25

def bigrams(toks):
    return {toks[i] + " " + toks[i+1] for i in range(len(toks) - 1)}

def main():
    y_tot = collections.Counter()
    watch_year = {w: collections.Counter() for w in WATCH}
    base_bg = collections.Counter(); rec_bg = collections.Counter()
    base_n = 0; rec_n = 0
    mark_single = collections.Counter()               # recent single-marker doc counts
    mark_pair = collections.Counter()                 # recent pair co-occurrence doc counts
    rec_docs = 0
    seen = set()
    n = 0
    for line in open(INP):
        r = json.loads(line); b = r["id"].split("v")[0]
        if b in seen:
            continue
        seen.add(b)
        pub = r["published"]
        if not pub or len(pub) < 7:
            continue
        y = int(pub[:4])
        if y < 2015 or y > 2026:
            continue
        toks = TOK.findall((r.get("abstract") or "").lower())
        tset = set(toks)
        bg = bigrams(toks)
        y_tot[y] += 1
        for w in WATCH:
            if w in bg:
                watch_year[w][y] += 1
        if y in BASE_YEARS:
            base_n += 1; base_bg.update(bg)
        if y in REC_YEARS:
            rec_n += 1; rec_bg.update(bg)
            present = [m for m in MARKERS if m in tset]
            rec_docs += 1
            for m in present:
                mark_single[m] += 1
            for i in range(len(present)):
                for j in range(i + 1, len(present)):
                    a, c = sorted((present[i], present[j]))
                    mark_pair[(a, c)] += 1
        n += 1
        if n % 50000 == 0:
            print(f"  {n} ...", flush=True)
    print(f"processed {n} abstracts; base_n={base_n} rec_n={rec_n}")

    # ---- n-gram viewer time series (through 2025, full years) ----
    years = list(range(2015, 2026))
    rows = []
    for w in WATCH:
        for y in years:
            tot = y_tot[y]
            rows.append({"phrase": w, "year": y, "n": watch_year[w][y],
                         "freq": 100 * watch_year[w][y] / tot if tot else 0})
    pd.DataFrame(rows).to_csv(os.path.join(DATA, "ngram_watch.csv"), index=False)

    # ---- data-driven top-rising bigrams ----
    keep = set()
    for d in (base_bg, rec_bg):
        for k, v in d.items():
            if v >= MINC:
                keep.add(k)
    recs = []
    for k in keep:
        bf = base_bg[k] / base_n
        rf = rec_bg[k] / rec_n
        recs.append({"bigram": k, "base_freq": bf, "rec_freq": rf,
                     "abs_excess": rf - bf, "ratio": (rf + 1e-9) / (bf + 1e-9),
                     "base_n": base_bg[k], "rec_n": rec_bg[k]})
    bd = pd.DataFrame(recs).sort_values("ratio", ascending=False)
    bd.to_csv(os.path.join(DATA, "bigram_discovery.csv"), index=False)

    # ---- co-occurrence lift matrix among markers (recent) ----
    M = len(MARKERS)
    lift = np.full((M, M), np.nan)
    for i in range(M):
        for j in range(M):
            if i == j:
                continue
            a, c = sorted((MARKERS[i], MARKERS[j]))
            pij = mark_pair[(a, c)] / rec_docs
            pi = mark_single[MARKERS[i]] / rec_docs
            pj = mark_single[MARKERS[j]] / rec_docs
            if pi > 0 and pj > 0:
                lift[i, j] = pij / (pi * pj)
    np.savez(os.path.join(DATA, "cooccur.npz"), lift=lift,
             markers=np.array(MARKERS), single=np.array([mark_single[m] for m in MARKERS]),
             rec_docs=rec_docs)

    print("\nTop 25 rising bigrams (ratio, min count %d):" % MINC)
    for _, r in bd.head(25).iterrows():
        print(f"  {r['bigram']:22s} base={r['base_freq']*100:5.2f}% rec={r['rec_freq']*100:5.2f}%  x{r['ratio']:5.1f}")
    print("\nwrote ngram_watch.csv, bigram_discovery.csv, cooccur.npz")

if __name__ == "__main__":
    main()
