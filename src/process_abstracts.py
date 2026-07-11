#!/usr/bin/env python3
"""
Single streaming pass over the harvested abstracts.

Produces:
  data/word_docfreq.parquet   -- per-(year) document frequency for every word that
                                  clears a min-count floor (for excess-word discovery)
  data/quarter_markers.csv    -- per-quarter doc-frequency for a hand-picked marker
                                  and control basket (for Fig 1 + delve-collapse)
  data/year_counts.csv        -- abstracts per year (denominators / Fig 4)

"Document frequency" = fraction of abstracts that contain the word at least once,
which is the standard quantity in marker-word / excess-word LLM studies.
"""
import json, os, re, collections
import pandas as pd

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data")
INP = os.path.join(DATA, "astroph_abstracts.jsonl")

TOKEN_RE = re.compile(r"[a-z]+")

# Hand-picked marker basket (from Liang 2024, Gray 2025, Kobak 2024).
# Grouped so we can show "strong/rare" markers separately from common ones.
MARKERS_STRONG = ["delve", "delves", "delving", "underscore", "underscores",
                  "underscoring", "intricate", "intricacies", "showcasing",
                  "showcases", "boasts", "tapestry", "realm", "realms",
                  "pivotal", "meticulous", "meticulously", "nuanced"]
MARKERS_SOFT = ["comprehensive", "leverage", "leveraging", "harness", "harnessing",
                "notably", "crucial", "robust", "seamless", "seamlessly",
                "garner", "encompass", "encompasses", "multifaceted", "insights",
                "unravel", "elucidate", "myriad", "plethora", "holistic"]
# Neutral control words -- astronomy content words that should NOT hockey-stick.
CONTROL = ["observed", "measured", "obtained", "presented", "galaxy", "stellar",
           "sample", "temperature", "redshift", "spectra"]

WATCH = sorted(set(MARKERS_STRONG + MARKERS_SOFT + CONTROL))
MIN_COUNT = 30  # min docs (in either baseline or recent) to keep a word for discovery

def qkey(published):
    # published like 2024-01-15T05:03:50Z
    y = int(published[:4]); m = int(published[5:7])
    q = (m - 1) // 3 + 1
    return y, q

def main():
    # per-year total docs, per-year doc-count per watched word,
    # per-quarter total + per-quarter watched word counts,
    # per-year full-vocab doc counts (for discovery)
    year_total = collections.Counter()
    quarter_total = collections.Counter()
    watch_year = collections.defaultdict(collections.Counter)      # word -> year -> n
    watch_quarter = collections.defaultdict(collections.Counter)   # word -> (y,q) -> n
    vocab_year = collections.defaultdict(collections.Counter)      # year -> word -> n
    seen = set()

    n = 0
    dup = 0
    with open(INP) as f:
        for line in f:
            r = json.loads(line)
            base = r["id"].split("v")[0]
            if base in seen:
                dup += 1
                continue
            seen.add(base)
            pub = r["published"]
            if not pub or len(pub) < 7:
                continue
            y = int(pub[:4]); yq = qkey(pub)
            if y < 2015 or y > 2026:
                continue
            toks = set(TOKEN_RE.findall((r.get("abstract") or "").lower()))
            year_total[y] += 1
            quarter_total[yq] += 1
            for w in toks:
                vocab_year[y][w] += 1
            for w in WATCH:
                if w in toks:
                    watch_year[w][y] += 1
                    watch_quarter[w][yq] += 1
            n += 1
            if n % 25000 == 0:
                print(f"  processed {n} ...", flush=True)

    print(f"processed {n} unique abstracts ({dup} duplicate versions skipped)")

    years = list(range(2015, 2027))
    # year_counts.csv
    pd.DataFrame({"year": years,
                  "n_abstracts": [year_total[y] for y in years]}).to_csv(
        os.path.join(DATA, "year_counts.csv"), index=False)

    # quarter_markers.csv (long format)
    rows = []
    for w in WATCH:
        grp = ("strong" if w in MARKERS_STRONG else
               "soft" if w in MARKERS_SOFT else "control")
        for (y, qn), tot in sorted(quarter_total.items()):
            cnt = watch_quarter[w].get((y, qn), 0)
            rows.append({"word": w, "group": grp, "year": y, "quarter": qn,
                         "yq": y + (qn - 1) / 4.0, "n": cnt, "total": tot,
                         "freq": cnt / tot if tot else 0.0})
    pd.DataFrame(rows).to_csv(os.path.join(DATA, "quarter_markers.csv"), index=False)

    # word_docfreq.parquet -- full vocab per-year doc freq, filtered by min count
    baseline_years = [2018, 2019, 2020, 2021]
    recent_years = [2024, 2025, 2026]
    keep = set()
    base_tot = sum(year_total[y] for y in baseline_years)
    rec_tot = sum(year_total[y] for y in recent_years)
    base_wc = collections.Counter()
    rec_wc = collections.Counter()
    for y in baseline_years:
        base_wc.update(vocab_year[y])
    for y in recent_years:
        rec_wc.update(vocab_year[y])
    allwords = set(base_wc) | set(rec_wc)
    recs = []
    for w in allwords:
        if len(w) < 3:
            continue
        b = base_wc[w]; rc = rec_wc[w]
        if b < MIN_COUNT and rc < MIN_COUNT:
            continue
        bf = b / base_tot
        rf = rc / rec_tot
        recs.append({"word": w, "base_n": b, "rec_n": rc,
                     "base_freq": bf, "rec_freq": rf,
                     "abs_excess": rf - bf,
                     "ratio": (rf + 1e-9) / (bf + 1e-9)})
    df = pd.DataFrame(recs).sort_values("abs_excess", ascending=False)
    df.to_parquet(os.path.join(DATA, "word_docfreq.parquet"), index=False)
    # also per-year for the watch words -> tidy csv
    wr = []
    for w in WATCH:
        for y in years:
            wr.append({"word": w, "year": y, "n": watch_year[w].get(y, 0),
                       "total": year_total[y],
                       "freq": watch_year[w].get(y, 0) / year_total[y] if year_total[y] else 0})
    pd.DataFrame(wr).to_csv(os.path.join(DATA, "year_markers.csv"), index=False)
    print("wrote year_counts.csv, quarter_markers.csv, year_markers.csv, word_docfreq.parquet")
    print(f"baseline docs={base_tot}, recent docs={rec_tot}, vocab kept={len(df)}")

if __name__ == "__main__":
    main()
