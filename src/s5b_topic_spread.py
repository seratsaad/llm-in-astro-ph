#!/usr/bin/env python3
"""Stage 5b: separate new instruments from style words by topic spread.

A dictionary lookup is a poor discriminator here: "underscores" is missing from
the system word list while "vera", "roman" and "navy" are in it. The property
that actually distinguishes the two families is *concentration*. A new
instrument or survey enters through one or two subfields; a change in writing
style enters everywhere at once.

For each candidate word this measures the normalised entropy of its 2024-2026
document counts across the eight broad concept classes, together with the share
taken by its single largest class. Style words sit near entropy 1; instrument
names sit low.

Writes data/discovered_with_spread.csv.
"""
import collections
import gzip
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from common import KG, DATA, TOKEN_RE, YEAR_MIN, id_to_ym
from s1_fulltext_features import clean_lines, split_sections

SHARDS = ["2024-2025", "2026"]
N_CAND = 400


def candidates():
    d = pd.read_csv(os.path.join(DATA, "discovered_markers.csv"))
    d = d.sort_values("disc_ratio", ascending=False).head(N_CAND)
    return d


def process(args):
    shard, words = args
    words = set(words)
    path = os.path.join(KG, "papers_ocr_markdowns_by_year",
                        f"papers_ocr_markdowns_{shard}.jsonl.gz")
    topics = pd.read_parquet(os.path.join(DATA, "abstract_features.parquet"),
                             columns=["arxiv_id", "topic"])
    topic_of = dict(zip(topics.arxiv_id, topics.topic))
    counts = collections.defaultdict(collections.Counter)   # word -> topic -> ndocs
    per_topic_docs = collections.Counter()
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            aid = rec["arxiv_id"]
            ym = id_to_ym(aid)
            if ym is None or ym[0] < 2024:
                continue
            tp = topic_of.get(aid)
            if tp is None or tp < 0:
                continue
            text = rec.get("ocr_markdown") or ""
            if len(text) < 3000:
                continue
            toks = set()
            for name, chunk in split_sections(clean_lines(text)):
                if name in ("references", "frontmatter", "appendix"):
                    continue
                toks |= {t for t in TOKEN_RE.findall(chunk.lower()) if t in words}
            per_topic_docs[tp] += 1
            for w in toks:
                counts[w][tp] += 1
    print(f"[{shard}] done", flush=True)
    return {w: dict(c) for w, c in counts.items()}, dict(per_topic_docs)


def main():
    cand = candidates()
    words = cand.word.tolist()
    with Pool(len(SHARDS)) as pool:
        parts = pool.map(process, [(s, words) for s in SHARDS])

    counts = collections.defaultdict(collections.Counter)
    docs = collections.Counter()
    for c, d in parts:
        for w, tc in c.items():
            counts[w].update(tc)
        docs.update(d)

    n_topics = max(docs) + 1
    base = np.array([docs.get(i, 0) for i in range(n_topics)], dtype=float)
    base = base / base.sum()

    rows = []
    for w in words:
        c = np.array([counts[w].get(i, 0) for i in range(n_topics)], dtype=float)
        if c.sum() < 20:
            continue
        # rate per topic, normalised by how big each topic is
        rate = np.divide(c, np.maximum(base * c.sum(), 1e-9))
        p = rate / rate.sum()
        ent = -(p * np.log(np.maximum(p, 1e-12))).sum() / np.log(n_topics)
        rows.append(dict(word=w, n_docs=int(c.sum()), topic_entropy=ent,
                         top_share=float(p.max())))
    spread = pd.DataFrame(rows)
    out = cand.merge(spread, on="word", how="inner")
    out.to_csv(os.path.join(DATA, "discovered_with_spread.csv"), index=False)

    out = out.sort_values("disc_ratio", ascending=False)
    print(f"\n{len(out)} candidates with topic spread measured\n")
    print("--- broad-spread gainers (entropy > 0.97): style vocabulary ---")
    a = out[out.topic_entropy > 0.97].head(30)
    print(a[["word", "base_df", "disc_ratio", "val_ratio", "topic_entropy",
             "is_seed"]].round(3).to_string(index=False))
    print("\n--- concentrated gainers (entropy < 0.93): instruments/subfields ---")
    b = out[out.topic_entropy < 0.93].head(20)
    print(b[["word", "base_df", "disc_ratio", "val_ratio", "topic_entropy",
             "top_share"]].round(3).to_string(index=False))
    seeds = out[out.is_seed]
    if len(seeds):
        print(f"\nseed markers: median entropy {seeds.topic_entropy.median():.3f}")
    print("wrote data/discovered_with_spread.csv")


if __name__ == "__main__":
    main()
