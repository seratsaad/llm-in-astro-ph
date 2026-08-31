#!/usr/bin/env python3
"""Stage 5c: freeze the discovered astronomy basket and count it per paper.

Selection uses ONLY quantities available at discovery time:
  - pre-2020 background present and stable (imposed in Stage 5);
  - excess ratio in the discovery years 2023-2024 >= DISC_MIN;
  - topic entropy > ENT_MIN (style words spread across all eight classes;
    instruments and subfield fashions concentrate).
The validation years 2025-2026 are reported but play no part in selection.

Output: data/expanded_features.parquet with per-paper, per-section counts of
the frozen expanded basket, aligned with fulltext_features.parquet.
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
from common import (KG, DATA, MARKERS, CONTROL, PLACEBO, TOKEN_RE,
                    YEAR_MIN, id_to_ym, quarter_index, QUARTER_MAX)
from s1_fulltext_features import clean_lines, split_sections, BODY_SECTIONS

SHARDS = ["2012-2015", "2016-2019", "2020-2023", "2024-2025", "2026"]
DISC_MIN = 1.6
ENT_MIN = 0.97
# Topic entropy cannot catch words that rise everywhere for non-style reasons:
# author-name fragments appear in citations across every subfield (accented
# names lose their diacritics in tokenisation: Jimenez -> jim + nez), and some
# new facilities are named after people and used field-wide. The dictionary
# requirement removes the fragments; the stoplist names the facilities.
PROPER_NOUN_STOPLIST = {
    "vera", "rubin", "roman", "nancy", "grace", "webb", "euclid", "ariel",
    "plato", "twinkle", "toltec", "colibri", "winter", "summer", "asia",
    "alabama", "navy", "het", "chakraborty", "jiao", "patel", "guan",
    "weaver", "lacy", "ramsey", "gent", "predehl", "groth", "haro", "mast",
    "cook", "mart", "tran", "copyright",
}
SECS = ("introduction", "methods", "results", "discussion", "conclusions")
SHORT = {"introduction": "intro", "methods": "methods", "results": "results",
         "discussion": "discussion", "conclusions": "conclusions"}


def frozen_basket():
    d = pd.read_csv(os.path.join(DATA, "discovered_with_spread.csv"))
    sel = d[(d.disc_ratio >= DISC_MIN) & (d.topic_entropy > ENT_MIN)
            & d.in_dict & (~d.word.isin(PROPER_NOUN_STOPLIST))
            & (~d.word.isin(CONTROL)) & (~d.word.isin(PLACEBO))]
    words = sorted(set(sel.word))
    path = os.path.join(DATA, "expanded_basket.json")
    with open(path, "w") as f:
        json.dump({"selection": {"disc_min": DISC_MIN, "ent_min": ENT_MIN},
                   "n_words": len(words),
                   "n_new": len(set(words) - set(MARKERS)),
                   "words": words}, f, indent=1)
    print(f"frozen basket: {len(words)} words "
          f"({len(set(words) - set(MARKERS))} beyond the 38 seeds)")
    held = sel[sel.val_ratio > 1.2]
    print(f"of these, {len(held)} also show >1.2x excess in held-out 2025-26 "
          f"(reported, not used for selection)")
    return words


def process(args):
    shard, words = args
    wordset = set(words)
    path = os.path.join(KG, "papers_ocr_markdowns_by_year",
                        f"papers_ocr_markdowns_{shard}.jsonl.gz")
    qmax = quarter_index(*QUARTER_MAX)
    rows = []
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            aid = rec["arxiv_id"]
            ym = id_to_ym(aid)
            if ym is None or ym[0] < YEAR_MIN:
                continue
            qi = quarter_index(*ym)
            if qi < 0 or qi > qmax:
                continue
            text = rec.get("ocr_markdown") or ""
            if len(text) < 3000:
                continue
            cnt = collections.Counter()
            for name, chunk in split_sections(clean_lines(text)):
                if name not in SECS:
                    continue
                s = SHORT[name]
                for t in TOKEN_RE.findall(chunk.lower()):
                    if t in wordset:
                        cnt[s] += 1
            rows.append((aid, *[cnt[SHORT[s]] for s in SECS]))
    print(f"[{shard}] done n={len(rows)}", flush=True)
    return rows


def main():
    words = frozen_basket()
    with Pool(len(SHARDS)) as pool:
        parts = pool.map(process, [(s, words) for s in SHARDS])
    rows = [r for p in parts for r in p]
    df = pd.DataFrame(rows, columns=["arxiv_id"] +
                      [f"E_{SHORT[s]}" for s in SECS]).drop_duplicates("arxiv_id")
    df.to_parquet(os.path.join(DATA, "expanded_features.parquet"), index=False)
    print("expanded_features:", df.shape)


if __name__ == "__main__":
    main()
