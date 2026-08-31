#!/usr/bin/env python3
"""Stage 5: discover marker candidates from the full-text vocabulary.

A phrase selected because it rose after 2022 cannot be validated by showing
that same rise. The split here is temporal and preregistered:

  discovery   rank every word by its 2023-2024 excess over a background
              extrapolated from 2015-2019;
  validation  the ranking is frozen, then evaluated on 2025-2026, which the
              ranking never saw.

Specificity is imposed before ranking: a candidate must have a measurable and
flat pre-2020 background, and must be ordinary English rather than an
instrument, survey or proper noun (which is what a rise in "nircam" would be).

Writes data/discovered_markers.csv.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from common import DATA, MARKERS, CONTROL

BASE_YEARS = list(range(2015, 2020))
DISC_YEARS = [2023, 2024]
VAL_YEARS = [2025, 2026]
MIN_BASE_DF = 0.002          # must occur in >= 0.2% of pre-2020 papers
MAX_BASE_SLOPE = 0.06        # |d log df / d year| in the known-negative era


def english_vocabulary():
    """Ordinary English words, used to exclude instruments and proper nouns."""
    for path in ("/usr/share/dict/words", "/usr/dict/words"):
        if os.path.exists(path):
            with open(path, encoding="utf-8", errors="ignore") as f:
                return {w.strip().lower() for w in f if w.strip()}
    return None


def load_counts():
    docs = {}
    vocab = {}
    for path in glob.glob(os.path.join(DATA, "ft_vocab_*.json")):
        with open(path) as f:
            blob = json.load(f)
        for y, n in blob["year_docs"].items():
            docs[int(y)] = docs.get(int(y), 0) + int(n)
        for y, counts in blob["vocab"].items():
            d = vocab.setdefault(int(y), {})
            for w, c in counts.items():
                d[w] = d.get(w, 0) + c
    return docs, vocab


def main():
    docs, vocab = load_counts()
    years = sorted(docs)
    print("documents per year:", {y: docs[y] for y in years})

    words = set()
    for y in BASE_YEARS + DISC_YEARS + VAL_YEARS:
        words |= set(vocab.get(y, {}))
    print(f"vocabulary considered: {len(words)}")

    eng = english_vocabulary()
    if eng is None:
        print("WARNING: no system dictionary; the proper-noun filter is disabled")

    def df_of(w, y):
        return vocab.get(y, {}).get(w, 0) / docs[y] if docs.get(y) else np.nan

    rows = []
    for w in words:
        base = np.array([df_of(w, y) for y in BASE_YEARS], dtype=float)
        if not np.all(base > 0) or base.mean() < MIN_BASE_DF:
            continue
        slope = np.polyfit(BASE_YEARS, np.log(base), 1)[0]
        if abs(slope) > MAX_BASE_SLOPE:
            continue                       # already drifting before any LLM existed
        b = base.mean()
        disc = np.mean([df_of(w, y) for y in DISC_YEARS])
        val = np.mean([df_of(w, y) for y in VAL_YEARS if y in docs])
        # background extrapolated with the measured pre-2020 slope
        exp_disc = b * np.exp(slope * (np.mean(DISC_YEARS) - np.mean(BASE_YEARS)))
        exp_val = b * np.exp(slope * (np.mean(VAL_YEARS) - np.mean(BASE_YEARS)))
        rows.append(dict(word=w, base_df=b, base_slope=slope,
                         disc_df=disc, val_df=val,
                         disc_ratio=disc / exp_disc, val_ratio=val / exp_val,
                         in_dict=(eng is None or w in eng),
                         is_seed=w in MARKERS, is_control=w in CONTROL))

    d = pd.DataFrame(rows)
    d = d.sort_values("disc_ratio", ascending=False).reset_index(drop=True)
    d["disc_rank"] = np.arange(1, len(d) + 1)
    d.to_csv(os.path.join(DATA, "discovered_markers.csv"), index=False)
    print(f"\n{len(d)} words passed the pre-2020 stability screen")

    top_words = d[d.in_dict].head(40)
    top_nondict = d[~d.in_dict].head(15)
    print("\n--- top 40 ordinary-English gainers (discovery: 2023-24) ---")
    print(top_words[["word", "base_df", "disc_ratio", "val_ratio", "is_seed"]]
          .round(4).to_string(index=False))
    print("\n--- top non-dictionary gainers (instruments / proper nouns, excluded) ---")
    print(top_nondict[["word", "base_df", "disc_ratio", "val_ratio"]]
          .round(4).to_string(index=False))

    # Does the frozen discovery ranking hold up on data it never saw?
    sel = d[d.in_dict].head(100)
    keep = sel[sel.val_ratio > 1.2]
    print(f"\nheld-out validation: {len(keep)}/100 top candidates still show "
          f">1.2x excess in {VAL_YEARS}")
    ctrl = d[d.is_control]
    print("control words, discovery and validation ratios:")
    print(ctrl[["word", "disc_ratio", "val_ratio"]].round(3).to_string(index=False))
    seeds = d[d.is_seed]
    print(f"\nseed markers recovered by the screen: {len(seeds)}/{len(MARKERS)}; "
          f"median discovery rank {seeds.disc_rank.median():.0f} of {len(d)}")
    print("wrote data/discovered_markers.csv")


if __name__ == "__main__":
    main()
