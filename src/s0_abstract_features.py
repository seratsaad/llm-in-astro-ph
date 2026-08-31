#!/usr/bin/env python3
"""Stage 0 (Phase A): build the abstract feature table.

One row per paper: first-submission quarter, abstract length in tokens, per-word
counts for the frozen marker / control / placebo baskets, and a broad topic
class. Nothing here depends on any model.

Writes data/abstract_features.parquet and data/cohort_summary.json.
"""
import collections
import gzip
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from common import (KG, DATA, WATCH, MARKERS, CONTROL, PLACEBO, TOKEN_RE,
                    YEAR_MIN, id_to_ym, quarter_index, QUARTER_MAX)


def topic_classes():
    """Modal broad concept class per paper (8 classes in the KG vocabulary)."""
    vocab = pd.read_csv(os.path.join(KG, "concepts_vocabulary.csv.gz"))
    label_to_class = dict(zip(vocab["label"], vocab["class"]))
    classes = sorted(vocab["class"].unique())
    cls_idx = {c: i for i, c in enumerate(classes)}

    counts = collections.defaultdict(collections.Counter)
    mapping = pd.read_csv(os.path.join(KG, "papers_concepts_mapping.csv.gz"),
                          dtype={"arxiv_id": str})
    for aid, label in zip(mapping["arxiv_id"].values, mapping["label"].values):
        c = label_to_class.get(label)
        if c is not None:
            counts[aid][c] += 1
    modal = {aid: cls_idx[c.most_common(1)[0][0]] for aid, c in counts.items()}
    return modal, classes


def main():
    modal, classes = topic_classes()
    print(f"topic classes: {classes}", flush=True)

    watch_idx = {w: i for i, w in enumerate(WATCH)}
    qmax = quarter_index(*QUARTER_MAX)

    rows, counts_rows = [], []
    seen = set()
    n_read = n_kept = n_nodate = n_dup = n_notopic = 0

    with gzip.open(os.path.join(KG, "abstracts_all.jsonl.gz"), "rt") as f:
        for line in f:
            rec = json.loads(line)
            n_read += 1
            aid = rec["arxiv_id"]
            if aid in seen:
                n_dup += 1
                continue
            seen.add(aid)
            ym = id_to_ym(aid)
            if ym is None:
                n_nodate += 1
                continue
            year, month = ym
            if year < YEAR_MIN:
                continue
            qi = quarter_index(year, month)
            if qi > qmax or qi < 0:
                continue
            abstract = rec.get("abstract") or ""
            toks = TOKEN_RE.findall(abstract.lower())
            if len(toks) < 20:          # truncated / withdrawn records
                continue
            c = np.zeros(len(WATCH), dtype=np.int32)
            for t in toks:
                j = watch_idx.get(t)
                if j is not None:
                    c[j] += 1
            topic = modal.get(aid, -1)
            if topic < 0:
                n_notopic += 1
            rows.append((aid, year, month, qi, len(toks), topic))
            counts_rows.append(c)
            n_kept += 1
            if n_kept % 25000 == 0:
                print(f"  {n_kept} kept / {n_read} read", flush=True)

    df = pd.DataFrame(rows, columns=["arxiv_id", "year", "month", "q", "L", "topic"])
    C = np.vstack(counts_rows)
    for w in WATCH:
        df["w_" + w] = C[:, watch_idx[w]]

    df["K_marker"] = df[["w_" + w for w in MARKERS]].sum(axis=1)
    df["K_control"] = df[["w_" + w for w in CONTROL]].sum(axis=1)
    df["K_placebo"] = df[["w_" + w for w in PLACEBO]].sum(axis=1)
    df["any_marker"] = (df["K_marker"] > 0).astype(int)

    os.makedirs(DATA, exist_ok=True)
    out = os.path.join(DATA, "abstract_features.parquet")
    df.to_parquet(out, index=False)

    summary = {
        "n_read": n_read, "n_kept": n_kept, "n_duplicate": n_dup,
        "n_old_style_id": n_nodate, "n_without_topic": n_notopic,
        "topic_classes": classes,
        "median_length": float(df.L.median()),
        "per_year": df.year.value_counts().sort_index().to_dict(),
        "any_marker_by_year": df.groupby("year").any_marker.mean().round(5).to_dict(),
        "control_any_by_year": df.groupby("year").apply(
            lambda g: float((g.K_control > 0).mean()), include_groups=False).round(5).to_dict(),
        "placebo_any_by_year": df.groupby("year").apply(
            lambda g: float((g.K_placebo > 0).mean()), include_groups=False).round(5).to_dict(),
    }
    with open(os.path.join(DATA, "cohort_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2)[:2500])
    print(f"\nwrote {out}  shape={df.shape}")


if __name__ == "__main__":
    main()
