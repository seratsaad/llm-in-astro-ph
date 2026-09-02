#!/usr/bin/env python3
"""Re-scan the OCR for writing-assistance declarations only.

Tokenisation and marker counting are unaffected by the declaration regexes, so
this pass skips them entirely and only re-derives D_i. Unlike the main pass it
examines *every* matching sentence rather than the first, so a false-positive
sentence early in a paper cannot mask a genuine declaration later.

Updates the `declared` column of data/fulltext_features.parquet in place and
rewrites data/declarations.csv.
"""
import gzip
import json
import os
import sys
from multiprocessing import Pool

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from common import KG, DATA, YEAR_MIN, id_to_ym, quarter_index, QUARTER_MAX
from s1_fulltext_features import (SHARDS, TOOL_RE, LLM_PHRASE_RE, WRITING_RE,
                                  USE_RE, SUBJECT_RE, SENT_SPLIT_RE,
                                  LLM_SUBJECT_MENTIONS)
import re


def scan(text):
    """Return (declared, evidence) considering every candidate sentence."""
    n = len(TOOL_RE.findall(text)) + len(LLM_PHRASE_RE.findall(text))
    if n == 0:
        return 0, ""
    about_llms = n > LLM_SUBJECT_MENTIONS
    near_miss = ""
    for sent in SENT_SPLIT_RE.split(text):
        if not (20 < len(sent) < 500):
            continue
        if not (TOOL_RE.search(sent) or LLM_PHRASE_RE.search(sent)):
            continue
        if not WRITING_RE.search(sent):
            continue
        if SUBJECT_RE.search(sent):
            continue
        if about_llms and not re.search(r"\b(we|authors?|manuscript|this paper)\b",
                                        sent, re.I):
            continue
        clean = " ".join(sent.split())[:400]
        if USE_RE.search(sent):
            return 1, clean            # keep looking only until a real one
        near_miss = near_miss or clean
    return 0, near_miss


def process(shard):
    path = os.path.join(KG, "papers_ocr_markdowns_by_year",
                        f"papers_ocr_markdowns_{shard}.jsonl.gz")
    qmax = quarter_index(*QUARTER_MAX)
    out = []
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
            d, ev = scan(text)
            if ev:
                out.append({"arxiv_id": aid, "year": ym[0],
                            "declared": d, "evidence": ev})
    print(f"[{shard}] done, {sum(r['declared'] for r in out)} declared", flush=True)
    return out


def main():
    with Pool(len(SHARDS)) as pool:
        parts = pool.map(process, SHARDS)
    rows = [r for p in parts for r in p]
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(DATA, "declarations.csv"), index=False)
    print(f"candidates {len(d)}, declared {int(d.declared.sum())}")
    print(d[d.declared == 1].groupby("year").size().to_string())

    ft_path = os.path.join(DATA, "fulltext_features.parquet")
    ft = pd.read_parquet(ft_path)
    pos = set(d.loc[d.declared == 1, "arxiv_id"])
    ft["declared"] = ft.arxiv_id.isin(pos).astype(int)
    ft.to_parquet(ft_path, index=False)
    print("updated declared column:", int(ft.declared.sum()))


if __name__ == "__main__":
    main()


def apply_judge_overrides(df):
    """Zero the declared flag for ids the language-model judge audit rejected."""
    p = os.path.join(DATA, "declaration_judge_overrides.csv")
    if os.path.exists(p):
        bad = set(pd.read_csv(p, dtype={"arxiv_id": str}).arxiv_id)
        df.loc[df.arxiv_id.isin(bad), "declared"] = 0
    return df
