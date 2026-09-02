#!/usr/bin/env python3
"""Reprocess the 2026 shard under the extended QUARTER_MAX and rebuild.

QUARTER_MAX was (2026, 2), which quarter_index reads as a month and maps to
2026Q1, so April onward never entered the corpus. With (2026, 6) the shard
covers through June. Only the 2026 shard can change, so the other four are
reused as they stand.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from common import DATA, MARKERS, CONTROL, PLACEBO
from s1_fulltext_features import process_shard, SHARDS

shard, n = process_shard("2026")
print(f"reprocessed {shard}: {n} papers", flush=True)

parts = [pd.read_parquet(os.path.join(DATA, f"ft_shard_{s}.parquet"))
         for s in SHARDS]
df = pd.concat(parts, ignore_index=True).drop_duplicates("arxiv_id")
df["K_marker"] = df[["w_" + w for w in MARKERS]].sum(axis=1)
df["K_control"] = df[["w_" + w for w in CONTROL]].sum(axis=1)
df["K_placebo"] = df[["w_" + w for w in PLACEBO]].sum(axis=1)
df.to_parquet(os.path.join(DATA, "fulltext_features.parquet"), index=False)
print("fulltext_features:", df.shape)
print(df[df.year >= 2025].groupby(["year", "month"]).size().to_string())
