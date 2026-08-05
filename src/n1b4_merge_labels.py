#!/usr/bin/env python3
"""Merge classification batch outputs into data/n1b_classified.json."""
import json, os, sys, glob, collections

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
SCRATCH = sys.argv[1]

years = {}
for line in open(os.path.join(DATA, "astroph_abstracts.jsonl")):
    r = json.loads(line)
    years[r["id"].split("v")[0]] = int(r["published"][:4])

labels = {}
for fn in sorted(glob.glob(os.path.join(SCRATCH, "labels_batch_*.json"))):
    for rec in json.load(open(fn)):
        labels[rec["id"]] = rec

purp_counts = collections.Counter()
tool_counts = collections.Counter()
for r in labels.values():
    for p in r["purposes"]:
        purp_counts[p] += 1
    tool_counts[r.get("tool", "generic")] += 1

writing = [i for i, r in labels.items() if "writing" in r["purposes"]]
writing_or_unclear = [i for i, r in labels.items()
                      if "writing" in r["purposes"] or r["purposes"] == ["unclear"]]
code_only = [i for i, r in labels.items() if r["purposes"] == ["code"]]
research_any = [i for i, r in labels.items() if "research" in r["purposes"]]
mention = [i for i, r in labels.items() if r["purposes"] == ["mention"]]

out = {"n_labeled": len(labels),
       "purpose_counts": dict(purp_counts),
       "tool_counts": dict(tool_counts),
       "writing_ids": writing,
       "writing_ids_2025": [i for i in writing if years.get(i) == 2025],
       "writing_or_unclear_ids": writing_or_unclear,
       "code_only_ids": code_only,
       "research_ids": research_any,
       "mention_ids": mention,
       "labels": labels}
json.dump(out, open(os.path.join(DATA, "n1b_classified.json"), "w"), indent=1)
print(f"labeled {len(labels)}: purposes {dict(purp_counts)}")
print(f"tools {dict(tool_counts)}")
print(f"writing {len(writing)} (2025: {len(out['writing_ids_2025'])}), "
      f"code-only {len(code_only)}, research {len(research_any)}, mention {len(mention)}")
