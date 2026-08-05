#!/usr/bin/env python3
"""Slice n1b_snippets.jsonl into classification batches for review."""
import json, os, sys

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
SCRATCH = sys.argv[1]
NB = int(sys.argv[2]) if len(sys.argv) > 2 else 6

recs = []
years = {}
for line in open(os.path.join(DATA, "astroph_abstracts.jsonl")):
    r = json.loads(line)
    base = r["id"].split("v")[0]
    years[base] = r["published"][:4]

nosrc = []
for line in open(os.path.join(DATA, "n1b_snippets.jsonl")):
    r = json.loads(line)
    if not r.get("ok") or not r.get("snippets"):
        nosrc.append(r["id"])
        continue
    sn = sorted(r["snippets"], key=lambda s: not s["in_ack_region"])
    recs.append({"id": r["id"], "year": years.get(r["id"], "?"),
                 "families": r["families"],
                 "snippets": [{"in_ack": s["in_ack_region"],
                               "text": s["snippet"][:500]} for s in sn[:6]]})
recs.sort(key=lambda r: r["id"])
per = (len(recs) + NB - 1) // NB
for b in range(NB):
    chunk = recs[b * per:(b + 1) * per]
    fn = os.path.join(SCRATCH, f"classify_batch_{b}.json")
    json.dump(chunk, open(fn, "w"), indent=1)
    print(fn, len(chunk))
json.dump(nosrc, open(os.path.join(SCRATCH, "no_source_ids.json"), "w"))
print("no-source papers:", len(nosrc))
