#!/usr/bin/env python3
"""
N1b stage 1b -- disclosure statements outside the acknowledgments.
Journal-mandated AI disclosures often sit in a dedicated statement section
(the standardized "During the preparation of this work the author(s) used ..."
template), which the ack: field misses. We search the ADS fulltext (full:)
with tight disclosure-template phrases, merge with the ack harvest, and fetch
arXiv source snippets for the newly found papers.
Appends to data/n1b_matched.json and data/n1b_snippets.jsonl.
"""
import json, os, re, time, urllib.parse, urllib.request

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data")
import importlib.util
spec = importlib.util.spec_from_file_location("n1b", os.path.join(HERE, "n1b_calibration.py"))
n1b = importlib.util.module_from_spec(spec); spec.loader.exec_module(n1b)

PHRASES = {
    "template": ['"During the preparation of this work"',
                 '"During the preparation of this manuscript"',
                 '"AI-assisted technologies"'],
    "body_chatgpt": ['"we used ChatGPT"', '"ChatGPT was used"',
                     '"the authors used ChatGPT"', '"the author used ChatGPT"',
                     '"with the help of ChatGPT"', '"assistance of ChatGPT"',
                     '"help of GPT-4"', '"assistance of GPT-4"'],
    "body_llm": ['"a large language model was used"',
                 '"assistance of a large language model"',
                 '"used a large language model to improve"',
                 '"language model for language editing"',
                 '"LLM was used to"', '"used an LLM to improve"'],
}

def main():
    matched = json.load(open(os.path.join(DATA, "n1b_matched.json")))
    ids_out = json.load(open(n1b.IDS_OUT))
    corpus = n1b.load_corpus()
    new = {}
    for fam, terms in PHRASES.items():
        q = 'full:(' + " OR ".join(terms) + ') database:astronomy'
        fam_ids = {}
        for year in (2023, 2024, 2025):
            ids, nfound = n1b.ads_query(q, year)
            for i in ids:
                fam_ids.setdefault(i, year)
            print(f"  {fam} {year}: ADS {nfound}, arXiv ids {len(ids)}", flush=True)
            time.sleep(0.3)
        ids_out["families"][fam] = fam_ids
        for pid in fam_ids:
            if pid in corpus:
                if pid not in matched:
                    new[pid] = True
                matched.setdefault(pid, [])
                if fam not in matched[pid]:
                    matched[pid].append(fam)
    json.dump(ids_out, open(n1b.IDS_OUT, "w"), indent=2)
    json.dump(matched, open(os.path.join(DATA, "n1b_matched.json"), "w"), indent=2)
    print(f"newly matched via body search: {len(new)}; total matched now {len(matched)}", flush=True)
    n1b.stage3(matched)

if __name__ == "__main__":
    main()
