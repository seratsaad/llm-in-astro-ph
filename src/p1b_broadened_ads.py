#!/usr/bin/env python3
"""
P1b (referee round 4, point 1) -- broaden the ADS harvest.

The first harvest queried nine families of model terms. The hand-read
statements show writing disclosures that name tools we never queried, in
particular Grammarly, DeepL, QuillBot and Writefull, and templates phrased
around language editing rather than around a model name. Here we add those
and re-run, then fetch source statements for anything new.

The added queries are blind to abstract vocabulary, so the enlarged sample
stays outcome-blind and the calibration stays unbiased.
Appends to data/n1b_matched.json and data/n1b_snippets.jsonl.
"""
import json, os, time

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data")
import importlib.util
spec = importlib.util.spec_from_file_location("n1b", os.path.join(HERE, "n1b_calibration.py"))
n1b = importlib.util.module_from_spec(spec); spec.loader.exec_module(n1b)

EXTRA = {
    "editing_tools": ['"Grammarly"', '"DeepL"', '"QuillBot"', '"Writefull"',
                      '"Trinka"', '"Paperpal"', '"Language Tool"', '"LanguageTool"'],
    "lang_template": ['"to improve the language"', '"improve language and readability"',
                      '"to improve readability"', '"language and readability"',
                      '"polish the language"', '"polish the English"',
                      '"improve the English"', '"English editing"',
                      '"language editing"', '"AI-assisted copy editing"',
                      '"grammar and readability"', '"improve the writing"'],
    "ack_phrases": ['"acknowledge the use of ChatGPT"', '"we used ChatGPT to"',
                    '"ChatGPT was used to"', '"with the help of ChatGPT"',
                    '"assistance of ChatGPT"', '"used ChatGPT for"',
                    '"thank ChatGPT"', '"aid of ChatGPT"'],
}


def main():
    matched = json.load(open(os.path.join(DATA, "n1b_matched.json")))
    before = len(matched)
    ids_out = json.load(open(n1b.IDS_OUT))
    corpus = n1b.load_corpus()
    new = {}
    for fam, terms in EXTRA.items():
        for field in ("ack", "full"):
            q = f'{field}:(' + " OR ".join(terms) + ') database:astronomy'
            fam_ids = {}
            for year in (2023, 2024, 2025):
                try:
                    ids, nfound = n1b.ads_query(q, year)
                except Exception as e:
                    print(f"  {fam}/{field} {year}: FAILED {e}", flush=True)
                    continue
                for i in ids:
                    fam_ids.setdefault(i, year)
                print(f"  {fam}/{field} {year}: ADS {nfound}, arXiv ids {len(ids)}",
                      flush=True)
                time.sleep(0.3)
            ids_out["families"][f"{fam}_{field}"] = fam_ids
            for pid in fam_ids:
                if pid in corpus:
                    if pid not in matched:
                        new[pid] = True
                    matched.setdefault(pid, [])
                    if fam not in matched[pid]:
                        matched[pid].append(fam)
    json.dump(ids_out, open(n1b.IDS_OUT, "w"), indent=2)
    json.dump(matched, open(os.path.join(DATA, "n1b_matched.json"), "w"), indent=2)
    print(f"\nmatched papers {before} -> {len(matched)} ({len(new)} new)", flush=True)
    n1b.stage3(matched)


if __name__ == "__main__":
    main()
