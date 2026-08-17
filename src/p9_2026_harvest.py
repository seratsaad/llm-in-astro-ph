#!/usr/bin/env python3
"""
P9 -- extend the disclosure harvest to the first half of 2026.

Runs every ADS query family from the earlier harvests (acknowledgment
families, full-text templates, editing tools, language-editing phrases) for
year 2026, matches against the corpus through June 2026, and fetches arXiv
source statements for the new papers. Appends to n1b_matched.json and
n1b_snippets.jsonl like the earlier stages.
"""
import json, os, re, time

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data")
import importlib.util
spec = importlib.util.spec_from_file_location("n1b", os.path.join(HERE, "n1b_calibration.py"))
n1b = importlib.util.module_from_spec(spec); spec.loader.exec_module(n1b)

QUERIES = dict(n1b.FAMILIES)
QUERIES["template"] = ['"During the preparation of this work"',
                       '"During the preparation of this manuscript"',
                       '"AI-assisted technologies"']
QUERIES["body_chatgpt"] = ['"we used ChatGPT"', '"ChatGPT was used"',
                           '"the authors used ChatGPT"', '"the author used ChatGPT"',
                           '"with the help of ChatGPT"', '"assistance of ChatGPT"',
                           '"help of GPT-4"', '"assistance of GPT-4"']
QUERIES["editing_tools"] = ['"Grammarly"', '"DeepL"', '"QuillBot"', '"Writefull"',
                            '"Trinka"', '"Paperpal"', '"LanguageTool"']
QUERIES["lang_template"] = ['"to improve the language"', '"improve language and readability"',
                            '"to improve readability"', '"polish the language"',
                            '"polish the English"', '"improve the English"',
                            '"English editing"', '"language editing"',
                            '"AI-assisted copy editing"', '"grammar and readability"']


def corpus_2026h1():
    out = {}
    for line in open(os.path.join(DATA, "astroph_abstracts.jsonl")):
        r = json.loads(line)
        y = int(r["published"][:4]); m = int(r["published"][5:7])
        if y == 2026 and m <= 6:
            out[re.sub(r"v\d+$", "", r["id"])] = True
    return out


def main():
    corpus = corpus_2026h1()
    print(f"2026 H1 corpus: {len(corpus)} papers", flush=True)
    matched = json.load(open(os.path.join(DATA, "n1b_matched.json")))
    before = len(matched)
    for fam, terms in QUERIES.items():
        for field in ("ack", "full"):
            q = f'{field}:(' + " OR ".join(terms) + ') database:astronomy'
            try:
                ids, nfound = n1b.ads_query(q, 2026)
            except Exception as e:
                print(f"  {fam}/{field}: FAILED {e}", flush=True)
                continue
            hits = [i for i in ids if i in corpus]
            print(f"  {fam}/{field} 2026: ADS {nfound}, matched {len(hits)}", flush=True)
            for pid in hits:
                matched.setdefault(pid, [])
                if fam not in matched[pid]:
                    matched[pid].append(fam)
            time.sleep(0.3)
    json.dump(matched, open(os.path.join(DATA, "n1b_matched.json"), "w"), indent=2)
    print(f"matched {before} -> {len(matched)}", flush=True)
    n1b.stage3(matched)


if __name__ == "__main__":
    main()
