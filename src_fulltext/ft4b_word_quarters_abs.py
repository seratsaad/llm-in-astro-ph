#!/usr/bin/env python3
"""
FT4 -- per-word quarterly token counts in the BODY of every paper, 2012-2026.

This is the series the word-level avoidance tests run on: for each of the 38
basket words (and 10 neutral controls), body tokens per quarter, plus the
total body words per quarter. Quarters come from the dated corpus (2015+);
2012-2014 enter as year bins for the pre-2022 baselines.
Output: data/ft_word_quarters_abs.json
"""
import gzip, json, re, os, csv, collections

KG   = "/Users/saad.104/Downloads/astroph_kg"
SH   = os.path.join(KG, "papers_ocr_markdowns_by_year")
DATA = os.path.join(os.path.dirname(__file__), "..", "data")

BASKET = """delve delves delving underscore underscores underscoring intricate
intricacies showcasing showcase showcases showcased boasts tapestry pivotal
meticulous meticulously nuanced garner garners garnered multifaceted commendable
noteworthy myriad plethora testament encompassing seamless seamlessly elucidate
elucidating unravel unraveling unravelling realm realms leveraging""".split()
CONTROL = """observed measured obtained presented galaxy stellar sample
temperature redshift spectra""".split()
WORDS = set(BASKET) | set(CONTROL)

REFS_RE = re.compile(r'^#{1,6}\s*(references|bibliography)\s*$', re.M | re.I)
ABS_RE  = re.compile(r'^#{1,6}\s*abstract\s*$', re.M | re.I)
HDR_RE  = re.compile(r'^#{1,6}\s+.*$', re.M)
WORD_RE = re.compile(r"[a-z]+")

SHARDS = ["papers_ocr_markdowns_2012-2015.jsonl.gz",
          "papers_ocr_markdowns_2016-2019.jsonl.gz",
          "papers_ocr_markdowns_2020-2023.jsonl.gz",
          "papers_ocr_markdowns_2024-2025.jsonl.gz",
          "papers_ocr_markdowns_2026.jsonl.gz"]


def main():
    month = {}
    for line in open(os.path.join(DATA, "astroph_abstracts.jsonl")):
        r = json.loads(line)
        month[re.sub(r'v\d+$', '', r["id"])] = r["published"][:7]
    year = {}
    with gzip.open(os.path.join(KG, "papers_year_mapping.csv.gz"), "rt") as fh:
        for row in csv.DictReader(fh):
            year[row["arxiv_id"]] = int(row["year"])

    counts = collections.defaultdict(collections.Counter)   # period -> word -> tokens
    words_tot = collections.Counter()                       # period -> body words
    npap = collections.Counter()
    n = 0
    for sh in SHARDS:
        print("shard:", sh, flush=True)
        with gzip.open(os.path.join(SH, sh), "rt") as fh:
            for line in fh:
                r = json.loads(line)
                aid = r["arxiv_id"].replace("astro-ph-", "")
                ym = month.get(aid)
                if ym:
                    y, m = int(ym[:4]), int(ym[5:7])
                    per = f"{y}.{(m-1)//3*25:02d}"          # quarter label, start month
                else:
                    y = year.get(r["arxiv_id"]) or year.get(aid)
                    if y is None or y > 2014:
                        continue
                    per = str(y)
                md = r["ocr_markdown"]
                mref = REFS_RE.search(md)
                doc = md[:mref.start()] if mref else md
                a = ABS_RE.search(doc)
                if not a:
                    continue
                after = doc[a.end():]
                nxt = HDR_RE.search(after)
                seg = after[:nxt.start()] if nxt else after[:3000]
                toks = WORD_RE.findall(seg.lower())
                words_tot[per] += len(toks)
                npap[per] += 1
                c = counts[per]
                for w in toks:
                    if w in WORDS:
                        c[w] += 1
                n += 1
                if n % 20000 == 0:
                    print(f"  [{n}]", flush=True)
    json.dump({"counts": {k: dict(v) for k, v in counts.items()},
               "body_words": dict(words_tot), "n_papers": dict(npap),
               "basket": BASKET, "control": CONTROL},
              open(os.path.join(DATA, "ft_word_quarters_abs.json"), "w"))
    print("DONE", n)


if __name__ == "__main__":
    main()
