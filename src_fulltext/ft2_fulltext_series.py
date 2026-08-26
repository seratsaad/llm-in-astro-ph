#!/usr/bin/env python3
"""
FT2 -- the full-baseline pass over YST's OCR corpus (433k papers, 1992-2026).

For every paper: segment the OCR markdown into abstract / body / references,
then count basket-word tokens, named-subset tokens, and neutral control-word
tokens in the body, and basket incidence in the abstract. Aggregate per
half-year (per year before 2015). Dates come from our own corpus where we
have them (month precision) and from the KG year mapping otherwise.

Controls carried through, so every caveat of the first quick look is testable:
  - neutral control words (drift in OCR quality / corpus composition)
  - named vs unnamed basket subsets (the mechanism test, per surface)
  - papers without an Abstract header are kept for the body series and
    flagged, so the segmentation cut can be tested for bias
Output: data/ft_series.json, data/ft_papers_2015plus.jsonl.gz
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
NAMED = set("""delve delves delving intricate pivotal showcasing realm
underscores intricacies meticulously""".split())
CONTROL = """observed measured obtained presented galaxy stellar sample
temperature redshift spectra""".split()
BSET, CSET = set(BASKET), set(CONTROL)

REFS_RE = re.compile(r'^#{1,6}\s*(references|bibliography)\s*$', re.M | re.I)
ABS_RE  = re.compile(r'^#{1,6}\s*abstract\s*$', re.M | re.I)
HDR_RE  = re.compile(r'^#{1,6}\s+.*$', re.M)
WORD_RE = re.compile(r"[a-z]+")


def split_doc(md):
    m = REFS_RE.search(md)
    doc = md[:m.start()] if m else md
    a = ABS_RE.search(doc)
    if not a:
        return None, doc            # no abstract header: body-only record
    after = doc[a.end():]
    nxt = HDR_RE.search(after)
    if nxt:
        return after[:nxt.start()], after[nxt.start():]
    return after[:3000], after[3000:]


def main():
    # month-precision dates from our harvested corpus (2015-2026)
    month = {}
    for line in open(os.path.join(DATA, "astroph_abstracts.jsonl")):
        r = json.loads(line)
        month[re.sub(r'v\d+$', '', r["id"])] = r["published"][:7]
    # year for everything else
    year = {}
    with gzip.open(os.path.join(KG, "papers_year_mapping.csv.gz"), "rt") as fh:
        for row in csv.DictReader(fh):
            year[row["arxiv_id"]] = int(row["year"])

    agg = collections.defaultdict(lambda: {
        "n": 0, "n_abs": 0, "abs_hit": 0, "abs_named_hit": 0,
        "body_words": 0, "body_hit": 0,
        "bt": 0, "bt_named": 0, "bt_ctrl": 0,
        "n_noabs": 0, "bt_noabs": 0, "bw_noabs": 0})
    out2 = gzip.open(os.path.join(DATA, "ft_papers_2015plus.jsonl.gz"), "wt")

    shards = sorted(f for f in os.listdir(SH) if f.endswith(".jsonl.gz"))
    n = 0
    for sh in shards:
        print("shard:", sh, flush=True)
        with gzip.open(os.path.join(SH, sh), "rt") as fh:
            for line in fh:
                r = json.loads(line)
                aid = r["arxiv_id"].replace("astro-ph-", "")
                ym = month.get(aid)
                if ym:
                    y = int(ym[:4])
                    per = f"{y}H1" if int(ym[5:7]) <= 6 else f"{y}H2"
                else:
                    y = year.get(r["arxiv_id"]) or year.get(aid)
                    if y is None:
                        continue
                    per = str(y)
                abstract, body = split_doc(r["ocr_markdown"])
                btoks = WORD_RE.findall(body.lower())
                bt = sum(1 for w in btoks if w in BSET)
                btn = sum(1 for w in btoks if w in NAMED)
                btc = sum(1 for w in btoks if w in CSET)
                d = agg[per]
                d["n"] += 1
                if abstract is None:
                    d["n_noabs"] += 1
                    d["bt_noabs"] += bt
                    d["bw_noabs"] += len(btoks)
                else:
                    at = set(WORD_RE.findall(abstract.lower()))
                    d["n_abs"] += 1
                    d["abs_hit"] += int(bool(at & BSET))
                    d["abs_named_hit"] += int(bool(at & NAMED))
                d["body_words"] += len(btoks)
                d["body_hit"] += int(bt > 0)
                d["bt"] += bt
                d["bt_named"] += btn
                d["bt_ctrl"] += btc
                if ym:
                    out2.write(json.dumps({
                        "id": aid, "ym": ym,
                        "abs_hit": (int(bool(at & BSET)) if abstract is not None else None),
                        "bt": bt, "btn": btn, "btc": btc,
                        "bw": len(btoks)}) + "\n")
                n += 1
                if n % 20000 == 0:
                    print(f"  [{n}]", flush=True)
    out2.close()
    json.dump(agg, open(os.path.join(DATA, "ft_series.json"), "w"), indent=1)
    print("DONE", n, "papers")


if __name__ == "__main__":
    main()
