#!/usr/bin/env python3
"""Full-text marker measurement on YST's OCR corpus.
For each paper: marker incidence in the BODY (references stripped) and in the
ABSTRACT, so the two surfaces can be compared within the same paper."""
import gzip, json, re, sys, os, collections

KG = "/Users/saad.104/Downloads/astroph_kg/papers_ocr_markdowns_by_year"
OUT = os.path.join(os.path.dirname(__file__), "..", "data")

BASKET = set("""delve delves delving underscore underscores underscoring intricate
intricacies showcasing showcase showcases showcased boasts tapestry pivotal
meticulous meticulously nuanced garner garners garnered multifaceted commendable
noteworthy myriad plethora testament encompassing seamless seamlessly elucidate
elucidating unravel unraveling unravelling realm realms leveraging""".split())

REFS_RE = re.compile(r'^#{1,6}\s*(references|bibliography)\s*$', re.M | re.I)
ABS_RE  = re.compile(r'^#{1,6}\s*abstract\s*$', re.M | re.I)
HDR_RE  = re.compile(r'^#{1,6}\s+.*$', re.M)
WORD_RE = re.compile(r"[a-z]+")

def split_doc(md):
    """Return (abstract_text, body_text) with references removed."""
    m = REFS_RE.search(md)
    doc = md[:m.start()] if m else md
    a = ABS_RE.search(doc)
    if not a:
        return "", doc
    after = doc[a.end():]
    nxt = HDR_RE.search(after)
    abstract = after[:nxt.start()] if nxt else after[:3000]
    body = after[nxt.start():] if nxt else ""
    return abstract, body

def main(shards):
    out = open(os.path.join(OUT, "ft_body_markers.jsonl"), "w")
    n = 0
    for sh in shards:
        with gzip.open(os.path.join(KG, sh), "rt") as fh:
            for line in fh:
                r = json.loads(line)
                abstract, body = split_doc(r["ocr_markdown"])
                at = set(WORD_RE.findall(abstract.lower()))
                bt_all = WORD_RE.findall(body.lower())
                bt = set(bt_all)
                out.write(json.dumps({
                    "id": r["arxiv_id"],
                    "abs_hit": int(bool(at & BASKET)),
                    "body_hit": int(bool(bt & BASKET)),
                    "abs_n": len(abstract.split()),
                    "body_n": len(bt_all),
                    "abs_markers": sorted(at & BASKET),
                    "body_markers": sorted(bt & BASKET),
                    "body_marker_tokens": sum(1 for w in bt_all if w in BASKET),
                }) + "\n")
                n += 1
                if n % 5000 == 0:
                    print(f"  [{n}]", flush=True)
    out.close()
    print("wrote data/ft_body_markers.jsonl", n)

if __name__ == "__main__":
    main(sys.argv[1:])
