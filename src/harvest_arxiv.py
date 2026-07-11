#!/usr/bin/env python3
"""
Harvest astro-ph abstracts from the arXiv API, month by month, 2015-2026.

Why month-chunked: the arXiv API cannot paginate reliably past ~30k results
for a single query, and it rate-limits. Chunking by submittedDate month keeps
every query to a few thousand results and lets us resume cleanly.

We keep the v1 submission date ("published" in the Atom feed) as the paper's
timestamp -- that is when it first appeared, which is what we want for a
time series of writing style.

Output: data/astroph_abstracts.jsonl  (one JSON object per paper, resumable)
"""
import json, time, sys, os, urllib.parse, urllib.request, xml.etree.ElementTree as ET
from datetime import date

OUT = os.path.join(os.path.dirname(__file__), "..", "data", "astroph_abstracts.jsonl")
PROGRESS = os.path.join(os.path.dirname(__file__), "..", "data", "harvest_progress.json")
API = "https://export.arxiv.org/api/query"
NS = {"a": "http://www.w3.org/2005/Atom",
      "os": "http://a9.com/-/spec/opensearch/1.1/",
      "arxiv": "http://arxiv.org/schemas/atom"}

PAGE = 1000           # results per request (arXiv max is ~2000; 1000 is safe)
SLEEP = 3.2           # polite delay between requests (arXiv asks for 3s)
MAX_RETRY = 5

def last_day(y, m):
    from calendar import monthrange
    return monthrange(y, m)[1]

def fetch(query, start, page):
    params = {
        "search_query": query,
        "start": str(start),
        "max_results": str(page),
        "sortBy": "submittedDate",
        "sortOrder": "ascending",
    }
    url = API + "?" + urllib.parse.urlencode(params)
    for attempt in range(MAX_RETRY):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "astrobites-llm-analysis/1.0 (mailto:seratmahmudsaad@gmail.com)"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception as e:
            wait = SLEEP * (attempt + 2)
            sys.stderr.write(f"  retry {attempt+1} after error: {e} (sleep {wait:.0f}s)\n")
            time.sleep(wait)
    raise RuntimeError(f"failed after {MAX_RETRY} retries: {url}")

def parse(xml_bytes):
    root = ET.fromstring(xml_bytes)
    tot_el = root.find("os:totalResults", NS)
    total = int(tot_el.text) if tot_el is not None else 0
    rows = []
    for e in root.findall("a:entry", NS):
        pid = e.findtext("a:id", default="", namespaces=NS)
        published = e.findtext("a:published", default="", namespaces=NS)
        title = (e.findtext("a:title", default="", namespaces=NS) or "").strip()
        summary = (e.findtext("a:summary", default="", namespaces=NS) or "").strip()
        prim = e.find("arxiv:primary_category", NS)
        primary = prim.get("term") if prim is not None else ""
        cats = [c.get("term") for c in e.findall("a:category", NS)]
        rows.append({
            "id": pid.rsplit("/", 1)[-1],
            "published": published,
            "primary_category": primary,
            "categories": cats,
            "title": title,
            "abstract": summary,
        })
    return total, rows

def load_progress():
    if os.path.exists(PROGRESS):
        return json.load(open(PROGRESS))
    return {"done_months": []}

def save_progress(p):
    json.dump(p, open(PROGRESS, "w"))

def main():
    prog = load_progress()
    done = set(tuple(x) for x in prog["done_months"])
    fout = open(OUT, "a")
    total_written = 0
    START = (2015, 1)
    END = (2026, 7)  # today is 2026-07; include partial July
    y, m = START
    while (y, m) <= END:
        if (y, m) in done:
            y, m = (y + 1, 1) if m == 12 else (y, m + 1)
            continue
        ld = last_day(y, m)
        q = f"cat:astro-ph* AND submittedDate:[{y}{m:02d}010000 TO {y}{m:02d}{ld:02d}2359]"
        start = 0
        got_month = 0
        month_total = None
        empty_retries = 0
        while True:
            data = fetch(q, start, PAGE)
            total, rows = parse(data)
            if month_total is None:
                month_total = total
            if not rows:
                # arXiv sometimes returns an empty page transiently; retry a few
                # times before concluding the month is actually exhausted.
                if start < (month_total or 0) and empty_retries < 4:
                    empty_retries += 1
                    time.sleep(SLEEP * 2)
                    continue
                break
            empty_retries = 0
            for row in rows:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()
            got_month += len(rows)
            total_written += len(rows)
            start += len(rows)
            time.sleep(SLEEP)
            if start >= total or len(rows) < PAGE:
                break
        sys.stderr.write(f"{y}-{m:02d}: wrote {got_month}/{month_total} (cum {total_written})\n")
        done.add((y, m))
        prog["done_months"] = sorted(list(done))
        save_progress(prog)
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    fout.close()
    sys.stderr.write(f"DONE. total abstracts written this run: {total_written}\n")

if __name__ == "__main__":
    main()
