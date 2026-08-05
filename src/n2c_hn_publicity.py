#!/usr/bin/env python3
"""
N2c -- second, independent publicity measure from Hacker News (Algolia API).
For each word in the decay set, count HN stories and comments that contain the
word together with an AI anchor term, per quarter 2023-2025. Same mechanical
criterion for every word, no manual labeling. The Algolia API is free and
unthrottled at this volume.
Output: data/n2c_hn_publicity.json
"""
import json, os, time, urllib.parse, urllib.request

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
API = "https://hn.algolia.com/api/v1/search"
UA = {"User-Agent": "academic-corpus-study/1.0"}

def quarters():
    out = []
    for y in (2023, 2024, 2025):
        for qi in range(4):
            t0 = time.mktime((y, 1 + 3 * qi, 1, 0, 0, 0, 0, 0, 0))
            if qi == 3:
                t1 = time.mktime((y + 1, 1, 1, 0, 0, 0, 0, 0, 0))
            else:
                t1 = time.mktime((y, 1 + 3 * (qi + 1), 1, 0, 0, 0, 0, 0, 0))
            out.append((y + 0.25 * qi, int(t0), int(t1)))
    return out

def count(query, t0, t1):
    url = API + "?" + urllib.parse.urlencode(
        {"query": query, "tags": "(story,comment)",
         "numericFilters": f"created_at_i>={t0},created_at_i<{t1}",
         "hitsPerPage": 0})
    req = urllib.request.Request(url, headers=UA)
    for a in range(4):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)["nbHits"]
        except Exception:
            time.sleep(2.0 * (a + 1))
    return None

def main():
    dec = json.load(open(os.path.join(DATA, "n2b_publicity.json"))) \
        if os.path.exists(os.path.join(DATA, "n2b_publicity.json")) else None
    if dec:
        words = sorted(dec["decay"].keys())
    else:
        words = """delve delves delving underscore underscores underscoring intricate
intricacies showcasing showcase showcases showcased boasts tapestry pivotal
meticulous meticulously nuanced garner garners garnered multifaceted commendable
noteworthy myriad plethora testament encompassing seamless seamlessly elucidate
elucidating unravel unraveling unravelling realm realms leveraging offering
highlighting""".split()
    qs = quarters()
    out = {}
    for w in words:
        series = {}
        for qk, t0, t1 in qs:
            n = count(f'{w} ChatGPT', t0, t1)
            series[str(qk)] = n
            time.sleep(0.35)
        total = sum(v for v in series.values() if v)
        peak_q = max(series, key=lambda k: series[k] or 0)
        out[w] = {"quarterly": series, "total": total,
                  "peak_q": float(peak_q), "peak": series[peak_q]}
        print(f"{w:14s} total {total:5d}  peak {series[peak_q]} @ {peak_q}", flush=True)
    json.dump(out, open(os.path.join(DATA, "n2c_hn_publicity.json"), "w"), indent=2)

if __name__ == "__main__":
    main()
