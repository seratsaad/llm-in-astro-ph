#!/usr/bin/env python3
"""
R7b -- journal-agnostic recheck of the r7 verification failures.
The first pass mapped journal names to ADS bibstems with regexes and its errors
(Statistical Science -> Sci, Nature Astronomy -> Natur, A&A Supplement -> A&A,
Soviet Astron. AJ -> AJ) produce false NOT_FOUNDs. Here each failure is
re-queried by volume + page + year window (+ first author when parsed), with
no journal constraint. Whatever still fails goes to hand inspection.
Output: data/r7b_recheck.jsonl, updated data/r7_summary.json
"""
import json, os, time, urllib.parse, urllib.request

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
TOKEN = open(os.path.expanduser("~/.ads/dev_key")).read().strip()
BASE = "https://api.adsabs.harvard.edu/v1/search/query"

def ads(q):
    url = BASE + "?" + urllib.parse.urlencode(
        {"q": q, "fl": "year,first_author,bibcode,pub", "rows": 5})
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {TOKEN}"})
    for a in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)["response"]["docs"]
        except Exception:
            time.sleep(2.0 * (a + 1))
    return None

def main():
    fails = []
    for line in open(os.path.join(DATA, "r7_refs.jsonl")):
        r = json.loads(line)
        if r.get("verdict") in ("NOT_FOUND", "YEAR_MISMATCH", "AUTHOR_MISMATCH"):
            fails.append(r)
    print(f"rechecking {len(fails)} failures", flush=True)
    out = open(os.path.join(DATA, "r7b_recheck.jsonl"), "w")
    tally = {}
    for r in fails:
        p = r["parsed"]
        parts = [f'volume:"{p["volume"]}"', f'page:"{p["page"]}"']
        if p["year"]:
            parts.append(f'year:{p["year"]-1}-{p["year"]+1}')
        if p["author"]:
            parts.append(f'author:"^{p["author"]}"')
        docs = ads(" ".join(parts))
        verdict = "STILL_FAILED"
        hit = None
        if docs:
            verdict = "VERIFIED_NOJOURNAL"
            hit = {"bibcode": docs[0].get("bibcode"), "pub": docs[0].get("pub"),
                   "first_author": docs[0].get("first_author")}
        elif docs is not None and p["author"]:
            # drop the author constraint (collaboration names, diacritics)
            docs2 = ads(" ".join(parts[:-1]))
            if docs2:
                verdict = "VERIFIED_VOLPAGE"
                hit = {"bibcode": docs2[0].get("bibcode"), "pub": docs2[0].get("pub"),
                       "first_author": docs2[0].get("first_author")}
        tally[verdict] = tally.get(verdict, 0) + 1
        out.write(json.dumps({"paper": r["paper"], "entry": r["entry"],
                              "parsed": p, "first_verdict": r["verdict"],
                              "recheck": verdict, "hit": hit}) + "\n")
        print(f"  {p['author']} {p['year']} {p['journal']} {p['volume']} {p['page']}"
              f" -> {verdict}" + (f" ({hit['pub'][:40]})" if hit else ""), flush=True)
        time.sleep(0.3)
    out.close()
    summ = json.load(open(os.path.join(DATA, "r7_summary.json")))
    summ["recheck"] = tally
    json.dump(summ, open(os.path.join(DATA, "r7_summary.json"), "w"), indent=2)
    print(tally)

if __name__ == "__main__":
    main()
