#!/usr/bin/env python3
"""
P13 (round 5, point 9 and the clustering minor) -- first-author geography and
a cluster-robust check on the headline incidence.

Fetches first-author affiliation and first-author name for every 2025 paper
from ADS in identifier batches. Produces:
  (a) per-country marker rates by FIRST-author country, the interpretable
      version of the equity figure,
  (b) a first-author-clustered bootstrap on the 2025 incidence, answering the
      independence concern about paper series from the same group.
Output: data/p13_firstauthor.json
"""
import json, os, re, time, urllib.parse, urllib.request
import numpy as np

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
TOKEN = open(os.path.expanduser("~/.ads/dev_key")).read().strip()
BASE = "https://api.adsabs.harvard.edu/v1/search/query"
CACHE = os.path.join(DATA, "p13_aff_cache.jsonl")

BASKET = set("""delve delves delving underscore underscores underscoring intricate
intricacies showcasing showcase showcases showcased boasts tapestry pivotal
meticulous meticulously nuanced garner garners garnered multifaceted commendable
noteworthy myriad plethora testament encompassing seamless seamlessly elucidate
elucidating unravel unraveling unravelling realm realms leveraging""".split())

COUNTRIES = ["USA", "United States", "China", "Germany", "United Kingdom", "UK",
             "Italy", "France", "Japan", "Spain", "India", "Australia", "Canada",
             "Netherlands", "Switzerland", "Brazil", "Korea", "Russia", "Iran",
             "Sweden", "Poland", "Austria", "Belgium", "Mexico", "Chile",
             "Portugal", "Israel", "Taiwan", "Denmark", "Norway", "Finland",
             "Turkey", "Greece", "Czech", "Hungary", "Argentina", "South Africa",
             "Ireland", "New Zealand", "Scotland", "England"]
ALIAS = {"United States": "USA", "UK": "United Kingdom",
         "England": "United Kingdom", "Scotland": "United Kingdom"}


def country_of(aff):
    for c in COUNTRIES:
        if re.search(r"\b" + re.escape(c) + r"\b", aff, re.I):
            return ALIAS.get(c, c)
    return None


def main():
    papers = {}
    for line in open(os.path.join(DATA, "astroph_abstracts.jsonl")):
        r = json.loads(line)
        if r["published"][:4] != "2025":
            continue
        b = re.sub(r"v\d+$", "", r["id"])
        papers[b] = int(bool(set(re.findall(r"[a-z]+", r["abstract"].lower())) & BASKET))

    have = {}
    if os.path.exists(CACHE):
        for line in open(CACHE):
            rec = json.loads(line)
            have[rec["id"]] = rec
    todo = [p for p in papers if p not in have]
    print(f"2025 papers {len(papers)}, cached {len(have)}, to fetch {len(todo)}",
          flush=True)
    fout = open(CACHE, "a")
    for i in range(0, len(todo), 40):
        chunk = todo[i:i + 40]
        q = "identifier:(" + " OR ".join(f'"arXiv:{x}"' for x in chunk) + ")"
        url = BASE + "?" + urllib.parse.urlencode(
            {"q": q, "fl": "identifier,aff,first_author", "rows": 50})
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {TOKEN}"})
        for a in range(4):
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    docs = json.load(r)["response"]["docs"]
                break
            except Exception:
                time.sleep(2.0 * (a + 1))
        else:
            continue
        for d in docs:
            pid = None
            for ident in d.get("identifier", []):
                m = re.match(r"arXiv:(\d{4}\.\d{4,5})$", ident)
                if m and m.group(1) in set(chunk):
                    pid = m.group(1)
            if not pid:
                continue
            affs = d.get("aff") or []
            rec = {"id": pid, "aff0": (affs[0] if affs else ""),
                   "fa": d.get("first_author", "")}
            fout.write(json.dumps(rec) + "\n")
            have[pid] = rec
        fout.flush()
        if (i // 40) % 25 == 0:
            print(f"  [{i}/{len(todo)}]", flush=True)
        time.sleep(0.25)
    fout.close()

    # (a) first-author country rates
    cc = {}
    for pid, rec in have.items():
        if pid not in papers:
            continue
        c = country_of(rec.get("aff0") or "")
        if not c:
            continue
        d = cc.setdefault(c, [0, 0])
        d[1] += 1
        d[0] += papers[pid]
    rows = {c: {"n": n, "marker": k, "pct": k / n * 100}
            for c, (k, n) in cc.items() if n >= 200}
    print("\nfirst-author country marker rates (n >= 200):")
    for c, v in sorted(rows.items(), key=lambda kv: -kv[1]["pct"]):
        print(f"  {c:15s} {v['pct']:5.2f}%  (n={v['n']})")

    # (b) first-author-clustered bootstrap of the 2025 incidence
    clusters = {}
    for pid, rec in have.items():
        if pid in papers:
            clusters.setdefault(rec.get("fa") or pid, []).append(papers[pid])
    keys = list(clusters)
    vals = [clusters[k] for k in keys]
    rng = np.random.default_rng(5)
    boots = []
    for _ in range(2000):
        idx = rng.integers(0, len(keys), len(keys))
        s = [x for j in idx for x in vals[j]]
        boots.append(np.mean(s) * 100)
    naive = np.mean([papers[p] for p in have if p in papers]) * 100
    lo, hi = np.percentile(boots, [16, 84])
    print(f"\n2025 incidence {naive:.2f}%  cluster boot 68% [{lo:.2f}, {hi:.2f}]"
          f"  ({len(keys)} first-author clusters)")
    json.dump({"country_rows": rows,
               "cluster_boot": {"incidence": naive, "ci68": [lo, hi],
                                "n_clusters": len(keys)}},
              open(os.path.join(DATA, "p13_firstauthor.json"), "w"), indent=1,
              default=float)


if __name__ == "__main__":
    main()
