#!/usr/bin/env python3
"""
Referee check R3 -- the latest-version-abstract systematic.
The arXiv API returns the abstract of the latest posted version, keyed to the v1
submission date. A pre-2022 paper revised after ChatGPT can therefore carry recent
language back to its original date, which smears the marker signal backward and
biases the measured rise downward. Here we measure the size of that effect.

For a random sample of pre-2022 papers (submitted 2018-2021) we read the arXiv
last-updated date and split into papers last revised before 2023 and papers revised
in 2023 or later. If the revised group has a higher marker rate on its (current)
abstract, that difference is the backward-smear.

Requests are batched (id_list up to 50 ids) with a polite delay and 429 backoff.
Output: data/c10_version_smear.json
"""
import json, os, re, time, urllib.parse, urllib.request
import xml.etree.ElementTree as ET

HERE = os.path.dirname(__file__); DATA = os.path.join(HERE, "..", "data")
API = "https://export.arxiv.org/api/query"
NS = {"a": "http://www.w3.org/2005/Atom"}
UA = {"User-Agent": "llm-astroph-smear-check/1.0 (mailto:rocketscience426@gmail.com)"}

MARKERS = {"delve", "delves", "delving", "underscore", "underscores", "underscoring",
           "intricate", "showcasing", "pivotal", "meticulous", "nuanced", "realm",
           "leveraging", "tapestry", "boasts", "garnered", "multifaceted", "garner",
           "elucidate", "comprehensively", "aligns", "advancing"}

def has_marker(text):
    return len(set(re.findall(r"[a-z]+", text.lower())) & MARKERS) > 0

def fetch_batch(ids):
    """Return list of (updated_year, abstract) for a batch of base arXiv ids."""
    url = API + "?" + urllib.parse.urlencode({"id_list": ",".join(ids), "max_results": len(ids)})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=90) as r:
                d = r.read().decode()
            entries = ET.fromstring(d).findall("a:entry", NS)
            out = []
            for e in entries:
                upd = e.findtext("a:updated", default="", namespaces=NS)[:4]
                ab = e.findtext("a:summary", default="", namespaces=NS)
                if upd and ab:
                    out.append((upd, ab))
            return out
        except urllib.error.HTTPError as ex:
            if ex.code == 429:
                time.sleep(15 * (attempt + 1)); continue
            time.sleep(5); continue
        except Exception:
            time.sleep(5); continue
    return []

def main():
    ids = []
    for line in open(os.path.join(DATA, "astroph_abstracts.jsonl")):
        r = json.loads(line)
        if 2018 <= int(r["published"][:4]) <= 2021:
            ids.append(re.sub(r"v\d+$", "", r["id"]))
    ids = ids[::max(1, len(ids) // 2500)][:2500]
    print(f"sampling {len(ids)} pre-2022 papers, batched", flush=True)

    grp = {"revised_post2022": {"n": 0, "hit": 0}, "not_revised_post2022": {"n": 0, "hit": 0}}
    B = 50
    for k in range(0, len(ids), B):
        for upd, ab in fetch_batch(ids[k:k + B]):
            key = "revised_post2022" if int(upd) >= 2023 else "not_revised_post2022"
            grp[key]["n"] += 1
            grp[key]["hit"] += 1 if has_marker(ab) else 0
        a, b = grp["revised_post2022"], grp["not_revised_post2022"]
        ra = 100 * a["hit"] / a["n"] if a["n"] else 0
        rb = 100 * b["hit"] / b["n"] if b["n"] else 0
        print(f"  {k+B}/{len(ids)}: revised {a['n']} ({ra:.2f}%)  not-revised {b['n']} ({rb:.2f}%)", flush=True)
        time.sleep(3.0)

    a, b = grp["revised_post2022"], grp["not_revised_post2022"]
    a["pct"] = 100 * a["hit"] / a["n"] if a["n"] else 0
    b["pct"] = 100 * b["hit"] / b["n"] if b["n"] else 0
    grp["smear_pp"] = a["pct"] - b["pct"]
    json.dump(grp, open(os.path.join(DATA, "c10_version_smear.json"), "w"), indent=2)
    print(f"\nrevised-post-2022:  {a['pct']:.2f}%  (n={a['n']})")
    print(f"not-revised:        {b['pct']:.2f}%  (n={b['n']})")
    print(f"backward-smear:     {grp['smear_pp']:+.2f} pp among pre-2022 papers")
    print("wrote c10_version_smear.json")

if __name__ == "__main__":
    main()
