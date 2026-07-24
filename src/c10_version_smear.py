#!/usr/bin/env python3
"""
Referee check R3 -- the latest-version-abstract systematic.
The arXiv API returns the abstract of the latest posted version, keyed to the v1
submission date. A pre-2022 paper revised after ChatGPT can therefore carry recent
language back to its original date, which smears the marker signal backward in time
and biases the measured rise downward. Here we measure the size of that effect.

For a random sample of pre-2022 papers (submitted 2018-2021) we fetch the arXiv
entry, read its last-updated date, and split into papers last revised before 2023
and papers revised in 2023 or later. If the revised group has a higher marker rate
on its (current) abstract, that difference is the backward-smear.

Output: data/c10_version_smear.json
"""
import json, os, re, time, urllib.parse, urllib.request
import xml.etree.ElementTree as ET

HERE = os.path.dirname(__file__); DATA = os.path.join(HERE, "..", "data")
API = "https://export.arxiv.org/api/query"
NS = {"a": "http://www.w3.org/2005/Atom"}

MARKERS = {"delve", "delves", "delving", "underscore", "underscores", "underscoring",
           "intricate", "showcasing", "pivotal", "meticulous", "nuanced", "realm",
           "leveraging", "tapestry", "boasts", "garnered", "multifaceted", "garner",
           "multifaceted", "elucidate", "comprehensively", "aligns", "advancing"}

def has_marker(text):
    toks = set(re.findall(r"[a-z]+", text.lower()))
    return len(toks & MARKERS) > 0

def fetch_updated(arxiv_id):
    """Return (updated_year, abstract) for a base arXiv id."""
    url = API + "?" + urllib.parse.urlencode({"id_list": arxiv_id, "max_results": 1})
    req = urllib.request.Request(url, headers={"User-Agent": "smear-check/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        d = r.read().decode()
    e = ET.fromstring(d).find("a:entry", NS)
    if e is None:
        return None, None
    upd = e.findtext("a:updated", default="", namespaces=NS)[:4]
    pub = e.findtext("a:published", default="", namespaces=NS)[:4]
    abs = e.findtext("a:summary", default="", namespaces=NS)
    return (upd, pub, abs)

def main():
    # pull pre-2022 ids from the local corpus, deterministic stride sample
    ids = []
    for line in open(os.path.join(DATA, "astroph_abstracts.jsonl")):
        r = json.loads(line)
        y = int(r["published"][:4])
        if 2018 <= y <= 2021:
            ids.append(re.sub(r"v\d+$", "", r["id"]))
    ids = ids[::max(1, len(ids) // 3000)][:3000]   # ~3000 evenly spaced
    print(f"sampling {len(ids)} pre-2022 papers", flush=True)

    grp = {"revised_post2022": {"n": 0, "hit": 0}, "not_revised_post2022": {"n": 0, "hit": 0}}
    done = 0
    for aid in ids:
        try:
            upd, pub, abs_ = fetch_updated(aid)
        except Exception:
            time.sleep(1.0); continue
        if not upd or not abs_:
            continue
        key = "revised_post2022" if int(upd) >= 2023 else "not_revised_post2022"
        grp[key]["n"] += 1
        grp[key]["hit"] += 1 if has_marker(abs_) else 0
        done += 1
        if done % 200 == 0:
            a = grp["revised_post2022"]; b = grp["not_revised_post2022"]
            ra = 100 * a["hit"] / a["n"] if a["n"] else 0
            rb = 100 * b["hit"] / b["n"] if b["n"] else 0
            print(f"  {done}: revised {a['n']} ({ra:.2f}%)  not-revised {b['n']} ({rb:.2f}%)", flush=True)
        time.sleep(0.34)   # arXiv rate limit

    a = grp["revised_post2022"]; b = grp["not_revised_post2022"]
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
