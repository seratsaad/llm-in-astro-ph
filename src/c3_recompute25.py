#!/usr/bin/env python3
"""
Citation audit restricted to papers submitted through 2025 (drops the 2026 bin).
arXiv side: recount instances/unique/checkable from c3_refs.jsonl (all previously
verified to resolve). DOI side: fresh 2,500-DOI sample (seed 42) from <=2025 papers,
checked against Crossref, misses auto-classified with the buckets established by the
earlier hand inspection. Output: data/c3_stats25.json
"""
import json, os, re, time, urllib.parse, urllib.request

HERE = os.path.dirname(__file__); DATA = os.path.join(HERE, "..", "data")
UA = {"User-Agent": "c3-recheck/1.0 (mailto:rocketscience426@gmail.com)"}

KNOWN_ID_ERRORS = {"10.3847/1538-4357/ac082c", "10.3847/2041-8213/ace280",
                   "10.1142/9789812834300", "10.5555/3294771.3294994",
                   "10.11648/j.xxxx.2025xxxx.xx"}
KNOWN_ARTIFACTS = {"10.1086/31138"}

def crossref_ok(doi):
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi, safe="")
    req = urllib.request.Request(url, headers=UA)
    try:
        with urllib.request.urlopen(req, timeout=40) as r:
            return r.status == 200
    except urllib.error.HTTPError as e:
        if e.code == 404: return False
        return None
    except Exception:
        return None

def handle_ok(doi):
    url = "https://doi.org/api/handles/" + urllib.parse.quote(doi, safe="/")
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
            return json.load(r).get("responseCode") == 1
    except Exception:
        return False

def bucket(d):
    dl = d.lower()
    if any(e in dl for e in KNOWN_ID_ERRORS): return "wrong identifier, real reference"
    if any(a in dl for a in KNOWN_ARTIFACTS) or ")/doi(" in dl or "10.48550/arxiv" in dl[8:]:
        return "extraction artifact"
    if dl.startswith("10.48550/arxiv"): return "arXiv DataCite (real)"
    if dl.startswith("10.5281/zenodo"): return "Zenodo (real)"
    if dl.startswith(("10.17909","10.26131","10.25574","10.18434","10.7910","10.21234",
                      "10.11570","10.5303","10.11316","10.18727","10.13039","10.71929",
                      "10.6084","10.25572","10.7936")): return "data archive / regional / funder (real)"
    return "OTHER"

def main():
    import math, random
    papers = 0; inst = 0; uniq = set(); checkable = 0; dois = set()
    for line in open(os.path.join(DATA, "c3_refs.jsonl")):
        r = json.loads(line)
        if not r.get("ok") or r["year"] > 2025: continue
        papers += 1
        new = [a for a in r.get("arxiv", []) if re.fullmatch(r"\d{4}\.\d{4,5}", a)]
        inst += len(new); uniq.update(new)
        if new: checkable += 1
        for d in r.get("doi", []):
            dois.add(d.lower().rstrip(".,;)}]"))
    print(f"papers<=2025={papers}  arxiv instances={inst}  unique={len(uniq)}  checkable papers={checkable}  unique DOIs={len(dois)}", flush=True)

    rng = random.Random(42)
    sample = rng.sample(sorted(dois), min(2500, len(dois)))
    missing = []
    for i, d in enumerate(sample):
        ok = crossref_ok(d)
        if ok is False: missing.append(d)
        if (i+1) % 250 == 0:
            print(f"  doi {i+1}/{len(sample)}  missing {len(missing)}", flush=True)
        time.sleep(0.55)

    cls = {}
    other = []
    for d in missing:
        b = bucket(d)
        cls[b] = cls.get(b, 0) + 1
        if b == "OTHER": other.append(d)
    # try the global handle for OTHERs (non-Crossref registries resolve there)
    other_resolved = 0; other_unresolved = []
    for d in other:
        if handle_ok(d): other_resolved += 1
        else: other_unresolved.append(d)
        time.sleep(0.3)
    cls.pop("OTHER", None)
    cls["other registry, resolves via handle (real)"] = cls.get("other registry, resolves via handle (real)", 0) + other_resolved

    ul = 3 / checkable * 100
    lam25 = checkable / 458; lam26 = checkable / 277
    out = {"papers": papers, "arxiv_instances": inst, "arxiv_unique": len(uniq),
           "checkable_papers": checkable, "unique_dois": len(dois),
           "doi_checked": len(sample), "doi_missing": len(missing),
           "classification": cls, "unresolved_inspect": other_unresolved,
           "per_paper_UL_pct": ul,
           "P0_biomed2025": math.exp(-lam25), "P0_biomed2026": math.exp(-lam26)}
    json.dump(out, open(os.path.join(DATA, "c3_stats25.json"), "w"), indent=2)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
