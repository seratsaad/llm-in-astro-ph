#!/usr/bin/env python3
"""
C3 -- Hallucinated-citation audit. Extract arXiv IDs (and DOIs) CITED in astro-ph
LaTeX source, then check whether each one actually resolves. LLM-fabricated references
look well-formed but point to non-existent papers. Astronomy is uniquely checkable:
ADS/arXiv are near-complete citation authorities.

Step 1 (this script, --collect): download source for a per-year sample, extract cited
arXiv IDs + DOIs -> data/c3_refs.jsonl
Step 2 (--verify): batch-check arXiv IDs against the arXiv API and DOIs against Crossref,
then report the non-resolving ("candidate hallucination") rate by year.
"""
import json, os, re, sys, time, urllib.parse, urllib.request
import harvest_source as H   # reuse fetch(), extract_text(), parallel machinery
from concurrent.futures import ThreadPoolExecutor

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
REFS = os.path.join(DATA, "c3_refs.jsonl")
VER = os.path.join(DATA, "c3_verify.json")

# new-style 2501.12345 and old-style astro-ph/0601001
ARXIV_NEW = re.compile(r"arxiv[:\s]*?(\d{4}\.\d{4,5})", re.I)
ARXIV_OLD = re.compile(r"(astro-ph|gr-qc|hep-ph|hep-th|math|cond-mat|physics)/(\d{7})", re.I)
DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+")

SAMPLE = {2023: 300, 2024: 450, 2025: 600, 2026: 450}  # 1800 recent papers: enough to test the biomedical per-paper rate (needs ~1400 for a 95% exclusion)

def collect():
    import threading
    H.SAMPLE = SAMPLE
    H.MAX_BYTES = 25_000_000   # raise cap: references live in tiny .bbl, but need the tarball
    done = set()
    if os.path.exists(REFS):
        done = {json.loads(l)["id"] for l in open(REFS)}
    targets = [(y, pid) for (y, pid) in H.sample_ids() if pid not in done]
    targets.sort(key=lambda t: (-t[0], t[1]))
    print(f"collect refs from {len(targets)} papers", flush=True)
    lock = threading.Lock(); fout = open(REFS, "a"); n = [0]

    def work(args):
        y, pid = args
        data, ct = H.fetch(pid)
        rec = {"id": pid, "year": y, "ok": data is not None}
        if data is not None:
            full, _ = H.extract_text(data)
            axv = set(m.lower() for m in ARXIV_NEW.findall(full))
            axv |= set(f"{a.lower()}/{b}" for a, b in ARXIV_OLD.findall(full))
            dois = set(d.rstrip(".,;)}").lower() for d in DOI_RE.findall(full))
            rec["arxiv"] = sorted(axv)
            rec["doi"] = sorted(dois)[:200]
            rec["n_arxiv"] = len(axv); rec["n_doi"] = len(dois)
        return rec

    with ThreadPoolExecutor(max_workers=H.WORKERS) as ex:
        for rec in ex.map(work, targets):
            with lock:
                fout.write(json.dumps(rec) + "\n"); fout.flush(); n[0] += 1
                if n[0] % 100 == 0:
                    print(f"  {n[0]}/{len(targets)}", flush=True)
    fout.close(); print("collect done", flush=True)

def arxiv_exists_batch(ids):
    """Return set of ids that EXIST (arXiv API id_list, up to 100 per call)."""
    exist = set()
    for i in range(0, len(ids), 80):
        chunk = ids[i:i+80]
        url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(
            {"id_list": ",".join(chunk), "max_results": len(chunk)})
        req = urllib.request.Request(url, headers=H.UA)
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                txt = r.read().decode("utf-8", "ignore")
            for cid in chunk:
                # arXiv returns an <entry> with the id if it exists
                if f"/abs/{cid}" in txt or f"arXiv.org/abs/{cid}" in txt or cid in txt:
                    exist.add(cid)
        except Exception as e:
            sys.stderr.write(f"arxiv batch err: {e}\n")
        time.sleep(3.1)
    return exist

def clean_doi(d):
    d = d.rstrip(".,;)}]").lower()
    # strip common LaTeX/url trailing artifacts
    for junk in ["\\", "{", "}", "%"]:
        d = d.split(junk)[0]
    return d

def crossref_exists(doi):
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi, safe="")
    req = urllib.request.Request(url + "?mailto=seratmahmudsaad@gmail.com", headers=H.UA)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status == 200
    except urllib.error.HTTPError as e:
        return False if e.code == 404 else None   # None = uncertain (rate limit etc)
    except Exception:
        return None

def verify_dois(recs, cap=2500):
    import collections, random
    alld = set()
    for r in recs:
        for d in r.get("doi", []):
            cd = clean_doi(d)
            if re.match(r"10\.\d{4,9}/\S{3,}", cd):
                alld.add(cd)
    alld = sorted(alld)
    random.seed(42)
    sample = alld if len(alld) <= cap else random.sample(alld, cap)
    print(f"\nDOIs: {len(alld)} unique; checking {len(sample)} via Crossref", flush=True)
    missing = []
    for i, d in enumerate(sample):
        ok = crossref_exists(d)
        if ok is False:
            missing.append(d)
        if (i + 1) % 200 == 0:
            print(f"  doi {i+1}/{len(sample)}  non-resolving so far {len(missing)}", flush=True)
        time.sleep(0.05)
    print(f"DOI non-resolving: {len(missing)}/{len(sample)} = {100*len(missing)/len(sample):.2f}% "
          f"(verify manually -- many are regex/format artifacts)")
    for d in missing[:25]:
        print("   miss:", d)
    return {"n_doi": len(alld), "checked": len(sample), "missing": missing}

def verify():
    recs = [json.loads(l) for l in open(REFS) if json.loads(l).get("ok")]
    # only new-style arxiv ids are cleanly checkable via id_list
    allids = set()
    for r in recs:
        for a in r.get("arxiv", []):
            if re.fullmatch(r"\d{4}\.\d{4,5}", a):
                allids.add(a)
    allids = sorted(allids)
    print(f"unique new-style arXiv IDs to verify: {len(allids)}", flush=True)
    exist = arxiv_exists_batch(allids)
    missing = set(allids) - exist
    json.dump({"n_ids": len(allids), "n_exist": len(exist),
               "missing": sorted(missing)}, open(VER, "w"))
    # per-year: fraction of cited (checkable) arXiv IDs that don't resolve
    import collections
    y_cited = collections.Counter(); y_missing = collections.Counter()
    y_papers = collections.Counter(); y_papers_with_missing = collections.Counter()
    for r in recs:
        y = r["year"]; y_papers[y] += 1
        checkable = [a for a in r.get("arxiv", []) if re.fullmatch(r"\d{4}\.\d{4,5}", a)]
        miss = [a for a in checkable if a in missing]
        y_cited[y] += len(checkable); y_missing[y] += len(miss)
        if miss: y_papers_with_missing[y] += 1
    print("\nyear  papers  cited_arxiv_ids  non_resolving  rate%   papers_with_missing")
    for y in sorted(y_papers):
        rate = 100*y_missing[y]/y_cited[y] if y_cited[y] else 0
        print(f"{y}   {y_papers[y]:>5}   {y_cited[y]:>8}        {y_missing[y]:>5}      {rate:5.2f}   {y_papers_with_missing[y]}")
    print("\nSample non-resolving arXiv IDs (verify manually -- could be typos/future/withdrawn):")
    for a in sorted(missing)[:25]:
        print("  ", a)
    doi_res = verify_dois(recs)
    json.dump({"arxiv": {"n_ids": len(allids), "n_exist": len(exist), "missing": sorted(missing)},
               "doi": doi_res}, open(VER, "w"))

if __name__ == "__main__":
    if "--collect" in sys.argv: collect()
    if "--verify" in sys.argv: verify()
