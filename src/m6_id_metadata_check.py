#!/usr/bin/env python3
"""
M6 (independent referee) -- do cited arXiv identifiers point at the works the
citations describe? Resolution alone would pass a plausible-but-wrong id.
Sample: every reference with a new-style arXiv id in the 57-paper bibliography
corpus already fetched for the identifier-free audit. For each entry we parse
the first-author surname and year from the bibitem text, fetch the id's real
metadata from the arXiv API in batches, and compare (surname prefix match on
any listed author for collaboration styles; year within 1 of either the arXiv
posting year or the cited year).
Output: data/m6_id_check.json (+ per-entry jsonl for mismatch inspection)
"""
import json, os, re, time, urllib.parse, urllib.request
import xml.etree.ElementTree as ET

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
ARXIV_RE = re.compile(r"(?<![\d.])(\d{4}\.\d{4,5})(?!\d)")
DOICTX_RE = re.compile(r"10\.\d{4,9}/\S*?(\d{4}\.\d{4,5})")
YEAR_RE = re.compile(r"\b((?:19|20)\d{2})[a-z]?\b")
NS = {"a": "http://www.w3.org/2005/Atom"}

def parse_author(body):
    lab = re.match(r"\\bibitem\[(.{0,120}?)\]\{", body)
    if lab:
        lt = re.sub(r"\\[a-zA-Z]+|[{}~]", " ", lab.group(1))
        m = re.search(r"([A-Z][A-Za-z'-]{2,})", lt)
        if m and m.group(1).lower() not in ("et", "al", "and", "protect"):
            return m.group(1).lower()
    body = re.sub(r"^\\bibitem(?:\[(?:[^\[\]]|\[[^\]]*\])*\])?\{[^}]*\}", "", body)
    body = re.sub(r"[{}~\\]", " ", body)
    m = re.match(r"\s*([A-Z][A-Za-z'-]+)\s*,", body)
    if m:
        return m.group(1).strip("-'").lower()
    m = re.match(r"\s*(?:[A-Z]\.[\s-]*)+([A-Z][A-Za-z'-]+)\s*[,.]", body)
    if m:
        return m.group(1).strip("-'").lower()
    m = re.match(r"\s*[A-Z][a-z]+\s+([A-Z][A-Za-z'-]+)\s*[,.]", body)
    return m.group(1).strip("-'").lower() if m else None

def main():
    cache_fn = os.path.join(DATA, "m6_entries_cache.json")
    if os.path.exists(cache_fn):
        entries = json.load(open(cache_fn))
        for e in entries:
            e["author"] = parse_author(e["entry"])
        run_meta_and_compare(entries)
        return
    entries = []
    for line in open(os.path.join(DATA, "r7_refs.jsonl")):
        r = json.loads(line)
        e = r.get("entry")
        if not e:
            continue
        m = ARXIV_RE.search(e)
        if not m:
            continue
        ym = YEAR_RE.search(e)
        entries.append({"paper": r["paper"], "id": m.group(1),
                        "author": parse_author(e), "year": int(ym.group(1)) if ym else None,
                        "entry": e[:200]})
    # NOTE: r7_refs.jsonl only kept no-id entries; re-split from source is not
    # stored. Fall back: use every arXiv id cited in the C3 audit sample with
    # its per-entry text unavailable -> restrict to r7 corpus entries that DO
    # carry ids (they were skipped in r7 output). If none, harvest from the
    # r7 sample papers' sources again.
    print(f"entries with ids in r7 output: {len(entries)}", flush=True)
    if len(entries) < 200:
        import importlib.util
        spec = importlib.util.spec_from_file_location("n1b", os.path.join(
            os.path.dirname(__file__), "n1b_calibration.py"))
        n1b = importlib.util.module_from_spec(spec); spec.loader.exec_module(n1b)
        spec2 = importlib.util.spec_from_file_location("r7", os.path.join(
            os.path.dirname(__file__), "r7_noid_refs.py"))
        r7 = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(r7)
        papers = sorted({json.loads(l)["paper"] for l in open(os.path.join(DATA, "r7_refs.jsonl"))
                         if "paper" in json.loads(l)})
        entries = []
        for k, pid in enumerate(papers):
            data, err = n1b.fetch(pid)
            if data is None:
                continue
            text = n1b.extract_text(data)
            for e in r7.split_bibitems(text):
                doi_ids = set(DOICTX_RE.findall(e))
                cands = [g for g in ARXIV_RE.findall(e) if g not in doi_ids]
                if not cands:
                    continue
                ym = YEAR_RE.search(e)
                entries.append({"paper": pid, "id": cands[0],
                                "author": parse_author(e),
                                "year": int(ym.group(1)) if ym else None,
                                "entry": re.sub(r"\s+", " ", e)[:200]})
            print(f"[{k+1}/{len(papers)}] {pid}: cumulative id-refs {len(entries)}", flush=True)
    json.dump(entries, open(os.path.join(DATA, "m6_entries_cache.json"), "w"))
    run_meta_and_compare(entries)

def run_meta_and_compare(entries):
    # arXiv API metadata in batches of 50
    mfn = os.path.join(DATA, "m6_meta_cache.json")
    meta = json.load(open(mfn)) if os.path.exists(mfn) else {}
    ids = sorted({e["id"] for e in entries} - set(meta))
    for i in range(0, len(ids), 50):
        chunk = ids[i:i + 50]
        url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(
            {"id_list": ",".join(chunk), "max_results": len(chunk)})
        for a in range(4):
            try:
                with urllib.request.urlopen(urllib.request.Request(
                        url, headers={"User-Agent": "academic-corpus-study/1.0"}), timeout=60) as r:
                    root = ET.fromstring(r.read())
                break
            except Exception:
                time.sleep(10 * (a + 1))
        else:
            continue
        for entry in root.findall("a:entry", NS):
            eid = entry.find("a:id", NS).text
            m = re.search(r"(\d{4}\.\d{4,5})", eid or "")
            if not m:
                continue
            auths = [a.find("a:name", NS).text for a in entry.findall("a:author", NS)]
            pub = entry.find("a:published", NS).text[:4]
            meta[m.group(1)] = {"authors": auths, "year": int(pub)}
        time.sleep(3.2)
        if (i // 50) % 5 == 0:
            print(f"  meta {len(meta)}", flush=True)
    json.dump(meta, open(mfn, "w"))

    ok = mism = unchecked = 0
    mismatches = []
    for e in entries:
        md = meta.get(e["id"])
        if md is None or e["author"] is None:
            unchecked += 1
            continue
        import unicodedata
        def fold(s):
            return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
        names = fold(" ".join(md["authors"]))
        cited = fold(e["author"])
        a_ok = cited[:5] in names or cited in ("collaboration", "team") or \
               cited in fold(e["entry"][:60]) and any(
                   fold(x).split()[-1][:5] in fold(e["entry"]) for x in md["authors"][:3])
        y_ok = e["year"] is None or abs(md["year"] - e["year"]) <= 1
        if a_ok and y_ok:
            ok += 1
        else:
            mism += 1
            mismatches.append({**e, "arxiv_authors": md["authors"][:3],
                               "arxiv_year": md["year"],
                               "fail": ("author" if not a_ok else "") + ("+year" if not y_ok else "")})
    json.dump({"n_entries": len(entries), "checked": ok + mism, "match": ok,
               "mismatch": mism, "unchecked": unchecked,
               "mismatches": mismatches},
              open(os.path.join(DATA, "m6_id_check.json"), "w"), indent=1)
    print(f"id-bearing refs {len(entries)}: match {ok}, mismatch {mism}, "
          f"unparseable/unfetched {unchecked}")

if __name__ == "__main__":
    main()
