#!/usr/bin/env python3
"""
R7 (referee point 7) -- verify identifier-free references.
The citation audit so far checked references carrying arXiv ids or DOIs, which
are BibTeX-mediated and cannot easily be fabricated. The biomedical failure
mode is the plain author-year reference with no identifier. Here we:
  stage 1: sample marker-flagged 2024-2025 papers, fetch arXiv source, extract
           bibliography entries, and split them by identifier presence;
  stage 2: parse identifier-free entries into (first author, year, journal,
           volume, page) where possible;
  stage 3: verify each parsed entry against ADS (bibstem+volume+page), matching
           year within 1 and first-author surname.
Failures are dumped for hand inspection (a parse artifact is not a fabrication).
Outputs: data/r7_refs.jsonl (per-entry), data/r7_summary.json
"""
import json, os, re, time, random, urllib.parse, urllib.request

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data")
import importlib.util
spec = importlib.util.spec_from_file_location("n1b", os.path.join(HERE, "n1b_calibration.py"))
n1b = importlib.util.module_from_spec(spec); spec.loader.exec_module(n1b)

TOKEN = open(os.path.expanduser("~/.ads/dev_key")).read().strip()
BASE = "https://api.adsabs.harvard.edu/v1/search/query"
OUT = os.path.join(DATA, "r7_refs.jsonl")

BASKET = set("""delve delves delving underscore underscores underscoring intricate
intricacies showcasing showcase showcases showcased boasts tapestry pivotal
meticulous meticulously nuanced garner garners garnered multifaceted commendable
noteworthy myriad plethora testament encompassing seamless seamlessly elucidate
elucidating unravel unraveling unravelling realm realms leveraging""".split())

N_PAPERS = 250
SEED = 20260805      # original 80-paper draw
SEED2 = 20260806     # extension draw to N_PAPERS (referee scale-up)

ARXIV_RE = re.compile(r"\b\d{4}\.\d{4,5}\b|arxiv|astro-ph[/.]", re.I)
DOI_RE = re.compile(r"\b10\.\d{4,9}/|\\doi\b|doi\.org|doi:", re.I)
BIBITEM_RE = re.compile(r"\\bibitem")
YEAR_RE = re.compile(r"\b((?:19|20)\d{2})[a-z]?\b")
# journal tokens -> ADS bibstem
JMAP = [
    (r"\\apjl\b|astrophys\.? ?j\.? ?lett|apj ?l\b|astrophysical journal letters", "ApJL"),
    (r"\\apjs\b|astrophys\.? ?j\.? ?suppl|apjs\b|astrophysical journal supplement", "ApJS"),
    (r"\\apj\b|astrophys\.? ?j\.?|(?<![a-z])apj\b|astrophysical journal", "ApJ"),
    (r"\\mnras\b|mon\.? ?not\.?|mnras\b|monthly notices", "MNRAS"),
    (r"\\aap\b|astron\.? ?astrophys|a ?& ?a\b|a&a\b|astronomy (?:&|and) astrophysics", "A&A"),
    (r"\\aj\b|astron\.? ?j\.?|(?<![a-z])aj\b|astronomical journal", "AJ"),
    (r"\\pasp\b|pasp\b|publ\.? ?astron\.? ?soc\.? ?pac", "PASP"),
    (r"\\pasj\b|pasj\b|publ\.? ?astron\.? ?soc\.? ?(?:jpn|japan)", "PASJ"),
    (r"\\araa\b|araa\b|annu\.? ?rev\.? ?astron", "ARA&A"),
    (r"\\nat\b|nature\b", "Natur"),
    (r"\\prd\b|phys\.? ?rev\.? ?d\b|physical review d", "PhRvD"),
    (r"\\prl\b|phys\.? ?rev\.? ?lett|physical review letters", "PhRvL"),
    (r"\\physrep\b|phys\.? ?rep\.?\b|physics reports", "PhR"),
    (r"\\jcap\b|jcap\b|j\.? ?cosmol\.? ?astropart", "JCAP"),
    (r"\\solphys\b|sol\.? ?phys\.?\b|solar physics", "SoPh"),
    (r"\\icarus\b|icarus\b", "Icar"),
    (r"science\b", "Sci"),
]
JMAP_C = [(re.compile(p, re.I), b) for p, b in JMAP]

def split_bibitems(text):
    items = []
    for mtx in re.finditer(r"\\begin\{thebibliography\}(.*?)\\end\{thebibliography\}",
                           text, re.S):
        body = mtx.group(1)
        parts = re.split(r"(?=\\bibitem)", body)
        items += [re.sub(r"\s+", " ", p).strip() for p in parts if p.strip().startswith(r"\bibitem")]
    return items

def parse_entry(e):
    # strip the \bibitem[...]{key} head
    head = re.match(r"\\bibitem(?:\[(?:[^\[\]]|\[[^\]]*\])*\])?\{[^}]*\}", e)
    body = e[head.end():] if head else e
    body_tex = body
    body = re.sub(r"[{}~]", " ", body)
    m = YEAR_RE.search(body)
    year = int(m.group(1)) if m else None
    # first author surname: first capitalized token before a comma
    am = re.match(r"\s*(?:\\[a-z]+\s*)?([A-Z][A-Za-z'`^\"\\-]+)\s*,", body)
    author = am.group(1).replace("\\", "").strip("-'`^\"") if am else None
    jname = None
    jpos = None
    for rx, bs in JMAP_C:
        mm = rx.search(body_tex)
        if mm:
            jname = bs; jpos = mm.end(); break
    vol = page = None
    if jname is not None:
        nums = re.findall(r"\b([A-L]?\d{1,4})\b", re.sub(r"[{}~,]", " ", body_tex[jpos:jpos + 60]))
        nums = [n for n in nums if not YEAR_RE.fullmatch(n)]
        if len(nums) >= 2:
            vol, page = nums[0], nums[1]
    return {"author": author, "year": year, "journal": jname,
            "volume": vol, "page": page}

def ads_check(p):
    q = f'bibstem:"{p["journal"]}" volume:"{p["volume"]}" page:"{p["page"]}"'
    url = BASE + "?" + urllib.parse.urlencode(
        {"q": q, "fl": "year,first_author,title,bibcode", "rows": 3})
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {TOKEN}"})
    for a in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                docs = json.load(r)["response"]["docs"]
            break
        except Exception:
            time.sleep(2.0 * (a + 1))
    else:
        return "ADS_ERR", None
    for d in docs:
        okyear = p["year"] is None or abs(int(d.get("year", 0)) - p["year"]) <= 1
        fa = (d.get("first_author") or "").split(",")[0].lower()
        oka = p["author"] is None or fa.startswith(p["author"].lower()[:4]) \
            or p["author"].lower().startswith(fa[:4] or "zzzz")
        if okyear and oka:
            return "VERIFIED", d.get("bibcode")
        if okyear:
            return "AUTHOR_MISMATCH", d.get("bibcode")
    return ("NOT_FOUND", None) if not docs else ("YEAR_MISMATCH", docs[0].get("bibcode"))

def main():
    rng = random.Random(SEED)
    flagged = []
    for line in open(os.path.join(DATA, "astroph_abstracts.jsonl")):
        r = json.loads(line)
        y = int(r["published"][:4])
        if y not in (2024, 2025):
            continue
        if set(re.findall(r"[a-z]+", r["abstract"].lower())) & BASKET:
            flagged.append(re.sub(r"v\d+$", "", r["id"]))
    sample = rng.sample(flagged, 80)
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            try:
                done.add(json.loads(line)["paper"])
            except Exception:
                pass
    if N_PAPERS > 80:
        rng2 = random.Random(SEED2)
        pool = [f for f in flagged if f not in set(sample) and f not in done]
        sample = sample + rng2.sample(pool, N_PAPERS - 80)
    print(f"marker-flagged 2024-25 papers: {len(flagged)}; sampled {len(sample)} "
          f"({len(done)} already processed)", flush=True)
    fout = open(OUT, "a")
    counts = {"total_refs": 0, "with_arxiv": 0, "with_doi_only": 0, "no_id": 0,
              "no_id_parsed": 0, "papers_ok": 0, "papers_nobib": 0}
    results = {}
    for k, pid in enumerate(sample):
        if pid in done:
            continue
        data, err = n1b.fetch(pid)
        if data is None:
            fout.write(json.dumps({"paper": pid, "err": err}) + "\n")
            continue
        text = n1b.extract_text(data)
        items = split_bibitems(text)
        if not items:
            counts["papers_nobib"] += 1
            fout.write(json.dumps({"paper": pid, "err": "NO_THEBIB"}) + "\n")
            continue
        counts["papers_ok"] += 1
        for e in items:
            counts["total_refs"] += 1
            if ARXIV_RE.search(e):
                counts["with_arxiv"] += 1
                continue
            if DOI_RE.search(e):
                counts["with_doi_only"] += 1
                continue
            counts["no_id"] += 1
            p = parse_entry(e)
            rec = {"paper": pid, "entry": e[:400], "parsed": p}
            if p["journal"] and p["volume"] and p["page"]:
                counts["no_id_parsed"] += 1
                verdict, bib = ads_check(p)
                rec["verdict"] = verdict; rec["ads_bibcode"] = bib
                results[verdict] = results.get(verdict, 0) + 1
                time.sleep(0.25)
            else:
                rec["verdict"] = "UNPARSED"
                results["UNPARSED"] = results.get("UNPARSED", 0) + 1
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fout.flush()
        print(f"[{k+1}/{N_PAPERS}] {pid}: refs so far {counts['total_refs']}, "
              f"no-id {counts['no_id']}, verdicts {results}", flush=True)
    fout.close()
    json.dump({"counts_this_run": counts, "verdicts_this_run": results},
              open(os.path.join(DATA, "r7_run_delta.json"), "w"), indent=2)
    print(json.dumps({"counts_this_run": counts, "verdicts_this_run": results}, indent=1))

if __name__ == "__main__":
    main()
