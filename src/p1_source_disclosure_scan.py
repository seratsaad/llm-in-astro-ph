#!/usr/bin/env python3
"""
P1 (referee round 4, points 1 and 7) -- outcome-blind disclosure scan.

The calibration sample so far reaches papers through NASA ADS, which indexes
published versions and lags for recent preprints, so it misses disclosures
that exist only in the arXiv source. Here we scan the arXiv source of a
random sample of 2025 astro-ph papers directly, in randomized order so that
any partial run is still a random sample, and extract every disclosure
statement we find.

The sample is drawn WITHOUT reference to abstract vocabulary. Selecting on
marker words would bias q upward catastrophically, so the draw is blind to
the outcome by construction.

Two products:
  (a) an enlarged, ADS-independent calibration sample for q,
  (b) an unbiased estimate of the true disclosure rate in 2025 astro-ph,
      which bounds how much the ADS route misses.
Output: data/p1_source_scan.jsonl (resumable, one record per paper)
"""
import json, os, re, random, threading
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data")
OUT = os.path.join(DATA, "p1_source_scan.jsonl")
import importlib.util
spec = importlib.util.spec_from_file_location("n1b", os.path.join(HERE, "n1b_calibration.py"))
n1b = importlib.util.module_from_spec(spec); spec.loader.exec_module(n1b)

SEED = 20260807
WORKERS = 4

# Broad disclosure vocabulary. Wider than the ADS families on purpose, since
# here we read the text ourselves and can discard false positives later.
TERMS = [
    r"chat-?gpt", r"\bgpt-?[0-9o]", r"\bopenai\b", r"\bclaude\b", r"\banthropic\b",
    r"\bgemini\b", r"\bllama\b", r"\bcopilot\b", r"\bdeepseek\b", r"\bqwen\b",
    r"\bmistral\b", r"\bgrok\b", r"\bperplexity\b", r"\bbard\b",
    r"large language model", r"\bllms?\b",
    r"generative (?:ai|artificial intelligence)", r"\bai-assisted\b",
    r"ai assistance", r"\bai tools?\b", r"\bgrammarly\b", r"\bdeepl\b",
    r"\bquillbot\b", r"\bwritefull\b", r"\btrinka\b", r"\bpaperpal\b",
    r"language editing", r"language polish", r"proofread",
    r"during the preparation of this work", r"during the preparation of this manuscript",
    r"ai-?assisted technolog",
]
TERM_RE = [(t, re.compile(t, re.I)) for t in TERMS]
ACK_RE = re.compile(
    r"\\(?:section|subsection|paragraph)\*?\{[^}]*(?:acknowledg|declaration|"
    r"statement|availability)[^}]*\}|\\acknowledgments|\\acknowledgements|"
    r"\\begin\{acknowledg", re.I)


def snippets_from(text):
    ack_pos = [m.start() for m in ACK_RE.finditer(text)]
    out, seen = [], []
    for tag, rx in TERM_RE:
        for m in rx.finditer(text):
            if any(abs(m.start() - s) < 150 for s in seen):
                continue
            seen.append(m.start())
            i, j = max(0, m.start() - 380), min(len(text), m.end() + 380)
            in_ack = any(p < m.start() and m.start() - p < 6000 for p in ack_pos)
            out.append({"term": tag, "in_ack_region": in_ack,
                        "snippet": re.sub(r"\s+", " ", text[i:j])})
            if len(out) >= 8:
                return out
    return out


def main():
    ids = []
    for line in open(os.path.join(DATA, "astroph_abstracts.jsonl")):
        r = json.loads(line)
        if r["published"][:4] == "2025":
            ids.append(re.sub(r"v\d+$", "", r["id"]))
    ids = sorted(set(ids))
    random.Random(SEED).shuffle(ids)          # randomized order: partial run stays random
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                pass
    todo = [i for i in ids if i not in done]
    print(f"2025 astro-ph papers: {len(ids)}; already scanned {len(done)}; "
          f"queue {len(todo)}", flush=True)

    lock = threading.Lock()
    fout = open(OUT, "a")
    n = [0]

    def work(pid):
        data, err = n1b.fetch(pid)
        if data is None:
            return {"id": pid, "ok": False, "err": err}
        text = n1b.extract_text(data)
        sn = snippets_from(text)
        return {"id": pid, "ok": True, "n_chars": len(text),
                "hit": bool(sn), "snippets": sn}

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for rec in ex.map(work, todo):
            with lock:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                n[0] += 1
                if n[0] % 250 == 0:
                    hits = sum(1 for line in open(OUT)
                               if json.loads(line).get("hit"))
                    print(f"  [{n[0]}/{len(todo)}] cumulative hits {hits}", flush=True)
    fout.close()
    print("scan complete", flush=True)


if __name__ == "__main__":
    main()
