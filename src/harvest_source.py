#!/usr/bin/env python3
"""
C2 -- "Caught in the source": scan arXiv LaTeX e-print source of astro-ph papers
for leaked LLM telltales (undisclosed AI use), the way LaTeXpOsEd (2510.03761) does.

Astronomy uniquely posts LaTeX source for ~every paper. LLM boilerplate that authors
paste and forget -- refusal strings, "as an AI language model", prompt notes -- survives
in `%` comments and body text even though it never reaches the compiled PDF.

We scan:
  * FULL source text  -> Tier-1 "hard leak" phrases (near-zero false positive anywhere)
  * COMMENT-only text  -> Tier-1 + Tier-2 prompt-residue phrases (hide in % comments)

Sampling: stride-sampled per year from data/astroph_abstracts.jsonl so we cover months
evenly. Output: data/source_scan.jsonl (one record per paper, resumable).
"""
import json, os, re, io, gzip, tarfile, time, sys, threading, urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data")
INP = os.path.join(DATA, "astroph_abstracts.jsonl")
OUT = os.path.join(DATA, "source_scan.jsonl")
EPRINT = "https://export.arxiv.org/e-print/"
UA = {"User-Agent": "astrobites-llm-analysis/1.0 (mailto:seratmahmudsaad@gmail.com)"}
SLEEP = 3.1
WORKERS = 4                # parallel download workers
REQ_SPACING = 0.6         # global min seconds between request starts (~1.6 req/s)
MAX_BYTES = 14_000_000    # skip tarballs larger than this (figure-heavy monsters)

_rl_lock = threading.Lock()
_next_req = [0.0]
def rate_limit():
    with _rl_lock:
        now = time.time()
        wait = max(0.0, _next_req[0] - now)
        _next_req[0] = max(now, _next_req[0]) + REQ_SPACING
    if wait > 0:
        time.sleep(wait)

# Per-year sample sizes. Recent years = where leaks are plausible; earlier = control
# (Tier-1 phrases should be ~0 before 2023, a built-in false-positive check).
SAMPLE = {2016: 250, 2018: 300, 2020: 300, 2022: 400,
          2023: 600, 2024: 800, 2025: 800, 2026: 500}

# ---- telltale patterns (case-insensitive) --------------------------------------
TIER1 = [  # hard writing leaks: pasted LLM output / refusal / meta strings
    r"as an ai language model", r"as a large language model", r"as an ai,? i",
    r"i'?m sorry,? but i can(?:'|no)t", r"i cannot fulfil", r"i cannot provide a",
    r"i can'?t assist with", r"regenerate response",
    r"as of my (last )?(knowledge|training) (update|cutoff|cut-off)",
    r"my (last )?knowledge cutoff", r"i do not have access to real-?time",
    r"here is (a|the|your) (rewritten|revised|reworded|paraphrased|polished) (version|text|paragraph)",
    r"here'?s (a|the|your) (rewritten|revised|reworded|paraphrased|polished) (version|text|paragraph)",
    r"certainly!? here (is|'?s)", r"sure!? here'?s", r"sure,? here is",
    r"as requested,? here (is|'?s)", r"note that i am an ai",
    r"as an ai assistant", r"i'?m happy to help you (write|rewrite|revise)",
    r"is there anything else you'?d like",
]
# Tier-2 = explicit AI-tool references in COMMENTS (a note that a tool was involved).
# Kept deliberately tight; generic phrases (make it more / tldr / let me know) removed
# because they matched LaTeX boilerplate and coauthor notes.
TIER2 = [
    r"\bchatgpt\b", r"\bchat-gpt\b", r"\bgpt-?4\b", r"\bgpt-?3(\.5)?\b",
    r"\bgpt-?5\b", r"\bopenai\b", r"\bgithub copilot\b",
    r"generated (by|with|using) (chatgpt|gpt|an? (llm|ai)|a language model)",
    r"(written|polished|edited|proofread) (by|with|using) (chatgpt|gpt|an? (llm|ai))",
    r"you are an? (expert|helpful|professional)",
    r"rewrite the following", r"please (rewrite|rephrase|proofread|polish) (the|this|following)",
]
TIER1_RE = [(p, re.compile(p, re.I)) for p in TIER1]
TIER2_RE = [(p, re.compile(p, re.I)) for p in TIER2]

COMMENT_RE = re.compile(r"(?<!\\)%(.*)$")  # first unescaped % to end of line

def strip_comments_collect(text):
    """Return (comment_text) -- everything after unescaped % on each line."""
    out = []
    for line in text.splitlines():
        m = COMMENT_RE.search(line)
        if m:
            out.append(m.group(1))
    return "\n".join(out)

def fetch(pid):
    req = urllib.request.Request(EPRINT + pid, headers=UA)
    for attempt in range(3):
        try:
            rate_limit()
            with urllib.request.urlopen(req, timeout=90) as r:
                cl = r.headers.get("Content-Length")
                if cl and int(cl) > MAX_BYTES:
                    return None, "SKIP_LARGE"           # don't download figure monsters
                # chunked read with hard cap (covers missing Content-Length)
                buf = io.BytesIO(); total = 0
                while True:
                    chunk = r.read(65536)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > MAX_BYTES:
                        return None, "SKIP_LARGE"
                    buf.write(chunk)
                return buf.getvalue(), r.headers.get("Content-Type", "")
        except Exception as e:
            if attempt == 2:
                return None, f"ERR:{e}"
            time.sleep(1.5 * (attempt + 1))
    return None, "ERR"

# Only author-written content; exclude .sty/.cls (boilerplate, cause false positives).
AUTHOR_EXT = (".tex", ".bbl", ".txt")

def _clean(b):
    """Decode bytes; return '' if the content looks binary (non-text file)."""
    s = b.decode("utf-8", "ignore")
    if not s:
        return ""
    sample = s[:4000]
    nonprint = sum(1 for ch in sample if ord(ch) < 9 or (13 < ord(ch) < 32))
    if len(sample) and nonprint / len(sample) > 0.02:
        return ""          # binary garbage -> drop
    return s

def extract_text(data):
    """Return (full_text, n_tex) from raw e-print bytes; handle tar.gz / gzip / raw."""
    texts = []
    n_tex = 0
    try:
        tf = tarfile.open(fileobj=io.BytesIO(data), mode="r:*")
        for m in tf.getmembers():
            if not m.isfile() or not m.name.lower().endswith(AUTHOR_EXT):
                continue
            try:
                s = _clean(tf.extractfile(m).read())
                if s:
                    texts.append(s)
                    if m.name.lower().endswith(".tex"):
                        n_tex += 1
            except Exception:
                pass
        return "\n".join(texts), n_tex
    except Exception:
        pass
    try:
        return _clean(gzip.decompress(data)), 1
    except Exception:
        pass
    return _clean(data), 0

def scan(full, comments):
    hits = []
    for pat, rx in TIER1_RE:
        m = rx.search(full)
        if m:
            i = max(0, m.start() - 40); j = min(len(full), m.end() + 40)
            hits.append({"tier": 1, "pattern": pat, "where": "body_or_comment",
                         "snippet": full[i:j].replace("\n", " ")})
    for pat, rx in TIER2_RE:
        m = rx.search(comments)
        if m:
            i = max(0, m.start() - 40); j = min(len(comments), m.end() + 40)
            hits.append({"tier": 2, "pattern": pat, "where": "comment",
                         "snippet": comments[i:j].replace("\n", " ")})
    return hits

def sample_ids():
    by_year = {}
    seen = set()
    for line in open(INP):
        r = json.loads(line)
        base = r["id"].split("v")[0]
        if base in seen:
            continue
        seen.add(base)
        pub = r["published"]
        if not pub or len(pub) < 7:
            continue
        y = int(pub[:4])
        if y in SAMPLE:
            by_year.setdefault(y, []).append(base)
    out = []
    for y, ids in by_year.items():
        ids.sort()
        n = SAMPLE[y]
        if len(ids) <= n:
            picked = ids
        else:
            stride = len(ids) / n
            picked = [ids[int(k * stride)] for k in range(n)]
        out.extend((y, pid) for pid in picked)
    return out

def process(args):
    y, pid = args
    data, ct = fetch(pid)
    rec = {"id": pid, "year": y}
    if data is None:
        rec["ok"] = (ct == "SKIP_LARGE")   # skipped-large is not an error, just excluded
        rec["err"] = ct
        rec["skipped_large"] = (ct == "SKIP_LARGE")
    else:
        full, n_tex = extract_text(data)
        comments = strip_comments_collect(full)
        rec["ok"] = True
        rec["bytes"] = len(data)
        rec["n_tex"] = n_tex
        rec["has_source"] = n_tex > 0 or len(full) > 200
        rec["comment_chars"] = len(comments)
        rec["hits"] = scan(full, comments)
    return rec

def main():
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                pass
    targets = [(y, pid) for (y, pid) in sample_ids() if pid not in done]
    targets.sort(key=lambda yp: (-yp[0], yp[1]))   # recent years first
    total = len(targets)
    print(f"to scan: {total} papers (already done: {len(done)}), {WORKERS} workers", flush=True)
    wlock = threading.Lock()
    fout = open(OUT, "a")
    n = [0]
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for rec in ex.map(process, targets):
            with wlock:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                n[0] += 1
                if rec.get("hits"):
                    print(f"[{n[0]}/{total}] {rec['id']} ({rec['year']}) HITS: "
                          + "; ".join(f"T{h['tier']}:{h['pattern']}" for h in rec["hits"]), flush=True)
                elif n[0] % 100 == 0:
                    print(f"[{n[0]}/{total}] ... scanning", flush=True)
    fout.close()
    print("DONE source scan", flush=True)

if __name__ == "__main__":
    main()
