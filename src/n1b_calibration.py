#!/usr/bin/env python3
"""
N1b -- enlarged and cleaned disclosure calibration (referee points 1-2).

Stage 1: ADS ack: search with a widened model-term list, grouped by term family,
         astronomy collection, 2023-2025 -> data/n1b_disclosed_ids.json
Stage 2: match to the astro-ph corpus by arXiv id.
Stage 3: fetch arXiv e-print source for every matched paper and extract the
         acknowledgment-region snippets that mention a model term
         -> data/n1b_snippets.jsonl (local only, contains text excerpts).

Classification into purpose classes (writing / code-only / research-method /
general-unspecified / false-positive) happens afterwards from the snippets.
Resumable: stage 3 appends and skips ids already present in the output.
"""
import json, os, re, io, gzip, tarfile, time, threading, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data")
TOKEN = open(os.path.expanduser("~/.ads/dev_key")).read().strip()
BASE = "https://api.adsabs.harvard.edu/v1/search/query"
IDS_OUT = os.path.join(DATA, "n1b_disclosed_ids.json")
SNIP_OUT = os.path.join(DATA, "n1b_snippets.jsonl")
EPRINT = "https://export.arxiv.org/e-print/"
UA = {"User-Agent": "astro-corpus-analysis/1.0 (mailto:seratmahmudsaad@gmail.com)"}

# Term families. Bare "Claude"/"Gemini"/"Mistral" are excluded on purpose
# (astronomer first names, the Gemini Observatory, the MISTRAL instrument);
# phrase forms carry the signal and stage-3 snippets catch the rest.
FAMILIES = {
    "chatgpt":  ['"ChatGPT"', '"Chat-GPT"'],
    "gpt":      ['"GPT-4"', '"GPT-4o"', '"GPT-3.5"', '"GPT-5"', '"OpenAI"',
                 '"generative pre-trained transformer"'],
    "claude":   ['"Anthropic"', '"Claude 3"', '"Claude 3.5"', '"Claude 3.7"',
                 '"Claude Sonnet"', '"Claude Opus"', '"Claude Haiku"', '"Claude AI"'],
    "gemini_ai":['"Google Gemini"', '"Gemini Pro"', '"Gemini 1.5"', '"Gemini 2"',
                 '"Gemini Advanced"'],
    "llama":    ['"LLaMA"', '"Llama 2"', '"Llama 3"'],
    "copilot":  ['"Copilot"'],
    "other_models": ['"DeepSeek"', '"Qwen"', '"Mistral AI"', '"Grok"', '"Bard"',
                     '"Perplexity"'],
    "llm_generic": ['"large language model"', '"large language models"',
                    '"LLM"', '"LLMs"'],
    "genai":    ['"generative AI"', '"generative artificial intelligence"',
                 '"AI-assisted"', '"AI assistance"', '"AI tools"',
                 '"AI language tools"', '"AI writing"'],
}

# regexes for snippet extraction from LaTeX source (case-insensitive)
TERM_RES = [
    ("chatgpt", r"chat-?gpt"),
    ("gpt", r"\bgpt-?[345o]"), ("gpt", r"\bopenai\b"),
    ("claude", r"\bclaude\b"), ("claude", r"\banthropic\b"),
    ("gemini_ai", r"\bgemini\b"),
    ("llama", r"\bllama\b"),
    ("copilot", r"\bcopilot\b"),
    ("other_models", r"\bdeepseek\b|\bqwen\b|\bmistral\b|\bgrok\b|\bbard\b|\bperplexity\b"),
    ("llm_generic", r"large language model|\bllms?\b"),
    ("genai", r"generative (ai|artificial intelligence)|ai-assisted|ai assistance|ai (language )?tools?|ai writing"),
]
TERM_COMP = [(fam, re.compile(rx, re.I)) for fam, rx in TERM_RES]
ACK_RE = re.compile(r"\\(?:section|subsection|paragraph)\*?\{[^}]*acknowledg[^}]*\}|\\acknowledgments|\\acknowledgements|\\begin\{acknowledg", re.I)

SLEEP = 0.3
WORKERS = 4
REQ_SPACING = 0.65
MAX_BYTES = 14_000_000
_rl_lock = threading.Lock(); _next_req = [0.0]

def rate_limit():
    with _rl_lock:
        now = time.time()
        wait = max(0.0, _next_req[0] - now)
        _next_req[0] = max(now, _next_req[0]) + REQ_SPACING
    if wait > 0:
        time.sleep(wait)

def ads_query(q, year):
    ids, start = [], 0
    while True:
        url = BASE + "?" + urllib.parse.urlencode(
            {"q": q, "fq": f"year:{year}", "fl": "identifier,bibcode", "rows": 200, "start": start})
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {TOKEN}"})
        for a in range(5):
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    resp = json.load(r)["response"]
                break
            except Exception:
                time.sleep(2.0 * (a + 1))
        else:
            raise RuntimeError(f"ADS query failed: {q} {year}")
        for d in resp["docs"]:
            for ident in d.get("identifier", []):
                m = re.match(r"arXiv:(\d{4}\.\d{4,5})$", ident)
                if m:
                    ids.append(m.group(1)); break
        start += len(resp["docs"])
        if start >= resp["numFound"] or not resp["docs"]:
            break
        time.sleep(SLEEP)
    return ids, resp["numFound"]

def stage1():
    if os.path.exists(IDS_OUT):
        print("stage 1: already done ->", IDS_OUT, flush=True)
        return json.load(open(IDS_OUT))
    out = {"families": {}, "per_year_found": {}}
    for fam, terms in FAMILIES.items():
        q = 'ack:(' + " OR ".join(terms) + ') database:astronomy'
        fam_ids = {}
        for year in (2023, 2024, 2025):
            ids, nfound = ads_query(q, year)
            for i in ids:
                fam_ids.setdefault(i, year)
            out["per_year_found"][f"{fam}_{year}"] = nfound
            print(f"  {fam} {year}: ADS {nfound}, arXiv ids {len(ids)}", flush=True)
            time.sleep(SLEEP)
        out["families"][fam] = fam_ids
    json.dump(out, open(IDS_OUT, "w"), indent=2)
    return out

def load_corpus():
    corpus = {}
    for line in open(os.path.join(DATA, "astroph_abstracts.jsonl")):
        r = json.loads(line)
        y = int(r["published"][:4])
        if 2023 <= y <= 2025:
            corpus[re.sub(r"v\d+$", "", r["id"])] = (y, r["abstract"])
    return corpus

def fetch(pid):
    req = urllib.request.Request(EPRINT + pid, headers=UA)
    for attempt in range(3):
        try:
            rate_limit()
            with urllib.request.urlopen(req, timeout=90) as r:
                cl = r.headers.get("Content-Length")
                if cl and int(cl) > MAX_BYTES:
                    return None, "SKIP_LARGE"
                buf = io.BytesIO(); total = 0
                while True:
                    chunk = r.read(65536)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > MAX_BYTES:
                        return None, "SKIP_LARGE"
                    buf.write(chunk)
                return buf.getvalue(), ""
        except Exception as e:
            if attempt == 2:
                return None, f"ERR:{e}"
            time.sleep(1.5 * (attempt + 1))
    return None, "ERR"

def extract_text(data):
    texts = []
    try:
        tf = tarfile.open(fileobj=io.BytesIO(data), mode="r:*")
        for m in tf.getmembers():
            if m.isfile() and m.name.lower().endswith((".tex", ".bbl", ".txt")):
                try:
                    s = tf.extractfile(m).read().decode("utf-8", "ignore")
                    sample = s[:4000]
                    bad = sum(1 for ch in sample if ord(ch) < 9 or (13 < ord(ch) < 32))
                    if not sample or bad / len(sample) <= 0.02:
                        texts.append(s)
                except Exception:
                    pass
        return "\n".join(texts)
    except Exception:
        pass
    try:
        return gzip.decompress(data).decode("utf-8", "ignore")
    except Exception:
        return data.decode("utf-8", "ignore")

def snippets_from(text):
    """All model-term mention snippets, flagged by whether they sit after an
    acknowledgment sectioning command (crude but effective ordering test)."""
    ack_pos = [m.start() for m in ACK_RE.finditer(text)]
    snips = []
    seen_spans = []
    for fam, rx in TERM_COMP:
        for m in rx.finditer(text):
            if any(abs(m.start() - s) < 120 for s in seen_spans):
                continue
            seen_spans.append(m.start())
            i, j = max(0, m.start() - 350), min(len(text), m.end() + 350)
            in_ack = any(p < m.start() and m.start() - p < 6000 for p in ack_pos)
            snips.append({"family": fam, "in_ack_region": in_ack,
                          "snippet": re.sub(r"\s+", " ", text[i:j])})
    return snips

def stage3(matched):
    done = set()
    if os.path.exists(SNIP_OUT):
        for line in open(SNIP_OUT):
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                pass
    todo = sorted(set(matched) - done)
    print(f"stage 3: fetching source for {len(todo)} papers ({len(done)} done)", flush=True)
    wlock = threading.Lock()
    fout = open(SNIP_OUT, "a")
    n = [0]
    def work(pid):
        data, err = fetch(pid)
        rec = {"id": pid, "families": matched[pid]}
        if data is None:
            rec["ok"] = False; rec["err"] = err
        else:
            rec["ok"] = True
            rec["snippets"] = snippets_from(extract_text(data))
        return rec
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for rec in ex.map(work, todo):
            with wlock:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n"); fout.flush()
                n[0] += 1
                if n[0] % 50 == 0:
                    print(f"  [{n[0]}/{len(todo)}]", flush=True)
    fout.close()
    print("stage 3 DONE", flush=True)

def main():
    ids = stage1()
    corpus = load_corpus()
    matched = {}
    for fam, fam_ids in ids["families"].items():
        for pid in fam_ids:
            if pid in corpus:
                matched.setdefault(pid, []).append(fam)
    print(f"union disclosed ids: {sum(len(v) for v in ids['families'].values())} raw, "
          f"{len(set(i for v in ids['families'].values() for i in v))} unique, "
          f"{len(matched)} matched to astro-ph corpus", flush=True)
    json.dump({k: v for k, v in matched.items()},
              open(os.path.join(DATA, "n1b_matched.json"), "w"), indent=2)
    stage3(matched)

if __name__ == "__main__":
    main()
