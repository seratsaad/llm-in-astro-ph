#!/usr/bin/env python3
"""
N3 -- version-diff design, stage 1 + 2.
Stage 1: for every corpus paper submitted 2015-2021 (v1 safely before ChatGPT),
batch-query the arXiv API for the last-updated date. Papers updated in 2023 or
later form the revised set. Checkpointed to data/n3_updated.jsonl.
Stage 2: for each revised paper, fetch the v1 abstract from the versioned abs
page, store alongside the latest abstract. Output data/n3_pairs.jsonl.
Analysis runs separately once pairs exist.
"""
import json, os, re, time, urllib.parse, urllib.request
import xml.etree.ElementTree as ET

HERE = os.path.dirname(__file__); DATA = os.path.join(HERE, "..", "data")
API = "https://export.arxiv.org/api/query"
NS = {"a": "http://www.w3.org/2005/Atom"}
UA = {"User-Agent": "llm-astroph-versiondiff/1.0 (mailto:rocketscience426@gmail.com)"}
CKPT = os.path.join(DATA, "n3_updated.jsonl")
PAIRS = os.path.join(DATA, "n3_pairs.jsonl")

def fetch_batch(ids):
    url = API + "?" + urllib.parse.urlencode({"id_list": ",".join(ids), "max_results": len(ids)})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=90) as r:
                d = r.read().decode()
            out = []
            for e in ET.fromstring(d).findall("a:entry", NS):
                aid = e.findtext("a:id", default="", namespaces=NS).split("/abs/")[-1]
                base = re.sub(r"v\d+$", "", aid)
                ver = re.search(r"v(\d+)$", aid)
                upd = e.findtext("a:updated", default="", namespaces=NS)[:10]
                ab = e.findtext("a:summary", default="", namespaces=NS)
                out.append({"id": base, "v": int(ver.group(1)) if ver else 1,
                            "updated": upd, "abstract_latest": ab})
            return out
        except urllib.error.HTTPError as ex:
            time.sleep(20 * (attempt + 1) if ex.code == 429 else 6)
        except Exception:
            time.sleep(6)
    return []

def stage1():
    done = set()
    if os.path.exists(CKPT):
        for line in open(CKPT):
            done.add(json.loads(line)["id"])
    ids = []
    for line in open(os.path.join(DATA, "astroph_abstracts.jsonl")):
        r = json.loads(line)
        if 2015 <= int(r["published"][:4]) <= 2021:
            b = re.sub(r"v\d+$", "", r["id"])
            if b not in done:
                ids.append(b)
    print(f"stage 1: {len(ids)} papers to query ({len(done)} already done)", flush=True)
    B = 50
    with open(CKPT, "a") as f:
        for k in range(0, len(ids), B):
            for rec in fetch_batch(ids[k:k + B]):
                f.write(json.dumps(rec) + "\n")
            if (k // B) % 40 == 0:
                f.flush()
                print(f"  stage1 {k + B}/{len(ids)}", flush=True)
            time.sleep(3.2)

def fetch_v1_abstract(base_id):
    """Scrape the v1 abstract from the versioned abs page."""
    url = f"https://arxiv.org/abs/{base_id}v1"
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                html = r.read().decode("utf-8", "ignore")
            m = re.search(r'<blockquote class="abstract[^"]*">\s*<span[^>]*>Abstract:</span>(.*?)</blockquote>', html, re.S)
            if m:
                txt = re.sub(r"<[^>]+>", " ", m.group(1))
                return re.sub(r"\s+", " ", txt).strip()
            return None
        except urllib.error.HTTPError as ex:
            if ex.code == 429: time.sleep(25 * (attempt + 1))
            else: return None
        except Exception:
            time.sleep(6)
    return None

def stage2():
    revised = []
    for line in open(CKPT):
        r = json.loads(line)
        if r["v"] >= 2 and r["updated"] >= "2023-01-01":
            revised.append(r)
    done = set()
    if os.path.exists(PAIRS):
        for line in open(PAIRS):
            done.add(json.loads(line)["id"])
    todo = [r for r in revised if r["id"] not in done]
    print(f"stage 2: {len(revised)} revised post-2022; {len(todo)} to fetch", flush=True)
    with open(PAIRS, "a") as f:
        for i, r in enumerate(todo):
            v1 = fetch_v1_abstract(r["id"])
            if v1:
                f.write(json.dumps({"id": r["id"], "updated": r["updated"],
                                    "v": r["v"], "abstract_v1": v1,
                                    "abstract_latest": r["abstract_latest"]}) + "\n")
            if (i + 1) % 50 == 0:
                f.flush(); print(f"  stage2 {i + 1}/{len(todo)}", flush=True)
            time.sleep(3.1)

if __name__ == "__main__":
    stage1()
    stage2()
    print("n3 harvest complete")
