#!/usr/bin/env python3
"""Patient GDELT cache refill: loop over missing (word_co, word_base) queries
at 75 s spacing until the cache is complete. Resumable and idempotent."""
import json, os, time, urllib.parse, urllib.request

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
CACHE = os.path.join(DATA, "n2b_gdelt_cache")
GDELT = "https://api.gdeltproject.org/api/v2/doc/doc"
UA = {"User-Agent": "academic-corpus-study/1.0"}
SLEEP = 75

WORDS = """delve delves delving underscore underscores underscoring intricate
intricacies showcasing showcase showcases showcased boasts tapestry pivotal
meticulous meticulously nuanced garner garners garnered multifaceted commendable
noteworthy myriad plethora testament encompassing seamless seamlessly elucidate
elucidating unravel unraveling unravelling realm realms leveraging offering
highlighting""".split()

def want():
    out = []
    for w in WORDS:
        for kind, q in (("co", f'"{w}" ("ChatGPT" OR "artificial intelligence")'),
                        ("base", f'"{w}"')):
            tag = f"{w}_{kind}"
            if not os.path.exists(os.path.join(CACHE, tag + ".json")):
                out.append((tag, q))
    return out

def main():
    os.makedirs(CACHE, exist_ok=True)
    rounds = 0
    while True:
        todo = want()
        if not todo:
            print("cache complete", flush=True)
            return
        rounds += 1
        print(f"round {rounds}: {len(todo)} queries missing", flush=True)
        for tag, q in todo:
            url = GDELT + "?" + urllib.parse.urlencode(
                {"query": q, "mode": "timelinevol", "format": "json",
                 "startdatetime": "20230101000000", "enddatetime": "20251231235959"})
            try:
                req = urllib.request.Request(url, headers=UA)
                with urllib.request.urlopen(req, timeout=60) as r:
                    txt = r.read().decode("utf-8", "ignore")
                d = json.loads(txt)
                if "timeline" in d:
                    json.dump(d, open(os.path.join(CACHE, tag + ".json"), "w"))
                    print(f"  ok {tag}", flush=True)
            except Exception as e:
                print(f"  miss {tag}: {e}", flush=True)
            time.sleep(SLEEP)

if __name__ == "__main__":
    main()
