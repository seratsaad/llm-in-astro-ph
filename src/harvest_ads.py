#!/usr/bin/env python3
"""
Harvest LLM-disclosure counts from NASA ADS, year by year, for astronomy papers.

Key idea: ADS parses the ACKNOWLEDGMENTS section (`ack:` field) from full text
where available. A paper that *used* an LLM discloses it in acknowledgments/
methods, whereas a paper *about* LLMs mentions them in title/abstract. So
`ack:"ChatGPT"` is a far cleaner "used it" signal than an abstract search.

We keep an unambiguous keyword set for the headline disclosure count, and track
confounded terms (Gemini = a telescope; Claude = a common personal name) SEPARATELY
so they never pollute the headline number.

Denominator each year = all astronomy papers (database:astronomy), so we can
report a disclosure *fraction*.

Output: data/ads_disclosure.json
"""
import json, time, os, urllib.parse, urllib.request

TOKEN = open(os.path.expanduser("~/.ads/dev_key")).read().strip()
BASE = "https://api.adsabs.harvard.edu/v1/search/query"
YEARS = list(range(2015, 2027))
SLEEP = 0.4

# Unambiguous LLM terms -> used for the headline "any disclosure" union.
UNAMBIG = [
    'ChatGPT', 'Chat-GPT', 'GPT-4', 'GPT-4o', 'GPT-3.5', 'GPT-3',
    'large language model', 'large language models',
    'GitHub Copilot', 'generative pre-trained transformer',
]
# Tracked individually but confounded -> reported with caveats, NOT in headline.
CONFOUNDED = ['Gemini', 'Claude', 'Copilot', 'OpenAI', 'Bard', 'LLM', 'Llama']

def q(query, fq=None):
    params = {"q": query, "rows": "0", "fl": "id"}
    if fq:
        params["fq"] = fq
    url = BASE + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {TOKEN}"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                d = json.load(r)
            return d["response"]["numFound"]
        except Exception as e:
            time.sleep(SLEEP * (attempt + 2))
            last = e
    raise RuntimeError(f"ADS query failed: {query} :: {last}")

def field_union(field, terms):
    inner = " OR ".join(f'"{t}"' for t in terms)
    return f'{field}:({inner})'

def main():
    out = {"years": YEARS, "total_astro": {}, "ack": {}, "full": {},
           "ack_confounded": {}, "abstract_confounded": {}}
    for y in YEARS:
        fq = f"year:{y}"
        out["total_astro"][y] = q("database:astronomy", fq)
        # headline: acknowledgments union (unambiguous)
        out["ack"][y] = q(field_union("ack", UNAMBIG) + " database:astronomy", fq)
        # cross-check: full-text union (unambiguous)
        out["full"][y] = q(field_union("full", UNAMBIG) + " database:astronomy", fq)
        # individual unambiguous terms in ack (for a stacked/detail view)
        for t in ['ChatGPT', 'GPT-4', 'large language model', 'GitHub Copilot']:
            out.setdefault("ack_terms", {}).setdefault(t, {})[y] = q(f'ack:"{t}" database:astronomy', fq)
        # confounded terms in ack (report separately, hand-check needed)
        for t in CONFOUNDED:
            out["ack_confounded"].setdefault(t, {})[y] = q(f'ack:"{t}" database:astronomy', fq)
        time.sleep(SLEEP)
        print(f"{y}: total={out['total_astro'][y]:>7}  ack_LLM={out['ack'][y]:>4}  full_LLM={out['full'][y]:>5}", flush=True)
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "data", "ads_disclosure.json"), "w"), indent=2)
    print("WROTE data/ads_disclosure.json")

if __name__ == "__main__":
    main()
