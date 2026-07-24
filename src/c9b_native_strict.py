#!/usr/bin/env python3
"""Tighter composition control: native-English affiliation AND no listed non-native
affiliation, to reduce the 'one native co-author on a large collaboration' leakage."""
import json, os, time, urllib.parse, urllib.request
TOKEN=open(os.path.expanduser("~/.ads/dev_key")).read().strip()
BASE="https://api.adsabs.harvard.edu/v1/search/query"; DATA=os.path.join(os.path.dirname(__file__),"..","data")
NATIVE=["USA","United Kingdom","Australia","Canada"]
NONNAT=["China","Germany","France","Italy","Spain","Japan","India","Iran","Russia","Netherlands","South Korea","Brazil","Poland","Turkey"]
MARKERS=["delve","delves","delving","underscore","underscores","underscoring","intricate","showcasing","pivotal","meticulous","nuanced","realm","leveraging","tapestry","boasts","garnered","multifaceted"]
def q(query,fq):
    url=BASE+"?"+urllib.parse.urlencode({"q":query,"fq":fq,"rows":"0"})
    r=urllib.request.Request(url,headers={"Authorization":f"Bearer {TOKEN}"})
    for a in range(5):
        try:
            with urllib.request.urlopen(r,timeout=60) as x: return json.load(x)["response"]["numFound"]
        except Exception: time.sleep(0.6*(a+2))
    return -1
nat="("+" OR ".join(f'aff:"{c}"' for c in NATIVE)+")"
excl=" ".join(f'-aff:"{c}"' for c in NONNAT)
bask='abs:('+" OR ".join(f'"{t}"' for t in MARKERS)+')'
out={}
for y in [2018,2019,2020,2021,2025]:
    fq=f"year:{y}"
    tot=q(f'{nat} {excl} database:astronomy',fq)
    mk=q(f'{nat} {excl} {bask} database:astronomy',fq)
    out[y]={"total":tot,"marker":mk,"marker_pct":100*mk/tot if tot>0 else 0}
    print(f"  strict-native {y}: n={tot}  marker={out[y]['marker_pct']:.2f}%",flush=True)
    time.sleep(0.3)
base=sum(out[y]["marker"] for y in (2018,2019,2020,2021))/sum(out[y]["total"] for y in (2018,2019,2020,2021))*100
print(f"strict-native: {base:.2f}% (2018-21) -> {out[2025]['marker_pct']:.2f}% (2025)")
json.dump(out,open(os.path.join(DATA,"c9b_native_strict.json"),"w"),indent=2)
