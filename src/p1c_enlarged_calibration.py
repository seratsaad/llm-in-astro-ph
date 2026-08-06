#!/usr/bin/env python3
"""
P1c (referee round 4, point 1) -- the enlarged calibration sample.

The broadened harvest and the direct source scan add papers whose disclosure
statements we have not read one by one. Classifying them with the validated
rule from p7 is defensible because that rule reproduces the calibration
itself on the labelled set, 21/180 against 22/187, a difference of 0.04
sigma, so its errors are uncorrelated with marker presence.

Sources combined here:
  (a) the 424 statements already labelled by hand and by the assistant,
  (b) the statements added by the broadened ADS harvest,
  (c) the statements found by the outcome-blind source scan of 2025 papers,
      which also gives an ADS-independent disclosure rate.
Papers labelled by hand keep their hand label. Everything else is classified
by the rule.
Output: data/p1c_enlarged.json
"""
import json, os, re, math
import importlib.util

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data")
spec = importlib.util.spec_from_file_location("p7", os.path.join(HERE, "p7_writing_rules.py"))
p7 = importlib.util.module_from_spec(spec); spec.loader.exec_module(p7)

BASKET = set("""delve delves delving underscore underscores underscoring intricate
intricacies showcasing showcase showcases showcased boasts tapestry pivotal
meticulous meticulously nuanced garner garners garnered multifaceted commendable
noteworthy myriad plethora testament encompassing seamless seamlessly elucidate
elucidating unravel unraveling unravelling realm realms leveraging""".split())


def main():
    cls = json.load(open(os.path.join(DATA, "n1b_classified.json")))
    hand = cls["labels"]
    hand_writing = set(cls["writing_ids"])

    # statements from the ADS route, original plus broadened
    stmt = {}
    for line in open(os.path.join(DATA, "n1b_snippets.jsonl")):
        r = json.loads(line)
        if r.get("ok") and r.get("snippets"):
            stmt[r["id"]] = " ".join(s["snippet"] for s in r["snippets"][:6])

    # statements from the outcome-blind source scan
    scan_seen = scan_hit = 0
    scan_stmt = {}
    fn = os.path.join(DATA, "p1_source_scan.jsonl")
    if os.path.exists(fn):
        for line in open(fn):
            r = json.loads(line)
            if not r.get("ok"):
                continue
            scan_seen += 1
            if r.get("hit"):
                scan_hit += 1
                scan_stmt[r["id"]] = " ".join(s["snippet"] for s in r["snippets"][:6])

    # corpus lookup
    year, hit = {}, {}
    for line in open(os.path.join(DATA, "astroph_abstracts.jsonl")):
        r = json.loads(line)
        b = re.sub(r"v\d+$", "", r["id"])
        y = int(r["published"][:4])
        if 2023 <= y <= 2025:
            year[b] = y
            hit[b] = int(bool(set(re.findall(r"[a-z]+", r["abstract"].lower())) & BASKET))

    writing, source_of = set(), {}
    for pid in hand_writing:
        if pid in hit:
            writing.add(pid); source_of[pid] = "hand"
    for pid, s in stmt.items():
        if pid in hand or pid not in hit:
            continue
        if p7.is_writing(s)[0]:
            writing.add(pid); source_of[pid] = "ads_broadened"
    for pid, s in scan_stmt.items():
        if pid in hand or pid in writing or pid not in hit:
            continue
        if p7.is_writing(s)[0]:
            writing.add(pid); source_of[pid] = "source_scan"

    def q(ids):
        ids = [i for i in ids if i in hit]
        k = sum(hit[i] for i in ids); n = len(ids)
        return k, n, (k / n if n else float("nan")), \
            (math.sqrt(k / n * (1 - k / n) / n) if n else float("nan"))

    out = {}
    print("calibration sample")
    for lab, ids in (("hand-labelled only", hand_writing),
                     ("enlarged, all sources", writing),
                     ("enlarged, 2025 only", {i for i in writing if year.get(i) == 2025})):
        k, n, qq, se = q(ids)
        out[lab.replace(" ", "_").replace(",", "")] = {"k": k, "n": n, "q": qq, "se": se}
        print(f"  {lab:24s} q = {k:3d}/{n:3d} = {qq:.4f} +- {se:.4f}")

    by_src = {}
    for pid in writing:
        by_src.setdefault(source_of[pid], []).append(pid)
    print("\n  contributions:", {k: len(v) for k, v in by_src.items()})
    out["by_source"] = {k: len(v) for k, v in by_src.items()}

    if scan_seen:
        rate = scan_hit / scan_seen
        sw = sum(1 for p in scan_stmt if p7.is_writing(scan_stmt[p])[0])
        out["source_scan"] = {"papers_scanned": scan_seen, "with_any_term": scan_hit,
                              "term_rate": rate,
                              "writing_by_rule": sw,
                              "writing_rate": sw / scan_seen}
        print(f"\noutcome-blind source scan of 2025 papers")
        print(f"  scanned {scan_seen}, any model term {scan_hit} ({rate*100:.1f}%), "
              f"writing disclosure {sw} ({sw/scan_seen*100:.2f}%)")

    out["writing_ids_enlarged"] = sorted(writing)
    out["writing_ids_enlarged_2025"] = sorted(i for i in writing if year.get(i) == 2025)
    json.dump(out, open(os.path.join(DATA, "p1c_enlarged.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
