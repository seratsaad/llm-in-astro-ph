#!/usr/bin/env python3
"""Stage 1 (Phase B): stream the OCR full text.

One pass over the year shards produces three things at once:

  1. per-paper body and per-section-group token counts + marker counts,
     with the abstract, references, tables and equations excluded so that
     Phase B is not simply re-reading the Phase A text;
  2. per-paper detection of an explicit writing-assistance declaration
     (the positive labels D_i for the positive-unlabelled layer);
  3. per-year document frequency for the whole vocabulary, which feeds
     marker discovery in Stage 5.

Shards are processed in parallel, one worker per shard.
"""
import collections
import gzip
import json
import os
import re
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from common import (KG, DATA, WATCH, MARKERS, CONTROL, PLACEBO, TOKEN_RE,
                    YEAR_MIN, id_to_ym, quarter_index, QUARTER_MAX)

SHARDS = ["2012-2015", "2016-2019", "2020-2023", "2024-2025", "2026"]

# ------------------------------------------------------------------ cleaning
HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)$")
# "2.1 Target selection" / "III.B Fitting" are subsections: they must inherit the
# enclosing section rather than resetting it, or most of the body lands in "other".
NUMPREFIX_RE = re.compile(r"^\s*((?:\d+|[IVXLC]+)(?:[.\-]\s*(?:\d+|[A-Z]|[ivx]+))*)[.\s]")
TABLE_RE = re.compile(r"^\s*\|")
EQ_BLOCK_RE = re.compile(r"\$\$.*?\$\$", re.S)
EQ_INLINE_RE = re.compile(r"\$[^$\n]{0,200}\$")
FIG_TAG_RE = re.compile(r"^\s*\[Figure:.*?\]\s*$")
AFFIL_RE = re.compile(r"^\s*(Affiliation|Email|E-mail)\s*:", re.I)
URL_RE = re.compile(r"https?://\S+|www\.\S+")
CITE_RE = re.compile(r"\(\s*[A-Z][A-Za-z'`-]+\s+(et al\.?)?\s*,?\s*\d{4}[a-z]?\s*\)")

SECTION_PATTERNS = [
    ("abstract", re.compile(r"^(abstract|key\s?words|keywords|contents)\b", re.I)),
    ("references", re.compile(r"^(references|bibliography|reference list)\b", re.I)),
    ("acknowledgements", re.compile(r"^(acknowledg|funding|author contribution|"
                                    r"data availability|software|facilit|"
                                    r"conflict of interest|orcid)", re.I)),
    ("appendix", re.compile(r"^(appendix|supplementary|supporting information)\b", re.I)),
    ("introduction", re.compile(r"^(introduction|background|motivation)\b", re.I)),
    ("methods", re.compile(r"^(methods?|methodology|observations?|data\b|"
                           r"sample|instrument|simulations?|modell?ing|"
                           r"target selection|reduction)", re.I)),
    ("results", re.compile(r"^(results?|analysis|measurements?)\b", re.I)),
    ("discussion", re.compile(r"^(discussion|interpretation|implications?)\b", re.I)),
    ("conclusions", re.compile(r"^(conclusions?|summary|concluding remarks)\b", re.I)),
]
# Section groups reported in the paper. Anything unmatched goes to "other".
BODY_SECTIONS = {"introduction", "methods", "results", "discussion",
                 "conclusions", "other"}

# ------------------------------------------------------------------ disclosure
# Unambiguous named tools, or an explicit statement of language-model use.
# Two astronomy-specific traps: "Gemini" is a telescope and "Claude" is a common
# given name. Both are required to carry a vendor or version qualifier, or they
# match acknowledgements of observatory staff and of colleagues named Claude.
TOOL_RE = re.compile(
    r"\b(chatgpt|chat\s?gpt|gpt-?[345](?:\.\d)?|gpt-?4o|openai|"
    r"anthropic|(?:anthropic\W{0,3})?claude(?:\s*(?:opus|sonnet|haiku|ai|"
    r"\d(?:\.\d)?))|"
    r"(?:google\W{0,3})?gemini\s*(?:pro|flash|advanced|\d(?:\.\d)?)|"
    r"copilot|llama\s?-?[23]|mistral\s?ai|deepseek|"
    r"perplexity\s?ai|writefull|grammarly|deepl)\b", re.I)
LLM_PHRASE_RE = re.compile(
    r"\b(large language model|language model|generative ai|generative artificial "
    r"intelligence|ai (?:writing )?assistant|ai[- ]generated text|llm)s?\b", re.I)
WRITING_RE = re.compile(
    r"\b(writing|wrote|write|prose|grammar|grammatical|language edit|"
    r"edit(?:ing|ed)?|polish|proofread|copy-?edit|phrasing|wording|style|"
    r"readability|manuscript|text|draft|translat)\w*\b", re.I)
# Sentences that are about LLMs as a research subject, not a writing tool.
SUBJECT_RE = re.compile(
    r"\b(we (?:train|fine-?tune|evaluate|benchmark|develop)|"
    r"our (?:model|agent|pipeline|system)|"
    r"astro(?:llama|sage|bert)|benchmark|fine-?tun|token(?:iz|s\b)|"
    r"embedding|transformer architecture|retrieval-augmented)\w*", re.I)
# Sentences that DENY use ("No generative AI was used ...") must not count.
NEG_RE = re.compile(
    r"(?i)\b(?:no|not|never|without)\b[\w\s,'()-]{0,60}"
    r"\b(?:use[dr]?|employ|appli|draft|writ|generat|creat)")
# An authorial cue: the tool is reported as having been *used by the authors*.
USE_RE = re.compile(
    r"\b((?:we|i|the authors?)\s+(?:also\s+|would like to\s+)?"
    r"(?:used?|have used|made use of|employ\w*|utili\w+|acknowledge|thank|"
    r"declare|disclose|applied)"
    r"|(?:made|make)\s+use\s+of"
    r"|(?:the\s+)?use\s+of"
    r"|(?:was|were|is|are|has been|have been)\s+"
    r"(?:used|employed|applied|utili\w+|adopted)"
    r"|with the (?:help|aid|assistance) of"
    r"|assisted by|aided by|assistance in|assistance with"
    r"|for (?:writing|language|grammar|editorial)\s+(?:assistance|help|support)"
    r"|(?:employed|used|applied)\s+(?:to|for|in)"
    r"|prepared using|generated using|polished|proofread)", re.I)
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
# Above this many tool/LLM mentions the paper is about language models, so a
# single sentence match is far likelier to be subject matter than a declaration.
LLM_SUBJECT_MENTIONS = 8


def clean_lines(text):
    """Strip tables, equations, URLs, affiliations and figure tags; keep prose."""
    text = EQ_BLOCK_RE.sub(" ", text)
    text = EQ_INLINE_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    out = []
    for line in text.split("\n"):
        if TABLE_RE.match(line) or FIG_TAG_RE.match(line) or AFFIL_RE.match(line):
            continue
        out.append(line)
    return out


def classify_header(head):
    """(section_or_None, is_subsection) for a header string."""
    m = NUMPREFIX_RE.match(head)
    depth, rest = 1, head
    if m:
        prefix = m.group(1)
        depth = len(re.split(r"[.\-]", prefix.strip().rstrip(".")))
        rest = head[m.end():].strip()
    rest = rest.strip("*_ .:")
    for name, pat in SECTION_PATTERNS:
        if pat.match(rest):
            return name, depth > 1
    return None, depth > 1


def split_sections(lines):
    """Yield (section_name, text) using markdown headers as boundaries.

    A subsection header ("2.1 Target selection") that matches no pattern
    inherits the enclosing section instead of resetting it.
    """
    current = "frontmatter"
    buf = []
    for line in lines:
        m = HEADER_RE.match(line)
        if m:
            level = len(m.group(1))
            head = m.group(2).strip()
            name, is_sub = classify_header(head)
            if buf:
                yield current, "\n".join(buf)
                buf = []
            if name is not None:
                current = name
            elif is_sub or level >= 3:
                pass                      # unlabelled subsection: keep the parent
            else:
                current = "other"
        else:
            buf.append(line)
    if buf:
        yield current, "\n".join(buf)


def detect_disclosure(text):
    """Return (declared, evidence).

    A declaration requires, within one sentence: a named tool or an explicit
    language-model phrase, writing/editing language, and an authorial cue that
    the authors themselves used it. Papers *about* language models are excluded.
    """
    n_mentions = len(TOOL_RE.findall(text)) + len(LLM_PHRASE_RE.findall(text))
    if n_mentions == 0:
        return 0, ""
    about_llms = n_mentions > LLM_SUBJECT_MENTIONS
    near_miss = ""
    for sent in SENT_SPLIT_RE.split(text):
        if not (20 < len(sent) < 500):
            continue
        if not (TOOL_RE.search(sent) or LLM_PHRASE_RE.search(sent)):
            continue
        if not WRITING_RE.search(sent):
            continue
        if SUBJECT_RE.search(sent):
            continue
        if NEG_RE.search(sent):
            continue
        if about_llms and not re.search(r"\b(we|authors?|manuscript|this paper)\b",
                                        sent, re.I):
            continue
        clean = " ".join(sent.split())[:400]
        if USE_RE.search(sent):
            return 1, clean
        near_miss = near_miss or clean
    # tool + writing language but no authorial cue: kept for the precision audit
    return 0, near_miss


def process_shard(shard):
    path = os.path.join(KG, "papers_ocr_markdowns_by_year",
                        f"papers_ocr_markdowns_{shard}.jsonl.gz")
    qmax = quarter_index(*QUARTER_MAX)
    watch_idx = {w: i for i, w in enumerate(WATCH)}
    marker_set = set(MARKERS)
    control_set = set(CONTROL)
    placebo_set = set(PLACEBO)

    rows, counts_rows, discl = [], [], []
    vocab_year = collections.defaultdict(collections.Counter)   # year -> word -> ndocs
    year_docs = collections.Counter()
    n = 0

    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            aid = rec["arxiv_id"]
            ym = id_to_ym(aid)
            if ym is None:
                continue
            year, month = ym
            if year < YEAR_MIN:
                continue
            qi = quarter_index(year, month)
            if qi < 0 or qi > qmax:
                continue
            text = rec.get("ocr_markdown") or ""
            if len(text) < 3000:
                continue

            lines = clean_lines(text)
            sec_tokens = collections.Counter()
            sec_markers = collections.Counter()
            sec_control = collections.Counter()
            sec_placebo = collections.Counter()
            body_tokens = []
            for name, chunk in split_sections(lines):
                if name in ("abstract", "references", "frontmatter",
                            "appendix", "acknowledgements"):
                    continue
                grp = name if name in BODY_SECTIONS else "other"
                toks = TOKEN_RE.findall(chunk.lower())
                sec_tokens[grp] += len(toks)
                sec_markers[grp] += sum(1 for t in toks if t in marker_set)
                sec_control[grp] += sum(1 for t in toks if t in control_set)
                sec_placebo[grp] += sum(1 for t in toks if t in placebo_set)
                body_tokens.extend(toks)

            L = len(body_tokens)
            if L < 500:                      # OCR failure or a non-article record
                continue

            c = np.zeros(len(WATCH), dtype=np.int32)
            for t in body_tokens:
                j = watch_idx.get(t)
                if j is not None:
                    c[j] += 1

            d, ev = detect_disclosure(text)
            if ev:
                discl.append({"arxiv_id": aid, "year": year,
                              "declared": d, "evidence": ev})

            SEC = ("introduction", "methods", "results", "discussion",
                   "conclusions", "other")
            row = [aid, year, month, qi, L, d]
            for s_ in SEC:
                row += [sec_tokens[s_], sec_markers[s_],
                        sec_control[s_], sec_placebo[s_]]
            rows.append(tuple(row))
            counts_rows.append(c)

            # discovery counts: document frequency of every plausible word
            uniq = {t for t in body_tokens if 3 <= len(t) <= 20}
            vy = vocab_year[year]
            for t in uniq:
                vy[t] += 1
            year_docs[year] += 1

            n += 1
            if n % 5000 == 0:
                print(f"[{shard}] {n}", flush=True)
                if n % 20000 == 0:
                    for y, cnt in vocab_year.items():
                        for w in [w for w, v in cnt.items() if v < 3]:
                            del cnt[w]

    SEC = ("introduction", "methods", "results", "discussion",
           "conclusions", "other")
    SHORT = {"introduction": "intro", "methods": "methods", "results": "results",
             "discussion": "discussion", "conclusions": "conclusions",
             "other": "other"}
    cols = ["arxiv_id", "year", "month", "q", "L", "declared"]
    for s_ in SEC:
        k = SHORT[s_]
        cols += [f"L_{k}", f"K_{k}", f"C_{k}", f"P_{k}"]
    df = pd.DataFrame(rows, columns=cols)
    C = np.vstack(counts_rows) if counts_rows else np.zeros((0, len(WATCH)), np.int32)
    for w in WATCH:
        df["w_" + w] = C[:, watch_idx[w]]

    out = os.path.join(DATA, f"ft_shard_{shard}.parquet")
    df.to_parquet(out, index=False)
    with open(os.path.join(DATA, f"ft_vocab_{shard}.json"), "w") as fh:
        json.dump({"year_docs": dict(year_docs),
                   "vocab": {str(y): {w: v for w, v in c.items() if v >= 10}
                             for y, c in vocab_year.items()}}, fh)
    with open(os.path.join(DATA, f"ft_disclosure_{shard}.json"), "w") as fh:
        json.dump(discl, fh, indent=1)
    print(f"[{shard}] DONE n={n}", flush=True)
    return shard, n


def main():
    os.makedirs(DATA, exist_ok=True)
    with Pool(len(SHARDS)) as pool:
        for shard, n in pool.imap_unordered(process_shard, SHARDS):
            print(f"finished {shard}: {n} papers", flush=True)

    parts = [pd.read_parquet(os.path.join(DATA, f"ft_shard_{s}.parquet"))
             for s in SHARDS]
    df = pd.concat(parts, ignore_index=True).drop_duplicates("arxiv_id")
    df["K_marker"] = df[["w_" + w for w in MARKERS]].sum(axis=1)
    df["K_control"] = df[["w_" + w for w in CONTROL]].sum(axis=1)
    df["K_placebo"] = df[["w_" + w for w in PLACEBO]].sum(axis=1)
    df.to_parquet(os.path.join(DATA, "fulltext_features.parquet"), index=False)
    print("fulltext_features:", df.shape)
    print(df.groupby("year").agg(n=("L", "size"), medL=("L", "median"),
                                 declared=("declared", "mean"),
                                 mk=("K_marker", "sum")).round(5))


if __name__ == "__main__":
    main()
