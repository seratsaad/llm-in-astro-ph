#!/usr/bin/env python3
"""Convert POST.md into a styled HTML with embedded figures, for Chrome -> PDF."""
import os, re, base64, html

ROOT = os.path.join(os.path.dirname(__file__), "..")
MD = os.path.join(ROOT, "POST.md")
OUT_HTML = os.path.join(ROOT, "POST.html")

def img_data_uri(path):
    full = os.path.join(ROOT, path)
    with open(full, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return "data:image/png;base64," + b64

def inline(t):
    t = html.escape(t)
    t = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', t)  # links first
    t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"\*(.+?)\*", r"<em>\1</em>", t)
    return t

def convert(md):
    blocks = re.split(r"\n\s*\n", md.strip())
    out = []
    for raw in blocks:
        lines = [l.strip() for l in raw.split("\n") if l.strip()]
        if not lines:
            continue
        m_img = re.match(r"!\[(.*?)\]\((.*?)\)$", lines[0])
        if m_img:  # image line, optionally followed by its caption line
            uri = img_data_uri(m_img.group(2))
            out.append(f'<div class="figwrap"><img src="{uri}"></div>')
            rest = " ".join(lines[1:]).strip()
            if rest.startswith("*"):
                out.append(f'<p class="caption">{inline(rest)}</p>')
            continue
        b = " ".join(lines).strip()
        if b.startswith("# "):
            out.append(f"<h1>{inline(b[2:].strip())}</h1>")
        elif b.startswith("## "):
            out.append(f"<h2>{inline(b[3:].strip())}</h2>")
        elif b.startswith("*") and b.endswith("*") and not b.startswith("**"):
            out.append(f'<p class="caption">{inline(b)}</p>')
        else:
            out.append(f"<p>{inline(b)}</p>")
    return "\n".join(out)

CSS = """
@page { size: A4; margin: 20mm 18mm 20mm 18mm; }
body { font-family: 'DejaVu Serif','Georgia',serif; color:#1a1a1a;
       font-size: 11.4pt; line-height: 1.62; max-width: 740px; margin: 0 auto; }
h1 { font-size: 21pt; line-height:1.25; margin: 0 0 6px 0; }
h2 { font-size: 14.5pt; margin: 26px 0 8px 0; }
p { margin: 0 0 11px 0; text-align: justify; }
em { font-style: italic; }
strong { font-weight: 700; }
a { color:#0563C1; text-decoration: underline; }
.figwrap { text-align:center; margin: 16px 0 4px 0; page-break-inside: avoid; }
.figwrap img { max-width: 100%; height:auto; }
.caption { font-size: 9.4pt; color:#555; text-align:center; line-height:1.4;
           margin: 0 auto 16px auto; max-width: 92%; }
/* the very first italic block (the standfirst note) */
body > .caption:first-of-type { text-align:left; font-size:10pt; color:#666;
           padding-bottom:6px; margin-bottom:18px; }
"""

def main():
    md = open(MD).read()
    body = convert(md)
    doc = f"""<!doctype html><html><head><meta charset="utf-8">
<style>{CSS}</style></head><body>{body}</body></html>"""
    open(OUT_HTML, "w").write(doc)
    print("wrote", OUT_HTML, f"({len(doc)//1024} KB)")

if __name__ == "__main__":
    main()
