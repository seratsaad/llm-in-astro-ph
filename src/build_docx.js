// Build POST.docx from POST.md: serif body, bold headings (no rules),
// centered figures, italic captions. No decorative lines.
const fs = require("fs");
const path = require("path");
const { Document, Packer, Paragraph, TextRun, ImageRun, AlignmentType } = require("docx");

const ROOT = path.join(__dirname, "..");
const MD = fs.readFileSync(path.join(ROOT, "POST.md"), "utf8");

const FONT = "DejaVu Serif";
const BODY = 22;      // 11 pt (half-points)
const CAP = 18;       // 9 pt
const H2 = 28;        // 14 pt
const TITLE = 40;     // 20 pt
const TARGET_W = 580; // px, image width to fit page

function pngSize(buf) {
  return { w: buf.readUInt32BE(16), h: buf.readUInt32BE(20) };
}

// inline **bold** / *italic* -> TextRun[]
function runs(text, opts = {}) {
  const base = { font: FONT, size: opts.size || BODY, italics: !!opts.italics, color: "1A1A1A" };
  const out = [];
  const re = /(\*\*([^*]+)\*\*|\*([^*]+)\*)/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(new TextRun({ ...base, text: text.slice(last, m.index) }));
    if (m[2] !== undefined) out.push(new TextRun({ ...base, text: m[2], bold: true }));
    else out.push(new TextRun({ ...base, text: m[3], italics: true }));
    last = re.lastIndex;
  }
  if (last < text.length) out.push(new TextRun({ ...base, text: text.slice(last) }));
  return out;
}

const children = [];
const blocks = MD.trim().split(/\n\s*\n/);

function addImage(imgRel) {
  const buf = fs.readFileSync(path.join(ROOT, imgRel));
  const { w, h } = pngSize(buf);
  const width = TARGET_W;
  const height = Math.round((h / w) * width);
  children.push(new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 160, after: 40 },
    children: [new ImageRun({ type: "png", data: buf, transformation: { width, height } })],
  }));
}
function addCaption(text) {
  children.push(new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 160 },
    children: runs(text.replace(/^\*/, "").replace(/\*$/, ""), { size: CAP, italics: true }),
  }));
}

for (const raw of blocks) {
  const lines = raw.split("\n").map((l) => l.trim()).filter((l) => l.length);
  if (!lines.length) continue;
  // image line (optionally followed by its caption line) in the same block
  const mFirst = lines[0].match(/^!\[(.*?)\]\((.*?)\)$/);
  if (mFirst) {
    addImage(mFirst[2]);
    const rest = lines.slice(1).join(" ").trim();
    if (rest.startsWith("*")) addCaption(rest);
    continue;
  }
  const b = lines.join(" ").trim();   // join wrapped lines with spaces (no \n in runs)
  if (b.startsWith("# ")) {
    children.push(new Paragraph({
      spacing: { after: 120 },
      children: [new TextRun({ text: b.slice(2), font: FONT, size: TITLE, bold: true, color: "1A1A1A" })],
    }));
  } else if (b.startsWith("## ")) {
    children.push(new Paragraph({
      spacing: { before: 300, after: 100 },
      children: [new TextRun({ text: b.slice(3), font: FONT, size: H2, bold: true, color: "1A1A1A" })],
    }));
  } else if (b.startsWith("*") && b.endsWith("*") && !b.startsWith("**")) {
    // standfirst note (fully italic block, not a figure caption)
    children.push(new Paragraph({
      alignment: AlignmentType.LEFT,
      spacing: { after: 200 },
      children: runs(b.replace(/^\*/, "").replace(/\*$/, ""), { size: CAP, italics: true }),
    }));
  } else {
    children.push(new Paragraph({
      alignment: AlignmentType.JUSTIFIED,
      spacing: { after: 120, line: 276 },
      children: runs(b),
    }));
  }
}

const doc = new Document({
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } },
    children,
  }],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(path.join(ROOT, "POST.docx"), buf);
  console.log("wrote POST.docx", (buf.length / 1024).toFixed(0), "KB");
});
