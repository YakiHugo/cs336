#!/usr/bin/env bash
# Download official CS336 lecture sources + assignment handouts, then extract
# searchable markdown under docs/ for text-first study (no videos required).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LEC_RAW="$ROOT/materials/raw/lectures"
ASN_RAW="$ROOT/materials/raw/assignments"
LEC_DOCS="$ROOT/docs/lectures"
ASN_DOCS="$ROOT/docs/handouts"
LEC_BASE="https://raw.githubusercontent.com/stanford-cs336/lectures/main"

mkdir -p "$LEC_RAW" "$ASN_RAW" "$LEC_DOCS" "$ASN_DOCS"

echo "==> Lectures (raw)"
for f in \
  lecture_01.py lecture_02.py lecture_03.pdf lecture_04.pdf lecture_05.pdf \
  lecture_06.py lecture_07.py lecture_08.pdf lecture_09.pdf lecture_10.py \
  lecture_11.pdf lecture_12.py lecture_13.py lecture_14.py lecture_15.pdf \
  lecture_16.pdf lecture_17.py \
  facts.py gpu_util.py lecture_util.py references.py README.md
do
  echo "  $f"
  curl -fsSL -o "$LEC_RAW/$f" "$LEC_BASE/$f"
done

echo "==> Assignment handouts (raw PDF)"
declare -a HANDOUTS=(
  "https://raw.githubusercontent.com/stanford-cs336/assignment1-basics/main/cs336_assignment1_basics.pdf"
  "https://raw.githubusercontent.com/stanford-cs336/assignment2-systems/main/cs336_assignment2_systems.pdf"
  "https://raw.githubusercontent.com/stanford-cs336/assignment3-scaling/main/cs336_assignment3_scaling.pdf"
  "https://raw.githubusercontent.com/stanford-cs336/assignment4-data/main/cs336_assignment4_data.pdf"
  "https://raw.githubusercontent.com/stanford-cs336/assignment5-alignment/main/cs336_spring2026_assignment5_alignment.pdf"
)
for url in "${HANDOUTS[@]}"; do
  name="$(basename "$url")"
  echo "  $name"
  curl -fsSL -o "$ASN_RAW/$name" "$url"
done

# Keep a copy next to the A1 starter if present
if [[ -d "$ROOT/assignments/assignment1-basics" ]]; then
  cp -f "$ASN_RAW/cs336_assignment1_basics.pdf" \
    "$ROOT/assignments/assignment1-basics/cs336_assignment1_basics.pdf"
fi

echo "==> Extract searchable markdown"
export ROOT
# Prefer uv ephemeral env with pypdf; fall back to any python that has pypdf
run_extract() {
  if command -v uv >/dev/null 2>&1; then
    uv run --with pypdf python - "$@"
  elif python3 -c "import pypdf" 2>/dev/null; then
    python3 - "$@"
  else
    echo "Need pypdf. Install with: uv run --with pypdf python -c 'import pypdf'" >&2
    exit 1
  fi
}

run_extract <<'PY'
from pathlib import Path
import re
import os
import pypdf

root = Path(os.environ["ROOT"])

def extract_pdf(src: Path, dest: Path) -> None:
    reader = pypdf.PdfReader(str(src))
    parts = [
        f"# {src.name}\n\n",
        "Extracted text for offline reading.\n",
        f"Local file: `{src.relative_to(root) if src.is_relative_to(root) else src}`\n",
    ]
    for i, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ""
        parts.append(f"\n\n---\n\n## Page {i}\n\n{text}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("".join(parts), encoding="utf-8")
    print(f"  PDF {src.name} -> {dest.relative_to(root)}")

hand_in = root / "materials/raw/assignments"
hand_out = root / "docs/handouts"
for pdf in sorted(hand_in.glob("*.pdf")):
    extract_pdf(pdf, hand_out / f"{pdf.stem}.md")

lec_in = root / "materials/raw/lectures"
lec_out = root / "docs/lectures"
simple_text = re.compile(r'^\s*text\(\s*f?(?P<q>["\'])(?P<body>(?:\\.|(?!\1).)*)(?P=q)')
section_fn = re.compile(r"^def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(")

def unescape(s: str) -> str:
    try:
        return bytes(s, "utf-8").decode("unicode_escape")
    except Exception:
        return s.replace("\\n", "\n").replace('\\"', '"').replace("\\'", "'")

for pdf in sorted(lec_in.glob("lecture_*.pdf")):
    extract_pdf(pdf, lec_out / f"{pdf.stem}.md")

for py in sorted(lec_in.glob("lecture_*.py")):
    lines = py.read_text(encoding="utf-8").splitlines()
    out = [
        f"# {py.stem.replace('_', ' ').title()}",
        "",
        f"Readable extraction from `{py.name}` (official executable lecture).",
        f"Source: https://github.com/stanford-cs336/lectures/blob/main/{py.name}",
        f"Interactive web view: https://cs336.stanford.edu/lectures/?trace={py.stem}",
        "",
        "---",
        "",
    ]
    for line in lines:
        m = section_fn.match(line)
        if m and m.group(1) != "main":
            out += ["", f"## `{m.group(1)}()`", ""]
            continue
        m = simple_text.match(line)
        if m:
            out.append(unescape(m.group("body")))
            out.append("")
            continue
        if "image(" in line and not line.strip().startswith("#"):
            out += [f"* [image] `{line.strip()}`", ""]
        if re.search(r"\b(link|article_link|post_link|video_link)\(", line) and "text(" not in line:
            out += [f"* [link] `{line.strip()}`", ""]
    dest = lec_out / f"{py.stem}.md"
    dest.write_text("\n".join(out), encoding="utf-8")
    print(f"  PY  {py.name} -> {dest.relative_to(root)}")

print("done")
PY

echo "==> Done"
echo "Text entrypoint: docs/text-first-study.md"
echo "Lectures:        docs/lectures/"
echo "Handouts:        docs/handouts/"
echo "Raw (gitignored): materials/raw/"
