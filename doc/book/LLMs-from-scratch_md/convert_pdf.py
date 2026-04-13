#!/usr/bin/env python3
"""
Convert "Build a Large Language Model (From Scratch)" PDF to Markdown.
One .md file per top-level chapter/appendix, based on PDF bookmarks.
Extracts images and inserts them at the correct positions.
"""

import pdfplumber
from pypdf import PdfReader
import re
import os
from collections import defaultdict

PDF_PATH = "D:/develop/github/UI_UX_LLM/doc/book/Sebastian Raschka - Build a Large Language Model (From Scratch) (2024, Manning Publications).pdf"
OUT_DIR = "D:/develop/github/UI_UX_LLM/doc/book/LLMs-from-scratch_md"

# ─── Font / size constants ────────────────────────────────────────────────────
BODY_SIZE = 14.8          # Verdana normal body
INLINE_CODE_SIZE = 11.9   # CourierNew inline code
CODE_BLOCK_SIZE = 8.9     # CourierNew code block
SMALL_BOLD_SIZE = 11.1    # Verdana-Bold: listing captions, code annotations

HEADING_SIZE_MAP = {
    # (min_size, max_size) -> markdown level
    (28.0, 99.0): 1,   # Chapter title  ~29.7
    (19.0, 28.0): 2,   # Section        ~21.0
    (15.0, 19.0): 3,   # Sub-section    ~16.7
}

LINE_THRESHOLD = 8.0  # points, chars within this Y range = same visual line

# Minimum image size to include (skip tiny decorative elements)
MIN_IMG_WIDTH = 100
MIN_IMG_HEIGHT = 60


def is_code_font(fontname: str) -> bool:
    return "courier" in fontname.lower()


def is_bold_font(fontname: str) -> bool:
    return "bold" in fontname.lower()


# ─── Outline helpers ──────────────────────────────────────────────────────────

def build_outline(reader: PdfReader):
    """Return flat list of all outline entries with depth and 0-based page."""
    items = []

    def walk(node, depth):
        for item in node:
            if isinstance(item, list):
                walk(item, depth + 1)
            else:
                try:
                    pg = reader.get_destination_page_number(item)
                    items.append({"title": item.title, "page": pg, "depth": depth})
                except Exception:
                    pass

    walk(reader.outline, 0)
    return items


def get_chapters(reader: PdfReader):
    """Top-level outline items with start/end page (0-based)."""
    all_items = build_outline(reader)
    top = [i for i in all_items if i["depth"] == 0]
    total = len(reader.pages)
    chapters = []
    for idx, item in enumerate(top):
        end = top[idx + 1]["page"] if idx + 1 < len(top) else total
        chapters.append({"title": item["title"], "start": item["page"], "end": end})
    return chapters


def get_sub_outline(all_items, start_page, end_page):
    """Outline entries within [start_page, end_page) for heading injection."""
    return [
        i for i in all_items
        if i["depth"] > 0 and start_page <= i["page"] < end_page
    ]


# ─── Per-page line extraction ─────────────────────────────────────────────────

def extract_lines(page):
    """
    Group chars by visual line (Y proximity) and return list of line-dicts:
      {y, segments: [{text, fontname, size}], is_code_block, is_inline_mixed}
    """
    chars = page.chars
    if not chars:
        return []

    buckets = defaultdict(list)
    for c in chars:
        key = round(c["top"] / LINE_THRESHOLD) * LINE_THRESHOLD
        buckets[key].append(c)

    lines = []
    for y in sorted(buckets.keys()):
        line_chars = sorted(buckets[y], key=lambda c: c["x0"])

        segments = []
        cur_fn = line_chars[0]["fontname"]
        cur_size = line_chars[0]["size"]
        cur_text = ""
        for c in line_chars:
            same_family = (is_code_font(c["fontname"]) == is_code_font(cur_fn) and
                           is_bold_font(c["fontname"]) == is_bold_font(cur_fn) and
                           abs(c["size"] - cur_size) < 1.5)
            if same_family:
                cur_text += c["text"]
            else:
                if cur_text:
                    segments.append({"text": cur_text, "fontname": cur_fn, "size": cur_size})
                cur_fn, cur_size, cur_text = c["fontname"], c["size"], c["text"]
        if cur_text:
            segments.append({"text": cur_text, "fontname": cur_fn, "size": cur_size})

        code_chars = sum(len(s["text"]) for s in segments if is_code_font(s["fontname"]))
        total_chars = sum(len(s["text"]) for s in segments)
        code_ratio = code_chars / total_chars if total_chars else 0

        is_code_block = (code_ratio > 0.85 and
                         any(abs(s["size"] - CODE_BLOCK_SIZE) < 1.5
                             for s in segments if is_code_font(s["fontname"])))
        is_inline_mixed = (0.05 < code_ratio < 0.85)

        lines.append({
            "y": y,
            "segments": segments,
            "is_code_block": is_code_block,
            "is_inline_mixed": is_inline_mixed,
        })

    return lines


def render_line_text(line) -> str:
    """Convert a line's segments to plain or markdown-inline text."""
    if line["is_code_block"]:
        return "".join(s["text"] for s in line["segments"])

    parts = []
    for seg in line["segments"]:
        text = seg["text"].strip()
        if not text:
            parts.append(" ")
            continue
        fn, size = seg["fontname"], seg["size"]
        if is_code_font(fn):
            parts.append(f"`{text}`")
        elif is_bold_font(fn) and abs(size - SMALL_BOLD_SIZE) < 1.5:
            parts.append(f"**{text}**")
        else:
            parts.append(seg["text"])

    return "".join(parts).strip()


def heading_level(seg) -> int:
    """Return markdown heading level (1-3) or 0 for non-heading."""
    if not is_bold_font(seg["fontname"]):
        return 0
    size = seg["size"]
    for (lo, hi), level in HEADING_SIZE_MAP.items():
        if lo <= size <= hi:
            return level
    return 0


# ─── Image extraction ─────────────────────────────────────────────────────────

def extract_page_images(page_plumber, page_pypdf, img_dir, page_num):
    """
    Extract images from a PDF page.
    Returns list of: {y: float, path: str (relative to img_dir), alt: str}
    Saves image files to img_dir.
    """
    os.makedirs(img_dir, exist_ok=True)

    # Build position map from pdfplumber (has y-coordinates in top-down system)
    plumber_by_name = {}
    for img in page_plumber.images:
        w = img.get("width", 0) or img.get("srcsize", [0, 0])[0]
        h = img.get("height", 0) or img.get("srcsize", [0, 0])[1]
        if w >= MIN_IMG_WIDTH and h >= MIN_IMG_HEIGHT:
            plumber_by_name[img.get("name", "")] = img

    result = []
    seen_data = set()  # deduplicate by content hash

    for pypdf_img in page_pypdf.images:
        try:
            pil_img = pypdf_img.image
            if pil_img is None:
                continue
            w, h = pil_img.size
            if w < MIN_IMG_WIDTH or h < MIN_IMG_HEIGHT:
                continue

            # Deduplicate: skip identical images
            img_data = pil_img.tobytes()
            data_hash = hash(img_data)
            if data_hash in seen_data:
                continue
            seen_data.add(data_hash)

            # Determine Y position
            name = pypdf_img.name
            y_pos = 0.0
            if name in plumber_by_name:
                y_pos = float(plumber_by_name[name].get("top", 0))

            # Save image
            safe_name = re.sub(r"[^a-z0-9]", "_", name.lower()).strip("_")
            filename = f"p{page_num:03d}_{safe_name}.png"
            out_path = os.path.join(img_dir, filename)
            pil_img.save(out_path)

            result.append({"y": y_pos, "filename": filename, "alt": ""})
        except Exception:
            pass

    return result


# ─── Page → Markdown ──────────────────────────────────────────────────────────

def page_to_markdown(page_plumber, page_pypdf, page_num, img_dir, img_rel_prefix) -> str:
    """
    Convert one PDF page to Markdown, including images.
    img_dir: absolute path where images are saved
    img_rel_prefix: relative path prefix used in Markdown img refs
    """
    text_lines = extract_lines(page_plumber)
    images = extract_page_images(page_plumber, page_pypdf, img_dir, page_num)

    # Build unified element list sorted by Y
    elements = []
    for line in text_lines:
        elements.append({"type": "text", "y": line["y"], "line": line})
    for img in images:
        elements.append({"type": "image", "y": img["y"],
                         "filename": img["filename"], "alt": img["alt"]})
    elements.sort(key=lambda e: e["y"])

    # Pre-scan: match each image to the next "Figure X.Y" bold caption
    img_captions = {}
    for i, elem in enumerate(elements):
        if elem["type"] != "image":
            continue
        for j in range(i + 1, min(i + 8, len(elements))):
            e2 = elements[j]
            if e2["type"] != "text":
                continue
            raw = "".join(s["text"] for s in e2["line"]["segments"]).strip()
            if re.match(r"figure\s+\d", raw, re.I):
                img_captions[elem["filename"]] = raw
                break

    # Render elements to Markdown
    md = []
    in_code = False
    prev_y = None

    for elem in elements:
        if elem["type"] == "image":
            # Close any open code block
            if in_code:
                md.append("```")
                md.append("")
                in_code = False

            fname = elem["filename"]
            alt = img_captions.get(fname, "Figure")
            # Trim alt to first sentence for cleanliness
            alt_short = re.split(r"(?<=\.)\s", alt)[0] if alt else "Figure"
            img_md = f"![{alt_short}]({img_rel_prefix}/{fname})"
            if md and md[-1] != "":
                md.append("")
            md.append(img_md)
            md.append("")
            prev_y = elem["y"]
            continue

        # ── Text element ──────────────────────────────────────────────────────
        line = elem["line"]
        raw = "".join(s["text"] for s in line["segments"]).strip()
        if not raw:
            if in_code:
                md.append("")
            continue

        # Code block line
        if line["is_code_block"]:
            if not in_code:
                md.append("```python")
                in_code = True
            md.append(raw)
            prev_y = elem["y"]
            continue

        # Close code block
        if in_code:
            md.append("```")
            md.append("")
            in_code = False

        # Heading detection
        first_seg = line["segments"][0]
        hlevel = heading_level(first_seg)
        if hlevel > 0:
            if md and md[-1] != "":
                md.append("")
            md.append(f"{'#' * hlevel} {raw}")
            md.append("")
            prev_y = elem["y"]
            continue

        # Body / inline-mixed
        rendered = render_line_text(line)
        if not rendered.strip():
            continue

        # Large Y gap → paragraph break
        if prev_y is not None and (elem["y"] - prev_y) > LINE_THRESHOLD * 3:
            if md and md[-1] != "":
                md.append("")

        # Listing caption or code annotation (Verdana-Bold 11.1)
        if (not line["is_inline_mixed"] and
                abs(first_seg["size"] - SMALL_BOLD_SIZE) < 1.5 and
                is_bold_font(first_seg["fontname"])):
            if md and md[-1] != "":
                md.append("")
            md.append(f"**{raw}**")
            md.append("")
        else:
            md.append(rendered)

        prev_y = elem["y"]

    # Close any open code block
    if in_code:
        md.append("```")
        md.append("")

    return "\n".join(md)


# ─── Heading merge ────────────────────────────────────────────────────────────

def merge_split_headings(md_text: str) -> str:
    """Merge heading lines split by PDF text wrapping into single headings."""
    lines = md_text.split("\n")
    result = []
    i = 0
    new_section_re = re.compile(r"^(?:\d+\.?\d*\s+\w|appendix\s+\w)", re.I)
    while i < len(lines):
        line = lines[i]
        m = re.match(r"^(#+)\s+(.*)", line)
        if m:
            level, text = m.group(1), m.group(2).strip()
            j = i + 1
            blanks = 0
            while j < len(lines) and lines[j].strip() == "" and blanks < 2:
                j += 1
                blanks += 1
            if j < len(lines):
                m2 = re.match(r"^(#+)\s+(.*)", lines[j])
                if m2 and m2.group(1) == level:
                    next_text = m2.group(2).strip()
                    if not new_section_re.match(next_text):
                        result.append(f"{level} {text} {next_text}")
                        i = j + 1
                        continue
        result.append(line)
        i += 1
    return "\n".join(result)


# ─── Chapter → Markdown ───────────────────────────────────────────────────────

def chapter_to_markdown(pdf_plumber, reader, chapter, all_outline,
                        chapter_img_dir, chapter_img_rel) -> str:
    """
    Convert all pages of a chapter to one Markdown string.
    chapter_img_dir: absolute path for this chapter's images
    chapter_img_rel: relative path from .md file to image dir
    """
    start, end = chapter["start"], chapter["end"]

    parts = []

    sub_outline = get_sub_outline(all_outline, start, end)
    page_to_sub = defaultdict(list)
    for item in sub_outline:
        page_to_sub[item["page"]].append(item)

    for page_idx in range(start, end):
        try:
            page_plumber = pdf_plumber.pages[page_idx]
            page_pypdf = reader.pages[page_idx]
        except IndexError:
            break

        content = page_to_markdown(
            page_plumber, page_pypdf,
            page_num=page_idx + 1,
            img_dir=chapter_img_dir,
            img_rel_prefix=chapter_img_rel,
        )
        if content.strip():
            parts.append(content)
            parts.append("")

    combined = "\n".join(parts).strip() + "\n"
    return merge_split_headings(combined)


# ─── File naming ──────────────────────────────────────────────────────────────

def chapter_filename(title: str, index: int) -> str:
    t = title.lower().strip()
    m = re.match(r"appendix\s+([a-e])\s+(.*)", t)
    if m:
        letter, name = m.group(1), m.group(2)
        slug = re.sub(r"[^a-z0-9]+", "_", name).strip("_")
        return f"appendix_{letter}_{slug[:40]}.md"
    m = re.match(r"(\d+)\s+(.*)", t)
    if m:
        num, name = int(m.group(1)), m.group(2)
        slug = re.sub(r"[^a-z0-9]+", "_", name).strip("_")
        return f"ch{num:02d}_{slug[:40]}.md"
    slug = re.sub(r"[^a-z0-9]+", "_", t).strip("_")
    return f"{index:02d}_{slug[:40]}.md"


def chapter_img_folder(filename: str) -> str:
    """Return subfolder name for this chapter's images, e.g. 'ch01'."""
    base = os.path.splitext(filename)[0]  # strip .md
    # Use first segment: ch01, appendix_a, 00, etc.
    return base.split("_")[0]


# ─── Chapters to skip / front matter ─────────────────────────────────────────

SKIP_TITLES = {"copyright", "contents"}

FRONT_MATTER_TITLES = {
    "build a large language model (from scratch)",
    "preface", "acknowledgments",
    "about this book", "about the author",
    "about the cover illustration",
}


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    reader = PdfReader(PDF_PATH)
    all_outline = build_outline(reader)
    chapters = get_chapters(reader)

    print(f"Found {len(chapters)} top-level sections in PDF")

    readme_entries = []
    front_matter_parts = []

    with pdfplumber.open(PDF_PATH) as pdf:
        file_index = 0
        for ch in chapters:
            title = ch["title"]
            title_lower = title.lower().strip()

            if title_lower in SKIP_TITLES:
                print(f"  [skip] {title}")
                continue

            # Front matter → group into one file (no image extraction for simplicity)
            if title_lower in FRONT_MATTER_TITLES:
                print(f"  + Front matter: {title}")
                sub_outline = get_sub_outline(all_outline, ch["start"], ch["end"])
                page_to_sub = defaultdict(list)
                for item in sub_outline:
                    page_to_sub[item["page"]].append(item)

                if title_lower == "build a large language model (from scratch)":
                    front_matter_parts.append(f"# {title}")
                else:
                    front_matter_parts.append(f"## {title}")
                front_matter_parts.append("")

                for page_idx in range(ch["start"], ch["end"]):
                    try:
                        page_pl = pdf.pages[page_idx]
                        page_py = reader.pages[page_idx]
                    except IndexError:
                        break

                    # Front matter: text only, no images for now
                    content = page_to_markdown(
                        page_pl, page_py,
                        page_num=page_idx + 1,
                        img_dir=os.path.join(OUT_DIR, "images", "front"),
                        img_rel_prefix="images/front",
                    )
                    if content.strip():
                        front_matter_parts.append(content)
                        front_matter_parts.append("")
                continue

            # Regular chapter / appendix
            filename = chapter_filename(title, file_index)
            folder = chapter_img_folder(filename)
            img_dir = os.path.join(OUT_DIR, "images", folder)
            img_rel = f"images/{folder}"

            print(f"  -> Converting: {title} (pages {ch['start']+1}-{ch['end']})")
            md_content = chapter_to_markdown(
                pdf, reader, ch, all_outline,
                chapter_img_dir=img_dir,
                chapter_img_rel=img_rel,
            )

            out_path = os.path.join(OUT_DIR, filename)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(md_content)

            # Count extracted images
            n_imgs = len(os.listdir(img_dir)) if os.path.isdir(img_dir) else 0
            readme_entries.append((filename, title))
            file_index += 1
            print(f"     [ok] {filename} ({len(md_content):,} chars, {n_imgs} images)")

    # Write front matter
    if front_matter_parts:
        fm_path = os.path.join(OUT_DIR, "00_front_matter.md")
        with open(fm_path, "w", encoding="utf-8") as f:
            f.write("\n".join(front_matter_parts).strip() + "\n")
        readme_entries.insert(0, ("00_front_matter.md", "Front Matter"))
        print(f"  [ok] 00_front_matter.md")

    # Write README
    readme_lines = [
        "# Build a Large Language Model (From Scratch)",
        "",
        "**Author:** Sebastian Raschka  ",
        "**Publisher:** Manning Publications, 2024",
        "",
        "## Contents",
        "",
    ]
    for fname, title in readme_entries:
        readme_lines.append(f"- [{title}]({fname})")

    with open(os.path.join(OUT_DIR, "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(readme_lines) + "\n")

    print(f"\nDone! Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
