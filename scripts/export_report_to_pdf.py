#!/usr/bin/env python3
import re
import os
import argparse
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

IMG_PATTERN = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")


def md_to_flowables(md_text: str, base_dir: Path):
    styles = getSampleStyleSheet()
    # Custom styles
    styles.add(ParagraphStyle(name='H1', parent=styles['Heading1'], spaceAfter=10))
    styles.add(ParagraphStyle(name='H2', parent=styles['Heading2'], spaceAfter=8))
    styles.add(ParagraphStyle(name='H3', parent=styles['Heading3'], spaceAfter=6))
    styles.add(ParagraphStyle(name='Body', parent=styles['BodyText'], leading=14))

    flow = []
    for raw_line in md_text.splitlines():
        line = raw_line.rstrip()
        if not line:
            flow.append(Spacer(1, 6))
            continue
        # Manual page break marker
        if line.strip() == '[[PAGEBREAK]]':
            flow.append(PageBreak())
            continue
        # Headings
        if line.startswith('# '):
            flow.append(Paragraph(line[2:].strip(), styles['H1']))
            continue
        if line.startswith('## '):
            flow.append(Paragraph(line[3:].strip(), styles['H2']))
            continue
        if line.startswith('### '):
            flow.append(Paragraph(line[4:].strip(), styles['H3']))
            continue
        # Images
        m = IMG_PATTERN.match(line.strip())
        if m:
            img_rel = m.group(1)
            img_path = (base_dir / img_rel).resolve()
            if img_path.exists():
                try:
                    # Scale to fit width with margins
                    max_width = 6.5 * inch
                    img = Image(str(img_path))
                    iw, ih = img.wrap(0, 0)
                    if iw > max_width:
                        scale = max_width / iw
                        img._restrictSize(max_width, ih * scale)
                    flow.append(img)
                    flow.append(Spacer(1, 10))
                except Exception:
                    # Fallback to a paragraph with the path
                    flow.append(Paragraph(f"[Image: {img_rel}]", styles['Body']))
            else:
                flow.append(Paragraph(f"[Missing image: {img_rel}]", styles['Body']))
            continue
        # Default paragraph
        # Escape bare angle brackets minimally
        safe = (line
                .replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;'))
        flow.append(Paragraph(safe, styles['Body']))
    return flow


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='Input Markdown file')
    ap.add_argument('--output', required=True, help='Output PDF path')
    ap.add_argument('--assets-root', default=None, help='Base directory for resolving relative images (defaults to input file dir)')
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    base_dir = Path(args.assets_root) if args.assets_root else in_path.parent

    md_text = in_path.read_text(encoding='utf-8')

    doc = SimpleDocTemplate(str(out_path), pagesize=A4, rightMargin=36, leftMargin=36, topMargin=48, bottomMargin=48)
    story = md_to_flowables(md_text, base_dir)
    doc.build(story)

    print(f"PDF written: {out_path}")


if __name__ == '__main__':
    main()
