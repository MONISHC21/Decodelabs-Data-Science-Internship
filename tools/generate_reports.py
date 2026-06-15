"""Generate professional PDF reports for each Week using ReportLab.

This script composes a multi-page PDF report for each week folder that
contains a `reports/WeekN_Report.md` file and a `charts/` folder of images.

Requirements: reportlab, pillow (PIL). Install in the project's venv if missing.

Usage:
    python tools/generate_reports.py

Outputs:
    Week_1_EDA_Feature_Engineering/reports/Week1_Report.pdf
    Week_2_Fraud_Detection/reports/Week2_Report.pdf
    Week_3_Customer_Segmentation/reports/Week3_Report.pdf
"""
from pathlib import Path
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
import textwrap
import sys
import os
from PIL import Image as PILImage


def md_to_paragraphs(md_text):
    # Very small markdown-to-plain conversion for titles and paragraphs
    lines = md_text.splitlines()
    out = []
    buf = []
    for ln in lines:
        ln = ln.strip()
        if ln.startswith('#'):
            if buf:
                out.append('\n'.join(buf))
                buf = []
            out.append(ln.lstrip('#').strip())
        else:
            buf.append(ln)
    if buf:
        out.append('\n'.join(buf))
    return out


def collect_images(charts_dir):
    imgs = []
    if not charts_dir.exists():
        return imgs
    for ext in ('*.png', '*.jpg', '*.jpeg'):
        for p in sorted(charts_dir.glob(ext)):
            imgs.append(p)
    return imgs


def build_report(week_dir: Path):
    reports_dir = week_dir / 'reports'
    charts_dir = week_dir / 'charts'
    outputs_dir = week_dir / 'outputs'
    models_dir = week_dir / 'models'

    # Find the markdown report
    md_files = list(reports_dir.glob('*.md')) if reports_dir.exists() else []
    if not md_files:
        print(f'No markdown report found in {reports_dir}, skipping')
        return
    md_path = md_files[0]

    pdf_path = reports_dir / (md_path.stem + '.pdf')

    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    normal = styles['Normal']
    heading = ParagraphStyle('Heading', parent=styles['Heading1'], spaceAfter=12)
    h2 = ParagraphStyle('H2', parent=styles['Heading2'], spaceAfter=8)

    story = []

    # Cover page
    title = week_dir.name.replace('_', ' ')
    story.append(Paragraph(title, heading))
    story.append(Spacer(1, 12))
    story.append(Paragraph('Project: Customer/Data Science Internship', normal))
    story.append(Paragraph(f'Path: {week_dir}', normal))
    story.append(Spacer(1, 24))

    # Read md and add sections
    md_text = md_path.read_text(encoding='utf-8')
    parts = md_to_paragraphs(md_text)
    for part in parts:
        if len(part) < 80 and part.isupper() == False and part.strip() and '\n' not in part and part == part.title():
            story.append(Paragraph(part, h2))
        else:
            # wrap long lines
            wrapped = '\n'.join(textwrap.wrap(part, 200))
            story.append(Paragraph(wrapped, normal))
        story.append(Spacer(1, 12))

    # KPI table
    kpi_data = [['KPI', 'Value']]
    # dataset size
    size_text = 'N/A'
    cleaned = outputs_dir / 'cleaned_customers.csv'
    if cleaned.exists():
        try:
            import pandas as pd
            dfc = pd.read_csv(cleaned)
            size_text = f"{dfc.shape[0]} rows × {dfc.shape[1]} cols"
        except Exception:
            size_text = 'Exists'
    kpi_data.append(['Dataset size', size_text])
    # features
    feat_text = 'N/A'
    if cleaned.exists():
        try:
            feat_text = ', '.join(list(dfc.select_dtypes(include=['number']).columns)[:6])
        except Exception:
            feat_text = 'numeric features'
    kpi_data.append(['Features (sample)', feat_text])
    # missing values
    miss = 'N/A'
    if cleaned.exists():
        miss = str(int(dfc.isna().sum().sum()))
    kpi_data.append(['Missing values', miss])
    # models count
    models_count = len(list(models_dir.glob('*'))) if models_dir.exists() else 0
    kpi_data.append(['Models trained', str(models_count)])

    t = Table(kpi_data, colWidths=[200, 300])
    t.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#003366')), ('TEXTCOLOR', (0, 0), (-1, 0), colors.white), ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)]))
    story.append(Spacer(1, 12))
    story.append(t)
    story.append(Spacer(1, 24))

    # Insert charts
    imgs = collect_images(charts_dir)
    for img in imgs:
        # validate image with Pillow first
        try:
            with PILImage.open(img) as pil:
                pil.load()
                w_px, h_px = pil.size
        except Exception as e:
            print('Skipping unreadable image', img, e)
            continue

        try:
            story.append(Paragraph(img.name.replace('_', ' ').replace('.png', ''), h2))
            # preserve aspect ratio, constrain width to 6in
            max_w = 6 * inch
            aspect = h_px / float(w_px) if w_px else 0.5
            draw_w = max_w
            draw_h = draw_w * aspect
            im = Image(str(img), width=draw_w, height=draw_h)
            story.append(im)
            story.append(Spacer(1, 12))
        except Exception as e:
            print('Failed to include image', img, e)

    story.append(PageBreak())

    # Appendix: include short code snippets (first 200 lines of md if any)
    story.append(Paragraph('Appendix', h2))
    snippet = '\n'.join(md_text.splitlines()[:50])
    story.append(Paragraph(snippet.replace('\n', '<br/>'), normal))

    doc.build(story)
    print('Wrote PDF:', pdf_path)


def main():
    repo = Path(__file__).resolve().parents[1]
    weeks = [p for p in repo.glob('Week_*') if p.is_dir()]
    for w in weeks:
        reports_dir = w / 'reports'
        if reports_dir.exists():
            build_report(w)


if __name__ == '__main__':
    main()
