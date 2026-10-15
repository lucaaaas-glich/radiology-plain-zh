"""Build a Chinese project report from the reviewed Markdown source.

Rendering is separate from inference. Final reports require the inspection
barrier to be released; developmental previews are visibly labelled on every page.
"""
import argparse
import html
import json
import re
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image


INK = colors.HexColor('#193746')
MUTED = colors.HexColor('#526673')
WIDTH = 491.28


def inline(text):
    text = html.escape(text)
    text = re.sub(r'\[([^\]]+)\]\((https?://[^)]+)\)',
                  r'<link href="\2" color="#336b86">\1</link>', text)
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'\1', text)
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    return re.sub(r'`([^`]+)`', r'\1', text)


def styles():
    normal = '/System/Library/Fonts/STHeiti Light.ttc'
    bold = '/System/Library/Fonts/STHeiti Medium.ttc'
    pdfmetrics.registerFont(TTFont('Chinese', normal, subfontIndex=0))
    pdfmetrics.registerFont(TTFont('Chinese-Bold', bold, subfontIndex=0))
    pdfmetrics.registerFontFamily('Chinese', normal='Chinese', bold='Chinese-Bold',
                                italic='Chinese', boldItalic='Chinese-Bold')
    base = {'fontName': 'Chinese', 'textColor': INK, 'wordWrap': 'CJK',
            'allowWidows': 0, 'allowOrphans': 0}
    return {
        'body': ParagraphStyle('body', **base, fontSize=10.3, leading=16.4, spaceAfter=8),
        'title': ParagraphStyle('title', **base, fontSize=24, leading=31, spaceAfter=16),
        'h2': ParagraphStyle('h2', **base, fontSize=15, leading=22, spaceBefore=16,
                             spaceAfter=9, keepWithNext=True),
        'h3': ParagraphStyle('h3', **base, fontSize=11.5, leading=17, spaceBefore=10,
                             spaceAfter=7, keepWithNext=True),
        'cell': ParagraphStyle('cell', **base, fontSize=9, leading=14),
        'caption': ParagraphStyle('caption', **base, fontSize=8.5, leading=13, spaceAfter=10),
        'quote': ParagraphStyle('quote', **base, fontSize=10.3, leading=16.4,
                                leftIndent=12, rightIndent=12, spaceAfter=9),
        'code': ParagraphStyle('code', **base, fontSize=8.6, leading=13,
                               leftIndent=8, rightIndent=8, spaceAfter=10),
    }


def build(source, output, stage):
    if output.exists():
        raise ValueError('Refusing to overwrite an existing report')
    if stage == 'final':
        barrier = json.loads(Path('examples/v2/final_inspection_barrier.json').read_text())
        if barrier['state'] != 'released':
            raise ValueError('Final inspection barrier is still held')
    sty = styles()
    stage_label = '开发排版预览 - 不是最终测试报告' if stage == 'development' else '第二版实测与讲解'
    story = [Paragraph('RADIOLOGY PLAIN CHINESE / V2',
                       ParagraphStyle('eyebrow', fontName='Helvetica', fontSize=9,
                                      textColor=MUTED, spaceAfter=10)),
             Paragraph(stage_label, sty['caption'])]
    lines = source.read_text().splitlines()
    title = '中文影像报告解释 - 第二版'
    i = 0
    while i < len(lines):
        line = lines[i].strip(); i += 1
        if not line:
            continue
        if line.startswith('```'):
            code = []
            while i < len(lines) and not lines[i].startswith('```'):
                code.append(html.escape(lines[i])); i += 1
            i += 1
            story.append(Paragraph('<br/>'.join(code), sty['code']))
            continue
        picture = re.fullmatch(r'!\[([^\]]*)\]\(([^)]+)\)', line)
        if picture:
            path = (source.parent/picture.group(2)).resolve()
            # Only local, already-created figures are embedded.
            if not path.is_file():
                raise ValueError('Missing local report figure: '+str(path))
            img = Image(str(path))
            img.drawHeight *= WIDTH/img.drawWidth
            img.drawWidth = WIDTH
            story.extend([img, Paragraph(inline(picture.group(1)), sty['caption'])])
            continue
        if line.startswith('|'):
            rows = [line]
            while i < len(lines) and lines[i].strip().startswith('|'):
                rows.append(lines[i].strip()); i += 1
            cells = [[Paragraph(inline(t.strip()), sty['cell']) for t in row.strip('|').split('|')]
                     for row in rows if not re.fullmatch(r'[|\s:\-]+', row)]
            columns = len(cells[0])
            if any(len(row) != columns for row in cells):
                raise ValueError('Unequal Markdown table columns')
            weights = {2: [0.38, 0.62], 3: [0.25, 0.38, 0.37],
                       4: [0.34, 0.22, 0.22, 0.22], 5: [0.28, 0.18, 0.18, 0.18, 0.18]}
            widths = [WIDTH*x for x in weights.get(columns, [1/columns]*columns)]
            table = Table(cells, colWidths=widths, repeatRows=1, hAlign='LEFT')
            table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#e8eff2')),
                ('VALIGN', (0,0), (-1,-1), 'TOP'),
                ('BOTTOMPADDING', (0,0), (-1,-1), 7),
                ('TOPPADDING', (0,0), (-1,-1), 7),
                ('LINEBELOW', (0,0), (-1,0), .6, colors.HexColor('#9ab2be')),
                ('LINEBELOW', (0,1), (-1,-1), .3, colors.HexColor('#dce4e8'))]))
            story.extend([table, Spacer(1,10)])
            continue
        if line.startswith('# '):
            title = line[2:]; story.append(Paragraph(inline(title), sty['title']))
        elif line.startswith('## '):
            story.append(Paragraph(inline(line[3:]), sty['h2']))
        elif line.startswith('### '):
            story.append(Paragraph(inline(line[4:]), sty['h3']))
        elif line.startswith('> '):
            story.append(Paragraph(inline(line[2:]), sty['quote']))
        else:
            story.append(Paragraph(inline(line), sty['body']))

    def footer(canvas, doc):
        canvas.setStrokeColor(colors.HexColor('#d4dfe4'))
        canvas.line(52, 41, 543.28, 41)
        canvas.setFont('Chinese', 8); canvas.setFillColor(MUTED)
        canvas.drawString(52, 27, stage_label+' / 项目AI复核，非临床验证')
        canvas.drawRightString(543.28, 27, str(doc.page))

    output.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(str(output), pagesize=(595.28,841.89),
        rightMargin=52, leftMargin=52, topMargin=45, bottomMargin=57,
        title=title, author='Radiology Plain Chinese research project')
    document.build(story, onFirstPage=footer, onLaterPages=footer)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--stage', required=True, choices=['development', 'final'])
    args = parser.parse_args()
    build(args.source, args.out, args.stage)
    print(args.out)


if __name__ == '__main__':
    main()
