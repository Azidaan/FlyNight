"""Create the final report as PDF from the same DOCX content and native TNR fonts."""
import json,hashlib
from pathlib import Path
from xml.sax.saxutils import escape
from docx import Document
from docx.text.paragraph import Paragraph as WordParagraph
from docx.table import Table as WordTable
from docx.oxml.ns import qn
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY,TA_CENTER,TA_LEFT
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle
from pypdf import PdfReader
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent;r=json.loads((OUT/'document_receipt.json').read_text(encoding='utf8'));source=ROOT/r['document'];dest=source.with_suffix('.pdf');assert not dest.exists()
for name,path in [('TNR','times.ttf'),('TNR-Bold','timesbd.ttf')]:pdfmetrics.registerFont(TTFont(name,str(Path('C:/Windows/Fonts')/path)))
normal=ParagraphStyle('Normal',fontName='TNR',fontSize=14,leading=21,firstLineIndent=1.25*cm,alignment=TA_JUSTIFY,spaceAfter=0,allowWidows=0,allowOrphans=0)
heading=ParagraphStyle('Heading',parent=normal,fontName='TNR-Bold',spaceBefore=10,spaceAfter=5,keepWithNext=True)
title=ParagraphStyle('Title',parent=heading,alignment=TA_CENTER,firstLineIndent=0,spaceAfter=12)
cellleft=ParagraphStyle('CellLeft',parent=normal,firstLineIndent=0,alignment=TA_LEFT)
cellcenter=ParagraphStyle('CellCenter',parent=cellleft,alignment=TA_CENTER)
cellbold=ParagraphStyle('CellBold',parent=cellcenter,fontName='TNR-Bold')
doc=Document(source);story=[];paragraph_count=0;table_count=0
def text(p):return ''.join(p._p.xpath('.//w:t/text()'))
for node in doc.element.body:
    if node.tag==qn('w:p'):
        p=WordParagraph(node,doc);value=text(p)
        if not value.strip():story.append(Spacer(1,5));continue
        style=title if p.style.name=='Title' else heading if p.style.name.startswith('Heading') else normal
        story.append(Paragraph(escape(value),style));paragraph_count+=1
    elif node.tag==qn('w:tbl'):
        t=WordTable(node,doc);data=[]
        for ri,row in enumerate(t.rows):
            data.append([Paragraph(escape(c.text),cellbold if ri==0 else cellleft if ci==0 else cellcenter) for ci,c in enumerate(row.cells)])
        widths=[c.width for c in t.columns];total=sum(widths);widths=[float(w)/total*16.5*cm for w in widths]
        tab=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT');tab.setStyle(TableStyle([('GRID',(0,0),(-1,-1),.5,colors.HexColor('#D9D9D9')),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E7E6E6')),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5)]));story.append(tab);story.append(Spacer(1,6));table_count+=1
def footer(canvas,doc):
    canvas.saveState();canvas.setFont('TNR',14);canvas.drawCentredString(10.5*cm,1*cm,str(doc.page));canvas.restoreState()
SimpleDocTemplate(str(dest),pagesize=(21*cm,29.7*cm),leftMargin=3*cm,rightMargin=1.5*cm,topMargin=2*cm,bottomMargin=2*cm,title=doc.core_properties.title,author='').build(story,onFirstPage=footer,onLaterPages=footer)
pdf=PdfReader(dest);pages=len(pdf.pages);full='\n'.join(p.extract_text() for p in pdf.pages);assert '17' in full and '0,036868' in full and '107' in full and 'Первичные материалы' in full
with (OUT/'pdf_receipt.json').open('x',encoding='utf8') as f:json.dump(dict(passed=True,pages=pages,paragraphs=paragraph_count,tables=table_count,pdf=dest.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),source_docx_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),font='Embedded Times New Roman14',line_spacing=1.5,first_line_cm=1.25,body_alignment='justify',word_render_unavailable=True,scope='PDF is the final independently rendered report; the DOCX remains an editable unverified intermediate.'),f,ensure_ascii=False,indent=2)
print('FINAL_IDENTITY_PDF_READY',pages,flush=True)
