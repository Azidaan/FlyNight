"""Final layout: complete short tables, intact source pairs and readable labels."""
from pathlib import Path
source=Path(__file__).with_name('build_pdf.py');code=source.read_text(encoding='utf8')
old="dest=source.with_suffix('.pdf')";assert code.count(old)==1;code=code.replace(old,"dest=ROOT/'dissertation/reports/Сохранение личности объектов — результаты проверок и продолжение.pdf'")
code=code.replace("OUT/'pdf_receipt.json'","OUT/'pdf_v3_receipt.json'")
code=code.replace('Spacer,Table,TableStyle','Spacer,Table,TableStyle,KeepTogether')
old="def text(p):return ''.join(p._p.xpath('.//w:t/text()'))"
new="""def text(p):return ''.join(p._p.xpath('.//w:t/text()'))
def clean(v):
    for a,b in [('из20','из 20'),('из6','из 6'),('FlyVis34','FlyVis 34'),('17потерянныхсвязей','17 потерянных связей'),('117передачномеров','117 передач номеров'),('отказа107всвязи136','отказа 107→136'),('результат0,094','результат 0,094'),('Рабочие26пар','Рабочие 26 пар')]:v=v.replace(a,b)
    return v
def markup(p):
    chunks=[]
    for child in p._p:
        value=clean(''.join(child.xpath('.//w:t/text()')))
        if child.tag==qn('w:hyperlink'):
            target=p.part.rels[child.get(qn('r:id'))].target_ref
            chunks.append('<link href="'+escape(target,{'"':'&quot;'})+'">'+escape(value)+'</link>')
        else:chunks.append(escape(value))
    return ''.join(chunks)"""
assert code.count(old)==1;code=code.replace(old,new)
old='story.append(Paragraph(escape(value),style))';assert code.count(old)==1
code=code.replace(old,"\n        if value.startswith('['):style=ParagraphStyle('SourceLabel',parent=normal,keepWithNext=True)\n        story.append(Paragraph(markup(p),style))")
code=code.replace('Paragraph(escape(c.text),','Paragraph(escape(clean(c.text)),')
code=code.replace('story.append(tab);','story.append(KeepTogether([tab]));')
exec(compile(code,str(source),'exec'))
