"""Preserve Word source hyperlinks in the final PDF; retain first layout draft."""
from pathlib import Path
source=Path(__file__).with_name('build_pdf.py');code=source.read_text(encoding='utf8')
old="dest=source.with_suffix('.pdf')";assert code.count(old)==1;code=code.replace(old,"dest=source.with_name(source.stem+' — итог.pdf')")
code=code.replace("OUT/'pdf_receipt.json'","OUT/'pdf_v2_receipt.json'")
old="def text(p):return ''.join(p._p.xpath('.//w:t/text()'))"
new="""def text(p):return ''.join(p._p.xpath('.//w:t/text()'))
def markup(p):
    chunks=[]
    for child in p._p:
        value=''.join(child.xpath('.//w:t/text()'))
        if child.tag==qn('w:hyperlink'):
            target=p.part.rels[child.get(qn('r:id'))].target_ref
            chunks.append('<link href="'+escape(target,{'"':'&quot;'})+'">'+escape(value)+'</link>')
        else:chunks.append(escape(value))
    return ''.join(chunks)"""
assert code.count(old)==1;code=code.replace(old,new)
old='story.append(Paragraph(escape(value),style))';assert code.count(old)==1;code=code.replace(old,'story.append(Paragraph(markup(p),style))')
exec(compile(code,str(source),'exec'))
