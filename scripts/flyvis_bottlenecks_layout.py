"""Editable second draw.io sheet: one horizontal strip of evidence-ranked bottlenecks."""
from pathlib import Path
import copy, hashlib, html, json, os, sys
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw, ImageFont
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
W,H=3900,1320
STEM='FlyVis_bottlenecks'
FONTS='C:/Windows/Fonts/'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def diagram_digest(diagram):
    d=copy.deepcopy(diagram);d.tail=None
    return hashlib.sha256(ET.tostring(d,encoding='utf8')).hexdigest()


def build(mx,out):
    out=Path(out)
    assert not any(d.get('id')=='flyvis-bottlenecks' for d in mx.findall('diagram'))
    paths=['runs/z13_box_supervision_followup_20261025/strict_action_gate.json',
           'runs/z13_box_supervision_followup_20261025/temporal_readiness.json',
           'runs/z13_spatial_set_readout_20261025/recovery2/training/internal_report.json',
           'tracking/z13_spatial.py',
           'dissertation/architecture/Пространственное_считывание_отложено_2026-10-25.md']
    evidence={p:sha(ROOT/p) for p in paths}
    load=lambda p:json.loads((ROOT/p).read_text(encoding='utf8'))
    reserve=load(paths[0]);temporal=load(paths[1]);trial=load(paths[2])
    assert reserve['complete'] and reserve['oracle_only'] and trial['complete'] and not trial['admit_final']
    fmt=lambda x:f'{100*x:+.2f}'.replace('.',',')
    b=reserve['paired'];d=trial['paired']['D-baseline'];t=temporal['eligible_oracle'];c=temporal['counts']
    cards=[dict(id='bn-choice',x=80,y=240,w=1100,h=740,fill='#FFE1DE',stroke='#B33838',rank=1,
        title='1  ВЫБОР ПОЛЕЗНОЙ ПОПРАВКИ',
        text='ПОДТВЕРЖДЁННАЯ НЕРЕШЁННАЯ ЗАДАЧА\n\nВыбрать реальную область либо сохранить исходное назначение.\n\n'
             +f'Идеальный запас рамочных действий: {fmt(b["mean_per_video"])} п.п. [{fmt(b["ci95"][0])}; {fmt(b["ci95"][1])}] на 26 видео. Это не качество модели.\n\n'
             +f'Последний считыватель: {fmt(d["mean_per_video"])} п.п.; интервал включает 0 на 104 видео. Критерий не пройден.\n\n'
             +'Нужно доказать: сигнал мухи даёт пользу сверх того же решения без мухи.',
        ref=paths[0],status='measured unresolved task'),
      dict(id='bn-identity',x=1400,y=300,w=1100,h=740,fill='#DFEBFF',stroke='#416FBA',rank=2,
        title='2  ЛИЧНОСТЬ ОБЪЕКТА ВО ВРЕМЕНИ',
        text='РАБОЧАЯ ГИПОТЕЗА, ПРИЧИНА НЕ ДОКАЗАНА\n\nСвязать текущую область с прежним объектом и отличить его от похожего соседа.\n\n'
             +'У FlyVis уже есть рекуррентное состояние и обновление на кадрах поиска.\n\n'
             +'Обучаемые соответствия всех реальных областей между наблюдениями в текущей серии не проверены. Прежние варианты трёх пиков не дали нужного эффекта.\n\n'
             +'Проверка: равные D / C / вход глаза / без мухи и прежний способ выбора.',
        ref=paths[1],status='hypothesis, not established cause'),
      dict(id='bn-pairs',x=2720,y=240,w=1100,h=740,fill='#FFF0BE',stroke='#AB790F',rank=3,
        title='3  ПОДГОТОВКА ВРЕМЕННЫХ ПАР',
        text='ПОДТВЕРЖДЁННОЕ ОГРАНИЧЕНИЕ НАБОРА ДАННЫХ\n\n'
             +f'Пары с областями в обоих запросах: {c["has_supported_prior_and_current"]} из {c["rows"]} ({100*c["has_supported_prior_and_current"]/c["rows"]:.1f}%).\n\n'.replace('.',',')
             +f'Идеальный запас на доступных парах: {fmt(t["mean_per_video"])} п.п. [{fmt(t["ci95"][0])}; {fmt(t["ci95"][1])}] на 26 видео.\n\n'
             +'Сначала восстановить предыдущие наблюдения и области для остальных случаев; проверить охват и одинаковые входы всех контролей.\n\n'
             +'Это охват подготовленных пар, не измерение памяти FlyVis.',
        ref=paths[1],status='measured data readiness limitation')]
    fonts={}
    def font(size,bold=False):
        k=size,bold
        if k not in fonts:fonts[k]=ImageFont.truetype(FONTS+('segoeuib.ttf' if bold else 'segoeui.ttf'),size)
        return fonts[k]
    def wrap(text,width,size,bold=False):
        lines=[];f=font(size,bold)
        for paragraph in text.split('\n'):
            line=''
            for word in paragraph.split():
                candidate=(line+' '+word).strip()
                if line and f.getlength(candidate)>width:lines.append(line);line=word
                else:line=candidate
            lines.append(line)
        return lines
    title='УЗКИЕ МЕСТА FLYVIS — ПРИОРИТЕТЫ НА 25.10.2026'
    subtitle='Одна полоса слева направо: от большего приоритета к меньшему. Стрелки показывают ранг, а не причинность или порядок слоёв.'
    footer='Порядок выполнения отдельный: подготовку временных данных начинают с №3. Пространственный считыватель v1 временно отложен; к нему можно вернуться позже. Результаты относятся к своим выборкам, не к HOTA.'
    diagram=ET.SubElement(mx,'diagram',id='flyvis-bottlenecks',name='Узкие места — приоритеты')
    model=ET.SubElement(diagram,'mxGraphModel',dx='1800',dy='1000',grid='1',gridSize='10',page='1',pageScale='1',pageWidth=str(W),pageHeight=str(H),arrows='1',connect='1',tooltips='1')
    root=ET.SubElement(model,'root');ET.SubElement(root,'mxCell',id='bn-root');ET.SubElement(root,'mxCell',id='bn-layer',parent='bn-root')
    def xmlbox(cid,x,y,w,h,value,style):
        cell=ET.SubElement(root,'mxCell',id=cid,parent='bn-layer',vertex='1',value=value,style=style)
        ET.SubElement(cell,'mxGeometry',x=str(x),y=str(y),width=str(w),height=str(h),**{'as':'geometry'})
        return cell
    xmlbox('bn-title',80,40,3740,75,html.escape(title),'text;html=1;align=left;fontSize=42;fontStyle=1;fontFamily=Segoe UI;')
    xmlbox('bn-subtitle',80,125,3740,75,html.escape(subtitle),'text;html=1;align=left;whiteSpace=wrap;fontSize=28;fontFamily=Segoe UI;')
    im=Image.new('RGB',(W,H),'#FAFBFD');draw=ImageDraw.Draw(im)
    draw.text((80,45),title,font=font(42,True),fill='#182B40')
    for i,line in enumerate(wrap(subtitle,3740,28)):draw.text((80,135+i*35),line,font=font(28),fill='#52647A')
    svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}"><rect width="100%" height="100%" fill="#FAFBFD"/>',
         '<defs><marker id="rank-arrow" markerWidth="14" markerHeight="14" refX="12" refY="7" orient="auto"><path d="M0,0 L14,7 L0,14 Z" fill="#5D537D"/></marker></defs>']
    def svgtext(x,y,lines,size,bold=False,fill='#182B40'):
        for i,line in enumerate(lines):svg.append(f'<text x="{x}" y="{y+i*(size+8)}" font-family="Segoe UI,Arial,sans-serif" font-size="{size}" font-weight="{700 if bold else 400}" fill="{fill}">{html.escape(line)}</text>')
    svgtext(80,85,[title],42,True);svgtext(80,162,wrap(subtitle,3740,28),28,fill='#52647A')
    for left,right in zip(cards,cards[1:]):
        a=(left['x']+left['w'],left['y']+left['h']/2);b=(right['x'],right['y']+right['h']/2)
        edge=ET.SubElement(root,'mxCell',id='rank-'+str(left['rank']),parent='bn-layer',edge='1',source=left['id'],target=right['id'],style='edgeStyle=orthogonalEdgeStyle;curved=1;rounded=1;endArrow=block;endFill=1;endSize=24;strokeColor=#5D537D;strokeWidth=7;exitX=1;exitY=0.5;entryX=0;entryY=0.5;')
        ET.SubElement(edge,'mxGeometry',relative='1',**{'as':'geometry'})
        draw.line((a,b),fill='#5D537D',width=7)
        import math
        angle=math.atan2(b[1]-a[1],b[0]-a[0]);size=26
        draw.polygon([b,(b[0]-size*math.cos(angle-.5),b[1]-size*math.sin(angle-.5)),(b[0]-size*math.cos(angle+.5),b[1]-size*math.sin(angle+.5))],fill='#5D537D')
        svg.append(f'<path d="M{a[0]},{a[1]} L{b[0]},{b[1]}" stroke="#5D537D" stroke-width="7" marker-end="url(#rank-arrow)"/>')
    for card in cards:
        x,y,w,h=(card[k] for k in ('x','y','w','h'))
        head=wrap(card['title'],w-65,34,True);body=wrap(card['text'],w-65,28)
        used=30+len(head)*42+24+len(body)*36+28
        assert used<h,(card['id'],used,h)
        text='<b>'+'<br>'.join(map(html.escape,head))+'</b><br><br>'+'<br>'.join(map(html.escape,body))
        obj=ET.SubElement(root,'object',id=card['id'],label=text,priority=str(card['rank']),evidence_status=card['status'],source=card['ref'],link=(ROOT/card['ref']).as_uri())
        cell=ET.SubElement(obj,'mxCell',parent='bn-layer',vertex='1',style=f'rounded=1;whiteSpace=wrap;html=1;align=left;verticalAlign=top;spacing=30;fontSize=28;fontFamily=Segoe UI;fillColor={card["fill"]};strokeColor={card["stroke"]};strokeWidth=3;')
        ET.SubElement(cell,'mxGeometry',x=str(x),y=str(y),width=str(w),height=str(h),**{'as':'geometry'})
        draw.rounded_rectangle((x,y,x+w,y+h),radius=28,fill=card['fill'],outline=card['stroke'],width=3)
        svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="28" fill="{card["fill"]}" stroke="{card["stroke"]}" stroke-width="3"/>')
        cy=y+30
        for line in head:draw.text((x+32,cy),line,font=font(34,True),fill='#182B40');cy+=42
        svgtext(x+32,y+64,head,34,True);cy+=24
        by=cy
        for line in body:draw.text((x+32,cy),line,font=font(28),fill='#182B40');cy+=36
        svgtext(x+32,by+28,body,28)
        card['text_fits']=True
    footer_lines=wrap(footer,3670,28)
    xmlbox('bn-footer',80,1110,3740,150,'<br>'.join(map(html.escape,footer_lines)),'rounded=1;html=1;whiteSpace=wrap;align=left;verticalAlign=middle;spacing=24;fontSize=28;fontFamily=Segoe UI;fillColor=#EEF1F7;strokeColor=#A5AFC0;')
    draw.rounded_rectangle((80,1110,3820,1260),radius=22,fill='#EEF1F7',outline='#A5AFC0',width=2)
    svg.append('<rect x="80" y="1110" width="3740" height="150" rx="22" fill="#EEF1F7" stroke="#A5AFC0"/>')
    for i,line in enumerate(footer_lines):draw.text((108,1138+36*i),line,font=font(28),fill='#34445A')
    svgtext(108,1166,footer_lines,28,fill='#34445A');svg.append('</svg>')
    # Atomic promotion avoids an interrupted overwrite of a preview open elsewhere.
    for suffix,value in [('.png',im),('_preview.png',im.resize((2340,792),Image.Resampling.LANCZOS))]:
        temp=out/(STEM+suffix+'.next.png');value.save(temp);os.replace(temp,out/(STEM+suffix))
    (out/(STEM+'.svg')).write_text('\n'.join(svg),encoding='utf8')
    dossier=dict(date='2026-10-25',sheet=2,orientation='horizontal single lane',priority_is_not_causality_or_execution_order=True,
        sources=evidence,modules=cards,canvas=[W,H],all_text_fits=True)
    (out/(STEM+'_sources.json')).write_text(json.dumps(dossier,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    return dict(name=diagram.get('name'),module_priorities=[c['rank'] for c in cards],horizontal_single_lane=True,all_text_fits=True,
        outputs={STEM+s:sha(out/(STEM+s)) for s in ('.png','_preview.png','.svg','_sources.json')})


def append_existing(out):
    out=Path(out);p=out/'FlyVis_architecture.drawio';receipt_path=out/'validation.json'
    receipt=json.loads(receipt_path.read_text(encoding='utf8'));assert sha(p)==receipt['outputs'][p.name]
    tree=ET.parse(p);mx=tree.getroot();assert len(mx.findall('diagram'))==1
    first_before=diagram_digest(mx.find('diagram'))
    info=build(mx,out);assert diagram_digest(mx.find('diagram'))==first_before
    ET.indent(mx);temp=p.with_suffix('.next.drawio');ET.ElementTree(mx).write(temp,encoding='utf-8',xml_declaration=True)
    check=ET.parse(temp).getroot();assert len(check.findall('diagram'))==2 and diagram_digest(check.find('diagram'))==first_before
    os.replace(temp,p)
    receipt.update(diagram_sheets=2,second_sheet=info,first_sheet_unchanged_when_adding_second=True,first_sheet_xml_sha256=first_before)
    receipt['outputs'][p.name]=sha(p)
    receipt_path.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    return receipt


if __name__=='__main__':print(json.dumps(append_existing(ROOT/'dissertation/architecture'),ensure_ascii=False))
