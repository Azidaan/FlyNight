"""Diagnostic annotation overlay; preserves native pixels in raw panels."""
import json,hashlib
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',13)
v='b2036451-aa924fd1';clock=read(ROOT/f'runs/identity_assignment_20261108/full_precision/runtime_inputs/{v}.json');labels=read(ROOT/f'data/bdd100k/labels/box_track_20/val/{v}.json');records=[]
crop=(440,320,530,405);scale=6;canvas=Image.new('RGB',(1140,1170),'white');draw=ImageDraw.Draw(canvas)
for ei,frame in enumerate([931,937]):
 anchor=clock['anchors'][str(frame)];row=next(r for i,r in enumerate(labels) if int(r.get('frameIndex',i))==anchor['frameIndex']);assert row['name']==anchor['name'];image=Image.open(ROOT/f'research_failure_localization_20261112/frames/574_{frame}_full.png').convert('RGB');raw=image.crop(crop).resize(((crop[2]-crop[0])*scale,(crop[3]-crop[1])*scale),Image.Resampling.NEAREST);marked=raw.copy();d=ImageDraw.Draw(marked);objects=[]
 for o in row.get('labels') or []:
  if 'box2d' not in o:continue
  b=o['box2d'];bb=[b[k] for k in ['x1','y1','x2','y2']]
  if bb[2]<crop[0] or bb[0]>crop[2] or bb[3]<crop[1] or bb[1]>crop[3]:continue
  ident=str(o['id']);color=(0,180,0) if ident=='00115019' else (225,30,40) if ident=='00115011' else (0,95,220);xy=[(bb[0]-crop[0])*scale,(bb[1]-crop[1])*scale,(bb[2]-crop[0])*scale,(bb[3]-crop[1])*scale];d.rectangle(xy,outline=color,width=2);objects.append(dict(id=ident,category=o['category'],box=bb,attributes=o.get('attributes',{})));d.text((max(0,xy[0]),max(0,xy[1]-16)),ident,fill=color,font=small,stroke_width=1,stroke_fill='white')
 y=ei*580;draw.text((15,y+8),f'Raw video frame {frame}; native pixel crop x440:530 y320:405, 6x nearest neighbor',fill='black',font=font);canvas.paste(raw,(15,y+40));canvas.paste(marked,(585,y+40));draw.text((585,y+550),'Green GT19; red GT11; blue all other annotated instances',fill='black',font=small);records.append(dict(raw_frame=frame,anchor=anchor,objects=objects))
draw.text((15,1160),'',fill='black');canvas.save(OUT/'native_context_574.png')
with (OUT/'overlay_manifest.json').open('x',encoding='utf8') as f:json.dump(dict(records=records,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),interpretation='Rectangles indicate annotation ambiguity only; no visible segmentation or pixel ownership supplied.'),f,ensure_ascii=False,indent=2)
