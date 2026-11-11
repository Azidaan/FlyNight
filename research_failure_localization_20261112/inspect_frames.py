"""Decode only source pixels up to named diagnostic frames; no model or tracker."""
from common import *
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
assert read(OUT/'stage2.json')['passed']
p=verify();ag=read(ROOT/'research_agreement_readout_20261112/protocol.json');native=read(NATIVE/'protocol.json')
out=OUT/'frames';out.mkdir(exist_ok=True);manifest=[]
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
for row in ag['screen']:
    v=row['video'];ref=ag['snapshots'][row['keys'][0]]['sequence'][-1][0];current=row['frame']
    wanted={ref,current}
    if row['old_id']==1102:wanted|={258,261,264}
    cap=cv2.VideoCapture(str(ROOT/native['videos'][v]['video_path']));assert cap.isOpened()
    frames={}
    for f in range(max(wanted)+1):
        ok,bgr=cap.read();assert ok
        if f in wanted:frames[f]=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
    cap.release()
    ann=annotations(v);r=next(x for x in p['selected'] if x['video']==v and x['frame']==current and x['old_id']==row['old_id'])
    target=r['reference_identity'];panels=[]
    for f,box in [(ref,row['reference_box']),(current,row['current_box'])]:
        image=frames[f];a=np.array(box);center=(a[:2]+a[2:])/2;extent=np.maximum(a[2:]-a[:2], [100,70])*1.6
        crop=np.r_[np.maximum(0,np.floor(center-extent/2)),np.minimum([1280,720],np.ceil(center+extent/2))].astype(int)
        raw=image.crop(tuple(crop));raw.save(out/f'{row["old_id"]}_{f}_crop.png');image.save(out/f'{row["old_id"]}_{f}_full.png')
        factor=min(640/raw.width,400/raw.height);size=(round(raw.width*factor),round(raw.height*factor))
        panel=Image.new('RGB',(680,470),'#eeeeee');panel.paste(raw.resize(size,Image.Resampling.NEAREST),(15,55));dr=ImageDraw.Draw(panel)
        dr.text((15,8),f'ID {row["old_id"]}, frame {f}; crop {crop.tolist()}',font=font,fill='black')
        def rect(b,color):
            q=(np.array(b).reshape(2,2)-crop[:2])*factor+[15,55];dr.rectangle(tuple(q.ravel()),outline=color,width=2)
        rect(box,'red')
        for obj in ann[f]:
            if obj['category']=='car' and str(obj['id'])==target:rect([obj['box2d'][k] for k in ['x1','y1','x2','y2']],'lime')
        panels.append(panel)
        manifest.append(dict(old_id=row['old_id'],video=v,frame=f,crop=crop.tolist(),display_scale=factor,raw_box=box,source_video=native['videos'][v]['video_path'],frame_png=f'frames/{row["old_id"]}_{f}_full.png',sha256=sha(out/f'{row["old_id"]}_{f}_full.png')))
    sheet=Image.new('RGB',(1360,470),'white');sheet.paste(panels[0]);sheet.paste(panels[1],(680,0));sheet.save(OUT/f'case_{row["old_id"]}.png')
    if row['old_id']==1102:
        sheet=Image.new('RGB',(1500,400),'white');dr=ImageDraw.Draw(sheet)
        for k,f in enumerate([252,258,261,264]):
            crop=(820,230,960,325);im=frames[f].crop(crop).resize((350,238),Image.Resampling.NEAREST);sheet.paste(im,(k*375,45));dr.text((k*375+5,5),f'1102: frame {f}',fill='black',font=font)
            frames[f].save(out/f'1102_{f}_full.png')
        sheet.save(OUT/'trace1102.png')
    print('FRAMES',row['old_id'],flush=True)
verify();save('frame_manifest.json',dict(passed=True,records=manifest,script_sha256=sha(__file__),method='Sequential OpenCV raw decode from frame0; no retinal preprocessing and no model calls; diagnostic visual inspection only.'))
