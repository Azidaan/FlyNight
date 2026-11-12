from common import *
import cv2,gzip
import numpy as np
from PIL import Image,ImageDraw,ImageFont
p=verify();assert read(OUT/'audit.json')['passed'];native=ROOT/'runs/domain_matched_native_20261109';np_=read(native/'protocol.json');framesdir=OUT/'frames';framesdir.mkdir(exist_ok=False)
specs=[dict(video='b1cd1e94-549d0bfe',frames=[493,500,504,518,522,528,534,540],crop=[525,350,655,435],ids=[773,801,832],target='00118997'),dict(video='b23b2649-8349d2a1',frames=[519,520,538,539,544],crop=[940,515,1065,590],ids=[232,474],target='00130234')]
inputs={np_['videos'][s['video']]['video_path']:sha(ROOT/np_['videos'][s['video']]['video_path']) for s in specs}
save('visual_protocol.json',dict(registered=now(),posthoc=True,reason='Reconstructed832 birth has negligible overlap with773; inspect whether initial physical identity is determinable. Also inspect first loss232.',specs=specs,inputs=inputs,method='Sequential raw video decode fromframe0. No learned model, no invented annotation. Exact annotation anchors only. Retain pixel crops enlarged with nearest-neighbor; boxes overlay public outputs and targetGT. Unknown physical identity stays unknown.'))
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18);manifest=[]
for spec in specs:
    v=spec['video'];path=ROOT/np_['videos'][v]['video_path'];cap=cv2.VideoCapture(str(path));assert cap.isOpened();images={}
    for f in range(max(spec['frames'])+1):
        ok,bgr=cap.read();assert ok
        if f in spec['frames']:images[f]=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
    cap.release()
    with np.load(native/'rollouts/C34'/(v+'.npz'),allow_pickle=False) as z:outputs=z['outputs']
    with gzip.open(native/'rollouts/C34'/(v+'.events.json.gz'),'rt',encoding='utf8') as h:assignments=json.load(h)['assignments']
    with np.load(ROOT/np_['videos'][v]['detections'],allow_pickle=False) as z:raw={k:z[k] for k in ['frame_idx','xyxy']}
    gt=read(ROOT/f'data/bdd100k/labels/box_track_20/val/{v}.json');by={int(x.get('frameIndex',i)):x for i,x in enumerate(gt)};anchors=read(LEGACY/'runtime_inputs'/(v+'.json'))['anchors'];ann={int(f):by[a['frameIndex']]['labels'] for f,a in anchors.items()};crop=spec['crop'];scale=3;pw=440;ph=330;sheet=Image.new('RGB',(pw*4,ph*((len(images)+3)//4)),(240,240,240));draw=ImageDraw.Draw(sheet);colors=['#ff7f00','#00aaff','#ff00ff']
    for n,f in enumerate(spec['frames']):
        image=images[f];pathcrop=framesdir/f'{v}_{f}.png';image.crop(tuple(crop)).save(pathcrop);x=(n%4)*pw;y=(n//4)*ph;sheet.paste(image.crop(tuple(crop)).resize(((crop[2]-crop[0])*scale,(crop[3]-crop[1])*scale),Image.Resampling.NEAREST),(x+10,y+45));draw.text((x+10,y+5),f'frame {f}; GT {"yes" if f in ann else "NO"}',font=font,fill='black');boxes=[]
        def rectangle(b,color):
            q=(np.array(b).reshape(2,2)-crop[:2])*scale+[x+10,y+45];draw.rectangle(tuple(q.ravel()),outline=color,width=2)
        for j,tid in enumerate(spec['ids']):
            aa=[a for a in assignments if a['frame']==f and a['tid']==tid]
            if aa:
                a=aa[0];rows=outputs[(outputs[:,0]==f)&(outputs[:,5]==tid)];b=rows[0][1:5].tolist() if len(rows) else raw['xyxy'][a['raw_id']].tolist();rectangle(b,colors[j]);boxes.append(dict(tid=tid,public=bool(len(rows)),box=b));draw.text((x+10+j*138,y+ph-30),str(tid)+('' if len(rows) else ' new'),font=font,fill=colors[j])
        target=[]
        for o in ann.get(f,[]):
            if str(o['id'])==spec['target']:
                b=[o['box2d'][k] for k in ['x1','y1','x2','y2']];rectangle(b,'#00ff00');target.append(b)
        manifest.append(dict(video=v,frame=f,crop=crop,pixel_sha256=sha(pathcrop),file=pathcrop.relative_to(ROOT).as_posix(),public_or_birth_boxes=boxes,target_gt=target))
    sheet.save(OUT/f'frames_{spec["ids"][0]}.png');assert sha(path)==inputs[path.relative_to(ROOT).as_posix()]
save('visual_manifest.json',dict(passed=True,records=manifest,source_hashes=inputs,script_sha256=sha(__file__)));print('FRAMES_READY',len(manifest))
