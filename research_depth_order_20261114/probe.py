from common import *
os.environ['CUDA_VISIBLE_DEVICES']=''
import time, subprocess
import numpy as np, cv2, torch
from PIL import Image, ImageDraw, ImageFont
torch.set_num_threads(2); cv2.setNumThreads(2)
SOURCE=ROOT/'third_party/depth_anything_v2'/('Depth-Anything-V2-'+read(OUT/'source_commit.json')['sha'])
WEIGHT=ROOT/'weights/depth_anything_v2_small/depth_anything_v2_vits.pth'
MASK=ROOT/'research_visible_part_2179_20261114/mask_review/masks_reviewed.npz'
FRAMES=[1159,1164,1165]
if sys.argv[1]=='prepare':
    paths=[WEIGHT,MASK,Path(__file__),OUT/'common.py']+list((SOURCE/'depth_anything_v2').rglob('*.py'))+[ROOT/f'research_visible_part_2179_20261114/frames/{f}.png' for f in FRAMES]
    save('protocol.json',dict(registered=now(),head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_commit=read(OUT/'source_commit.json')['sha'],inputs={rel(p):sha(p) for p in paths},frames=FRAMES,model='Depth Anything V2 Small, official vits, frozen pretrained weights, CPU two threads, full frame, input_size518',question='Does predicted inverse depth put the manually reviewed farther target car behind its nearer neighboring car on old2179?',expected='Median inverse depth target < neighbor on all three frames; rank separation >0.75 on every frame.',acceptance=['median(target)<median(neighbor) on all 3 frames','P(inverse_depth_neighbor>inverse_depth_target)+0.5*ties>0.75 on all 3 frames','All inputs unchanged; no training, threshold adjustment, tracker or FlyVis-core changes.'],evaluation='Existing reviewed visible cores only. Target farther and neighbor nearer by scene occlusion interpretation, not measured 3D ground truth. Occluder combines people and bin: report combined without fabricated separation. Save maps before evaluation. Compare full target-reference/query ROI median as contamination diagnostic. No across-frame raw-scale comparison. This is one familiar episode, not independent validation.',budget='3 CPU inferences + 1 same-frame warm-up, <=600 seconds. Weights and source only on E.',references=['https://github.com/DepthAnything/Depth-Anything-V2','https://github.com/DepthAnything/Depth-Anything-V2/issues/174','https://arxiv.org/abs/2406.09414']))
    cat=read(ROOT/'experiments/project_history.json')
    cat['experiments'].append(dict(id='depth_order_2179_20261114',title='Глубина видимых частей перекрывающихся машин',area='own_tracker',state='running',started='2026-11-14',question='Различает ли готовая малая модель глубины дальнюю целевую машину и ближнего соседа?',hypothesis='Порядок по относительной глубине может стать дополнительным признаком сопоставления.',expected='Устойчивая разница на трёх известных кадрах, без обучения.',acceptance=[dict(criterion=s,met=None) for s in read(OUT/'protocol.json')['acceptance']],evidence=[dict(kind='file',ref=rel(OUT/'protocol.json'))]))
    (ROOT/'experiments/project_history.json').write_text(json.dumps(cat,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    journal('Зарегистрирована проверка depth_order_2179_20261114: готовая Depth Anything V2 Small, три прежних кадра 1159, 1164, 1165; сравнение глубины ручных видимых частей машин. Полный трекер и ядро FlyVis не запускаются; модель не обучается. Протокол research_depth_order_20261114/protocol.json; условие продолжения — правильный порядок медиан и вероятность парного ранжирования выше 0,75 на каждом кадре. Из официальных источников загружены веса и исходники только на E.')
    raise SystemExit
p=read(OUT/'protocol.json')
for n,h in p['inputs'].items(): assert sha(ROOT/n)==h,n
sys.path.insert(0,str(SOURCE))
from depth_anything_v2.dpt import DepthAnythingV2
model=DepthAnythingV2(encoder='vits',features=64,out_channels=[48,96,192,384])
model.load_state_dict(torch.load(WEIGHT,map_location='cpu',weights_only=True));model.eval()
times=[];maps={};images={}
with torch.inference_mode():
    model.infer_image(cv2.imread(str(ROOT/f'research_visible_part_2179_20261114/frames/{FRAMES[0]}.png')),518)
    for f in FRAMES:
        im=cv2.imread(str(ROOT/f'research_visible_part_2179_20261114/frames/{f}.png'));images[f]=im
        start=time.perf_counter();d=model.infer_image(im,518);times.append(time.perf_counter()-start)
        assert d.shape==im.shape[:2] and np.isfinite(d).all();maps[str(f)]=d
with (OUT/'depth_maps.npz').open('xb') as handle: np.savez_compressed(handle,**maps)
z=np.load(MASK);results=[]
case=read(ROOT/'research_yolo26seg_20261114/case2179.json')['events'][0]
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',21)
for f in FRAMES:
    d=maps[str(f)];regions={k:d[z[f'{f}_{k}'].astype(bool)] for k in ['target','neighbor','occluder','background']}
    a=np.sort(regions['target']);b=regions['neighbor'];rank=float(((np.searchsorted(a,b,'left')+np.searchsorted(a,b,'right'))*.5/len(a)).mean())
    stats={k:dict(n=len(v),median=float(np.median(v)),q10=float(np.quantile(v,.1)),q90=float(np.quantile(v,.9))) for k,v in regions.items()}
    bbox=case['roles'][0 if f==1159 else -1]['box'];x1,y1,x2,y2=np.round(bbox).astype(int)
    results.append(dict(frame=f,regions=stats,pair_rank_probability=rank,correct_median_order=stats['target']['median']<stats['neighbor']['median'],gate=stats['target']['median']<stats['neighbor']['median'] and rank>.75,full_roi_median=float(np.median(d[y1:y2,x1:x2])),roi_source='1159reference' if f==1159 else '1165query;1164diagnostic uses next frame ROI',seconds=times[len(results)]))
    # Colors show inverse depth: cool farther, warm nearer; percentiles fixed per full frame for display only.
    lo,hi=np.quantile(d,[.02,.98]);disp=cv2.applyColorMap(np.uint8(np.clip((d-lo)/(hi-lo),0,1)*255),cv2.COLORMAP_TURBO)
    view=images[f].copy()
    for k,col in [('target',(0,255,0)),('neighbor',(255,100,0)),('occluder',(0,80,255))]:
        contours,_=cv2.findContours(z[f'{f}_{k}'].astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(view,contours,-1,col,1)
    crop=(860,170,1280,450);l,t,r,b=crop
    panels=[cv2.resize(im[t:b,l:r],(630,420)) for im in [images[f],disp,view]]
    sheet=Image.new('RGB',(1890,470),'#20232a');draw=ImageDraw.Draw(sheet)
    for j,(im,title) in enumerate(zip(panels,['Исходный кадр','Глубина: тёплое ближе, холодное дальше','Ручные видимые части для проверки'])):
        sheet.paste(Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB)),(630*j,50));draw.text((630*j+8,12),title,font=font,fill='white')
    sheet.save(OUT/f'{f}_depth_comparison.png')
for n,h in p['inputs'].items(): assert sha(ROOT/n)==h,n
save('result.json',dict(records=results,scientific_passed=all(r['gate'] for r in results),seconds_median=float(np.median(times)),device='CPU, 2threads',parameters=sum(x.numel() for x in model.parameters()),depth_maps_sha256=sha(OUT/'depth_maps.npz'),source_unchanged=True,scope='One familiar episode; ordinal scene interpretation only, no metric depth truth, no identity or tracker metric.'))
print(json.dumps(results),flush=True)
