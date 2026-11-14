from common import *
import numpy as np,time,cv2,torch
if sys.argv[1]=='prepare':
    p=read(OUT/'protocol_v2.json')
    save('resolution_protocol.json',dict(registered=now(),parent_sha256=sha(OUT/'protocol_v2.json'),reason='Post-result diagnostic: full-frame518 reversed target/neighbor depth on1164/1165. Test whether doubling input spatial resolution reverses that failure. Not a replacement of original acceptance or independent validation.',inputs={**p['inputs'],rel(Path(__file__)):sha(__file__),rel(OUT/'result.json'):sha(OUT/'result.json')},input_size=1036,frames=p['frames'],criteria='Same median order and rank>0.75 on all three frames; compare both results, do not select thresholds.',budget='CPU two threads, three frames<=600 seconds, same guarded memory reserve.'))
    journal('После отрицательной пробы глубины зарегистрирована диагностика разрешения: те же кадры и веса, размер входа 1036 вместо 518. Цель — проверить потерю различия при уменьшении изображения; это последующая диагностика, исходный отрицательный результат не заменяется. Протокол research_depth_order_20261114/resolution_protocol.json.')
    raise SystemExit
p=read(OUT/'resolution_protocol.json')
for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
# Import only preparation and class definitions of the frozen CPU adapter.
prefix=(OUT/'probe_v2.py').read_text(encoding='utf8').split('times=[];maps={};images={}')[0]
exec(compile(prefix,str(OUT/'probe_v2.py'),'exec'),globals())
rp=read(OUT/'resolution_protocol.json');maps={};times=[]
with torch.inference_mode():
    for frame in rp['frames']:
        im=cv2.imread(str(ROOT/f'research_visible_part_2179_20261114/frames/{frame}.png'));t=time.perf_counter();maps[str(frame)]=model.infer_image(im,1036);times.append(time.perf_counter()-t)
with (OUT/'resolution_maps.npz').open('xb') as f:np.savez_compressed(f,**maps)
z=np.load(MASK);rows=[]
for frame,t in zip(rp['frames'],times):
    d=maps[str(frame)];assert np.isfinite(d).all();a=np.sort(d[z[f'{frame}_target'].astype(bool)]);b=d[z[f'{frame}_neighbor'].astype(bool)];rank=float(((np.searchsorted(a,b,'left')+np.searchsorted(a,b,'right'))*.5/len(a)).mean());rows.append(dict(frame=frame,target_median=float(np.median(a)),neighbor_median=float(np.median(b)),rank_probability=rank,seconds=t,gate=bool(np.median(a)<np.median(b) and rank>.75)))
for n,h in rp['inputs'].items():assert sha(ROOT/n)==h,n
save('resolution_result.json',dict(records=rows,passed=all(r['gate'] for r in rows),map_sha256=sha(OUT/'resolution_maps.npz'),source_unchanged=True,diagnostic_only=True))
print(json.dumps(rows),flush=True)
