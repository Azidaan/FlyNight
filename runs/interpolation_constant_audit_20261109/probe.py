"""New numerical invariant, not another model or identity-quality experiment."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,hashlib,subprocess,sys
from pathlib import Path
from datetime import datetime,timedelta,timezone
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent;MSK=timezone(timedelta(hours=3));STOP=datetime(2026,11,9,6,35,tzinfo=MSK)
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
from system_memory_guard import guard_entrypoint
remaining=(STOP-datetime.now(MSK)).total_seconds()
if remaining<=0:raise SystemExit('Save/stop reached')
guard_entrypoint(timeout=min(600,remaining))
import numpy as np
from tracking.observed_spatial_reader import ObservedSpatialReader
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,v):
    with (OUT/n).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def verify(p):
    for path,h in p['sources'].items():assert sha(ROOT/path)==h,path
def register():
    files=[Path(__file__),ROOT/'tracking/observed_spatial_reader.py',ROOT/'tracking/channel_balanced_correspondence.py',ROOT/'tracking/hex_background_filter.py',ROOT/'runs/common_state_memory_20261109/protocol.json',ROOT/'runs/z13_local_peak_20261021/geometry.json',ROOT/'tests/test_observed_spatial_reader.py',ROOT/'tests/test_channel_balanced_correspondence.py']
    write('protocol.json',dict(registered=datetime.now(MSK).isoformat(),head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),deadline='2026-11-09T06:40:00+03:00',scientific_save_stop=STOP.isoformat(),selection='All19 already valid common query geometries from completed memory study. No real feature maps, GT, detector, tracker or neural model read/executed. The previous13 tests are not repeated.',math='A constant field must remain the same exact constant under a linear interpolant. Normalized spatial correlation with a wholly constant current patch is undefined and must abstain. If interpolation creates floating-point variation, subsequent scale-invariant normalization can amplify numerical noise. This is an implementation hypothesis, not yet a measured cause of real wrong identities.',synthetic='Reference8channels are deterministic sin/cos fields of eye coordinates. Current maps are exactly constant per channel, .1/.3/1/10 times fixed channel multipliers. Same frozen19 queries and geometry; log every score and interpolation peak-to-peak. No epsilon/variance threshold or fitting.',acceptance='Every constant-current score must be null; any non-null value disproves the implemented invariant. If no failure, no interpolation fix is justified by this probe. If failure, retain it and require an exact constant-preserving expression plus affine-field and real-grid controls before adoption.',literature=[dict(url='https://scribblethink.org/Work/nvisionInterface/nip.html',role='Lewis NCC equation2: centered constant patch has zero norm; no valid normalized identity evidence.'),dict(url='https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.LinearNDInterpolator.html',role='Official implementation uses Qhull triangles and linear barycentric interpolation; exact partition of unity defines constant reproduction. Current local sampler is inspected directly, not assumed identical to every SciPy version.')],sources={p.relative_to(ROOT).as_posix():sha(p) for p in files}))
    path=ROOT/'experiments/project_history.json';cat=read(path);assert not any(x['id']==OUT.name for x in cat['experiments']);cat['experiments'].append(dict(id=OUT.name,title='Ровный сигнал после интерполяции — проверка ложного сходства',area='flyvis',state='running',started='2026-11-09',question='Может ли интерполяция создать колебания на ровной карте, которые считыватель примет за пространственное сходство?',hypothesis='Округление взвешенной суммы может нарушить точное сохранение константы; нормировка по малой вариации способна его усилить.',expected='Новый строгий математический пример на сохранённой геометрии, без реальных карт и повторения13готовыхтестов.',acceptance=[dict(criterion='Все оценки с ровной текущей картой неопределены; ни одна не становится свидетельством сходства.',met=None)],evidence=[dict(kind='file',ref=f'runs/{OUT.name}/protocol.json',claim='Математическая проверка до нового вычисления')]))
    path.write_text(json.dumps(cat,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    path=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';text=path.read_text(encoding='utf8');now=datetime.now(MSK);entry=f'**{now:%d.%m.%Y, %H:%M} MSK** — Проверка сохранения ровного сигнала при интерполяции.\n\n*Постановка (`{OUT.name}`).* На19готовыхгеометриях проверить новое строгое свойство: ровная текущаякарта послеинтерполяции остаётся ровной, корреляция не должна давать число. Восемь аналитическихканалов прошлого, четыре константныхуровня текущего. Это гипотеза численнойошибки, не уже найденная причина потериID. Нет реальныхкарт, разметки, ядра, детектора, трекера или подборапорога. Старые13тестовнеповторяются. Источники — формулаNCCLewis и официальное описание линейной интерполяцииSciPy, обе первичные страницы открыты. Условия `runs/{OUT.name}/protocol.json`.\n\n';path.write_text(text.replace('## Журнал изменений\n','## Журнал изменений\n\n'+entry,1),encoding='utf8');print('Registered constant reproduction invariant',flush=True)
def run():
    p=read(OUT/'protocol.json');verify(p)
    if (OUT/'results.json').exists():raise SystemExit('Ready numerical probe exists')
    g=read(ROOT/'runs/z13_local_peak_20261021/geometry.json');reader=ObservedSpatialReader(g['points'],g['eye_size'],[1280,720]);x,y=reader.points.T;reference=np.column_stack([np.sin(x/(17+c))+np.cos(y/(23+c))+np.sin(x*y/(301+13*c)) for c in range(8)]);cases=read(ROOT/'runs/common_state_memory_20261109/protocol.json')['cases'];records=[]
    for c in cases:
        plan=c['commonplan']
        if len(plan['cells'])<3:continue
        q=np.array(plan['query']);si=reader.mesh.find_simplex(q);tr=reader.mesh.transform[si];w=np.einsum('...ij,...j->...i',tr[...,:2,:],q-tr[...,2,:]);w=np.concatenate([w,1-w.sum(-1,keepdims=True)],-1)
        for level in [.1,.3,1.,10.]:
            constants=level*(1+np.arange(8)/10);current=np.tile(constants,(721,1));result=reader.compare(reference,current,plan);patch=np.einsum('...v,...vc->...c',w,current[reader.mesh.simplices[si]]).transpose(1,0,2);scores=result['scores'];finite=[v for v in scores if v is not None]
            records.append(dict(video=c['video'],frame=c['frame'],old=c['old'],level=level,cells=len(plan['cells']),scores=scores,nonnull_scores=len(finite),maximum_abs_score=max(map(abs,finite)) if finite else None,maximum_generated_ptp=float(np.ptp(patch,axis=1).max()),maximum_value_error=float(np.max(abs(patch-constants)))))
    verify(p);write('results.json',dict(passed=True,invariant_met=all(r['nonnull_scores']==0 for r in records),grids=len(records),records=records,nonnull_scores=sum(r['nonnull_scores'] for r in records),maximum_abs_score=max((r['maximum_abs_score'] for r in records if r['maximum_abs_score'] is not None),default=None),no_real_feature_or_model_calls=True));print('CONSTANT INVARIANT',read(OUT/'results.json')['invariant_met'],flush=True)
if __name__=='__main__':{'register':register,'run':run}[sys.argv[1]]()
