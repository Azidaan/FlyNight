"""Preregistered fixed-template support gate; no image or neural scoring."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import sys,json,hashlib,subprocess
from pathlib import Path
from datetime import datetime,timezone,timedelta
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
BASE=ROOT/'runs/identity_channel_head_20261109';MSK=timezone(timedelta(hours=3));STOP=datetime(2026,11,9,23,54,tzinfo=MSK)
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
if not (OUT/'status.json').exists():(OUT/'status.json').write_text(json.dumps(dict(stage='starting',updated=datetime.now(MSK).isoformat())),encoding='utf8')
from system_memory_guard import guard_entrypoint
remaining=(STOP-datetime.now(MSK)).total_seconds();assert remaining>0;guard_entrypoint(timeout=min(1800,remaining))
import numpy as np
import psutil
psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
from tracking.observed_spatial_reader import ObservedSpatialReader
from tracking.full_template_correspondence import relative_template_plan

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def cast(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,dict):return {k:cast(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [cast(x) for x in v]
    return v
def write(n,v):
    p=OUT/n;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8') as f:json.dump(cast(v),f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def verify(p):
    for name,value in p['sources'].items():assert sha(ROOT/name)==value,name
def journal(title,body):
    p=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';s=p.read_text(encoding='utf8');stamp=datetime.now(MSK).strftime('%d.%m.%Y, %H:%M MSK')
    p.write_text(s.replace('## Журнал изменений\n','## Журнал изменений\n\n**'+stamp+'** — '+title+'\n\n'+body+'\n\n',1),encoding='utf8')

def register():
    old=read(BASE/'protocol.json');assert read(BASE/'audit.json')['passed']
    assert not any(read(BASE/'summary.json')['point_pilot_gates'].values())
    inventory=ROOT/'runs/historical_direct_query_inventory_20261109/results.json'
    idx={(r['video'],r['uid']):r for r in read(inventory)['records']}
    edges=old['edges']
    for edge in edges:
        source=idx[(edge['video'],edge['uid'])];candidate=next(k for k in source['candidates'] if k['raw_id']==edge['raw_id'])
        assert source['frame']==edge['frame'] and candidate['box']==edge['current_box']
        edge['offsets']=candidate['plan']['offsets']
        assert np.array(edge['offsets']).shape==(19,2) and np.array_equal(edge['offsets'][9],[0.,0.])
    files=[Path(__file__),ROOT/'tracking/full_template_correspondence.py',ROOT/'tests/test_full_template_correspondence.py',
        ROOT/'tracking/observed_spatial_reader.py',ROOT/'tracking/channel_identity_head.py',ROOT/'tracking/channel_balanced_correspondence.py',ROOT/'tracking/hex_linear_interpolation.py',
        inventory,BASE/'protocol.json',BASE/'features.json',BASE/'predictions.json',BASE/'summary.json',BASE/'audit.json',BASE/'maps_complete.json',ROOT/'runs/z13_local_peak_20261021/geometry.json']
    protocol=dict(registered=datetime.now(MSK).isoformat(),head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        authorization='research_identity_synthesis_20261109/authorization.json',save_stop=STOP.isoformat(),hard_stop='2026-11-09T23:59:00+03:00',
        action='Replace candidate-box intersection across all shifts by the whole previous template, keeping all-shift eye-hull validity. Same affine candidate alignment, physical19 offsets and head objective. No new core, detector, images or maps.',
        diagnostic_number=1,maximum_diagnostics=3,
        mechanism='Old condition requires T_k(x)+s in B_k for EVERY s. For a rectangle this is exactly T_k(x) in [B_lo-min_s(s),B_hi-max_s(s)]. It discards reference cells before measuring activity. New support removes this candidate-box gate only; all surviving reference cells are used at every shift.',
        strongest_alternative='Extra reference cells may include background or an occluder and merely inflate chance similarity. Match image representations, geometry-derived support controls, and a fixed common cell count across shifts. Never call a larger support proof of identity.',
        mathematical_gate='Six synthetic tests, exact old-cell reconstruction and old-query parity <=1e-10, old cells subset of full cells for every edge. Proceed to signal comparison only if at least one of91 edges gains a cell. Otherwise stop this action without rescoring ready maps.',
        edges=edges,representations=['C34','R134_spatial','R134_temporal'],
        immutable='Same91 strict-last edges,15 own/76 foreign,32 videos, same folds, lambda1,zero threshold and unchanged equal-channel shift selection. Old features and G predictions reused. No earlier-labelled cases or prior15 videos added.',
        geometry_control='G_ready is reused unchanged. New G_support augments the same seven geometry features with log1p(full_support_count) and full_support_count/reference_box_cell_count, standardized on training videos only. All paired old/full image models receive that SAME G_support. This isolates support from changes in available geometric coverage information.',
        comparison_models='G_ready, G_support; for each ofthree34-channel representations, old/full support crossed with mean/channels heads:14 reported models, no hidden feature or threshold search.',
        model='Exactly class-balanced mean logistic loss + ||w||^2/2, free intercept, deterministic damped Newton, fixedlambda1,decision z>0. Train feature-valid rows within31 videos; undefined feature or loss of a class falls back to G_support. Stored old-feature scores are not recalculated.',
        validation='32 leave-one-complete-video-out folds. Historical development population, not a new official independent split. No label, ID, owner or video identifier is a predictive feature.',
        primary='G_support+C34 full-channel head versus its old-support counterpart at the same G_support.',
        pilot_admission='A full-support arm may enter a bounded actual-ID pilot only if own accepts are NO LOWER than G_ready,G_support and its same-representation/same-kind OLD-support counterpart; foreign accepts must be STRICTLY LOWER than all three; balanced accuracy must be higher than all three. Evaluate allsix full arms. Primary C34channels fixed. If an exploratory arm passes, disclose selection; rank by balanced accuracy, then fewer foreign accepts, then scalar before34weights, then protocol representation order. No thresholds tuned.',
        admission_change_from_prior='Previous version required a strict increase of own accepts. This distinct support protocol instead requires preservation of own accepts plus strict rejection improvement: preventing foreign-ID transfers is itself part of the goal. This change is registered BEFORE any new support counts or scores. It does not change the previous failed gate or its status.',
        uncertainty='Paired video-cluster bootstrap4000 draws,seed20261109. Descriptive95% CIs; for the four fixed primary contrasts (oldC34channels,G_support,fullR134spatialchannels,fullR134temporalchannels) also report98.75% intervals as Bonferroni family check. No confirmed FlyVis or ID benefit without later actual independent tracker evidence.',
        stop='If no full-support arm passes, stop this support version; do not change thresholds,lambda,channels,cell rule or samples based on outcomes. This is action-specific, not closure of FlyVis.',
        literature=dict(title='Fully-Convolutional Siamese Networks for Object Tracking',authors='Bertinetto, Valmadre, Henriques, Vedaldi, Torr',year=2016,url='https://arxiv.org/pdf/1606.09549',section='2.1, equation2 and discussion of full template cross-correlation within the valid search domain',borrowed='Fixed spatial template translated within a larger search field; not their learned embedding or their tracker implementation.',limit='Paper does not validate our FlyVis maps, candidate-affine warp, Pearson normalization, learned identity refusal or global ID assignment.'),
        sources={p.relative_to(ROOT).as_posix():sha(p) for p in files})
    write('protocol.json',protocol)
    catalog_path=ROOT/'experiments/project_history.json';catalog=read(catalog_path);assert not any(c['id']==OUT.name for c in catalog['experiments'])
    catalog['experiments'].append(dict(id=OUT.name,title='Целый пространственный шаблон без усечения всеми сдвигами',area='flyvis',state='running',started='2026-11-09',
        question='Удаляет ли пересечение поддержки всех сдвигов полезные части шаблона, и поможет ли их сохранение отвергать чужую рамку без потери своих?',
        hypothesis='Правило усечения по текущей рамке заранее убирает границы прошлого объекта. Сохранение целого шаблона в поле глаза может дать более различимые пространственные совпадения. Альтернатива — больше фона и случайного сходства.',
        expected='Сначала точная математика и готовая геометрия, затем одна проверка изменённого считывания с одинаковыми входными и геометрическими контролями.',
        acceptance=[dict(criterion=protocol['mathematical_gate'],met=None),dict(criterion=protocol['pilot_admission'],met=None)],
        obtained='',interpretation='',decision='',cannot_claim='',evidence=[dict(kind='file',ref='runs/'+OUT.name+'/protocol.json',claim='Механизм, условия и критерии до новой геометрии и сигналов.')],tags=['fly-problem-P1']))
    temp=catalog_path.with_suffix('.full_template.tmp');temp.write_text(json.dumps(catalog,ensure_ascii=False,indent=2)+'\n',encoding='utf8');temp.replace(catalog_path)
    journal('Зафиксирована проверка целого шаблона вместо пересечения всех сдвигов',f'*Постановка (`{OUT.name}`).* Меняется только ограничение шаблона рамкой кандидата; остаются все сдвиги, карты, минимальная поддержка и ограничение полем глаза. Материалы прежнего отрицательного опыта переиспользуются:91 пара,32 видео. Сначала математическая формула и воспроизведение прежних координат, затем сравнение сигналов при наличии добавленных ячеек. Для сигналов заранее заданы одинаковая расширенная геометрия и старые/полные шаблоны на FlyVis и двух34-канальных входах, штраф1 и порог0. Новый допуск требует не снижать принятие своих и строго снизить чужие; прежний строгий критерий прошлого опыта не меняется. Источник `runs/{OUT.name}/protocol.json`. Литература: Bertinetto и соавторы2016, раздел2.1, формула2; перенос на FlyVis — гипотеза. Ядро не меняется.')
    print('Registered full-template action before geometry or signal scores',flush=True)

def probe():
    p=read(OUT/'protocol.json');verify(p)
    import pytest
    assert pytest.main(['tests/test_full_template_correspondence.py','-q'])==0
    g=read(ROOT/'runs/z13_local_peak_20261021/geometry.json');reader=ObservedSpatialReader(g['points'],g['eye_size'],[1280,720]);records=[];error=0.
    for e in p['edges']:
        args=(e['reference_box'],e['sequence'][3][1],e['current_box'],e['sequence'][-1][1],e['offsets'])
        old=relative_template_plan(reader,*args,clip_to_candidate=True);full=relative_template_plan(reader,*args)
        np.testing.assert_array_equal(old['cells'],e['cells'])
        error=max(error,float(np.max(np.abs(old['query']-np.array(e['query'])))))
        assert set(old['cells'])<=set(full['cells'])
        positions=np.searchsorted(full['cells'],old['cells']);np.testing.assert_array_equal(full['query'][positions],old['query'])
        records.append(dict(video=e['video'],uid=e['uid'],raw_id=e['raw_id'],label=e['label'],old_count=len(old['cells']),full_count=len(full['cells']),reference_count=full['reference_cells'],candidate_area_fraction=full['candidate_area_fraction'],plan=full))
    assert error<=1e-10;changed=sum(r['full_count']>r['old_count'] for r in records);gate=changed>0
    old=np.array([r['old_count'] for r in records]);full=np.array([r['full_count'] for r in records]);ref=np.array([r['reference_count'] for r in records])
    verify(p)
    write('geometry_results.json',dict(passed=True,gate=gate,records=records,changed_edges=changed,total_edges=len(records),old_median=np.median(old),full_median=np.median(full),reference_median=np.median(ref),median_reference_retained_old=np.median(old/ref),median_reference_retained_full=np.median(full/ref),old_query_parity_max_error=error,old_cells_preserved=True,fixed_support_across_shifts=True,no_image_or_neural_scores=True))
    (OUT/'status.json').write_text(json.dumps(dict(stage='geometry_complete',gate=gate,changed_edges=changed,total=91,updated=datetime.now(MSK).isoformat())),encoding='utf8')
    print('GEOMETRY',changed,'/91 changed; medians',np.median(old),np.median(full),'parity',error,flush=True)

if __name__=='__main__':{'register':register,'probe':probe}[sys.argv[1]]()
