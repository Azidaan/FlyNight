"""Exact annotation-frame audit of stored reference states and raw candidates.

No interpolation, core execution, descriptor rescoring or runtime GT access.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import hashlib,json,sys,subprocess
from collections import Counter,defaultdict
from pathlib import Path
from datetime import datetime,timedelta,timezone
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
PARENT=ROOT/'runs/background_patch_gate_20261108';MSK=timezone(timedelta(hours=3))
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
from system_memory_guard import guard_entrypoint
guard_entrypoint(timeout=600)
import numpy as np
from datasets.bdd100k_loader import CLASS_NAMES
def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(name,value):
    with (OUT/name).open('x',encoding='utf8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
def journal(title,body):
    path=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';text=path.read_text(encoding='utf8')
    entry=f'**{datetime.now(MSK):%d.%m.%Y, %H:%M} MSK** — {title}.\n\n{body}\n\n'
    path.write_text(text.replace('## Журнал изменений\n','## Журнал изменений\n\n'+entry,1),encoding='utf8')
def register():
    p=read(PARENT/'protocol.json');videos=sorted({e['video'] for e in p['events']})
    sources=[Path(__file__),PARENT/'protocol.json',PARENT/'results.json',ROOT/'runs/decoder34_identity_gate_20261108/results.json',ROOT/'research_bbox_confirmation_inventory_20261108/export.py',ROOT/'datasets/bdd100k_loader.py']
    for video in videos:
        sources += [ROOT/f'data/bdd100k/labels/box_track_20/train/{video}.json',ROOT/f'runs/train_cache/{video}/alignment.json',ROOT/f'runs/bbox_singleton_pilot_20261108/labels/{video}.json',ROOT/f'runs/bbox_confirmation_inventory_20261108/native/{video}.npz']
    write('protocol.json',dict(registered=datetime.now(MSK).isoformat(),head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),deadline='2026-11-09T07:40:00+03:00',
        action='Choose whether to implement a protected initial/stable object memory instead of latest-only Kalman-state template. No such change is accepted by this audit.',
        hypothesis='The latest stored reference may already describe a different object before recovery. Stored reference boxes are Kalman states; current boxes are raw detections.',
        strongest_alternative='The old template still describes the right object; the centered spatial comparison and common motion/background cannot distinguish instances.',
        scope='All200 preselected fit20 events. Preserve old conditional trajectory labels. New exact-frame box matches are separate evaluator evidence, never runtime inputs.',
        rule='Use actual aligned annotation frames only; same class, non-crowd. Unique direct match means exactly one GT box with IoU>=0.5. Zero means unmatched, multiple means ambiguous, unannotated means unknown. No new threshold selection.',
        mathematical_check='For any deterministic descriptor and identical stored history, two latent identities producing the same observation are unidentifiable by changing only the comparison. A reliable previous template can help only when its observation differs; absent annotations cannot prove contamination.',
        decision='A directly matched latest reference ID differing from the earlier unique trajectory target is positive contamination evidence. Otherwise report only tested coverage; do not claim all histories clean. Check direct current raw-box versus saved native-output identity independently.',
        literature=dict(title='Learning Spatio-Temporal Transformer for Visual Tracking',authors='Bin Yan, Houwen Peng, Jianlong Fu, Dong Wang, Huchuan Lu',year=2021,paper='https://arxiv.org/pdf/2103.17154',code='https://raw.githubusercontent.com/researchmm/Stark/main/lib/test/tracker/stark_st.py',role='Section3.2 and tracker lines42-54,84-92 retain initial template and gate dynamic updates. This motivates testing memory quality, not copying its threshold or claiming a STARK reproduction.',access='CVF PDF returned403; arXiv PDF and official source opened successfully.'),
        sources={path.relative_to(ROOT).as_posix():sha(path) for path in sources}))
    path=ROOT/'experiments/project_history.json';cat=read(path);assert not any(c['id']==OUT.name for c in cat['experiments'])
    cat['experiments'].append(dict(id=OUT.name,title='Память объекта — принадлежит ли последний шаблон прежней машине',area='flyvis',state='running',started='2026-11-08',question='Ошибается сравнение или в прошлом шаблоне уже другая машина?',hypothesis='Последнее состояние Калмана может загрязнять память цели до пропуска.',expected='Новые прямые подтверждения по точному кадру отдельно от прежних меток траектории; без пересчёта карт.',acceptance=[dict(criterion='Все200 сохранённых событий проверены по фактической разметке без интерполяции, прежние метки не перезаписаны; каждый вывод содержит IoU и источник.',met=None)],evidence=[dict(kind='file',ref=f'runs/{OUT.name}/protocol.json',claim='Фиксированные правила до сверки')]))
    path.write_text(json.dumps(cat,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    journal('Зарегистрирована проверка принадлежности последнего шаблона',f'*Постановка (`{OUT.name}`).* Все200 прежних fit20 событий; прямое сопоставление рамки состояния Калмана и сырого кандидата с реальной разметкой точного кадра. Прежнее правило IoU≥0,5, неоднозначное и неизвестное сохраняются. Результат определит целесообразность защищённой памяти вместо последнего наблюдения. Штатные карты, оценки и назначения не пересчитываются. Источник `runs/{OUT.name}/protocol.json`.')
    print('Registered200-event exact-frame audit')
def overlap(box,objects):
    a=np.asarray(box,float);b=np.asarray([x['box'] for x in objects],float).reshape(-1,4)
    inter=np.maximum(0,np.minimum(a[2:],b[:,2:])-np.maximum(a[:2],b[:,:2])).prod(1)
    union=np.maximum(0,a[2:]-a[:2]).prod()+np.maximum(0,b[:,2:]-b[:,:2]).prod(1)-inter
    values=inter/np.maximum(union,1e-12)
    # Independent scalar implementation, including empty/degenerate boxes.
    for obj,value in zip(objects,values):
        q=obj['box'];w=max(0,min(a[2],q[2])-max(a[0],q[0]));h=max(0,min(a[3],q[3])-max(a[1],q[1]));u=max(0,a[2]-a[0])*max(0,a[3]-a[1])+max(0,q[2]-q[0])*max(0,q[3]-q[1])-w*h
        assert abs(value-(w*h/u if u>0 else 0))<1e-12
    return values
def match(frame,box,cls,annotations):
    if frame not in annotations:return dict(status='unannotated',identity=None,frame=frame,box=box,overlaps=[])
    objects=[o for o in annotations[frame] if o['category']==CLASS_NAMES[cls]]
    values=overlap(box,objects);qualified=np.flatnonzero(values>=.5)
    status='unique' if len(qualified)==1 else 'ambiguous' if len(qualified)>1 else 'unmatched'
    return dict(status=status,identity=objects[int(qualified[0])]['id'] if status=='unique' else None,frame=frame,box=box,
        overlaps=[dict(identity=o['id'],iou=float(v),box=o['box'],attributes=o['attributes']) for o,v in sorted(zip(objects,values),key=lambda x:-x[1]) if v>0])
def run():
    if (OUT/'results.json').exists():raise SystemExit('Ready audit exists; no repetition.')
    protocol=read(OUT/'protocol.json')
    for path,digest in protocol['sources'].items():assert sha(ROOT/path)==digest,path
    grouped=defaultdict(list)
    for e in read(PARENT/'protocol.json')['events']:grouped[e['video']].append(e)
    records=[];totals=Counter();known_counts=Counter();contamination=[];current_disagreement=[]
    for video,events in grouped.items():
        gt=read(ROOT/f'data/bdd100k/labels/box_track_20/train/{video}.json');aligned=read(ROOT/f'runs/train_cache/{video}/alignment.json')['matches'];assert len(gt)==len(aligned)
        annotations={}
        for fr,a in zip(gt,aligned):
            frame=int(a['video_frame_index']);assert frame not in annotations
            annotations[frame]=[dict(id=str(o['id']),category=o['category'],box=[o['box2d'][k] for k in ['x1','y1','x2','y2']],attributes=o.get('attributes') or {}) for o in fr.get('labels') or [] if o['category'] in CLASS_NAMES.values() and not (o.get('attributes') or {}).get('crowd',False)]
        labels=read(ROOT/f'runs/bbox_singleton_pilot_20261108/labels/{video}.json')['rows']
        with np.load(ROOT/f'runs/bbox_confirmation_inventory_20261108/native/{video}.npz',allow_pickle=False) as z:native=z['outputs']
        for e in events:
            label=labels[f'{e["frame"]}|{e["old_tid"]}|{e["new_tid"]}'];history=e['row_data']['history'];cls=int(e['row_data']['cls'])
            past=[match(int(h['frame']),h['box'],cls,annotations) for h in history];current=match(int(e['frame']),e['box'],cls,annotations)
            out=native[(native[:,0]==e['frame']) & (native[:,5]==e['new_tid'])];assert len(out)==1
            native_current=match(int(e['frame']),out[0,1:5].astype(float).tolist(),cls,annotations)
            reference=past[-1];direct_kind='unknown' if not reference['identity'] or not current['identity'] else 'own' if reference['identity']==current['identity'] else 'foreign'
            ref_conflict=bool(reference['identity'] and label['target'] and reference['identity']!=label['target'])
            cur_conflict=bool(current['identity'] and native_current['identity'] and current['identity']!=native_current['identity'])
            row=dict(video=video,frame=e['frame'],old=e['old_tid'],new=e['new_tid'],conditional_label=label,reference_history=past,current=current,native_current=native_current,direct_kind=direct_kind,reference_disagrees_with_prior_target=ref_conflict,current_raw_disagrees_with_native_output=cur_conflict,
                source_annotation=f'data/bdd100k/labels/box_track_20/train/{video}.json',source_alignment=f'runs/train_cache/{video}/alignment.json')
            records.append(row);totals['events']+=1;totals['reference_'+reference['status']]+=1;totals['current_'+current['status']]+=1;totals['direct_'+direct_kind]+=1
            if label['known']:
                known_counts['conditional_known']+=1;known_counts['reference_'+reference['status']]+=1;known_counts['current_'+current['status']]+=1;known_counts['direct_'+direct_kind]+=1
            if ref_conflict:contamination.append([video,e['frame'],e['old_tid'],e['new_tid']])
            if cur_conflict:current_disagreement.append([video,e['frame'],e['old_tid'],e['new_tid']])
        print('Audited',video,len(records),'/',200,flush=True)
    assert len(records)==200
    for path,digest in protocol['sources'].items():assert sha(ROOT/path)==digest,path
    write('results.json',dict(passed=True,records=records,totals=dict(totals),conditional_known_counts=dict(known_counts),reference_conflicts=contamination,raw_native_conflicts=current_disagreement,sources_stable=True,no_interpolation=True,old_scores_unchanged=True,independent_scalar_iou_passed=True))
    print('SUMMARY',dict(totals),'KNOWN',dict(known_counts),'REFERENCE_CONFLICTS',contamination,'CURRENT_CONFLICTS',current_disagreement)
{'register':register,'run':run}[sys.argv[1]]()
