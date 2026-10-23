"""Saved-artifact audit of where D's interventions change visible outputs."""
from pathlib import Path
import sys,json,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,sha
from scripts import z13_localization_labels_20261023 as prior
RUN=ROOT/'runs/z13_effect_path_audit_20261023';OLD=ROOT/'runs/z13_center_hota_20261021';IDENT='z13_effect_path_audit_v1'


def journal(title,text):
    old=prior.IDENT
    try:prior.IDENT=IDENT;prior.journal(title,text)
    finally:prior.IDENT=old


def prepare():
    assert not RUN.exists();report=read(OLD/'eval/report.json');videos=sorted(report['per_video']['D_masa']);assert len(videos)==35
    names=['scripts/z13_effect_path_audit_20261023.py','runs/z13_center_hota_20261021/eval/report.json','runs/z13_center_hota_20261021/result_review_20261022.json','tracking/z13_readout.py']
    for v in videos:names.extend([f'runs/fly_signatures/{v}/signatures.npz',f'runs/z13_center_hota_20261021/D/signatures/{v}/signatures.npz',f'runs/z13_center_hota_20261021/D/signatures/{v}/signatures.json',f'runs/chain_filter_rank/repro/dev/{v}.json',f'runs/z13_center_hota_20261021/eval/D_masa/{v}.json'])
    write(RUN/'protocol.json',dict(id=IDENT,created=prior.now().isoformat(),videos=videos,method='Read saved outputs only; no tracker/inference/video/TrackEval. Count exact raw-frame differences, geometry ignoring ID, complete trajectories up to global ID renaming; same checks on already stitched annotated-frame outputs. Bind to existing per-video HOTA and existing aggregate intervention counters.',action='Determine whether further work should target selection before tracker output, or later persistence/stitching. No threshold or model tuning on dev.',limits='Post-hoc reused dev35; no new validation claim. Aggregate event counters cannot attribute every correction to a frame or prove causal neutralization by MASA. Float-exact output comparison is not a performance metric.',sha256={n:sha(ROOT/n) for n in names}))
    cp=ROOT/'experiments/project_history.json';cat=read(cp);assert not any(c['id']==IDENT for c in cat['experiments'])
    cat['experiments'].append(dict(id=IDENT,title='З13: путь эффекта D в сохранённых выходах стенда',area='flyvis',state='ready',started=f'{prior.now():%Y-%m-%d}',question='На каком уровне сохранённых выходов виден эффект1162поправок D?',hypothesis='Часть поправок может не менять видимый выход; другая часть может исчезать после отбора размеченных кадров или сшивки. Это версии, не установленная причина.',expected='Только чтение35готовых пар, хеши, точное сравнение и равенство при переименовании ID; без новой HOTA/моделей/видео/GPU.',acceptance=[dict(criterion='Все35пар прочитаны, источники неизменны; старые HOTA/события сверены, различены геометрия и идентичности.',met=None)],summary='Постановка аудита сохранённых артефактов; параметры модели не подбираются.',done='',obtained='',interpretation='',decision='',cannot_claim='Агрегаты не позволяют приписать каждый эффект конкретной поправке; сравнение до/после сшивки не изолирует её причинный вклад от выборки кадров. Dev35 уже использован.',metrics=[],commits=[],files=[names[0]],evidence=[dict(kind='file',ref='runs/z13_effect_path_audit_20261023/protocol.json',claim='Заранее заданный аудит сохранённых выходов')]))
    write(cp,cat);journal('постановка аудита пути эффекта D','*Постановка.* В сохранённой HOTA D против стенда4видео лучше,31равно,0хуже; это уже записано22.10, не новый результат. Проверяется незакрытый пункт: где1162события перестают менять видимый выход. Только чтение исходных и сшитых траекторий35видео, без нового расчёта HOTA или изменения по dev. Протокол `runs/z13_effect_path_audit_20261023/protocol.json`. От ответа зависит выбор узла дальнейшей правки; это одна диагностика, не серия перебора порога.')


def sorted_rows(rows):return tuple(sorted(tuple(r) for r in rows))


def tracks(rows):
    groups={}
    for r in rows:groups.setdefault(r[5],[]).append(tuple(r[:5])+tuple(r[6:]))
    return tuple(sorted(sorted_rows(v) for v in groups.values()))


def compare(a,b):
    import numpy as np
    frames=sorted(set(a[:,0])|set(b[:,0]));diff=[];geometry=[]
    for f in frames:
        x=a[a[:,0]==f];y=b[b[:,0]==f]
        if sorted_rows(x)!=sorted_rows(y):diff.append(int(f))
        if sorted_rows(np.delete(x,5,1))!=sorted_rows(np.delete(y,5,1)):geometry.append(int(f))
    return dict(rows_before=len(a),rows_after=len(b),different_frames=diff,geometry_different_frames=geometry,equal_up_to_global_id_renaming=tracks(a)==tracks(b))


def stitched(path):
    import numpy as np
    rows=[]
    for fr in read(path):
        for l in fr.get('labels',[]):
            b=l['box2d'];rows.append([fr['frameIndex'],b['x1'],b['y1'],b['x2'],b['y2'],l['id'],l.get('score'),l['category']])
    return np.asarray(rows,dtype=object).reshape(-1,8)


def run():
    import numpy as np
    p=read(RUN/'protocol.json');prior.verify_sources(p);report=read(OLD/'eval/report.json');review=read(OLD/'result_review_20261022.json');rows={};totals={}
    for v in p['videos']:
        with np.load(ROOT/f'runs/fly_signatures/{v}/signatures.npz') as z:a=z['outputs'].copy()
        with np.load(OLD/f'D/signatures/{v}/signatures.npz') as z:b=z['outputs'].copy()
        event=read(OLD/f'D/signatures/{v}/signatures.json')['z13_events']
        for key,value in event.items():totals[key]=totals.get(key,0)+value
        rows[v]=dict(events=event,raw=compare(a,b),stitched=compare(stitched(ROOT/f'runs/chain_filter_rank/repro/dev/{v}.json'),stitched(OLD/f'eval/D_masa/{v}.json')),hota_gain=report['per_video']['D_masa'][v]['HOTA']-report['per_video']['A_masa'][v]['HOTA'])
    assert totals==review['events']['D'];prior.verify_sources(p)
    summary=dict(videos=len(rows),events=totals,raw_changed_videos=sum(bool(r['raw']['different_frames']) for r in rows.values()),raw_geometry_changed_videos=sum(bool(r['raw']['geometry_different_frames']) for r in rows.values()),raw_trajectory_changed_videos=sum(not r['raw']['equal_up_to_global_id_renaming'] for r in rows.values()),stitched_changed_videos=sum(bool(r['stitched']['different_frames']) for r in rows.values()),stitched_trajectory_changed_videos=sum(not r['stitched']['equal_up_to_global_id_renaming'] for r in rows.values()),hota_better=sum(r['hota_gain']>0 for r in rows.values()),hota_equal=sum(r['hota_gain']==0 for r in rows.values()),hota_worse=sum(r['hota_gain']<0 for r in rows.values()))
    write(RUN/'report.json',dict(summary=summary,per_video=rows,source_sha256=p['sha256'],limits=p['limits']));print(json.dumps(summary))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--prepare',action='store_true');a.add_argument('--run',action='store_true');x=a.parse_args()
    if x.prepare:prepare()
    elif x.run:run()
    else:a.error('Choose prepare/run')
