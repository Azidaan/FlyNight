"""Service recovery: bind absent crop metadata to the immutable episode archive."""
from pathlib import Path
import inspect
import hashlib
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts import z13_region_choice_gate_20261025 as base
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
RUN=base.RUN/'recovery1'


def crop_metadata(p):
    labels=read(ROOT/'runs/z13_candidate_reserve_20261021/evaluate/labels.json')
    result={};offset=0
    for v in p['videos']:
        with np.load(ROOT/f'runs/z13_data/{v}.npz') as d:
            off,crop,frame=d['ep_offset'],d['fr_crop'],d['fr_frame']
        for l in labels:
            if l['video']!=v:continue
            e=l['episode']-offset;assert 0<=e<len(off)-1
            k=int(off[e])+4+l['pos'];assert k<off[e+1]
            result[l['label']]=dict(video=v,pos=l['pos'],kind=l['kind'],pred=l['pred'],crop=crop[k].tolist(),frame=int(frame[k]))
        offset+=len(off)-1
    assert len(result)==8326
    return result


def hydrate_crops(queries,p):
    metadata=crop_metadata(p);count=0
    for v in p['videos']:
        for q in queries[v]:
            l=q['label'];m=metadata[l['label']]
            assert all(m[k]==l[k] for k in ('video','pos','kind','pred')) and q['frame']==m['frame']
            l['crop']=m['crop'];count+=1
    assert count==8326
    write(RUN/'crop_binding.json',dict(at=now().isoformat(),rows=count,all_crops_bound_to_original=True))


def worker_source():
    s=inspect.getsource(base.worker)
    old="queries,centers=original.load_queries(read(OLD/'protocol.json'));metadata={}"
    assert s.count(old)==1
    return s.replace(old,"queries,centers=original.load_queries(read(OLD/'protocol.json'));hydrate_crops(queries,p);metadata={}")


def prepare():
    from scripts.z13_hota_deadline_guard import WindowsProcesses
    from scripts.z13_localization_labels_20261023 import verify_sources
    assert not RUN.exists()
    p,e=read(base.RUN/'protocol.json'),read(base.RUN/'controller_result.json');assert e['exit_code']==1 and e['all_owned_ended'] and not e['remaining']
    ps=WindowsProcesses();assert all(ps.creation(int(pid))!=b for pid,b in e['owned'].items())
    assert not list((base.RUN/'videos').glob('*.json')),'Keep completed videos'
    verify_sources(p);metadata=crop_metadata(p)
    names=[Path(__file__).relative_to(ROOT).as_posix(),'scripts/z13_region_choice_gate_recovery1_console.ps1',
        (base.RUN/'protocol.json').relative_to(ROOT).as_posix(),(base.RUN/'controller_result.json').relative_to(ROOT).as_posix(),
        'scripts/z13_presence_scope_20261025.py']
    replacement=dict(p,created=now().isoformat(),service_recovery=dict(error="KeyError('crop'): compact reserve labels omit crop metadata; original episode archives contain it",change='Add only the original frame crop keyed by label/video/position/frame. All8326 metadata records checked. No changed scores, support rule, candidate interface, labels, cost function, criteria or deadlines.',completed_videos=0,original_save_at=p['save_at'],original_hard_stop=p['hard_stop']),executed_worker_sha256=hashlib.sha256(worker_source().encode()).hexdigest(),sha256={**p['sha256'],**{n:sha(ROOT/n) for n in names}})
    write(RUN/'protocol.json',replacement);write(RUN/'crop_metadata_preflight.json',dict(at=now().isoformat(),rows=len(metadata),source='Original immutable fr_crop arrays'))
    write(RUN/'status.json',dict(state='ready',stage='Восстановлены исходные метаданные вырезки',done=0,total=26,updated=now().isoformat()))
    base.journal('служебное восстановление прямого выбора области','*Ошибка.* Повторно допущена известная служебная ошибка: короткий список reserve не содержит crop. Исходный exit1, источники и журналы сохранены;готовых видео0.\n\n*Исправление.* Добавлен только исходный fr_crop по точному эпизоду/кадру8326строк; scientific параметры и критерий прежние. Сроки '+p['save_at']+' / '+p['hard_stop']+' не продлеваются. Проверка метаданных: recovery1/crop_metadata_preflight.json. Не повторять этот адресный сбой при следующих адаптациях load_queries.')


def worker():
    p=read(RUN/'protocol.json');source=worker_source();assert hashlib.sha256(source.encode()).hexdigest()==p['executed_worker_sha256']
    ns={**vars(base),'RUN':RUN,'hydrate_crops':hydrate_crops};exec(compile(source,str(Path(__file__)),'exec'),ns);ns['worker']()


def finish():
    # Rebind the output root and explicitly publish the preserved failure/version.
    source=inspect.getsource(base.finish)
    old="ns={**vars(pub),'RUN':RUN};exec"
    new="code=code.replace(\"names=['tracking/z13_region_choice.py'\",\"names=EXTRA+['tracking/z13_region_choice.py'\");ns={**vars(pub),'RUN':RUN,'EXTRA':RECOVERY_FILES};exec"
    assert source.count(old)==1
    source=source.replace(old,new).replace('runs/z13_region_choice_gate_20261025/publication.json','runs/z13_region_choice_gate_20261025/recovery1/publication.json')
    extra=[Path(__file__).relative_to(ROOT).as_posix(),'scripts/z13_region_choice_gate_recovery1_console.ps1']
    extra += [(base.RUN/name).relative_to(ROOT).as_posix() for name in ('protocol.json','controller_result.json','status.json','monitor_failure.json')]
    ns={**vars(base),'RUN':RUN,'RECOVERY_FILES':extra};exec(compile(source,str(Path(__file__)),'exec'),ns);ns['finish']()


if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    elif '--worker' in sys.argv:
        try:worker()
        except Exception as exc:write(RUN/'status.json',dict(state='failed',stage=repr(exc),updated=now().isoformat()));raise
    elif '--finish' in sys.argv:
        try:finish()
        except Exception as exc:write(RUN/'monitor_failure.json',dict(at=now().isoformat(),error=repr(exc)));raise
    else:raise RuntimeError('Explicit mode required')
