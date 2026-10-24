"""Service recovery: obtain crop metadata from the existing map-label sidecar.

The reserve label sidecar has no crop key. Its matching map sidecar does; every
crop is verified against the original episode archive before any replay resumes.
"""
from pathlib import Path
import hashlib
import inspect
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from scripts import z13_region_readout_20261025 as base
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
RUN=base.RUN/'recovery1'


def worker_source():
    source=inspect.getsource(base.worker)
    changes=[
        ('maps={}; reproduction={}', 'maps={}; map_labels={}; reproduction={}'),
        ("labels={x['label']:x for x in read(MAPS/arm/split/'labels.json')}",
         "labels={x['label']:x for x in read(MAPS/arm/split/'labels.json')};map_labels[split,arm]=labels"),
        ("to_frame(c,label['crop'],g['eye_size'])", "to_frame(c,labels[idx]['crop'],g['eye_size'])"),
        ("l=q['label'];crop=l['crop']", "l=q['label'];crop=map_labels[split,'D'][l['label']]['crop']")]
    for old,new in changes:
        assert source.count(old)==1,old
        source=source.replace(old,new)
    return source


def prepare():
    from scripts.z13_hota_deadline_guard import WindowsProcesses
    from scripts.z13_localization_labels_20261023 import verify_sources
    assert not RUN.exists()
    p,e=read(base.RUN/'protocol.json'),read(base.RUN/'controller_result.json')
    assert e['exit_code']==1 and e['all_owned_ended'] and not e['remaining']
    ps=WindowsProcesses();assert all(ps.creation(int(pid))!=birth for pid,birth in e['owned'].items())
    verify_sources(p)
    assert not list((base.RUN/'videos').glob('*.json')), 'Preserve completed jobs rather than repeat them'
    checks=[]
    for split,videos in [('fit',p['fit20']),('evaluate',p['evaluate26'])]:
        labels=read(ROOT/'runs/z13_candidate_reserve_20261021'/split/'labels.json')
        sides={arm:{x['label']:x for x in read(base.MAPS/arm/split/'labels.json')} for arm in ('C','D','R1')}
        offset=0
        for video in videos:
            with np.load(ROOT/f'runs/z13_data/{video}.npz') as z:
                off,crops,frames=z['ep_offset'],z['fr_crop'],z['fr_frame']
            subset=[l for l in labels if l['video']==video]
            for label in subset:
                e=label['episode']-offset;k=int(off[e])+4+label['pos']
                for arm,side in sides.items():
                    m=side[label['label']]
                    assert m['video']==video and m['pos']==label['pos'] and m['kind']==label['kind']
                    assert np.array_equal(m['pred'],label['pred'])
                    assert np.array_equal(m['crop'],crops[k]), (split,video,label['label'],arm)
            checks.append(dict(split=split,video=video,rows=len(subset),all3_crops_equal_original=True))
            offset+=len(off)-1
    source=worker_source();compile(source,str(Path(__file__)),'exec')
    names=[Path(__file__).relative_to(ROOT).as_posix(),'scripts/z13_region_readout_recovery1_console.ps1',
           (base.RUN/'protocol.json').relative_to(ROOT).as_posix(),
           (base.RUN/'controller_result.json').relative_to(ROOT).as_posix()]
    replacement=dict(p,created=now().isoformat(),service_recovery=dict(
        error="KeyError('crop'): reserve labels intentionally omit crop; canonical crop exists in map-label metadata",
        change='Only metadata access path in worker; all46 videos and every3-arm crop/prediction checked against original episode arrays. Readout, normalization, assignment, calibration grid, thresholds and tolerances unchanged.',
        completed_videos_retained=0,original_save_at=p['save_at'],original_hard_stop=p['hard_stop']),
        sha256={**p['sha256'],**{n:sha(ROOT/n) for n in names}},
        executed_worker_sha256=hashlib.sha256(source.encode()).hexdigest())
    write(RUN/'protocol.json',replacement)
    write(RUN/'crop_binding.json',dict(at=now().isoformat(),checks=checks,all_equal=True,source_hashes_unchanged=True))
    write(RUN/'executed_worker_source.txt',source)
    write(RUN/'executed_replay_source.txt',base.replay_source())
    write(RUN/'status.json',dict(state='ready',stage='Служебное восстановление: исходные вырезки сверены',done=0,total=46,updated=now().isoformat()))
    base.journal('исправление служебного пути вырезки регионального чтения',
        '*Ошибка.* Исходный региональный проход остановился до первого видео: KeyError(crop), потому что короткий список reserve не содержит вырезку. Старые код1, источники и журналы сохранены.\n\n'
        '*Проверено.* recovery1/crop_binding.json: все46 видео, все строки и три варианта имеют ту же вырезку и прогноз в sidecar полных карт и исходном архиве эпизода.\n\n'
        '*Исправление.* Меняется только адрес чтения метаданных. Формула считывания, карты, пороги, сетка alpha, критерии и допуски прежние. Готовых видео0, завершённое не повторяется. Исходные сроки '+p['save_at']+' / '+p['hard_stop']+' не продлеваются.')


def worker():
    p=read(RUN/'protocol.json')
    source=worker_source();assert hashlib.sha256(source.encode()).hexdigest()==p['executed_worker_sha256']
    namespace={**vars(base),'RUN':RUN}
    exec(compile(source,str(Path(__file__)),'exec'),namespace)
    namespace['worker']()


if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    elif '--worker' in sys.argv:
        try:worker()
        except Exception as error:write(RUN/'status.json',dict(state='failed',stage=repr(error),updated=now().isoformat()));raise
    else:raise ValueError('Choose --prepare or --worker')
