import sys,time,ast,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_identity_assignment_run_20261108.common import *
from research_singleton_verify_full_recovery1_20261108.status_io import for_run
RUN=ROOT/'runs/singleton_native_confirm_20261108';BASE=ROOT/'runs/identity_assignment_20261108'


def main():
    import psutil
    assert read(RUN/'queue_result.json')['passed'];assert datetime.now(MSK)<datetime(2026,11,8,13,55,tzinfo=MSK)
    for path in Path(__file__).parent.glob('*.py'):ast.parse(path.read_text(encoding='utf-8-sig'))
    queries=read(ROOT/'research_identity_assignment_protocol_20261108/queries.json')['rows'];fit=sorted({q['video'] for q in queries if q['split']=='fit20'});exports=read(BASE/'export_complete.json')['videos'];videos={}
    for video in fit:
        raw=f'runs/train_cache/{video}/detections/raw_video.npz';baseline=f'runs/fly_signatures/{video}/signatures.npz';export=exports[video]['path'];assert sha(ROOT/export)==exports[video]['sha256']
        videos[video]=dict(raw=raw,baseline=baseline,export=export,sources={n:sha(ROOT/n) for n in [raw,baseline,export,str(Path(export).with_suffix('.npz')).replace('\\','/')]})
    names=['tracking/native_confirm_identity.py','tests/test_native_confirm_identity.py','tracking/xywh_bytetrack.py','tracking/spatial_identity_assignment.py','tracking/own_template_verification.py','tracking/z13_region_assignment.py','tracking/foveated_eyes.py','tracking/offline_association.py','research_identity_assignment_run_20261108/replay.py','research_identity_assignment_run_20261108/full_planner.py','research_identity_assignment_run_20261108/common.py','research_singleton_verify_20261108/common.py','research_singleton_verify_full_recovery1_20261108/status_io.py','runs/z13_local_peak_20261021/geometry.json','research_identity_assignment_protocol_20261108/protocol_v2.json']
    names.extend(p.relative_to(ROOT).as_posix() for p in Path(__file__).parent.glob('*.py'))
    names.extend(p.relative_to(ROOT).as_posix() for p in (ROOT/'.venv/Lib/site-packages/ultralytics/trackers').rglob('*.py') if p.name in ['byte_tracker.py','bot_sort.py','matching.py','stracks.py'])
    protocol=dict(registered=datetime.now(MSK).isoformat(),deadline='2026-11-08T14:00:00+03:00',sources={n:sha(ROOT/n) for n in names},videos=videos,native_config=read(ROOT/'research_identity_assignment_protocol_20261108/protocol_v2.json')['native_config'],
        purpose='Recover missing native internal confirmations once; preserve all native physical outputs exactly. This is infrastructure, not a repeated baseline quality evaluation.',
        acceptance='Complete native output must equal the already cached signature outputs for every fit20 video. At archived query frames, all first costs/matches, predicted means/covariances and histories/search prefixes must match exactly. Fail before inspecting labels if any parity breaks.',
        eligibility='Previously public Lost row, unmatched native first, same class, positive IoU, exactly two supported real candidates with exactly one geometrically valid candidate; that candidate is emitted now by a never-public formerly unconfirmed track; common analytic DIS hull>=3 unique reference cells. Two existing public working IDs excluded.',
        next='Evaluate causal ID-only link on these metadata with no-FlyVis controls. No quality claim or acceptance from exposure. Unknown identity never negative; no GT used in runtime eligibility.',
        new_detector_calls=0,new_core_calls=0,new_image_reads=0)
    write(RUN/'native_trace_protocol.json',protocol,new=True)
    plan=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';text=plan.read_text(encoding='utf-8');stamp=datetime.now(MSK).strftime('%d.%m.%Y, %H:%M MSK');entry=f'\n\n**{stamp}** — Первая проблема: восстановление недостающих штатных подтверждений.\n\n*Получено.* В сохранённых fit20 запросах 49 свободных единственных рёбер, 4 относятся к первой штатной выдаче нового подтверждённого фрагмента; личности этих четырёх в прежней выборке неизвестны (`runs/singleton_native_confirm_20261108/fit_inventory.json`). Это нижняя оценка, не качество связывания.\n\n*Решение.* Одним проходом готовых детекций восстановить отсутствующие внутренние события между запросами на тех же 20 fit-видео. Полный выход обязан точно совпасть с уже сохранённым B0; готовые метрики и зрительные карты не пересчитывать. Протокол и хеши до прохода: `runs/singleton_native_confirm_20261108/native_trace_protocol.json`. Предварительно проверить метки личности на единичных примерах кода. Проблемы 2–4 отложены.\n';marker='## Журнал изменений';assert text.count(marker)==1;plan.write_text(text.replace(marker,marker+entry,1),encoding='utf-8',newline='\n')
    for_run(RUN)('native_metadata_registered',current_problem=1,deferred=[2,3,4])
    with (RUN/'metadata_controller.log').open('x',encoding='utf-8') as out,(RUN/'metadata_controller.err').open('x',encoding='utf-8') as err:
        p=subprocess.Popen([sys.executable,'-u','research_singleton_native_confirm_20261108/metadata_controller.py'],cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.IDLE_PRIORITY_CLASS)
    root=dict(pid=p.pid,created=psutil.Process(p.pid).create_time(),script='research_singleton_native_confirm_20261108/metadata_controller.py');write(RUN/'metadata_started.json',root,new=True)
    roots=read(BASE/'deadline_roots.json');roots['roots'].append(root)
    for attempt in range(10):
        try:write(BASE/'deadline_roots.json',roots);break
        except PermissionError:
            if attempt==9:raise
            time.sleep(.1)
    write(RUN/'metadata_deadline_registered.json',dict(passed=True,root=root,deadline=protocol['deadline']),new=True);print(root)


if __name__=='__main__':main()
