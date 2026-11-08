"""Preserve the path-verification failure and resume missing stages without reruns."""
import sys,time,ast,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_native_confirm_full_recovery1_20261108.common import *


def main():
    import psutil,numpy as np
    assert not (FULL/'features_complete.json').exists();failure=read(RUN/'children/full_features.ended.json');assert failure['returncode']==1 and failure['guardian']['reason'] is None and failure['all_owned_ended']
    for pid,birth in failure['guardian']['owned'].items():
        try:p=psutil.Process(int(pid));assert abs(p.create_time()-birth)>=.01,(pid,'Failed feature worker still alive')
        except psutil.NoSuchProcess:pass
    for path in Path(__file__).parent.glob('*.py'):ast.parse(path.read_text(encoding='utf-8-sig'))
    norm=read(ROOT/'runs/spatial_identity_maps_20261107/image_normalization.json');width={'C':8,'R1':34,'IMAGE':len(norm['mean']),'DIS':2};partial={}
    for path in (FULL/'features').rglob('*.npy'):
        kind=path.name.split('.')[-2];value=np.load(path,allow_pickle=False);assert value.shape==(721,width[kind]) and np.isfinite(value).all();partial[path.relative_to(ROOT).as_posix()]=dict(sha256=sha(path),shape=list(value.shape),dtype=str(value.dtype))
    write(RECOVERY/'partial_maps.json',dict(passed=True,files=partial,scope='Maps saved before a later verification-path failure. Recorded now, not retroactively attributed checksums.'),new=True)
    diagnosis=dict(registered=datetime.now(MSK).isoformat(),error='Post-feature verifier assumed the old decoded float64 container still existed. Its original released.json proves it was deliberately freed; the actual reader reconstructed a new container inFULL. No OOM; minimum available memory13,062,410,240bytes.',failed_guardian=failure['guardian'],partial_maps=len(partial),native_trace_reused=True,catalog_check_reused=True,repair='Prepare only required retinal views and verify cached views numerically. GPU reads prepared views; missing-view fallback explicitly raises. Preserve all original sources, logs, failed exit codes, saved maps and decoded first-video scratch.',science_parameters_unchanged=True)
    write(RECOVERY/'diagnosis.json',diagnosis,new=True)
    sources=read(RUN/'full_phase.json')['sources'].copy()
    for path in Path(__file__).parent.glob('*.py'):sources[path.relative_to(ROOT).as_posix()]=sha(path)
    for path in [RECOVERY/'diagnosis.json',RECOVERY/'partial_maps.json',FULL/'native_trace_complete.json']:sources[path.relative_to(ROOT).as_posix()]=sha(path)
    sources.update({n:r['sha256'] for n,r in partial.items()})
    write(RECOVERY/'phase.json',dict(registered=datetime.now(MSK).isoformat(),sources=sources),new=True)
    plan=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';text=plan.read_text(encoding='utf-8');stamp=datetime.now(MSK).strftime('%d.%m.%Y, %H:%M MSK');entry=f'\n\n**{stamp}** — Первая проблема: исправление пути проверки временного входа без повторения готовых карт.\n\n*Сбой исполнения.* Полный нулевой выход20видео уже совпал. После сохранения карт первого видео проверка обратилась к старому декодированному контейнеру, ранее освобождённому по протоколу; фактический читатель восстановил вход в новой папке. Ошибка `FileNotFoundError`, не нехватка памяти; прежний журнал и код1сохранены (`runs/native_confirm_risk_20261108/children/full_features.ended.json`; `full_recovery1/diagnosis.json`).\n\n*Восстановление.* Проверены и сохранены {len(partial)} готовых карт. Новая служебная версия восстанавливает только нужные вырезы с сохранением родной точности кадров и численно сверяет используемые готовые вырезы. GPUчитает подготовленные входы; повторное создание полных временных видео запрещено. Исходники и критерии опыта не меняются, нулевой трекер и готовые карты не повторяются (`runs/native_confirm_risk_20261108/full_recovery1/phase.json`).\n';marker='## Журнал изменений';assert text.count(marker)==1;plan.write_text(text.replace(marker,marker+entry,1),encoding='utf-8',newline='\n')
    with (RECOVERY/'controller.log').open('x',encoding='utf-8') as out,(RECOVERY/'controller.err').open('x',encoding='utf-8') as err:child=subprocess.Popen([sys.executable,'-u','research_native_confirm_full_recovery1_20261108/controller.py'],cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.IDLE_PRIORITY_CLASS)
    root=dict(pid=child.pid,created=psutil.Process(child.pid).create_time(),script='research_native_confirm_full_recovery1_20261108/controller.py');write(RECOVERY/'started.json',root,new=True);roots=read(BASE/'deadline_roots.json');roots['roots'].append(root)
    for attempt in range(10):
        try:write(BASE/'deadline_roots.json',roots);break
        except PermissionError:
            if attempt==9:raise
            time.sleep(.1)
    write(RECOVERY/'deadline_registered.json',dict(passed=True,root=root,deadline='2026-11-08T14:00:00+03:00'),new=True);print(dict(root=root,ready_maps=len(partial)))


if __name__=='__main__':main()
