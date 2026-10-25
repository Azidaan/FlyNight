"""Publication-only recovery: normalize CRLF in Git's index, preserve working hashes."""
from pathlib import Path
import inspect
import sys
import time
import os
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import z13_region_product_20261025 as science
from scripts import z13_region_product_recovery1_20261025 as recovery
from scripts.z13_candidate_reserve_20261021 import read,write,now,sha
from scripts.z13_hota_deadline_guard import WindowsProcesses
OUT=ROOT/'runs/z13_hidden_consoles_20261025'


def publish(stage):
    code=inspect.getsource(recovery.publish)
    old="code=inspect.getsource(science.publish);old="
    new='code=inspect.getsource(science.publish).replace("[\'git\',\'add\',\'-f\',", "[\'git\',\'-c\',\'core.autocrlf=input\',\'add\',\'-f\',");old='
    assert code.count(old)==1;code=code.replace(old,new)
    old="    names += [p.relative_to(ROOT).as_posix() for p in RUN.glob('*.json')]"
    assert code.count(old)==1
    code=code.replace(old,old+"\n    names += EXTRA_SERVICE_FILES")
    extra=[Path(__file__).relative_to(ROOT).as_posix(),'scripts/z13_region_product_recovery1_20261025.py',
        'AGENTS.md','scripts/z13_hide_progress_consoles_20261025.py','runs/z13_hidden_consoles_20261025/result.json',
        'runs/z13_hidden_consoles_20261025/publication_recovery.json']
    ns=dict(vars(recovery));ns['EXTRA_SERVICE_FILES']=extra
    exec(compile(code,str(Path(__file__)),'exec'),ns);ns['publish'](stage)


def prepare():
    assert not (OUT/'publication_recovery.json').exists()
    c=read(OUT/'result.json');assert c['hidden']==11 and c['processes_stopped'] is False and all(not p['visible_after'] for p in c['consoles'])
    write(OUT/'publication_recovery.json',dict(at=now().isoformat(),science_unchanged=True,
        service='Use per-command core.autocrlf=input when adding own files: Git check rejected CRLF in a live status snapshot. Working files/protocol hashes remain unchanged; no global Git setting or check relaxed.',
        previous_start_tests='runs/z13_region_product_20261025/start_checks_1.log:61passed',
        preference='25Oct author requested hidden consoles, no foreground windows; supersedes16Oct visible requirement. AGENTS updated; existing11visible windows hidden.'))
    science.journal('скрытый запуск консолей по решению автора',
        '*Решение автора25.10.* Консоли больше не выводятся поверх приложений: новые наблюдатели/контроллеры запускаются скрыто, видимое окно только по прямой просьбе. '
        '*Сделано.* Проверены35проектных консолей, скрыты11видимых; ни один процесс не остановлен, расчёт продолжает продвигаться. '
        'AGENTS.md заменяет правило16.10. Источник runs/z13_hidden_consoles_20261025/result.json. '
        '*Служебная публикация.*61тест пройден; git diff --check отклонилCRLF одного текущегоstatus.json. '
        'Отдельный публикационный помощник нормализует окончания строк только при добавлении собственных файлов в индекс; рабочие научные файлы и хеши не меняются. '
        'Источник runs/z13_hidden_consoles_20261025/publication_recovery.json.')
    science.handoff('Дополнение автора25.10: все будущие консоли скрыто (-WindowStyle Hidden), без перехвата фокуса;11старыхвидимых окон скрыты. '
        'Текущий z13_region_product_v1 продолжает runs/z13_region_product_20261025/recovery1 с прежним hard05:12:33МСК. '
        'ИзвестныйCRLF сбой публикации исправляет отдельный scripts/z13_region_product_publication_recovery_20261025.py; scientific script и сроки неизменны. '
        'Публикационный watcher после конца recovery1 проверяет сохранённый полный итог и делает проверенный scopedpush. Старые ошибки не удалять.')


def watch():
    ps=WindowsProcesses();me=dict(pid=os.getpid(),birth=ps.creation(os.getpid()),at=now().isoformat())
    write(OUT/'publication_watcher.json',me)
    target=read(recovery.RUN/'pipeline.json')
    while ps.creation(target['pid'])==target['birth']:time.sleep(2)
    if (science.RUN/'publication.json').exists() and read(science.RUN/'publication.json')['push_completed']:return
    assert (science.RUN/'finish_recorded.json').exists(),'Scientific stage did not complete; do not publish a verdict'
    assert read(recovery.RUN/'controller_result.json')['exit_code']==0
    assert read(science.RUN/'independent_audit.json')['all24_models_reinferred_exact']
    publish('final')
    write(OUT/'publication_completed.json',dict(at=now().isoformat(),scientific_publication=read(science.RUN/'publication.json')['commit'],source_sha256=sha(Path(__file__))))


if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    elif '--start' in sys.argv:publish('start')
    elif '--watch' in sys.argv:watch()
    else:raise RuntimeError('Explicit mode required')
