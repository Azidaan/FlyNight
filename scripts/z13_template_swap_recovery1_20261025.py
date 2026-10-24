"""Service-only recovery: create output directories, retain original deadline/code."""
from pathlib import Path
from datetime import datetime
import hashlib
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.z13_template_swap_20261025 import RUN as OLD,W,IDENT,ARMS,read,write,sha,now,journal
RUN=OLD/'recovery1'


def worker_namespace():
    source=(ROOT/'scripts/z13_template_swap_20261025.py').read_text(encoding='utf-8')
    ns={'__name__':'template_swap_service_recovery','__file__':str(ROOT/'scripts/z13_template_swap_20261025.py')}
    exec(compile(source,ns['__file__'],'exec'),ns)
    elapsed=(now()-datetime.fromisoformat(read(OLD/'controller.json')['started'])).total_seconds()
    ns.update(RUN=RUN,T0=time.monotonic()-elapsed)
    return ns


def prepare():
    from scripts.z13_hota_deadline_guard import WindowsProcesses
    from scripts.z13_localization_labels_20261023 import verify_sources
    assert not RUN.exists()
    p,e=read(OLD/'protocol.json'),read(OLD/'controller_result.json')
    assert e['exit_code']==1 and e['worker_ended'] and e['all_owned_ended'] and not e['remaining']
    ps=WindowsProcesses();assert all(ps.creation(int(pid))!=birth for pid,birth in e['owned'].items())
    assert not list(OLD.glob('*/chunk_*.json'))
    assert now()<datetime.fromisoformat(p['save_at'])
    verify_sources(p)
    for name in ('donors.json','coverage.json'):
        destination=RUN/name;destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes((OLD/name).read_bytes());assert sha(destination)==sha(OLD/name)
    for arm in ARMS:(RUN/arm).mkdir(parents=True,exist_ok=False)
    additions=['scripts/z13_template_swap_recovery1_20261025.py','scripts/z13_finish_template_swap_recovery1_20261025.py','scripts/z13_template_swap_recovery1_console.ps1',
        (OLD/'protocol.json').relative_to(ROOT).as_posix(),(OLD/'controller.json').relative_to(ROOT).as_posix(),
        (OLD/'controller_result.json').relative_to(ROOT).as_posix(),(OLD/'finish_recorded.json').relative_to(ROOT).as_posix()]
    source=(ROOT/'scripts/z13_template_swap_20261025.py').read_text(encoding='utf-8')
    replacement=dict(p,created=now().isoformat(),stop_request_path=(RUN/'stop.request').relative_to(ROOT).as_posix(),
        recovery=dict(error='FileNotFoundError saving first chunk: arm output directory had not been created',
            change='Create D/C/R1 output directories before running identical frozen worker in a separate recovery folder. No scientific source edits, tolerance change, data/model change or completed-stage repeat.',
            original_started=read(OLD/'controller.json')['started'],original_save_at=p['save_at'],original_hard_stop=p['hard_stop'],retained_completed_groups=0,
            remaining_original_seconds=(datetime.fromisoformat(p['save_at'])-now()).total_seconds()),
        executed_source_sha256=hashlib.sha256(source.encode()).hexdigest(),sha256={**p['sha256'],**{n:sha(ROOT/n) for n in additions}})
    write(RUN/'protocol.json',replacement)
    write(RUN/'status.json',dict(state='ready',stage='Восстановлена запись карт; исходный срок прежний',done=0,total=822,updated=now().isoformat()))
    cp=ROOT/'experiments/project_history.json';cat=read(cp);card=next(c for c in cat['experiments'] if c['id']==IDENT)
    card['state']='ready';card['done']='';card['evidence'].append(dict(kind='file',ref=(RUN/'protocol.json').relative_to(ROOT).as_posix(),claim='Только создание выходных каталогов; исходный срок и код неизменны'))
    write(cp,cat)
    journal('служебное восстановление записи изолированной подмены',
        '*Ошибка.* Первый блок не сохранён: отсутствовал выходной каталог D. Сохранены исходные код1, stderr, controller_result.json и неполный finish_recorded.json. Полных блоков0; научного вывода нет.\n\n'
        '*Исправление.* До запуска создаются три выходных каталога в recovery1. Исполняется неизменный исходный код с другим адресом вывода; три донорских фильтра, веса, нормировки и допуски прежние. Исходный срок '+p['hard_stop']+' не продлевается.\n\n'
        '*Окно прогресса.* Исходный UTF8-скрипт сохранён; отдельный ASCII-загрузчик читает его с явной UTF8-кодировкой для Windows PowerShell5. Источник старой ошибки не удаляется.')


def worker():
    from scripts.z13_localization_labels_20261023 import verify_sources
    p=read(RUN/'protocol.json');verify_sources(p)
    source=(ROOT/'scripts/z13_template_swap_20261025.py').read_text(encoding='utf-8')
    assert hashlib.sha256(source.encode()).hexdigest()==p['executed_source_sha256']
    assert p['save_at']==p['recovery']['original_save_at'] and p['hard_stop']==p['recovery']['original_hard_stop']
    worker_namespace()['worker']()


if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    elif '--worker' in sys.argv:
        try:worker()
        except Exception as exc:
            write(RUN/'status.json',dict(state='failed',stage=repr(exc),updated=now().isoformat()));raise
    else:raise RuntimeError('Explicit mode required')
