"""Service-only skip of undefined ground truth fields on already-unknown rows."""
from pathlib import Path
import sys,inspect,hashlib,subprocess,os
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import z13_spatial_bins_gate_20261025 as base
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
from scripts.z13_hota_deadline_guard import WindowsProcesses
RUN=base.RUN/'recovery1'

def source():
    s=inspect.getsource(base.worker);old="if not o['own_after'] or o['harmful']:bad.append(j)"
    assert s.count(old)==1
    return s.replace(old,"if l['kind']>=0 and (not o['own_after'] or o['harmful']):bad.append(j)")

def service_read(path):
    path=Path(path)
    if path==base.RUN/'protocol.json':return read(RUN/'protocol.json')
    if path==base.RUN/'controller_result.json':return read(RUN/'controller_result.json')
    return read(path)

def prepare():
    assert not RUN.exists();p=read(base.RUN/'protocol.json');c=read(base.RUN/'controller_result.json');ps=WindowsProcesses()
    assert c['exit_code']==1 and c['all_owned_ended'] and not c['remaining'] and all(ps.creation(int(pid))!=b for pid,b in c['owned'].items())
    assert not list((base.RUN/'videos').glob('*.json')) and "KeyError: 'own_after'" in (base.RUN/'worker.stderr.log').read_text(encoding='utf-8')
    p=dict(p,service_recovery='Unknown rows have no own_after field; exclude their undefined semantic correctness before reading it. They already never enter the known-query denominator. All known actions, bins, criteria and original deadline unchanged; completed videos0.',
        inherited_parent_deadlines={k:p[k] for k in ('pipeline_save_at','pipeline_hard_stop')},pipeline_save_at=p['save_at'],pipeline_hard_stop=p['hard_stop'],
        executed_worker_sha256=hashlib.sha256(source().encode()).hexdigest(),sha256={**p['sha256'],Path(__file__).relative_to(ROOT).as_posix():sha(Path(__file__))})
    write(RUN/'protocol.json',p)
    base.journal('служебное восстановление неизвестных исходов пространственного учёта','*Ошибка.* До сохранения первого видео чтение own_after у неизвестного исхода дало KeyError; у этих строк поле отсутствует по определению. Старые exit1 и журналы сохранены. '
        '*Исправление.* Только не читать недоступную истинную правильность у kind<0; эти строки и прежде не входят в долю известных запросов. Известные исходы, решётка, критерий и первоначальный save '+p['save_at']+' / hard '+p['hard_stop']+' неизменны. '
        'Унаследованные поля pipeline_save_at от предшественника сохранены отдельно; фактический контроллер с первого запуска использовал новые save_at/hard_stop. Источник runs/z13_spatial_bins_gate_20261025/recovery1/protocol.json.')

def publish():
    code=inspect.getsource(base.template.publish).replace('Measure duplicate-time exposure before a FlyVis state intervention','Check retinal spatial-bin feasibility with known-outcome bookkeeping')
    needle='    assert not set(names)&foreign';assert code.count(needle)==1
    code=code.replace(needle,"    names += ['scripts/z13_spatial_bins_gate_20261025.py']\n"+needle)
    ns=dict(vars(base));ns['__file__']=__file__;exec(compile(code,__file__,'exec'),ns);ns['publish']()

def worker():
    code=source();assert hashlib.sha256(code.encode()).hexdigest()==read(RUN/'protocol.json')['executed_worker_sha256']
    ns={**vars(base),'read':service_read};exec(compile(code,__file__,'exec'),ns);ns['worker']()

def pipeline():
    ps=WindowsProcesses();write(RUN/'pipeline.json',dict(pid=os.getpid(),birth=ps.creation(os.getpid()),at=now().isoformat()))
    try:
        with (RUN/'controller.stdout.log').open('w',encoding='utf-8') as out,(RUN/'controller.stderr.log').open('w',encoding='utf-8') as err:
            child=subprocess.Popen([str(ROOT/'.venv/Scripts/python.exe'),'-u','scripts/z13_bounded_recovery_controller_20261022.py','--run',str(RUN),'--script',__file__],cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.IDLE_PRIORITY_CLASS|subprocess.CREATE_NO_WINDOW);child.wait()
        ns={**vars(base),'read':service_read,'publish':publish};exec(compile(inspect.getsource(base.finish),__file__,'exec'),ns);ns['finish']()
    except Exception as e:write(RUN/'pipeline_failure.json',dict(at=now().isoformat(),error=repr(e)));raise

if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    elif '--pipeline' in sys.argv:pipeline()
    elif '--worker' in sys.argv:worker()
    else:raise RuntimeError('Explicit mode required')
