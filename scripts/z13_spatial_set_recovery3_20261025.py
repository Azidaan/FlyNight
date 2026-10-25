"""CPU service recovery: assemble transient inputs in RAM, not synchronous memmaps."""
from pathlib import Path
import os,sys,inspect,hashlib,ctypes,subprocess
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
from scripts import z13_spatial_set_recovery2_20261025 as previous
RUN=previous.RUN;SERVICE=RUN/'training/recovery3'


def available_memory():
    class State(ctypes.Structure):
        _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong),('total',ctypes.c_ulonglong),('available',ctypes.c_ulonglong),('total_page',ctypes.c_ulonglong),('available_page',ctypes.c_ulonglong),('total_virtual',ctypes.c_ulonglong),('available_virtual',ctypes.c_ulonglong),('extended',ctypes.c_ulonglong)]
    s=State();s.length=ctypes.sizeof(s);assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s));return int(s.available)


def ram_load_data(torch,arm,entries,include26=False):
    from scripts.z13_region_assignment_recovery2_20261025 import namespace
    from scripts.z13_default_class_trial_20261025 import load,merge
    ns=namespace();basearm='D' if arm in ('geometry','D_mean') else arm.split('_')[0]
    data=merge(torch,load(torch,ns,basearm,'fit84',entries),load(torch,ns,basearm,'fit20',entries));splits=['fit84','fit20']
    if include26:
        ev=load(torch,ns,basearm,'train26',entries);ev=dict(ev,video=ev['video']+20);data=merge(torch,data,ev);splits.append('train26')
    N=len(data['uid']);K=data['mask'].shape[1];shapes=dict(full=(N,721,68),pooled=(N,K,68),retinal_mask=(N,K,721),centers=(N,K,2),scales=(N,K,2))
    required=sum(int(np.prod(s))*(1 if k=='retinal_mask' else 4) for k,s in shapes.items())
    # Check available RAM before allocation; do not fall back to another disk writer.
    available=available_memory();assert available>required*1.2+1024**3,(available,required)
    arrays={k:np.zeros(shape,dtype=np.bool_ if k=='retinal_mask' else np.float32) for k,shape in shapes.items()}
    lookup={int(u):i for i,u in enumerate(data['uid'].tolist())};seen=set();source_hashes={};groups=0
    for split in splits:
        for marker in sorted((RUN/'features'/basearm/split).glob('chunk_*.json')):
            path=marker.with_suffix('.npz');h=sha(path);assert h==read(marker)['sha256'];source_hashes[path.relative_to(ROOT).as_posix()]=h
            with np.load(path) as z:
                idx=np.array([lookup[int(u)] for u in z['uid']]);assert not seen.intersection(idx.tolist());seen.update(idx.tolist())
                arrays['full'][idx]=z['full'];w=z['retinal_mask'].shape[1]
                for k in ('pooled','retinal_mask','centers','scales'):arrays[k][idx,:w]=z[k]
            groups+=1
            if groups%10==0:status('training',stage='Сборка сохранённых входов в оперативной памяти',variant=basearm+'/'+split,groups_done=groups,rows_done=len(seen),rows_total=N)
    assert len(seen)==N and np.isfinite(arrays['full']).all()
    write(RUN/'training/ram_inputs'/basearm/('all130.json' if include26 else 'fit104.json'),dict(at=now().isoformat(),rows=N,shapes={k:list(v) for k,v in shapes.items()},source_sha256=source_hashes,uid=data['uid'].tolist(),memory_bytes=required,available_before=available,transient_only=True,original_gpu_chunks_unchanged=True))
    data.update({('x' if k=='pooled' else k):v for k,v in arrays.items()});return data


def source():
    s=previous.source();a=s.index('def load_data(');b=s.index('\n\ndef take(',a)
    new=inspect.getsource(ram_load_data).replace('def ram_load_data(','def load_data(',1)
    s=s[:a]+new+s[b:]
    old='torch.from_numpy(np.array(v[idx],copy=True))';assert s.count(old)==1
    s=s.replace(old,'torch.from_numpy(v[idx])') # Advanced integer indexing already returns an owned copy.
    return s


def prepare():
    from scripts.z13_hota_deadline_guard import WindowsProcesses
    from scripts.z13_localization_labels_20261023 import verify_sources
    assert not (SERVICE/'protocol.json').exists();p=read(RUN/'training/protocol.json');verify_sources(p)
    assert read(RUN/'features/report.json')['complete'] and read(RUN/'features/controller_result.json')['all_owned_ended']
    assert not list((RUN/'training').glob('internal/*/seed*/fold*/checkpoint.pt')) and not list((RUN/'training').glob('internal/*/seed*/fold*/done.json'))
    c=read(RUN/'training/controller.json');ps=WindowsProcesses();assert ps.creation(c['worker_pid'])==c['worker_birth']
    rows=ps.children();candidates=[pid for pid,parent in rows if parent==c['worker_pid'] and ps.creation(pid) is not None];matches=[]
    # The venv launcher also owns a conhost child. Select the actual Python
    # interpreter by its readable Python stack rather than counting all children.
    for pid in candidates:
        probe=subprocess.run([str(ROOT/'.venv/Scripts/py-spy.exe'),'dump','--pid',str(pid)],text=True,encoding='utf8',stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=10)
        if probe.returncode==0 and 'Python v' in probe.stdout:matches.append((pid,probe.stdout))
    assert len(matches)==1,matches
    real=[matches[0][0]];stack=matches[0][1]
    assert 'flush (numpy' in stack and 'load_data' in stack,stack
    write(SERVICE/'flush_stall_diagnostic.json',dict(at=now().isoformat(),controller=c,real_worker_pid=real[0],real_worker_birth=ps.creation(real[0]),stack=stack,status=read(RUN/'training/status.json'),available_memory=available_memory(),models_completed=0,checkpoints=0,temporary_npy_paths=[str(f.relative_to(ROOT)) for f in (RUN/'training/consolidated/D/fit104').glob('*.npy')],cause='Main thread blocked in NumPy memmap.flush on transient5.68GB cache; no model has started. Original feature chunks are complete and retained. Stat of a flushing temporary file also blocks, so do not stat it during recovery.'))
    assert ps.creation(c['worker_pid'])==c['worker_birth']
    stopped=subprocess.run(['taskkill','/PID',str(c['worker_pid']),'/T','/F'],capture_output=True,creationflags=subprocess.CREATE_NO_WINDOW)
    write(SERVICE/'scoped_stop.json',dict(at=now().isoformat(),pid=c['worker_pid'],birth=c['worker_birth'],returncode=stopped.returncode,stdout=stopped.stdout.decode(errors='replace'),stderr=stopped.stderr.decode(errors='replace')))
    assert stopped.returncode==0
    # Controllers must first record the interruption and prove their complete tree ended.
    for _ in range(40):
        import time
        if (RUN/'controller_result.json').exists() and (RUN/'training/controller_result.json').exists():break
        time.sleep(.5)
    for folder in (RUN,RUN/'training'):
        e=read(folder/'controller_result.json');assert e['all_owned_ended'] and not e['remaining']
        assert all(ps.creation(int(pid))!=birth for pid,birth in e['owned'].items())
    q=dict(p,device='cpu',stop_request_path=(SERVICE/'stop.request').relative_to(ROOT).as_posix(),service_recovery=dict(at=now().isoformat(),cause='Synchronous flush of transient memory mapped arrays blocks before first model.',fix='Same source chunks copied into ordinary in-memory NumPy arrays with allUID/hash checks. No transient consolidation files need persistence; original GPU chunks are retained. Remove duplicate copy after advanced integer indexing; values identical.',original_cpu_save_at=p['save_at'],original_cpu_hard_stop=p['hard_stop'],completed_models=0))
    q['executed_service_sha256']=hashlib.sha256(source().encode()).hexdigest()
    names=[Path(__file__).relative_to(ROOT).as_posix(),(RUN/'controller_result.json').relative_to(ROOT).as_posix(),(RUN/'training/controller_result.json').relative_to(ROOT).as_posix(),(SERVICE/'flush_stall_diagnostic.json').relative_to(ROOT).as_posix()]
    q['sha256']={**p['sha256'],**{n:sha(ROOT/n) for n in names}}
    write(SERVICE/'protocol.json',q);write(SERVICE/'status.json',dict(state='ready',stage='Сборка тех же входов в RAM',updated=now().isoformat()))
    # Old status and protocol are kept as evidence; active worker writes its current
    # progress to the established training/status location, with separate controller.
    write(SERVICE/'previous_training_status.json',read(RUN/'training/status.json'))
    ns=namespace(verify=False)
    ns['journal']('CPU-восстановление сборки входов без синхронного memmap.flush','*Получено.* GPU-этап завершён19:54:39МСК:1863блока,старые peak-проверки и точные новые mean-пары пройдены; features/controller_result.json exit0/all_owned_ended. CPU начал19:54:44, но до первого обучения заблокировался в numpy.memmap.flush временного массива5681364768байт. Стек/PID/birth и старые статусы сохранены в training/recovery3/flush_stall_diagnostic.json; готовых моделей0.\n\n*Исправление.* Завершено только своё прежнее CPU-дерево после проверки идентичностей. Те же проверенные GPU-блоки собираются в обычные массивы оперативной памяти; дублирующее копирование после целочисленного индексирования убрано без изменения значений. Старые временныефайлы/логи/исходники/коды не удалены. Веса, примеры, разбиения, обучение и критерии неизменны; CPU-срок '+p['save_at']+' / '+p['hard_stop']+' сохранён, бюджет не продлён.')
    cat=read(ROOT/'experiments/project_history.json');card=next(x for x in cat['experiments'] if x['id']==p['id']);card['evidence'].append(dict(kind='file',ref=(SERVICE/'protocol.json').relative_to(ROOT).as_posix(),claim='CPU-служебное восстановление памяти; все1863GPUблока переиспользуются,0моделей повторено, прежний срок'));write(ROOT/'experiments/project_history.json',cat)
    ns['handoff']('ТЕКУЩИЙ spatial-set CPUrecovery3: '+SERVICE.relative_to(ROOT).as_posix()+'. GPU1863блока полностью готовы в recovery2/features,exit0;ихНЕповторять. training/recovery3 имеет отдельныйcontroller иprotocol,но первичные training/status,internal,report,price по-прежнему в recovery2/training. Предыдущее root/training дерево остановлено поbirth: дообученияflush5,68GB; старыеexit/logs/массивысохранены. ТеперьсборкаRAM,данные/модели/критериипрежние;CPU save '+p['save_at']+' hard '+p['hard_stop']+'. Не путатьстарыйrooterrorсновымCPUcontroller. После75моделей только при внутреннемдопуске15итоговых;никакогоавтоHOTA. ВсеGPUисточникихешированы;57чужихправок/.gitignoreсохранить,всёскрыто. Итог→независимыйаудит/карточка/revisions/журнал/тесты/preserving graph/коммит/push.')


def namespace(verify=True):
    s=source()
    if verify:assert hashlib.sha256(s.encode()).hexdigest()==read(SERVICE/'protocol.json')['executed_service_sha256']
    ns={'__name__':'spatial_set_ram_recovery','__file__':str(ROOT/'scripts/z13_spatial_set_readout_20261025.py'),'available_memory':available_memory};exec(compile(s,__file__,'exec'),ns);return ns


if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    elif '--worker' in sys.argv:
        os.environ['CUDA_VISIBLE_DEVICES']='';ns=namespace()
        try:
            ns['train_worker']();write(SERVICE/'status.json',dict(state='completed',stage='Результат CPU сохранён',updated=now().isoformat()))
        except Exception as exc:
            write(SERVICE/'failure.json',dict(at=now().isoformat(),error=repr(exc)));write(SERVICE/'status.json',dict(state='failed',stage=repr(exc),updated=now().isoformat()));raise
    else:raise RuntimeError('Explicit mode required')
