"""Service-only CUDA restoration after loading CPU metadata; original deadlines."""
from pathlib import Path
import os,sys,hashlib,copy
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
BASE=ROOT/'runs/z13_spatial_set_readout_20261025';RUN=BASE/'recovery1'


def source():
    s=(ROOT/'scripts/z13_spatial_set_readout_20261025.py').read_text(encoding='utf8')
    pairs=[("RUN=ROOT/'runs/z13_spatial_set_readout_20261025'","RUN=ROOT/'runs/z13_spatial_set_readout_20261025/recovery1'"),
      ("queries,_=ns['queries_and_crops']();geom=ns['geometry']()","queries,_=ns['queries_and_crops']();os.environ['CUDA_VISIBLE_DEVICES']='0';assert torch.cuda.device_count()==1;geom=ns['geometry']()"),
      ("stop=min(now()+timedelta(seconds=BUDGETS[phase]),datetime.fromisoformat(p['save_at']))","stop=min(now()+timedelta(seconds=BUDGETS[phase]),datetime.fromisoformat(p['save_at']))\n        if phase=='features':stop=min(stop,datetime.fromisoformat(p['service_recovery']['original_feature_save_at']))"),
      ("write(folder/'protocol.json',q);write(RUN/'status.json'","write(folder/'protocol.json',q);write(folder/'status.json',dict(state='ready',stage='Инициализация этапа',updated=now().isoformat()));write(RUN/'status.json'"),
      ("str(Path(__file__))],cwd=ROOT,env=env","str(ROOT/'scripts/z13_spatial_set_recovery1_20261025.py')],cwd=ROOT,env=env")]
    for a,b in pairs:assert s.count(a)==1,(a,s.count(a));s=s.replace(a,b)
    return s


def device_check():
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    import torch
    torch.set_num_threads(2);torch.set_default_device('cpu');before=os.environ['CUDA_VISIBLE_DEVICES']
    from scripts.z13_assignment_effect_20261024 import load_queries
    after=os.environ['CUDA_VISIBLE_DEVICES'];assert before=='0' and after==''
    os.environ['CUDA_VISIBLE_DEVICES']='0';assert torch.cuda.device_count()==1 and torch.cuda.is_available()
    x=torch.ones(3,device='cuda:0');assert float(x.sum())==3.
    write(BASE/'recovery_device_check.json',dict(at=now().isoformat(),before=before,after_metadata_import=after,after_restore=os.environ['CUDA_VISIBLE_DEVICES'],device_count=torch.cuda.device_count(),tensor_device=str(x.device),passed=True))


def prepare():
    from scripts.z13_hota_deadline_guard import WindowsProcesses
    from scripts.z13_localization_labels_20261023 import verify_sources
    assert not (RUN/'protocol.json').exists();p=read(BASE/'protocol.json');verify_sources(p);ps=WindowsProcesses()
    for folder in (BASE,BASE/'features'):
        e=read(folder/'controller_result.json');assert e['exit_code']==1 and e['all_owned_ended'] and not e['remaining']
        assert all(ps.creation(int(pid))!=b for pid,b in e['owned'].items())
    assert not list((BASE/'features').glob('*/*/chunk_*.json')) and read(BASE/'recovery_device_check.json')['passed']
    q=copy.deepcopy(p);q['stop_request_path']=(RUN/'stop.request').relative_to(ROOT).as_posix()
    old=read(BASE/'features/protocol.json');q['service_recovery']=dict(at=now().isoformat(),cause='queries_and_crops imports CPU assignment-effect helper after the earlier environment restoration; helper masks CUDA again.',fix='Restore CUDA0 immediately after CPU metadata returns, before querying count/loading network. Initialize phase status before launching child.',original_feature_save_at=old['save_at'],original_feature_hard_stop=old['hard_stop'],original_global_save_at=p['save_at'],original_global_hard_stop=p['hard_stop'],completed_gpu_chunks=0)
    q['executed_service_sha256']=hashlib.sha256(source().encode()).hexdigest()
    extras=[Path(__file__).relative_to(ROOT).as_posix(),(BASE/'recovery_device_check.json').relative_to(ROOT).as_posix(),(BASE/'controller_result.json').relative_to(ROOT).as_posix(),(BASE/'features/controller_result.json').relative_to(ROOT).as_posix()]
    q['sha256']={**p['sha256'],**{n:sha(ROOT/n) for n in extras}}
    write(RUN/'protocol.json',q);write(RUN/'status.json',dict(state='ready',stage='Восстановление видимости GPU после CPU-метаданных',updated=now().isoformat()))
    ns=namespace(verify=False)
    ns['journal']('служебное восстановление GPU пространственного считывания','*Сбой.* Первый features закончился exit1 при загрузке сети: CPU-импорт queries_and_crops сбросил CUDA_VISIBLE_DEVICES после прежнего восстановления. Первый запуск и все журналы/исходники сохранены. Новых GPU-блоков0.\n\n*Проверено.* recovery_device_check.json воспроизвёл0→пусто и восстановил cuda0 с реальным тензором.\n\n*Исправление.* Отдельная версия scripts/z13_spatial_set_recovery1_20261025.py восстанавливает устройство после возврата CPU-метаданных и создаёт статус до запуска дочернего процесса. Протокол/хеш исполняемого кода: recovery1/protocol.json. Веса, данные, входы, критерии и исходные сроки GPU '+old['save_at']+' / '+old['hard_stop']+' и всего дерева '+p['save_at']+' / '+p['hard_stop']+' сохранены. Предыдущий неудачный результат не подменён успехом.')
    cat=read(ROOT/'experiments/project_history.json');card=next(x for x in cat['experiments'] if x['id']==p['id']);card['state']='running';card['evidence'].append(dict(kind='file',ref=(RUN/'protocol.json').relative_to(ROOT).as_posix(),claim='Служебное восстановление CUDA после CPU-импорта; научный протокол и прежние сроки сохранены'));write(ROOT/'experiments/project_history.json',cat)
    ns['handoff']('ТЕКУЩИЙ spatial-set recovery1: '+str(RUN.relative_to(ROOT))+'. Первые root/features exit1 из-за маскированияCUDA;всеPIDпоbirthзакончились,0новыхchunks. Devicecheckpassed, отдельнаяслужебнаяверсиявосстанавливаетCUDAПОСЛЕqueries_and_crops. GPUдедлайнпрежний '+old['save_at']+' / '+old['hard_stop']+', общий '+p['save_at']+' / '+p['hard_stop']+'. Не запускатьдубль;читайrecovery1/controller/status/features/logs,послепризнаковавтоматическиCPU10800свобщемокне. Научныеисточникинеизменны;57чужихправок/.gitignoreсохранить,всёскрыто. Результатещёнеизмерен.')


def namespace(verify=True):
    s=source()
    if verify:assert hashlib.sha256(s.encode()).hexdigest()==read(RUN/'protocol.json')['executed_service_sha256']
    ns={'__name__':'spatial_set_service_recovery','__file__':str(ROOT/'scripts/z13_spatial_set_readout_20261025.py')};exec(compile(s,__file__,'exec'),ns);return ns


if __name__=='__main__':
    if '--device-check' in sys.argv:device_check()
    elif '--prepare' in sys.argv:prepare()
    elif '--worker' in sys.argv:
        ns=namespace();phase=os.environ.get('Z13_SPATIAL_SET_PHASE','pipeline')
        try:ns[{'features':'feature_worker','training':'train_worker','pipeline':'pipeline'}[phase]]()
        except Exception as exc:
            folder=RUN/phase if phase!='pipeline' else RUN
            write(folder/'failure.json',dict(at=now().isoformat(),error=repr(exc)))
            write(folder/'status.json',dict(state='failed',stage=repr(exc),updated=now().isoformat()));raise
    else:raise RuntimeError('Explicit mode required')
