"""Separate CPU training/recording of version2; frozen optimizer and comparisons."""
from pathlib import Path
import os,sys,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['CUDA_VISIBLE_DEVICES']='-1';os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
from scripts.z13_filter_descriptor_fixed_20261024 import RUN as CACHE,TRAIN,IDENT,read,write,sha,now,journal


def source():
    text=(ROOT/'scripts/z13_filter_descriptor_20261023.py').read_text(encoding='utf-8')
    old="RUN=ROOT/'runs/z13_filter_descriptor_20261023'";assert text.count(old)==1
    text=text.replace(old,"RUN=ROOT/'runs/z13_filter_descriptor_fixed_20261024/train'")
    text=text.replace("IDENT='z13_filter_descriptor_v1'","IDENT='z13_filter_descriptor_fixed_v2'")
    text=text.replace('from scripts.z13_filter_descriptor_cache_20261023 import RUN as CRUN','from scripts.z13_filter_descriptor_fixed_20261024 import RUN as CRUN')
    text=text.replace('runs/z13_filter_descriptor_cache_20261023/report.json','runs/z13_filter_descriptor_fixed_20261024/cache/report.json')
    return text


def namespace():
    ns={'__name__':'fixed_descriptor_cpu_training','__file__':str(ROOT/'scripts/z13_filter_descriptor_20261023.py')};exec(compile(source(),ns['__file__'],'exec'),ns);ns.update(journal=journal);return ns


def prepare():
    from scripts.z13_record_filter_descriptor_20261024 import ended
    ended(CACHE);assert read(CACHE/'report.json')['verification_policy'].startswith('Original fixed')
    ns=namespace();ns['prepare']();p=read(TRAIN/'protocol.json');p['cache_execution_controller']=CACHE.relative_to(ROOT).as_posix();p['executed_training_source_sha256']=hashlib.sha256(source().encode()).hexdigest();p['service_sha256']={n:sha(ROOT/n) for n in ['scripts/z13_filter_descriptor_fixed_train_20261024.py','scripts/z13_filter_descriptor_fixed_train_console.ps1']};p['sha256'].update(p['service_sha256']);write(TRAIN/'protocol.json',p)


def worker():
    import torch
    torch.cuda.is_available=lambda:False;torch.cuda.device_count=lambda:0;torch.set_default_device('cpu');torch.set_num_threads(2)
    p=read(TRAIN/'protocol.json');assert p['executed_training_source_sha256']==hashlib.sha256(source().encode()).hexdigest()
    assert p['service_sha256']['scripts/z13_filter_descriptor_fixed_train_20261024.py']==sha(Path(__file__))
    ns=namespace();ns['worker']();write(TRAIN/'status.json',dict(state='completed',stage='Версия2:36CPUмоделей и парные итоги сохранены',updated=now().isoformat(),models_done=36,models_total=36))


def record():
    text=(ROOT/'scripts/z13_record_filter_descriptor_20261024.py').read_text(encoding='utf-8')
    text=text.replace('runs/z13_filter_descriptor_20261023/','runs/z13_filter_descriptor_fixed_20261024/train/').replace('runs/z13_filter_descriptor_cache_20261023/','runs/z13_filter_descriptor_fixed_20261024/cache/')
    ns={'__name__':'fixed_descriptor_record','__file__':str(ROOT/'scripts/z13_record_filter_descriptor_20261024.py')};exec(compile(text,ns['__file__'],'exec'),ns);ns.update(RUN=TRAIN,CRUN=CACHE,IDENT=IDENT,journal=journal);ns['main']()


if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    elif '--record' in sys.argv:record()
    elif '--worker' in sys.argv:
        try:worker()
        except Exception as e:write(TRAIN/'status.json',dict(state='failed',stage=repr(e),updated=now().isoformat()));raise
    else:raise RuntimeError('Explicit training mode required')
