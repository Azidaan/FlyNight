"""Inference packing16 instead of8; no changed examples, thresholds or deadlines."""
from pathlib import Path
import sys,os,hashlib,copy
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
from scripts import z13_spatial_set_recovery1_20261025 as prior
BASE=prior.BASE;PREV=prior.RUN;RUN=BASE/'recovery2'


def source():
    s=prior.source()
    old="oldpath=OLD/'features'/arm/split/f'chunk_{bi:05d}.npz'\n                    assert sha(oldpath)==read(oldpath.with_suffix('.json'))['sha256']\n                    with np.load(oldpath) as z:oldrows={int(u):(z['x'][i],z['mask'][i]) for i,u in enumerate(z['uid'])}"
    new="oldrows={};oldpaths=[]\n                    for oi in (2*bi,2*bi+1):\n                        oldpath=OLD/'features'/arm/split/f'chunk_{oi:05d}.npz'\n                        if not oldpath.exists():continue\n                        assert sha(oldpath)==read(oldpath.with_suffix('.json'))['sha256'];oldpaths.append(oldpath)\n                        with np.load(oldpath) as z:oldrows.update({int(u):(z['x'][i],z['mask'][i]) for i,u in enumerate(z['uid'])})"
    changes=[("RUN=ROOT/'runs/z13_spatial_set_readout_20261025/recovery1'","RUN=ROOT/'runs/z13_spatial_set_readout_20261025/recovery2'"),
        ("str(ROOT/'scripts/z13_spatial_set_recovery1_20261025.py')","str(ROOT/'scripts/z13_spatial_set_recovery2_20261025.py')"),
        ('batches=[sorted(byep)[i:i+8] for i in range(0,len(byep),8)]','batches=[sorted(byep)[i:i+16] for i in range(0,len(byep),16)]'),
        (old,new),("uid_sha256=sha(oldpath)","uid_sha256={f.relative_to(ROOT).as_posix():sha(f) for f in oldpaths}")]
    for a,b in changes:assert s.count(a)==1,(a,s.count(a));s=s.replace(a,b)
    return s


def namespace(verify=True):
    s=source()
    if verify:assert hashlib.sha256(s.encode()).hexdigest()==read(RUN/'protocol.json')['executed_service_sha256']
    ns={'__name__':'spatial_set_packing_recovery','__file__':str(ROOT/'scripts/z13_spatial_set_readout_20261025.py')};exec(compile(s,__file__,'exec'),ns);return ns


def prepare():
    from scripts.z13_hota_deadline_guard import WindowsProcesses
    from scripts.z13_localization_labels_20261023 import verify_sources
    assert not (RUN/'protocol.json').exists();p=read(PREV/'protocol.json');verify_sources(p);ps=WindowsProcesses()
    for folder in (PREV,PREV/'features'):
        e=read(folder/'controller_result.json');assert e['exit_code']==1 and e['all_owned_ended'] and not e['remaining']
        assert all(ps.creation(int(pid))!=b for pid,b in e['owned'].items())
    assert not read(PREV/'features/price.json')['proceed']
    markers=sorted((PREV/'features').glob('*/*/chunk_*.json'));assert len(markers)==2
    assert [m.relative_to(PREV/'features').as_posix() for m in markers]==['D/fit20/chunk_00000.json','D/fit20/chunk_00001.json']
    chunks=[];source_hashes={}
    for m in markers:
        path=m.with_suffix('.npz');assert sha(path)==read(m)['sha256'];source_hashes[path.relative_to(ROOT).as_posix()]=sha(path)
        with np.load(path) as z:chunks.append({k:z[k].copy() for k in z.files})
    width=max(a['retinal_mask'].shape[1] for a in chunks);parts=[]
    for d in chunks:
        a={}
        for k,v in d.items():
            if k not in ('uid','full'):
                shape=list(v.shape);shape[1]=width-v.shape[1]
                pad=np.ones(shape,dtype=v.dtype) if k=='scales' else np.zeros(shape,dtype=v.dtype)
                v=np.concatenate([v,pad],1)
            a[k]=v
        parts.append(a)
    arrays={k:np.concatenate([d[k] for d in parts],0) for k in parts[0]};assert len(set(arrays['uid'].tolist()))==len(arrays['uid'])
    dest=RUN/'features/D/fit20/chunk_00000.npz';dest.parent.mkdir(parents=True,exist_ok=True);np.savez(dest,**arrays)
    with np.load(dest) as z:
        offset=0
        for old in chunks:
            n=len(old['uid']);w=old['retinal_mask'].shape[1]
            for k,v in old.items():assert np.array_equal(z[k][offset:offset+n] if k in ('uid','full') else z[k][offset:offset+n,:w],v),(k,offset)
            offset+=n
    write(dest.with_suffix('.json'),dict(sha256=sha(dest),rows=len(arrays['uid']),seconds=0.,reused_without_GPU=True,source_sha256=source_hashes,max_center_px=max(read(m)['max_center_px'] for m in markers),max_score=max(read(m)['max_score'] for m in markers),same_full_mean_exact=True))
    q=copy.deepcopy(p);q['stop_request_path']=(RUN/'stop.request').relative_to(ROOT).as_posix()
    q['service_recovery'].update(at=now().isoformat(),cause2='Conservative original GPU price4795.67s exceeded remaining4521.99s; price gate preserved. No training yet.',fix2='Pack16 independent frozen inference episodes rather than8; keep all same examples/time prefixes, old numeric checks and original price formula. Reuse first two8episode chunks as one16chunk without any recomputation. All new chunks retain old UID/choices and canonical-center/score checks.',physical_inference_batch=16,training_batch_unchanged=128,reused_rows=len(arrays['uid']),completed_previous_gpu_chunks=2)
    q['executed_service_sha256']=hashlib.sha256(source().encode()).hexdigest()
    names=[Path(__file__).relative_to(ROOT).as_posix(),(PREV/'features/price.json').relative_to(ROOT).as_posix(),(PREV/'controller_result.json').relative_to(ROOT).as_posix(),(PREV/'features/controller_result.json').relative_to(ROOT).as_posix()]
    q['sha256']={**p['sha256'],**{n:sha(ROOT/n) for n in names},**source_hashes}
    write(RUN/'protocol.json',q);write(RUN/'status.json',dict(state='ready',stage='Пакеты16, прежний GPU срок и готовые46строк сохранены',updated=now().isoformat()))
    ns=namespace(verify=False)
    ns['journal']('упаковка замороженного вывода после ограничения стоимости','*Получено.* recovery1/features/price.json: консервативная оценка4795,673с, осталось4521,986с; gatefalse. Два блока46строк сохранены, исходные центры/оценки прошли старые допуски, разность среднего двух проходов не использовалась как критерий эффективности. Старые exit1/источники/журналы сохранены.\n\n*Исправление исполнения.* Пакет замороженного GPU-вывода8→16 независимых эпизодов; обучающий пакет128, веса/входы/истории/критерии/priceформула прежние. Два готовых8-блока точно объединены в один16-блок, без GPU-повтора. Все UID/маски и выходы сверяются с прежним производителем. Непройденная оценка стоимости не отменена; новая упаковка снова проходит прежний pricegate в оставшемся сроке '+q['service_recovery']['original_feature_save_at']+'. Глобальный срок '+q['save_at']+' не продлён. Научного результата ещё нет.')
    cat=read(ROOT/'experiments/project_history.json');card=next(x for x in cat['experiments'] if x['id']==p['id']);card['state']='running';card['evidence'].append(dict(kind='file',ref=(RUN/'protocol.json').relative_to(ROOT).as_posix(),claim='Упаковка16 только замороженного вывода, готовые46строк переиспользованы; прежние критерии и сроки'));write(ROOT/'experiments/project_history.json',cat)
    ns['handoff']('ТЕКУЩИЙ spatial-set recovery2: '+str(RUN.relative_to(ROOT))+'. recovery1 исправилCUDA, но price4795,67>4521,99с остановил. Готовые46строк/2блока точно объединены безGPUповтора. Новаяслужебнаяупаковка16замороженныхэпизодов; обучающийbatch128и все критерии прежние. Повторить исходныйpricegateна2новыхблоках, безослабления. GPUsave '+q['service_recovery']['original_feature_save_at']+' hard '+q['service_recovery']['original_feature_hard_stop']+'; общие '+q['save_at']+' / '+q['hard_stop']+'. Проверять recovery2 controller/status/features/logs,нестарыеошибки;после признаков автоматическиCPU10800свобщемокне. Чужие57правок/.gitignoreсохранить,всёскрыто. Не запускатьдубль. Старыеисточникинеизменны.')


if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    elif '--worker' in sys.argv:
        ns=namespace();phase=os.environ.get('Z13_SPATIAL_SET_PHASE','pipeline')
        try:ns[{'features':'feature_worker','training':'train_worker','pipeline':'pipeline'}[phase]]()
        except Exception as exc:
            folder=RUN/phase if phase!='pipeline' else RUN
            write(folder/'failure.json',dict(at=now().isoformat(),error=repr(exc)))
            write(folder/'status.json',dict(state='failed',stage=repr(exc),updated=now().isoformat()));raise
    else:raise RuntimeError('Explicit mode required')
