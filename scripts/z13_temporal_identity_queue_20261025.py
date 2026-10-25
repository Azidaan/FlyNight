"""Local bounded continuation: audit frozen features, then run only the registered learner."""
from pathlib import Path
from datetime import datetime,timedelta
import os,sys,time,subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import z13_temporal_identity_trial_20261025 as S
from scripts.z13_temporal_identity_trial_recovery1_20261025 import retry_write
from scripts.z13_hota_deadline_guard import WindowsProcesses


def feature_audit():
    import numpy as np
    f=S.FEATURES;p=S.read(f/'protocol.json');r=S.read(f/'report.json');c=S.read(f/'controller_result.json');ps=WindowsProcesses()
    assert r['complete'] and c['exit_code']==0 and c['all_owned_ended'] and not c['remaining']
    assert all(ps.creation(int(pid))!=birth for pid,birth in c['owned'].items())
    refs={}
    for videos in p['splits'].values():
        for v in videos:
            schedule={x['video_frame_index'] for x in S.read(ROOT/f'runs/train_cache/{v}/alignment.json')['matches']}
            for row in S.read(S.REF/'inputs'/(v+'.json'))['rows']:
                if row['previous']:
                    assert row['previous']['frame'] in schedule
                    refs[row['uid']]=row
    seen={a:set() for a in ('D','C','R1')};binding={};maximum={a:[0.,0.] for a in seen};groups=0
    for name,digest in r['manifest'].items():
        path=ROOT/name;assert S.sha(path)==digest;marker=S.read(path);npz=path.with_suffix('.npz');assert S.sha(npz)==marker['sha256']
        arm=Path(name).parts[-3];maximum[arm][0]=max(maximum[arm][0],marker['max_center_px']);maximum[arm][1]=max(maximum[arm][1],marker['max_score'])
        assert marker['max_center_px']<.5 and marker['max_score']<.001 and marker['all_prior_masks_exact'] and marker['save_reload_exact']
        with np.load(npz) as z:
            assert all(np.isfinite(z[k]).all() for k in z.files)
            for i,u in enumerate(z['uid']):
                u=int(u);assert u in refs and u not in seen[arm];seen[arm].add(u);ref=refs[u];n=int(z['previous_mask'][i].sum())
                assert n==len(ref['previous']['choices'])+1
                assert z['previous_mask'][i,:n].all() and not z['previous_mask'][i,n:].any()
                assert int(z['previous_frame'][i])==ref['previous']['frame']<ref['current_frame']
                assert int(z['gap_frames'][i])==ref['current_frame']-ref['previous']['frame']
                b=(n,int(z['previous_frame'][i]),int(z['gap_frames'][i]))
                if u in binding:assert binding[u]==b
                else:binding[u]=b
        groups+=1
    assert len(refs)==18824 and all(v==set(refs) for v in seen.values()) and groups==r['groups']
    retry_write(f/'independent_audit.json',dict(at=S.now().isoformat(),all_bindings_verified=True,rows_each_arm={a:len(v) for a,v in seen.items()},groups=groups,
        maximum_old_center_px_and_score=maximum,all_arrays_and_markers_hash_verified=True,
        original_source_hashes_checked_by_worker=True,no_GT_object_file_opened_by_worker=True,feature_report_sha256=S.sha(f/'report.json')))
    path=ROOT/'experiments/project_history.json';catalog=S.read(path);card=next(c for c in catalog['experiments'] if c['id']=='z13_temporal_identity_features_v1')
    card.update(state='infrastructure',done=S.now().strftime('%Y-%m-%d'),summary='Равные признаки прошлых опорных областейD/C/R1 подготовлены; текущие ответы сохранены в прежних допусках.',
        obtained='runs/z13_temporal_identity_20261025/features/report.json: rows_each_arm='+str(r['rows_each_arm'])+',groups='+str(groups)+',seconds='+str(r['seconds'])+'; independent_audit.json:maximum_old_center_px_and_score='+str(maximum)+'.',
        interpretation='Подтверждена техническая готовность одинаковых прошлых областей на наблюдаемой сетке; это ещё не результат обучения личности или HOTA.',
        decision='Передать в зарегистрированный ограниченный опыт считывателя после точного воспроизведения12исходныхголов и обязательных проверок каталога.')
    for i,a in enumerate(card['acceptance']):a.update(met=True,evidence='runs/z13_temporal_identity_20261025/features/'+('independent_audit.json' if i==0 else 'price.json'))
    card['confirmed_results']=[];retry_write(path,catalog)
    S.note('итог равных признаков прошлых областей','*Получено.* '+card['obtained']+'\n\n*Вывод.* '+card['interpretation']+'\n\n*Решение.* '+card['decision']+'\n\nHOTA и реализованная моделью доля потолка не измерены;434блока прежней иной выборки сохранены.')


def queue():
    ps=WindowsProcesses();base=S.BASE;receipt=base/'queue.json';assert not receipt.exists()
    retry_write(receipt,dict(pid=os.getpid(),birth=ps.creation(os.getpid()),started=S.now().isoformat(),script=Path(__file__).relative_to(ROOT).as_posix()))
    p=S.read(S.FEATURES/'protocol.json')
    try:
        while not (S.FEATURES/'controller_result.json').exists():
            if S.now()>datetime.fromisoformat(p['hard_stop'])+timedelta(seconds=30):raise TimeoutError('Feature controller missing after original deadline')
            time.sleep(2)
        assert S.read(S.FEATURES/'controller_result.json')['exit_code']==0,'Feature extraction failed; do not train'
        feature_audit()
        assert S.read(S.RUN/'original_regression.json')['all_exact']
        env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONIOENCODING':'utf-8'}
        for label,args in [('catalog',['scripts/validate_project_history.py']),('tests',['-m','pytest','tests/test_project_history.py','tests/test_z13_temporal_identity_readout.py','-q'])]:
            with (base/(label+'_learner_preflight.log')).open('w',encoding='utf8') as log:
                result=subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),*args],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.IDLE_PRIORITY_CLASS)
            assert result.returncode==0,(label,result.returncode)
        retry_write(base/'learner_preflight.json',dict(at=S.now().isoformat(),catalog_ok=True,tests_ok=True,old12heads_exact=True,matched_features_verified=True))
        with (S.RUN/'controller.stdout.log').open('w',encoding='utf8') as out,(S.RUN/'controller.stderr.log').open('w',encoding='utf8') as err:
            child=subprocess.Popen([str(ROOT/'.venv/Scripts/python.exe'),'-u','scripts/z13_bounded_recovery_controller_20261022.py','--run',str(S.RUN),'--script',str(ROOT/'scripts/z13_temporal_identity_trial_recovery1_20261025.py')],cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.IDLE_PRIORITY_CLASS|subprocess.CREATE_NO_WINDOW)
            retry_write(S.RUN/'launch.json',dict(pid=child.pid,birth=ps.creation(child.pid),at=S.now().isoformat()))
            code=child.wait()
        retry_write(base/'queue_result.json',dict(at=S.now().isoformat(),state='computed' if code==0 else 'failed',training_controller_exit=code))
    except Exception as e:
        retry_write(base/'queue_failure.json',dict(at=S.now().isoformat(),error=repr(e)));raise


if __name__=='__main__':
    if '--feature-audit' in sys.argv:feature_audit()
    elif '--queue' in sys.argv:queue()
