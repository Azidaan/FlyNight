"""Хеши/разбиения/потери всех30голов, CPU-повтор первого полногоdev-видео."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
import sys,subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261028 import RUN,service
read,write,sha,now=service.read,service.write,service.sha,service.now


def main():
    import numpy as np,torch
    from tracking.history_pair_features import from_cache
    from tracking.history_pair_model import HistoryPairHead
    source=RUN/'history_factorial_training';folder=RUN/'history_factorial_training_audit'
    c=read(source/'controller_result.json');assert c['exit_code']==0 and c['all_owned_ended'] and not any(service.matching(k,v) for k,v in c['owned'].items())
    r=read(source/'report.json');p=read(source/'protocol.json');history=read(ROOT/'experiments/project_history.json')
    card=next(x for x in history['experiments'] if x['id']=='history_factorial_training_v1')
    if not r['complete']:
        write(folder/'report.json',dict(passed=False,full_training=False,scientific_verdict=None,reason='Цена остановила полную матрицу; сохранена первая эпоха, не научное отрицание истории',updated=now()))
        card.update(state='inconclusive',done='2026-10-28',obtained='runs/continuation_20261028/history_factorial_training/report.json: complete=false, первая эпоха сохранена; price.json не допустил остальные эпохи.',interpretation='Полной обученной матрицы нет; качество истории поHOTA не оценено.',decision='Сохранить цену и частичный checkpoint; прежний бюджет не продлевать.',cannot_claim='Нет полного обучения/HOTA; остановка по стоимости не означает бесполезности представлений.')
    else:
        torch.set_num_threads(2)
        manifest=read(source/'data_manifest.json');assert sha(source/'data_manifest.json')==r['data_manifest_sha256']
        assert len(r['completed'])==len(p['variants'])*len(p['seeds'])==30
        assert {(e['variant'],e['seed']) for e in r['completed']}=={(v,s) for v in p['variants'] for s in p['seeds']}
        train_counts=np.zeros(2,np.int64);dev=[]
        for v,e in manifest['videos'].items():
            pair=RUN/'history_pair_dataset/videos'/v
            assert sha(pair/'inputs.npz')==e['inputs_sha256'] and sha(pair/'targets.npz')==e['targets_sha256']
            with np.load(pair/'targets.npz') as z:y=z['identity'];assert np.array_equal(z['known'],y>=0)
            if e['split']=='fit84':train_counts+=np.bincount(y[y>=0],minlength=2)
            else:assert e['split']=='fit20';dev.append(v)
        assert np.array_equal(train_counts,manifest['train_counts'])
        weights=train_counts.sum()/(2*train_counts);assert np.array_equal(weights,manifest['class_weights'])
        first=next(e['video'] for e in read(RUN/'history_visual_inputs/report.json')['completed'] if e['split']=='fit20')
        pair=RUN/'history_pair_dataset/videos'/first;visual=RUN/'history_visual_inputs/videos'/first
        with np.load(pair/'inputs.npz') as z:x={k:z[k] for k in z.files}
        with np.load(visual/'index.npz') as z:index={k:z[k] for k in z.files}
        embeddings=np.load(visual/'embeddings.npy',mmap_mode='r');images=np.load(visual/'images.npy',mmap_mode='r')
        initial={s:set() for s in p['seeds']};parameters=set();max_diff=0.;verified_pairs=0
        for m in r['completed']:
            name=f"{m['variant']}_seed{m['seed']}";out=source/'models'/name
            assert sha(out/'best.pt')==m['best_sha256'] and len(m['epochs'])==p['epochs']
            initial[m['seed']].add(m['initial_sha256']);parameters.add(m['parameters'])
            checkpoint=torch.load(out/'best.pt',map_location='cpu',weights_only=False)
            assert checkpoint['protocol_sha256']==sha(source/'protocol.json') and checkpoint['variant']==m['variant'] and checkpoint['seed']==m['seed']
            assert all(torch.isfinite(v).all() for v in checkpoint['state'].values())
            assert checkpoint['epoch']==min(m['epochs'],key=lambda e:e['dev_loss'])['epoch']
            loss_sum=0.;n=0
            for v in dev:
                f=out/'dev_predictions'/(v+'.npz');assert sha(f)==m['dev_prediction_sha256'][v]
                with np.load(f) as z:rows=z['rows'];scores=z['logits']
                with np.load(RUN/'history_pair_dataset/videos'/v/'targets.npz') as z:y=z['identity']
                assert np.array_equal(rows,np.flatnonzero(y>=0)) and np.isfinite(scores).all()
                yy=y[rows];scores=scores.astype(np.float64)
                loss_sum+=float(((np.logaddexp(0,scores)-yy*scores)*weights[yy]).sum());n+=len(rows)
            assert abs(loss_sum/n-m['best_dev_loss'])<=p['audit_loss_tolerance']
            model=HistoryPairHead();model.load_state_dict(checkpoint['state']);model.eval()
            with np.load(out/'dev_predictions'/(first+'.npz')) as z:rows=z['rows'];expected=z['logits']
            actual=[]
            with torch.no_grad():
                for start in range(0,len(rows),p['batch_size']):
                    f=from_cache(x,index,embeddings,images,rows[start:start+p['batch_size']],m['variant'])
                    actual.append(model({k:torch.from_numpy(v) for k,v in f.items()}).numpy())
            diff=float(np.max(np.abs(np.concatenate(actual)-expected))) if len(rows) else 0.
            assert diff<=p['audit_forward_tolerance'];max_diff=max(max_diff,diff);verified_pairs+=len(rows)
            write(folder/'status.json',dict(state='running',model=name,done=sum(len(v) for v in initial.values()),models=30,updated=now()),required=False)
        assert all(len(v)==1 for v in initial.values()) and len(parameters)==1 and not torch.cuda.is_initialized()
        write(folder/'report.json',dict(passed=True,full_training=True,models=30,parameters=next(iter(parameters)),same_initialization_within_seed=True,unknown_excluded=True,class_weights_fit84_only=True,all_dev_losses_replayed=True,first_full_dev_video=first,forward_pairs=verified_pairs,max_cpu_gpu_difference=max_diff,hota_measured=False,scientific_verdict=None,updated=now()))
        card.update(state='infrastructure',done='2026-10-28',obtained='runs/continuation_20261028/history_factorial_training/report.json и history_factorial_training_audit/report.json:30голов/3начальныхзначения/10вариантов/6эпох; общая архитектура, проверены источники/разбиения/всеdevпотери/CPUповтор первого полногоdevвидео.',interpretation='Обученные сопоставимые варианты подготовлены. Ошибка пар не подтверждает пользу в полном трекере.',decision='Интегрировать замороженные головы в первое назначение; нулевойконтроль/короткаяполнаяцена/полнаяHOTA безподборапоval.',cannot_claim='Это неHOTA и не доказательство пользы истории/ядраFlyVis. Последняя наблюдённая личность может отличаться от исходной после ошибки трекера; окно ограничено6наблюдениями/60кадрами.')
        for a in card['acceptance']:a.update(met=True,evidence='runs/continuation_20261028/history_factorial_training_audit/report.json')
    write(ROOT/'experiments/project_history.json',history)
    from scripts.b123_safe_journal_20261027 import journal
    journal('history_factorial_training_finished','Обучение сопоставимых представлений истории завершило свой этап',
        'Источник runs/continuation_20261028/history_factorial_training_audit/report.json; full_training='+str(r['complete'])+'. '+card['interpretation']+' '+card['decision'])
    for i,args in enumerate([['scripts/validate_project_history.py'],['-m','pytest','tests/test_project_history.py','-q']]):
        with (folder/f'check_{i}.log').open('w',encoding='utf-8') as out:code=subprocess.run([sys.executable,*args],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,timeout=300).returncode
        assert code==0


if __name__=='__main__':main()
