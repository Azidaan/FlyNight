"""One preregistered CPU comparison on existing Z13 train predictions only."""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SRC = ROOT / 'runs/z13_spatial_v2_20261013'
OUT = ROOT / 'runs/z13_tree_quality_probe_20261015'
FEATURES = ['presence', 'peak', 'peak_margin', 'entropy', 'local_mass',
            'variance_x', 'variance_y', 'peak_mean_distance', 'shift_x',
            'shift_y', 'shift_norm', 'border_distance']

def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

def write(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')

def stamp():
    return datetime.now(timezone(timedelta(hours=3))).isoformat()

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def prepare():
    OUT.mkdir(exist_ok=True)
    assert not (OUT / 'protocol.json').exists(), 'Do not overwrite a protocol'
    files = [SRC / 'protocol.json', ROOT / 'tracking/z13_spatial.py', Path(__file__)]
    for arm in 'BCD':
        for name in ('fold0', 'fold1', 'full'):
            files.extend([SRC / arm / name / 'predictions.json', SRC / arm / name / 'identity.json'])
        files.append(SRC / arm / 'threshold_predictions.json')
    protocol = dict(id='z13_tree_quality_probe_v1', created=stamp(),
        scope='CPU train-only diagnostic; frozen localization; no GPU/dev/test/heldout/HOTA',
        authorization='Author 15.10: test useful idea and continue autonomously until 20:00 MSK',
        deadline='2026-10-15T20:00:00+03:00', features=FEATURES,
        fit='Existing out-of-localizer-training predictions on 104 train videos, known labels only',
        evaluation='Existing 26 train calibration videos, already used in development, not independent confirmation',
        target='kind=1 AND err<=0.25*scale AND err<err_pred; unknown excluded, never relabelled negative',
        models={'tree': dict(max_depth=4, min_samples_leaf=100, random_state=0),
                'histgb': dict(max_iter=200, max_leaf_nodes=15, learning_rate=.1,
                               min_samples_leaf=100, l2_regularization=1., early_stopping=False, random_state=0)},
        primary='histgb; shallow tree descriptive, no winner selection after results',
        controls='Saved neural quality scorer on exactly the same B/C/D rows; no retraining',
        acceptance=['D histgb minus D neural AP paired video bootstrap CI lower >0',
                    'D histgb minus B histgb AP paired video bootstrap CI lower >0',
                    'D histgb meets retained 90% useful corrections with >=50 labelled selections'],
        stop='If any criterion fails, no HOTA rerun for this tree replacement and no parameter sweep',
        bootstrap=dict(repetitions=2000, seed=20261015, unit='video'),
        caveats=['Calibration set previously examined; all intervals exploratory',
                 'Quality fit predictions use 52-video localizers, evaluation uses 104-video localizer',
                 'Quality scorer changes refusal only, cannot create a better position',
                 'Passing diagnostic is not proof of HOTA or FlyVis novelty'],
        source_sha256={str(p.relative_to(ROOT)):digest(p) for p in files})
    write(OUT / 'protocol.json', protocol)
    catalog_path = ROOT / 'experiments/project_history.json'
    catalog = read(catalog_path)
    assert not any(c['id']==protocol['id'] for c in catalog['experiments'])
    catalog['experiments'].append(dict(id=protocol['id'], title='З13: деревья вместо нейросетевой оценки качества положения',
        area='flyvis', state='planned', started='2026-10-15',
        question='Поможет ли деревообразное считывание качества отобрать полезные положения без изменения локализатора?',
        hypothesis='Нелинейные пороги по тем же 12 признакам могут улучшить отбор; это не добавляет отсутствующий сигнал цели.',
        expected='Одна CPU-проба HistGB и поясняющего малого дерева; B/C/D; прежние train104/26, без GPU и dev.',
        acceptance=[dict(criterion=x, met=None) for x in protocol['acceptance']],
        cannot_claim='Не проверяется HOTA, причинный узел всего провала, независимое обобщение или новизна FlyVis.',
        evidence=[dict(kind='json',ref=str((OUT/'protocol.json').relative_to(ROOT)),claim='Критерии, настройки и источники до расчёта')]))
    write(catalog_path,catalog)
    plan=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md'
    text=plan.read_text(encoding='utf-8')
    marker='## Журнал изменений\n\n'
    assert text.count(marker)==1
    entry=f"**{datetime.now():%d.%m.%Y, %H:%M} MSK** — постановка CPU-пробы деревьев качества З13\n\n*Постановка (`z13_tree_quality_probe_v1`).* Автор разрешил самостоятельно исследовать идею дерева и провести полезную пробу до 20:00 MSK. Меняется только оценка качества сохранённого положения: прежние 12 наблюдаемых признаков, HistGB как основной вариант и дерево глубины 4 для объяснения. B/C/D, обучение оценщика на прежних внеобучающих прогнозах train104, оценка на прежних train26; неизвестная идентичность исключается. Критерии и хеши до расчёта — `runs/z13_tree_quality_probe_20261015/protocol.json`. Оценка на уже использованных train26 диагностическая; прежние критерии и итоги HOTA не переписываются. Ни dev, ни отложенные видео, ни GPU не открываются. При непрохождении — без сетки настроек и без нового HOTA ради этого дерева.\n\n"
    plan.write_text(text.replace(marker,marker+entry,1),encoding='utf-8')
    print('Prepared protocol and card',flush=True)

def groups_for(job, provenance):
    groups=[]
    for video in job['predict']:
        p=ROOT/'runs/z13_data'/f'{video}.npz'
        provenance[str(p.relative_to(ROOT))]=digest(p)
        with np.load(p) as z:
            groups.extend([video]*(len(z['ep_offset'])-1))
    return np.asarray(groups)

def run():
    from sklearn.tree import DecisionTreeClassifier, export_text
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import average_precision_score, brier_score_loss
    from threadpoolctl import threadpool_limits
    from scripts.z13_train import success
    from scripts.z13_threshold_diagnostic import exact_curve
    protocol=read(OUT/'protocol.json')
    deadline=datetime.fromisoformat(protocol['deadline'])
    assert datetime.now(deadline.tzinfo)<deadline
    for p,h in protocol['source_sha256'].items():
        assert digest(ROOT/p)==h,p
    assert not (OUT/'report.json').exists()
    report=dict(started=stamp(), arms={}, diffs={}, data_sha256={}, scope=protocol['scope'])
    all_scores={}; all_y={}; common_keys=None; groups=None
    with threadpool_limits(limits=2):
        for arm in 'BCD':
            assert datetime.now(deadline.tzinfo)<deadline
            write(OUT/'status.json',dict(state='running',arm=arm,updated=stamp()))
            full=read(SRC/arm/'full/identity.json')['config']['job']
            train=[]
            for fold in ('fold0','fold1'):
                job=read(SRC/arm/fold/'identity.json')['config']['job']
                assert not set(job['fit'])&set(job['predict'])
                assert not (set(job['fit'])|set(job['predict']))&set(full['predict'])
                train.extend(read(SRC/arm/fold/'predictions.json'))
            train=[r for r in train if r['kind']>=0]
            rows=read(SRC/arm/'full/predictions.json')
            neural=read(SRC/arm/'threshold_predictions.json')
            key=lambda r:(r['label'],r['episode'],r['search_position'],r['kind'])
            assert [key(r) for r in rows]==[key(r) for r in neural]
            assert all(r['features']==q['features'] and r['center']==q['center'] for r,q in zip(rows,neural))
            if common_keys is None:
                common_keys=[key(r) for r in rows]
                ep_groups=groups_for(full,report['data_sha256'])
                groups=ep_groups[[r['episode'] for r in rows]]
            else: assert common_keys==[key(r) for r in rows]
            x=np.asarray([r['features'] for r in train]); y=np.array([success(r) for r in train],int)
            xx=np.asarray([r['features'] for r in rows]); yy=np.array([success(r) for r in rows],int)
            ok=np.array([r['kind']>=0 for r in rows]); all_y[arm]=yy[ok]
            assert x.shape[1]==xx.shape[1]==12 and np.isfinite(x).all() and np.isfinite(xx).all()
            ar=dict(fit_rows=len(y),fit_successes=int(y.sum()),evaluation_rows=len(rows),known=int(ok.sum()),unknown=int((~ok).sum()),models={})
            scores={'neural':np.array([r['p_present'] for r in neural])}
            for name,cls in [('tree',DecisionTreeClassifier),('histgb',HistGradientBoostingClassifier)]:
                t=time.perf_counter(); model=cls(**protocol['models'][name]).fit(x,y)
                scores[name]=model.predict_proba(xx)[:,1]
                ar[name+'_fit_predict_seconds']=time.perf_counter()-t
                import joblib
                joblib.dump(model,OUT/f'{arm}_{name}.joblib')
                if name=='tree':
                    (OUT/f'{arm}_tree.txt').write_text(export_text(model,feature_names=FEATURES),encoding='utf-8')
            for name,s in scores.items():
                ranked=[{**r,'p_present':float(v)} for r,v in zip(rows,s)]
                curve=exact_curve(ranked)
                write(OUT/f'{arm}_{name}_curve.json',curve)
                tau=float(np.sort(s)[::-1][int(np.ceil(.01*len(s)))-1]); selected=s>=tau
                ar['models'][name]=dict(AP=float(average_precision_score(yy[ok],s[ok])),
                    brier=float(brier_score_loss(yy[ok],s[ok])),
                    original_gate={k:v for k,v in curve.items() if k!='curve'},
                    top1=dict(tau=tau,all=int(selected.sum()),known=int((selected&ok).sum()),
                        unknown=int((selected&~ok).sum()),useful=int(yy[selected&ok].sum()),
                        precision=float(yy[selected&ok].mean()) if (selected&ok).any() else None))
                all_scores[arm+'_'+name]=s[ok]
            np.savez_compressed(OUT/f'{arm}_scores.npz',**scores,known=ok,useful=yy,video=groups)
            report['arms'][arm]=ar
            print(arm,json.dumps(ar['models']),flush=True)
        g=groups[ok]; vids=np.unique(g)
        indices={v:np.flatnonzero(g==v) for v in vids}
        rng=np.random.default_rng(protocol['bootstrap']['seed'])
        comparisons=[('D_histgb','D_neural'),('D_histgb','B_histgb'),('D_histgb','C_histgb')]
        samples={a+'-'+b:[] for a,b in comparisons}
        for _ in range(protocol['bootstrap']['repetitions']):
            ids=np.concatenate([indices[v] for v in rng.choice(vids,len(vids),replace=True)])
            for a,b in comparisons:
                ya,yb=all_y[a[0]][ids],all_y[b[0]][ids]
                if ya.sum()==0 or yb.sum()==0: continue
                samples[a+'-'+b].append(float(average_precision_score(ya,all_scores[a][ids])-average_precision_score(yb,all_scores[b][ids])))
        for a,b in comparisons:
            k=a+'-'+b
            report['diffs'][k]=dict(AP=float(average_precision_score(all_y[a[0]],all_scores[a])-average_precision_score(all_y[b[0]],all_scores[b])),
                ci95=np.quantile(samples[k],[.025,.975]).tolist(),replicates=len(samples[k]),videos=len(vids))
    report['criteria']=[report['diffs']['D_histgb-D_neural']['ci95'][0]>0,
        report['diffs']['D_histgb-B_histgb']['ci95'][0]>0,
        report['arms']['D']['models']['histgb']['original_gate']['tau'] is not None]
    report['passed']=all(report['criteria']); report['finished']=stamp()
    report['limitations']=protocol['caveats']
    for p,h in {**protocol['source_sha256'],**report['data_sha256']}.items(): assert digest(ROOT/p)==h
    write(OUT/'report.json',report)
    write(OUT/'status.json',dict(state='completed',updated=stamp(),passed=report['passed']))
    print('RESULT',json.dumps(dict(diffs=report['diffs'],criteria=report['criteria'])),flush=True)

if __name__=='__main__':
    if '--prepare' in sys.argv: prepare()
    else: run()
