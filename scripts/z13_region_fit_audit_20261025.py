"""Short CPU audit of fit/calibration errors; no model update or evaluation tuning."""
from pathlib import Path
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']=os.environ['OPENBLAS_NUM_THREADS']='2'
import sys
import time
from datetime import timedelta,datetime
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
RUN=ROOT/'runs/z13_region_assignment_20261025/recovery2'
OUT=ROOT/'runs/z13_region_fit_audit_20261025'


def prepare():
    assert not OUT.exists()
    p=read(RUN/'protocol.json')
    names=[Path(__file__).relative_to(ROOT).as_posix(),'tracking/z13_filter_descriptor.py',
           'tracking/z13_region_assignment.py','scripts/z13_region_assignment_recovery2_20261025.py']
    start=now()
    write(OUT/'protocol.json',dict(at=start.isoformat(),budget_seconds=180,device='cpu',
        timezone='Europe/Moscow',save_at=(start+timedelta(seconds=180)).isoformat(),hard_stop=(start+timedelta(seconds=200)).isoformat(),
        stop_request_path=(OUT/'stop.request').relative_to(ROOT).as_posix(),
        question='Do the trained regional models identify beneficial actions even on their fit84 rows, or mostly prefer unchanged?',
        method='All12 frozen models; only fit84 and fit20. Existing calibrated policy, raw positive-logit decisions, and forced regional argmax on known queries with at least one beneficial action. Count good/harmful choices and average positive/negative cross-entropy separately. No new training or threshold selection.',
        interpretation='High forced-choice recovery with low positive-logit recovery supports investigating action/no-action supervision; low forced recovery leaves representation, model capacity and missing assignment context unresolved. Counts alone do not establish imbalance as a cause.',
        stop='180seconds wall budget checked between each model; preserve partial rows. No GPU, train26, dev or HOTA.',
        sha256={n:sha(ROOT/n) for n in names},seeds=p['seeds']))
    from scripts import z13_localization_labels_20261023 as j
    j.IDENT='z13_region_assignment_v1'
    j.journal('заранее записан короткий разбор обучения регионального узла',
        '*Проверка.* После завершения12моделей: CPU, до180с, без изменения весов и порогов. '
        'Только обучающие fit84 и калибровочные fit20: доля запросов с полезным доступным действием, '
        'восстановление таких действий с прежним выбором и при принудительном выборе региона, '
        'отдельная ошибка для запросов с полезным действием и без него. '
        'Высокое восстановление при принудительном выборе и низкое при обычном поддержит проверку '
        'обучения решения о вмешательстве; низкое восстановление не позволит свести проблему к одному порогу. '
        'Дисбаланс не объявляется доказанной причиной по одному числу. '
        'Источник постановки: runs/z13_region_fit_audit_20261025/protocol.json. '
        'Это разбор той же правки, не дополнительная оценка на train26 и не новый подбор гиперпараметров.')


def main():
    import torch
    from scripts import z13_region_assignment_recovery2_20261025 as source
    from tracking.z13_filter_descriptor import make_ranker
    torch.set_num_threads(2);torch.set_default_device('cpu')
    p=read(OUT/'protocol.json')
    assert all(sha(ROOT/n)==h for n,h in p['sha256'].items())
    controller=read(RUN/'training/controller_result.json')
    assert controller['exit_code']==0 and controller['all_owned_ended'] and not controller['remaining']
    assert read(RUN/'training/report.json')['complete']
    ns=source.namespace();began=time.monotonic();results={}
    for variant in ('D','C','R1','geometry'):
        arm='D' if variant=='geometry' else variant
        for split in ('fit84','fit20'):
            d=ns['load_split'](torch,arm,split)
            for seed in p['seeds']:
                if time.monotonic()-began>=p['budget_seconds'] or (OUT/'stop.request').exists() or now()>=datetime.fromisoformat(p['save_at']):
                    write(OUT/'report.json',dict(complete=False,results=results,reason='Original180s budget ended'))
                    write(OUT/'status.json',dict(state='partial',done=len(results),total=24,updated=now().isoformat()));return
                write(OUT/'status.json',dict(state='running',stage='Fit/calibration audit; no training',variant=f'{variant}/{split}/seed{seed}',done=len(results),total=24,updated=now().isoformat()))
                folder=RUN/'training'/variant/f'seed{seed}'
                done=read(folder/'done.json');m=read(folder/'report.json')
                assert sha(folder/'model.pt')==done['model_sha256'] and sha(folder/'report.json')==done['report_sha256']
                model=make_ranker(torch,activity=variant!='geometry').cpu()
                model.load_state_dict(torch.load(folder/'model.pt',map_location='cpu',weights_only=True))
                logits=ns['predict'](torch,model,d);known=d['kind'].numpy()>=0
                good=d['good'].numpy();harm=d['harm'].numpy();positive=known&good.any(1)
                forced_good=positive_logit_good=calibrated_good=0
                loss_positive=[];loss_negative=[]
                from tracking.z13_region_assignment import apply_policy
                for i,z in enumerate(logits):
                    if not known[i]:continue
                    z=np.asarray(z,float);k=int(z.argmax());den=float(np.logaddexp(0,np.logaddexp.reduce(z)))
                    if positive[i]:
                        numerator=float(np.logaddexp.reduce(z[good[i,:len(z)]]))
                        loss_positive.append(den-numerator)
                        forced_good+=int(good[i,k])
                        positive_logit_good+=int(z[k]>0 and good[i,k])
                        chosen,_=apply_policy(z,m['tau']);calibrated_good+=int(chosen>=0 and good[i,chosen])
                    else:loss_negative.append(den)
                results[f'{variant}/{split}/seed{seed}']=dict(rows=len(known),known=int(known.sum()),positive=int(positive.sum()),
                    forced_good=forced_good,positive_logit_good=positive_logit_good,calibrated_good=calibrated_good,
                    positive_loss_mean=float(np.mean(loss_positive)),negative_loss_mean=float(np.mean(loss_negative)),
                    tau=m['tau'],model_sha256=done['model_sha256'])
                write(OUT/'partial.json',dict(at=now().isoformat(),results=results))
    assert not torch.cuda.is_initialized()
    write(OUT/'report.json',dict(at=now().isoformat(),complete=True,seconds=time.monotonic()-began,results=results,
        cuda_initialized=False,source_protocol_sha256=sha(OUT/'protocol.json'),claim='Descriptive fit/calibration audit only; no causal diagnosis or new model gain.'))
    write(OUT/'status.json',dict(state='completed',done=24,total=24,updated=now().isoformat()))


if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    elif '--audit' in sys.argv or '--worker' in sys.argv:main()
    else:raise RuntimeError('Explicit preparation or audit required')
