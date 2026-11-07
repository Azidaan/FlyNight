"""Независимая проверка окон и компонент по сохранённым векторам."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/flyvis_tau_half_components_20261104'
NEW=ROOT/'runs/flyvis_tau_half_20261104'
OLD=ROOT/'runs/flyvis_y1_latency_fp64_20261104'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    report=json.loads((OUT/'report.json').read_text(encoding='utf-8'))
    previous=json.loads((NEW/'report.json').read_text(encoding='utf-8'))
    mask=np.load(OLD/'geometry.npz')['mask']
    biggest=0.
    for row in report['rows']:
        case=row['case'];axis=case['axis'];other=1-axis
        for arm,base in [('stock',OLD),('fast',NEW)]:
            source=base/'outputs'/(case['id']+'.npz')
            assert sha(source)==row['hashes'][arm]
            # Прямое суммирование по выбранным точкам вместо mean основного скрипта.
            vectors=np.load(source)['flow'][0,1]
            trace=vectors[:,mask,:].sum(axis=1)/int(mask.sum())
            for phase,(start,end) in {'before':(31,55),'after_first5':(55,60),
                                      'after_later19':(60,79),'after_all':(55,79)}.items():
                expected=case['speed'] if start<55 else (-case['speed'] if case['event']=='reverse' else 0)
                residual=trace[start:end,axis]-expected
                crossing=trace[start:end,other]
                independent={'axis_abs_error':np.abs(residual).sum()/(end-start),
                             'cross_abs_error':np.abs(crossing).sum()/(end-start),
                             'axis_squared_error':np.dot(residual,residual)/(end-start),
                             'cross_squared_error':np.dot(crossing,crossing)/(end-start),
                             'epe':np.sqrt(residual**2+crossing**2).sum()/(end-start)}
                for key,value in independent.items():
                    difference=abs(float(value)-row['phases'][phase][arm][key]);biggest=max(biggest,difference)
                    assert difference < 1e-10,(case['id'],phase,arm,key,difference)
    plateau={}
    for ct in (1.,.25):
        vals=[x for x in previous['rows'] if x['event']=='reverse' and x['contrast']==ct and x['mode']=='hold' and x['aggregation']=='last']
        assert len(vals)==16
        plateau[str(ct)]={arm:float(np.mean([x[arm]['plateau_over_speed'] for x in vals])) for arm in ['stock','fast']}
    audit=dict(complete=True,passed=True,cases=len(report['rows']),both_models=True,all_four_windows=True,
               max_difference=biggest,plateau_over_speed=plateau,report_sha256=sha(OUT/'report.json'),
               interpretation='Проверка арифметики и исходных хешей, не независимый новый прогон модели.')
    (OUT/'independent_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(audit,ensure_ascii=False))


if __name__=='__main__':main()
