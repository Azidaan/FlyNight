"""Посмертный разбор уже сохранённого сравнения; новых прогонов FlyVis нет."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NEW = ROOT / 'runs/flyvis_tau_half_20261104'
OLD = ROOT / 'runs/flyvis_y1_latency_fp64_20261104'
OUT = ROOT / 'runs/flyvis_tau_half_components_20261104'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(flow, axis, speed, event, start, end):
    q = flow[start:end]
    target = np.zeros_like(q)
    if start < 55:
        target[:, axis] = speed
    elif event == 'reverse':
        target[:, axis] = -speed
    diff = q-target
    square_axis = float(np.mean(diff[:, axis]**2))
    square_cross = float(np.mean(diff[:, 1-axis]**2))
    squared = float(np.mean(np.sum(diff**2, axis=1)))
    assert abs(square_axis+square_cross-squared) < 1e-11
    result = dict(epe=float(np.linalg.norm(diff,axis=1).mean()),
                  axis_abs_error=float(np.abs(diff[:,axis]).mean()),
                  cross_abs_error=float(np.abs(diff[:,1-axis]).mean()),
                  axis_squared_error=square_axis, cross_squared_error=square_cross,
                  squared_error=squared, axis_signed_output=float(q[:,axis].mean()),
                  output_magnitude=float(np.linalg.norm(q,axis=1).mean()))
    if event == 'reverse' or start < 55:
        result['wrong_axis_sign_fraction'] = float((q[:,axis]*target[:,axis] < 0).mean())
    else:
        baseline = flow[21:31].mean(axis=0)
        result['baseline_adjusted_abs_axis'] = float(np.abs(q[:,axis]-baseline[axis]).mean())
        result['baseline_axis'] = float(baseline[axis])
    return result


def main():
    OUT.mkdir(exist_ok=False)
    protocol=json.loads((NEW/'protocol.json').read_text(encoding='utf-8'))
    previous=json.loads((NEW/'report.json').read_text(encoding='utf-8'))
    assert previous['complete'] and previous['passed']
    mask=np.load(NEW/'geometry.npz')['mask']
    rows=[]
    for case in protocol['cases']:
        paths={arm: folder/'outputs'/(case['id']+'.npz') for arm,folder in [('stock',OLD),('fast',NEW)]}
        marks={arm:json.loads(path.with_suffix('.json').read_text(encoding='utf-8')) for arm,path in paths.items()}
        assert all(sha(path)==marks[arm]['sha256'] for arm,path in paths.items())
        assert marks['stock']['input_sha256']==marks['fast']['input_sha256']
        flow={arm:np.load(path)['flow'][0,1][:,mask].mean(axis=1) for arm,path in paths.items()}
        phases={'before':(31,55),'after_first5':(55,60),'after_later19':(60,79),'after_all':(55,79)}
        phases={name:{arm:metrics(value,case['axis'],case['speed'],case['event'],*window)
                      for arm,value in flow.items()} for name,window in phases.items()}
        row=dict(case=case,phases=phases,hashes={arm:marks[arm]['sha256'] for arm in paths})
        rows.append(row)
        oldrow=next(x for x in previous['rows'] if x['id']==case['id'] and x['mode']=='hold' and x['aggregation']=='last')
        for arm in paths:
            combined=(24*phases['before'][arm]['epe']+24*phases['after_all'][arm]['epe'])/48
            assert abs(combined-oldrow[arm]['epe_nominal_px'])<1e-10
    groups={}
    for event in ['reverse','stop']:
        for contrast in [1.,.25]:
            subset=[r for r in rows if r['case']['event']==event and r['case']['contrast']==contrast]
            assert len(subset)==16
            phases={}
            for phase in subset[0]['phases']:
                phases[phase]={}
                for arm in ['stock','fast']:
                    keys=subset[0]['phases'][phase][arm]
                    phases[phase][arm]={key:float(np.mean([r['phases'][phase][arm][key] for r in subset])) for key in keys}
            groups[f'{event}_{contrast}']=phases
    report=dict(complete=True, type='exploratory_post_hoc', cases=len(rows), groups=groups, rows=rows,
      verified_original_epe=True, protocol_sha256=sha(NEW/'protocol.json'), earlier_report_sha256=sha(NEW/'report.json'),
      metric_scope='Средний по пространству вектор; анализ временных окон после выбора по имеющимся результатам; '
        'ошибка без калибровки амплитуды, сравнение на тех же 4 изображениях и одной модели. '
        'Квадраты ошибок по двум осям складываются точно; средние абсолютные ошибки не разлагают EPE на сумму. '
        'Сравнение не устанавливает причину на уровне отдельных нейронов, не оценивает HOTA и не является независимым тестом.')
    (OUT/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(groups,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
