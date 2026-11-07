"""Фиксированная перекрёстная проверка масштабирования сохранённого выхода."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib,json,sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/flyvis_tau_gain_loso_20261104'
FAST=ROOT/'runs/flyvis_tau_half_20261104'
STOCK=ROOT/'runs/flyvis_y1_latency_fp64_20261104'
sys.path.insert(0,str(ROOT))
from tracking.flyvis_latency_metrics import response_metrics

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,obj):(OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def register():
    OUT.mkdir(exist_ok=False)
    original=json.loads((FAST/'protocol.json').read_text(encoding='utf-8'))
    info=dict(id='flyvis_tau_gain_loso_v1',registered_msk=datetime.now(ZoneInfo('Europe/Moscow')).isoformat(),
      purpose='Предварительный контроль восстановления масштаба при ускорении ядра; без нового запуска сети.',
      source_protocol_sha256=sha(FAST/'protocol.json'),source_report_sha256=sha(FAST/'report.json'),
      baseline_report_sha256=sha(STOCK/'analysis.json'),script_sha256=sha(Path(__file__)),
      split='4-кратная перекрёстная проверка по исходным изображениям: 3 для одного общего коэффициента, четвёртое для оценки; все четыре ранее изучались и не являются независимым финальным тестом.',
      fitting='Для каждой версии отдельно неотрицательный скаляр без сдвига по МНК на векторах 31:79, все скорости, направления, события и оба контраста в трёх обучающих сценах; целевые глобальные перемещения известны из генератора. При неположительном или неопределённом коэффициенте — отказ.',
      controls='stock без и с калибровкой, fast без и с калибровкой, нулевой вектор; все используют одинаковые разбиения; ядро и считыватель не меняются.',
      metric='Предварительные средние по 16 случаям разворота каждого контраста: EPE усреднённых пространственных векторов за кадры31:79; среднее СКО временного шума неподвижного выхода кадров21:31; задержка исходным критерием с нормированием.',
      gate='Все коэффициенты положительны, все случаи идентифицируемы; fast calibrated против stock calibrated: средняя задержка меньше не менее чем на 1 кадр при контрасте1, EPE и неподвижный временной шум не хуже на обоих контрастах. Это предварительный, не финальный гейт.',
      caveat='Кадры искусственные и все четыре изображения прежде изучены; отбор метода уже исследовательский. Независимые данные, трекинг и HOTA не оценены. Метки используются только для подгонки коэффициентов, не как вход модели.',cases=original['cases'])
    write('protocol.json',info)
    catalog=ROOT/'experiments/project_history.json';data=json.loads(catalog.read_text(encoding='utf-8'))
    assert not any(c['id']==info['id'] for c in data['experiments'])
    data['experiments'].append(dict(id=info['id'],title='Предварительная перекрёстная проверка масштаба сигнала FlyVis',area='flyvis',state='planned',started='2026-11-04',tags=['Y1','motion-diagnostic'],
      question='Можно ли компенсировать снижение масштаба сигнала без потери преимущества в задержке и роста шума?',
      hypothesis='Отдельная линейная калибровка может уменьшить часть ошибки, но может усилить шум и потерять преимущество на новых сценах.',
      expected='Дешёвое парное сравнение по отдельным изображениям; результат не является финальным тестом.',
      acceptance=[dict(criterion=info['gate'],met=None)],done='',obtained='',interpretation='',decision='',cannot_claim='',metrics=[],
      evidence=[dict(kind='json',ref='runs/flyvis_tau_gain_loso_20261104/protocol.json',claim='Разбиения и критерий до вычисления')]))
    (OUT/'project_history_before.json').write_text(catalog.read_text(encoding='utf-8'),encoding='utf-8')
    catalog.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('registered')

def target(case):
    v=np.zeros((79,2),float);v[31:55,case['axis']]=case['speed']
    if case['event']=='reverse':v[55:,case['axis']]=-case['speed']
    return v

def study():
    p=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    assert p['script_sha256']==sha(Path(__file__)) and p['source_report_sha256']==sha(FAST/'report.json')
    previous=json.loads((FAST/'report.json').read_text(encoding='utf-8'))
    mask=np.load(FAST/'geometry.npz')['mask']; data=[]
    for c in p['cases']:
        signals={}
        for arm,folder in [('stock',STOCK),('fast',FAST)]:
            path=folder/'outputs'/(c['id']+'.npz');record=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
            assert sha(path)==record['sha256']
            signals[arm]=np.load(path)['flow'][0,1][:,mask].mean(axis=1)
        data.append(dict(case=c,signals=signals,truth=target(c)))
    gains={}
    for scene in range(4):
        train=[x for x in data if x['case']['scene']!=scene]; gains[str(scene)]={}
        for arm in ['stock','fast']:
            numerator=sum(float(np.sum(x['signals'][arm][31:79]*x['truth'][31:79])) for x in train)
            denominator=sum(float(np.sum(x['signals'][arm][31:79]**2)) for x in train)
            assert denominator>0 and numerator>0
            gains[str(scene)][arm]=numerator/denominator
    rows=[]
    for x in data:
        c=x['case'];ref=next(z for z in previous['rows'] if z['id']==c['id'] and z['mode']=='hold' and z['aggregation']=='last')
        results={}
        for arm in ['stock','fast']:
            for mode,k in [('raw',1.),('calibrated',gains[str(c['scene'])][arm])]:
                f=x['signals'][arm]*k; s=f[21:31]; truth=x['truth']
                met=response_metrics(f,c['axis'],c['speed'],c['event'])
                results[arm+'_'+mode]=dict(epe=float(np.linalg.norm(f[31:79]-truth[31:79],axis=1).mean()),
                    static_noise=float(np.sqrt(np.mean(np.sum((s-s.mean(axis=0))**2,axis=1)))),
                    static_output=float(np.linalg.norm(s,axis=1).mean()),latency=met['latency_frames'],eligible=met['eligible'])
                if mode=='raw':assert abs(results[arm+'_'+mode]['epe']-ref[arm]['epe_nominal_px'])<1e-10
            assert results[arm+'_raw']['latency']==results[arm+'_calibrated']['latency']
        results['zero']=dict(epe=float(np.linalg.norm(x['truth'][31:79],axis=1).mean()),static_noise=0.,static_output=0.,latency=None,eligible=False)
        rows.append(dict(case=c,arms=results))
    groups={}
    for event in ['reverse','stop']:
        for contrast in [1.,.25]:
            subset=[x for x in rows if x['case']['event']==event and x['case']['contrast']==contrast];assert len(subset)==16
            group={}
            for arm in ['stock_raw','stock_calibrated','fast_raw','fast_calibrated','zero']:
                group[arm]={key:float(np.mean([x['arms'][arm][key] for x in subset])) for key in ['epe','static_noise','static_output']}
                eligible=[x['arms'][arm]['latency'] for x in subset if x['arms'][arm]['eligible'] and x['arms'][arm]['latency'] is not None]
                group[arm].update(identifiable=len(eligible),latency_mean=float(np.mean(eligible)) if eligible else None)
            groups[event+'_'+str(contrast)]=group
    a,b=groups['reverse_1.0'],groups['reverse_0.25']
    checks=dict(complete_coverage=all(g[z]['identifiable']==16 for g in [a,b] for z in ['stock_calibrated','fast_calibrated']),
      latency_reduction=a['stock_calibrated']['latency_mean']-a['fast_calibrated']['latency_mean']>=1,
      normal_epe=a['fast_calibrated']['epe']<=a['stock_calibrated']['epe'],low_epe=b['fast_calibrated']['epe']<=b['stock_calibrated']['epe'],
      normal_noise=a['fast_calibrated']['static_noise']<=a['stock_calibrated']['static_noise'],low_noise=b['fast_calibrated']['static_noise']<=b['stock_calibrated']['static_noise'])
    write('report.json',dict(complete=True,preliminary_gate=all(checks.values()),checks=checks,gains=gains,groups=groups,rows=rows,protocol_sha256=sha(OUT/'protocol.json')))
    print(json.dumps(dict(checks=checks,gains=gains,groups=groups),ensure_ascii=False))

if __name__=='__main__':{'register':register,'study':study}[sys.argv[1]]()
