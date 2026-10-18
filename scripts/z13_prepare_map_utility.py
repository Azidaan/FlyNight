"""Register the second utility implementation before its full-map extraction."""
from datetime import datetime,timedelta,timezone
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.z13_parallel_next import read,write,sha
RUN=ROOT/'runs/z13_map_utility_20261018'


def main():
    if RUN.exists():raise RuntimeError('Already prepared')
    now=datetime.now(timezone(timedelta(hours=3)));assert now.date().isoformat()=='2026-10-18'
    src=read(ROOT/'runs/z13_spatial_v2_20261013/D/full/identity.json')['config']['job']
    limited=read(ROOT/'runs/z13_parallel_next_20261018/utility/report.json');assert not limited['primary_pass']
    paths=['tracking/z13_response_utility.py','scripts/z13_map_utility.py','scripts/z13_map_utility_controller.py',
           'scripts/z13_prepare_map_utility.py','scripts/z13_record_map_utility.py','tests/test_z13_response_utility.py',
           'tracking/z13_utility.py','scripts/z13_target_center_probe.py','scripts/z13_target_model.py',
           'scripts/z13_train.py','tracking/z13_target_center.py','tracking/z13_readout.py',
           'runs/z13_spatial_v2_20261013/D/full/model.pt','runs/z13_spatial_v2_20261013/D/full/identity.json',
           'runs/z13_target_center_20261018/report.json','runs/z13_parallel_next_20261018/utility/report.json']
    paths += [f'runs/z13_target_center_20261018/{a}/center_rows.json' for a in ('D','C','R1')]
    paths += [f'runs/z13_parallel_next_20261018/utility/{a}_rows.json' for a in ('D','C','R1')]
    paths += [f'runs/z13_data/{v}.npz' for v in src['fit'][:20]+src['predict']]
    hashes=dict(read(ROOT/'runs/z13_target_center_20261018/protocol.json')['sha256'])
    hashes.update({p:sha(ROOT/p) for p in paths})
    stop=min(now+timedelta(hours=4),now.replace(hour=23,minute=50,second=0,microsecond=0))
    protocol=dict(id='z13_map_utility_v1',created_at=now.isoformat(),sha256=hashes,fit20=src['fit'][:20],videos=src['predict'],
       authorization='Author authorized justified long runs during18.10.2026 while away; bounded CPU branch parallel to single GPU.',
       save_stop_at=stop.isoformat(),hard_stop_at=(stop+timedelta(seconds=30)).isoformat(),
       action='Replace limited utility inputs by eight predetermined map statistics plus same observed geometry. Fixed initial D/C/R1 centers and labels.',
       features='peak_z, second separated peak_z, gap_z, normalized entropy, winner mass,7-neighbor mass, weighted x/y spread;6 previous geometry features. No feature selection or tuning.',
       estimator='Same StandardScaler + LogisticRegression C1 lbfgs max_iter500. Leave entire video out; only known outcomes in fitting. Unknowns tracked separately.',
       capture='Original frozen CPU evaluation, same normalization fit20 and lambdas; tap returned ridge scores without modifying them. Save each arm/video. Completed videos/arms reused.',
       reproduction='For every fixed source row: identical video/label/kind/useful; center difference<0.5px, peak difference<0.001. Use saved original centers/outcomes for final comparison.',
       primary='At1%: D map-utility minus original, geometry and limited utility all lower95%video CIs>0.2/5% secondary, not rescue. D/C/R1 map-head comparisons separate.',
       counterargument='Map shape may describe a sharp response to the wrong object; it is not target identity or a calibrated absence probability.',
       limitations='Second implementation after failed limited head. Reused train26 is exploratory. One seed, no risk guarantee for unknowns, no dev/HOTA/heldout75. No conclusion about all representations if it fails.',
       device='cpu',hota=False,dev=False,heldout75=False,chat_automation=False,
       literature=['https://arxiv.org/html/2003.12565','https://proceedings.mlr.press/v97/geifman19a.html'])
    RUN.mkdir();write(RUN/'protocol.json',protocol);write(RUN/'status.json',dict(state='ready',stage='registered before full-map extraction',done=0,total=3))
    path=ROOT/'experiments/project_history.json';catalog=read(path)
    catalog['experiments'].append(dict(id=protocol['id'],title='З13: оценка полезности по полной карте ответа',area='flyvis',state='ready',started=now.date().isoformat(),
       question='Добавляют ли неоднозначность и форма карты ответа пользу отбора сверх геометрии и высоты пика?',
       hypothesis='Восемь заранее заданных статистик карты могут отделять полезные центры от уверенных неверных; это гипотеза.',
       expected=protocol['primary'],acceptance=[dict(criterion=c,met=None) for c in (protocol['reproduction'],protocol['primary'])],
       tags=['Z13','fly-problem-P8'],runs=['runs/z13_map_utility_20261018'],files=['scripts/z13_map_utility.py','tracking/z13_response_utility.py'],
       evidence=[dict(kind='json',ref='runs/z13_map_utility_20261018/protocol.json',claim='Фиксированные признаки/контроли до извлечения'),
                 dict(kind='json',ref='runs/z13_parallel_next_20261018/utility/report.json',claim='Неуспех ограниченного оценщика, основание следующей реализации')]))
    path.write_text(json.dumps(catalog,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    from scripts.z13_record_next import journal
    journal('постановка z13_map_utility_v1','После ограниченного оценщика без доказанного преимущества над геометрией задана вторая реализация внутри считывателя FlyVis: восемь заранее записанных статистик полной карты ответа и та же геометрия. Данные, центры, ядра D/C/R1 и λ прежние; исходный вывод воспроизводится на каждой строке. Обучение логистического выхода с исключением целого видео, без перебора признаков и параметров. Основной отбор1%,2/5% вторичные. Четыре часа CPU плюс30с завершения в пределах18.10.2026, параллельно единственному GPU-прогону. Каждый вариант/видео сохраняется отдельно; dev,HOTA,75отложенных и новые загрузки не используются. Источник условий — runs/z13_map_utility_20261018/protocol.json. Полная карта не является вероятностью присутствия; неудача не диагностирует всё представление.')
    print('Map utility protocol frozen; hashes:',len(hashes))


if __name__=='__main__':main()
