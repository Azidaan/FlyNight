"""Audit the displayed comparison from primary tables and detector metadata."""
from pathlib import Path
import sys,json,csv,subprocess,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,sha
from scripts.z13_localization_labels_20261023 import now,journal


def main():
    import torch
    out=ROOT/'reports/stand_hota_comparability_20261024.json';assert not out.exists()
    legacy=read(ROOT/'runs/z13_shared_masa_hota_v1_20261015/full/eval/report.json');protocol=read(ROOT/'runs/z13_shared_masa_hota_v1_20261015/full/protocol.json')
    weights=ROOT/'weights/baseline_seed42/best.pt';ck=torch.load(weights,map_location='cpu',weights_only=False);args=ck['train_args'];assert Path(args['data']).resolve()==(ROOT/'training/configs/generated/full_step5.yaml').resolve()
    trainfile=ROOT/'training/configs/generated/full_step5_train.txt';trainids={Path(s.replace(chr(92),'/')).parent.name for s in trainfile.read_text(encoding='utf-8').splitlines() if s.strip()}
    videos=protocol['videos'];overlap=sorted(set(videos)&trainids);assert len(videos)==len(overlap)==35
    path=ROOT/'experiments/results/tracker_zoo_960_tau025.csv'
    with path.open(encoding='utf-8-sig',newline='') as f:old={r['трекер']:float(r['значение']) for r in csv.DictReader(f) if r['метрика']=='HOTA'}
    oldvideos=sorted(p.stem for p in (ROOT/'runs/tracker_zoo_960_tau025/bytetrack').glob('*.json'));assert len(oldvideos)==200 and not set(oldvideos)&trainids
    p=legacy['pooled'];d=legacy['diff']['D_masa-A_masa|all'];head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()
    paths=['weights/baseline_seed42/best.pt','training/configs/generated/full_step5.yaml','training/configs/generated/full_step5_train.txt','experiments/results/tracker_zoo_960_tau025.csv','runs/z13_shared_masa_hota_v1_20261015/full/eval/report.json','runs/z13_shared_masa_hota_v1_20261015/full/protocol.json','scripts/tracker_zoo.py','scripts/fly_lifecycle_alive.py']
    record=dict(at=now().isoformat(),commit=head,old_external=old,old_external_videos=oldvideos,old_external_detector_training_overlap=0,current_videos=videos,current_detector_training_overlap=overlap,detector_data=args['data'],detector_training_imgsz=args['imgsz'],same_video_A=p['A_masa|all']['HOTA'],same_video_D=p['D_masa|all']['HOTA'],same_video_A_without_masa=p['A_none|all']['HOTA'],paired_D_A=d,relative_D_A_percent=100*d['pooled_gain']/p['A_masa|all']['HOTA'],
      verdict='Numbers exist, but mixed protocols cannot rank methods. All35 current videos occur in the detector training list, whereas200historical val videos do not. This audit confirms membership and checkpoint training metadata; it does not estimate the magnitude or cause of the score gap.',
      scientific_text_review=dict(subject_vs_tool='FlyVis is subject;detector/tracker/MASA are tools of the evaluated system.',duplication='This is a correction of the displayed comparison; previous measured numbers remain unchanged.',actual_vs_planned='Only existing checkpoint metadata,200saved video filenames,35protocol IDs,CSV and primary HOTA report read; no new performance run.'),sources_sha256={n:sha(ROOT/n) for n in paths})
    write(out,record)
    cp=ROOT/'experiments/project_history.json';cat=read(cp)
    suffix=' Дополнение24.10: все35dev-видео есть в списке обучения YOLO26 full_step5, подтверждённом train_args.data самих весов. Это отложенные данные для трекерных модулей, но не полностью новые видео всей системы. Старые ByteTrack/BoT-SORT/TrackTrack измерены на200других val-видео без пересечения со списком обучения детектора; единый рейтинг с61,65/61,67 недопустим. Источник reports/stand_hota_comparability_20261024.json; это не оценка величины смещения.'
    changed=[]
    for card in cat['experiments']:
        if card['id'] in ('fly_spatial_hota_policy_z13_v1','z13_center_hota_v1'):
            oldtext=card.get('cannot_claim','');card.setdefault('revisions',[]).append(dict(date=f'{now():%Y-%m-%d}',commit=head,field='cannot_claim',old=oldtext,reason='Автор потребовал перепроверить несопоставимую таблицу; подтверждено пересечение с обучением детектора.'));card['cannot_claim']=oldtext+suffix;card.setdefault('evidence',[]).append(dict(kind='file',ref=out.relative_to(ROOT).as_posix(),claim='Разные протоколы и подтверждённое обучение детектора на35видео'));changed.append(card['id'])
    assert set(changed)=={'fly_spatial_hota_policy_z13_v1','z13_center_hota_v1'};write(cp,cat)
    text=f'''Числа прежней таблицы существуют в первичных файлах, однако таблица объединяла разные условия и не является рейтингом систем.

| Группа измерения | Видео | Детектор видел эти видео при обучении | Обработка | Что можно сравнивать |
|---|---:|---|---|---|
| Прежние ByteTrack / BoT-SORT / TrackTrack | 200 val | Нет пересечения со списком full_step5_train | Последовательность размеченных кадров; без общей сшивки MASA | Эти три метода между собой в том прежнем протоколе |
| Наш стенд / тот же стенд с D | 35 dev из исходного train BDD100K | Да,35из35; подтверждено метаданными весов | Исходные кадры около30Гц; одинаковая сшивка MASA | Добавку D относительно того же стенда |

Общая HOTA стенда {record['same_video_A']:.8f}, с прежней D {record['same_video_D']:.8f}; разность {d['pooled_gain']:+.8f} пункта, или {record['relative_D_A_percent']:+.6f}% относительно стенда. Средняя разность по35видео {d['mean_per_video']:+.8f},95%интервал [{d['ci95'][0]:+.8f};{d['ci95'][1]:+.8f}]; подтверждения положительной добавки D в этом опыте нет. Общая разность и средняя разность по видео — разные способы агрегации.

Высокая HOTA целого стенда не является измерением вклада FlyVis. Пересечение с обучением детектора ограничивает вывод о переносе всей системы; само по себе оно не доказывает, какая доля разрыва объясняется этим пересечением. Для ответа нужно новое сравнение на одинаковых видео,детекциях,частоте,оценщике и с явно одинаковой дополнительной обработкой.

Источник и полная проверка: reports/stand_hota_comparability_20261024.json. Предыдущие отчёты и числа сохранены. Доля измеренного потолка неизвестна.
'''
    (ROOT/'reports/stand_hota_comparability_20261024.md').write_text(text,encoding='utf-8')
    # Preserve the unfinished descriptor work while prioritizing the author's new request.
    cache=ROOT/'runs/z13_filter_descriptor_cache_20261023';end=read(cache/'recovery_gpu_2/controller_result.json');assert end['all_owned_ended'] and end['exit_code']==1
    chunks={p.relative_to(ROOT).as_posix():sha(p) for p in cache.glob('*/*/chunk_*.json')}
    write(cache/'remaining_recorded_20261024.json',dict(at=now().isoformat(),completed_chunks=chunks,total_groups=3720,parity_failure=read(cache/'parity_failure.json'),next='Investigate original-frame3 peak reproduction without relaxing tolerances; user requested fair tracker comparison first.',efficacy_not_computed=True,all_owned_ended=True))
    card=next(c for c in cat['experiments'] if c['id']=='z13_filter_descriptor_v1');oldtext=card.get('summary','');card.setdefault('revisions',[]).append(dict(date=f'{now():%Y-%m-%d}',commit=head,field='summary',old=oldtext,reason='Сохранён незавершённый технический этап перед новой задачей автора.'));card.update(state='inconclusive',summary='Эффективность не измерена: проверка воспроизведения вторичных пиков не прошла; остаток сохранён.',obtained='runs/z13_filter_descriptor_cache_20261023/parity_failure.json: D,fit84,эпизод7128; центр19,113170573px иscore0,250649105 за допуском,средний образец отличается лишь1,944e-6. runs/z13_filter_descriptor_cache_20261023/remaining_recorded_20261024.json содержит хеши готовых блоков и остаток. Обучение36моделей не запускалось.',interpretation='Это незавершённая техническая проверка, не отрицательный результат механизма. Причина расхождения пиков пока не установлена.',decision='Сохранить готовые блоки без повторения; допуски не ослаблять. По новому запросу автора приоритет — общее сравнение трекеров на одинаковых условиях.')
    write(cp,cat)
    journal('исправлена несопоставимая таблица HOTA и принят запрос общего сравнения','Подтверждено первичными CSV/JSON и train_args.data весов: ByteTrack45,7326/BoT-SORT54,9252/TrackTrack53,3390 относятся к200val; стенд61,6548/D61,6727 — к35другим dev,все35входят в обучение детектора. Разные видео,частота иMASA исключают единый рейтинг. Добавка прежнейD +0,017911 пункта (+0,029051%),по видео +0,008680[−0,029900;+0,056076],не подтверждена. Источник reports/stand_hota_comparability_20261024.json; ограничения двух карточек дополнены с revisions,старые числа не менялись. Новый запрос автора: «Хочу результаты на одинаковых условиях». Проверяется готовность общей проверки на7имеющихсяval-видео без обучения детектора на них; никаких75отложенных. Параллельно остаток описания цели сохранён после непройденного воспроизведения D/fit84/7128 (parity_failure.json),без оценки эффективности и ослабления допусков. Приоритет новой задачи автора. Сводка: HOTA заново не вычислялась,доля потолка неизвестна.')
    print(json.dumps(dict(overlap=len(overlap),external_videos=len(oldvideos),descriptor_completed_chunks=len(chunks)),ensure_ascii=False))


if __name__=='__main__':main()
