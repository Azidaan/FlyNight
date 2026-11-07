"""Новое прямое разрешение, первичные источники и критерии до этапов."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import copy,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.recovery_research_service_20261027 import RUN,read,write,sha
from scripts.b123_safe_journal_20261027 import journal
MSK=timezone(timedelta(hours=3))

def main():
    assert not RUN.exists();RUN.mkdir()
    (RUN/'ready_jobs').mkdir();(RUN/'jobs_done').mkdir()
    auth=dict(created=datetime.now(MSK).isoformat(),user_request='Занимаемся всем дальше',
        attachment='C:/Users/Егор/.codex/attachments/95248e41-1c7b-4770-83f5-97077db5bd3d/Вставленный текст.txt',
        budget_question='До какого времени по Москве продолжаем новые расчёты на одном GPU?',user_answer='28.10.2026 7:40',
        hard_stop='2026-10-28T07:40:00+03:00',save_at='2026-10-28T07:30:00+03:00',controller_stop='2026-10-28T07:39:30+03:00',single_gpu=True,
        scope='Обучаемый допуск продолжений и измерение резерва ошибок HOTA; затем проверка обоснованной замены с полным трекером. '
              'Сначала переиспользовать прежние проверенные результаты и не повторять готовое. Б1–Б3 сохраняют прежние ограничения. '
              'Новые наборы данных не загружать; основной базис не менять. Меньшие бюджеты этапов не продлевать. '
              'Строгое разделение обучения/разработки и исторических val, короткая цена до серии, независимая остановка и один GPU.',
        inherited_limits=['Не удалять .git/index.lock','Не менять основное окружение','Не удалять и не перемещать файлы графа','Сохранять дерево гипотез побайтно'])
    write(RUN/'authorization.json',auth);write(RUN/'registry.json',dict(processes=[]))
    selection=read(ROOT/'runs/train_cache/selection.json')['videos'];weights='weights/baseline_seed42/best.pt';weight_sha=sha(ROOT/weights)
    inventory=[]
    for split in ['train','dev']:
        for v in selection[split]:
            folder=ROOT/'runs/train_cache'/v
            names=[folder/'detections/raw_video.npz',folder/'detections/raw_video.json',folder/'alignment.json',ROOT/f'data/bdd100k/videos/train/{v}.mov',ROOT/f'data/bdd100k/labels/box_track_20/train/{v}.json']
            assert all(p.is_file() for p in names),v
            meta=read(names[1])['identity'];assert meta['weights_sha256']==weight_sha and meta['imgsz']==960 and meta['conf']==.01
            frames=[int(x['video_frame_index']) for x in read(names[2])['matches']]
            inventory.append(dict(video=v,split=split,cache=names[0].relative_to(ROOT).as_posix(),cache_size=names[0].stat().st_size,
                metadata_sha256=sha(names[1]),alignment_sha256=sha(names[2]),monotonic=all(b>a for a,b in zip(frames,frames[1:]))))
    assert len(set(selection['train']) & set(selection['dev']))==0
    assert all(x['monotonic'] for x in inventory),'Нужна отдельная проверка конфликтных привязок до публикации данных'
    write(RUN/'cache_inventory.json',dict(videos=inventory,train=len(selection['train']),dev=len(selection['dev']),detector_sha256=weight_sha,
        detector_seen_these_training_videos=True,not_detector_quality_validation=True,no_inference_run=True))
    old=read(ROOT/'runs/oracle_ladder/report.json');legacy={}
    for name,group in old['sets'].items():
        legacy[name]=dict(videos=group['videos'],hota={k:v['all']['HOTA'] for k,v in group['summary'].items()},
            paired_hota={k:v for k,v in group['bootstrap_vs_base'].items() if k.endswith('|all|HOTA')})
    write(RUN/'prior_error_budget_review.json',dict(source='runs/oracle_ladder/report.json',source_sha256=sha(ROOT/'runs/oracle_ladder/report.json'),
        implementation='scripts/oracle_detection_ladder.py',implementation_sha256=sha(ROOT/'scripts/oracle_detection_ladder.py'),results=legacy,
        correction='Утверждение об отсутствии прежнего измерения резерва было слишком широким. Лестница уже проверяла геометрию, порог, ложные рамки и пропуски. '
                   'Нужна актуализация сопоставимости, а не повтор старого опыта.',
        limitations=['test35 относится к train_cache и видео, виденным детектором: не доказательство качества детектора на новых видео',
                     'val7 — лишь7исторических видео; иной прежний MotionSourceByteTrack и другой базис',
                     'Исправления вводились в детекции на размеченных кадрах до состояния трекера; новый выходной анализ имеет другой смысл',
                     'Это вмешательства с разметкой, не достижимая прибавка, не глобальный максимум HOTA; эффекты не складываются']))
    catalog_path=ROOT/'experiments/project_history.json';before=catalog_path.read_bytes();catalog=json.loads(before)
    specs=[]
    sources=['scripts/recovery_research_service_20261027.py','tracking/recovery_admission_features.py','tracking/recovery_admission_oracle.py',
        'tracking/detection_supported_kalman.py','tracking/recovery_detection_support.py','tracking/xywh_bytetrack.py','tracking/offline_association.py',
        'scripts/run_bdd_video_rate_experiment.py','evaluation/hota_eval.py','runs/train_cache/selection.json',weights]
    config=read(ROOT/'runs/b123_20261027/recovery_motion_control/protocol.json')['configs']['xywh']
    data=dict(id='recovery_admission_data_v1',created=datetime.now(MSK).isoformat(),budget_seconds=7200,author_stop=auth['save_at'],device='cpu',
        videos={s:selection[s] for s in ['train','dev']},config=config,detector_sha256=weight_sha,feature_names_source='tracking/recovery_admission_features.py',
        action='Неизменный XYWH на сохранённых30Гцобнаружениях. Те же кандидаты Калман+поддержка текущим обнаружением. '
               'Сохранить причинные числовые признаки, текущую область и последнее реально назначенное изображение области в прошлом; JPEG95,128×384,BGR. '
               'Реальная история обновляется только после решения текущего кадра. Метки готовятся отдельно прежним PastIdentityAdmissionOracle: unknown=-1,known refuse=0,admit=1.',
        hypothesis='Внешний вид реальной прошлой области и текущего кандидата может добавить различимость к числовому контролю; это будущая проверка, а не установленная польза.',
        counterargument='Прежние числовые признаки после детектора уже часто не давали пользы. Поэтому их повтор — лишь контроль; новый проверяемый вход — изображения конкретного прошлого и текущего наблюдения. Метка допуска не равна глобальной дельте HOTA.',
        cost_gate='После первого полного видео:2*время*(165-1)+120с должны помещаться в остаток7200с; иначе остановить без продления. Первое видео не повторять.',
        acceptance=['Все130train/35dev раздельны; исторический val не используется для обучения или выбора порогов',
                    'Кандидаты и базис получены одним неизменным трекером; признаки не получают разметку/будущие кадры',
                    'unknown сохранён отдельно, изображения и метаданные имеют строго прошлое происхождение; охват и цена сохранены'],
        limitation='Подготовка данных, не обучение и не HOTA. YOLO видел эти train-видео; на них не оценивается обобщение детектора. '
                   'Отбор меток по известной прошлой личности неполон; неизвестные не являются отрицательными. Это параллельный модуль всего проекта, не вклад ядра FlyVis.',
        sha256={s:sha(ROOT/s) for s in sources+['scripts/recovery_admission_dataset_20261027.py']})
    write(RUN/'admission_dataset/protocol.json',data)
    first=read(ROOT/'runs/b123_20261027/detection_supported_recovery/report.json');second=read(ROOT/'runs/b123_20261027/supported_transfer/recovery2/report.json')
    groups=dict(development20=list(first['per_video']['baseline']),transfer20=list(second['per_video']['baseline']))
    assert not set(groups['development20']) & set(groups['transfer20'])
    rows=[]
    for group,report,folder in [('development20',first,ROOT/'runs/b123_20261027/recovery_motion_control/baseline'),('transfer20',second,ROOT/'runs/b123_20261027/supported_transfer/recovery2/baseline')]:
        for v in groups[group]:rows.append(dict(video=v,group=group,baseline=(folder/(v+'.json')).relative_to(ROOT).as_posix(),baseline_metrics=report['per_video']['baseline'][v]))
    reserve_sources=['scripts/hota_error_budget_20261027.py','tracking/recovery_detection_support.py','evaluation/hota_eval.py','scripts/z13_continue.py']+[x['baseline'] for x in rows]+[f'data/bdd100k/labels/box_track_20/val/{x["video"]}.json' for x in rows]
    reserve=dict(id='hota_error_budget_current_v1',created=datetime.now(MSK).isoformat(),budget_seconds=3600,author_stop=auth['save_at'],device='cpu',sources=rows,groups=groups,
        matching_iou=.5,arms=['identity','geometry','remove_unmatched','identity_geometry','identity_missing'],
        action='На том же сохранённом выходе отдельно изменить толькоID, только рамки, удалить несопоставленные рамки; проверить геометрию при исправленныхID и добавление отсутствующихGTпри исправленныхID. '
               'Сопоставление взаимно однозначное по классу/IoU>=0.5, crowd исключены. Базовые индивидуальные метрики переиспользуются без повторного детектора/трекера.',
        hypothesis='Запас геометрии, ложных рамок и личности может отличаться на актуальном выходе; знак и величина не заданы.',
        counterargument='Уже есть старая лестница с другим конвейером. Новая проверка нужна только для сравнения доступных вмешательств на текущих одинаковых выходах. '
                        'Сопоставление по0.5не оптимизирует HOTA на всех порогах; удаление unmatched может убирать частично верные рамки. Добавление GTпосле исправленияIDимеет условный смысл.',
        cost_gate='После первого полного видео:2*время*39+120с должны помещаться в остаток3600с; иначе сохранить остаток и не продлевать.',
        acceptance=['Все40видео, сохранённый базис и одинаковая разметка; отдельные результаты прежних двух20частей',
                    'Вариант личности меняет толькоID, геометрия толькоbox2d; уникальностьIDв кадре, первичные JSONи хеши',
                    'Парные95%интервалы HOTAпо видео и условные сравнения; не выдавать за реальную модель, максимум или сумму независимых вкладов'],
        limitation='Только диагностика на историческом val, не независимое итоговое испытание. Не глобальный максимум/верхняя границаHOTA, не доказательство обучаемости, '
                   'не разложение всех ошибок на непересекающиеся причины. Пропуски при условно исправленной личности не тождественны вкладу детектора; восстановление после перекрытия отдельно ещё требует протокола.',
        sha256={s:sha(ROOT/s) for s in reserve_sources})
    write(RUN/'error_budget/protocol.json',reserve)
    for protocol,title,script,order in [(reserve,'Резерв HOTA — геометрия, личность и пропуски на одинаковом выходе','scripts/hota_error_budget_20261027.py','010_error_budget'),
                                          (data,'Допуск продолжений — раздельные данные и реальные области во времени','scripts/recovery_admission_dataset_20261027.py','020_admission_dataset')]:
        assert not any(c['id']==protocol['id'] for c in catalog['experiments'])
        folder='error_budget' if protocol is reserve else 'admission_dataset'
        catalog['experiments'].append(dict(id=protocol['id'],title=title,area='flyvis',state='ready',started='2026-10-27',question=protocol['action'],hypothesis=protocol['hypothesis'],
            expected=protocol['action'],summary=protocol['counterargument'],obtained='',interpretation='',decision='',cannot_claim=protocol['limitation'],
            acceptance=[dict(criterion=x,met=None) for x in protocol['acceptance']],metrics=[],commits=[],files=[script],
            evidence=[dict(kind='file',ref=f'runs/recovery_research_20261027/{folder}/protocol.json',claim='Действие,контроли,критерии,стоимость до расчёта')]))
        specs.append(dict(name=order,folder=f'runs/recovery_research_20261027/{folder}',script=script,script_sha256=sha(ROOT/script),protocol_sha256=sha(RUN/folder/'protocol.json')))
    assert catalog_path.read_bytes()==before;write(catalog_path,catalog);write(RUN/'jobs_prepared.json',specs)
    journal('recovery_research_authorization_20261027','новое окно до28.10 07:40 и подготовка двух направлений',
        '*Решение автора.* «Занимаемся всем дальше»; ответ о сроке «28.10.2026 7:40». ОдинGPU,сохранение07:30/остановка07:39:30,новые данные не загружать. '
        'Разрешение сохранено в runs/recovery_research_20261027/authorization.json. Прежнее ночное окно закончено; старые источники и контроллеры не меняются.\n\n'
        '*Поправка объяснения.* Резерв уже измерялся прежней лестницей: первичный runs/oracle_ladder/report.json,код scripts/oracle_detection_ladder.py. '
        'В val7 база54,573090→геометрия59,410122; это диагностическое вмешательство в детекции до трекера,не достижимаяприбавка и не актуальный выходной опыт. '
        'Поэтому утверждение «резерв ещё не измеряли» слишком широкое; нужно проверить сопоставимость. Часть test35виделдетектор,использовать её как новое качество детектора нельзя. '
        'Сверка и границы: runs/recovery_research_20261027/prior_error_budget_review.json.\n\n'
        '*Подготовлено до запуска.* Карточки hota_error_budget_current_v1 и recovery_admission_data_v1,CPU3600/7200сс измеренной ценой первого полного видео. '
        'Числовой контроль дополняется реальными прошлой и текущей областями; неизвестная метка отдельная. '
        'Сначала обязательные проверки каталога,затем публикация готовых заданий в новую очередь с независимым контроллером. '
        'GPUобучение ещё не запускалось,Б1–Б3 не объявляются проверенными,граф новых технических узлов не получает.')
    print(json.dumps(dict(prepared=True,run=str(RUN),train=len(selection['train']),dev=len(selection['dev']),jobs=len(specs))))

if __name__=='__main__':main()
