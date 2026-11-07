"""Зарегистрировать новое действие с пикселями до выгрузки и обучения."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.recovery_research_service_20261027 import RUN,read,write,sha,now
from scripts.b123_safe_journal_20261027 import journal

def main():
    folder=RUN/'pixel_box_dataset';assert not folder.exists()
    assert read(RUN/'learned_validation/independent_audit.json')['passed']
    files=['scripts/pixel_box_dataset_20261027.py','tracking/pixel_box_refinement.py','training/configs/generated/full_step5_train.txt','runs/train_cache/selection.json']
    selection=read(ROOT/files[-1])['videos'];videos={s:selection[s] for s in ['train','dev']}
    for s,vs in videos.items():
        for v in vs:files += [f'runs/recovery_research_20261027/admission_dataset/{s}/{v}/baseline.json',f'runs/train_cache/{v}/alignment.json',f'data/bdd100k/labels/box_track_20/train/{v}.json']
    p=dict(id='pixel_box_refinement_v1',created=now(),budget_seconds=5400,author_stop=read(RUN/'authorization.json')['save_at'],device='cpu',videos=videos,
        size=96,context=1.5,query_stride=10,query_offset=2,detector_training_list='training/configs/generated/full_step5_train.txt',
        action='Обучить ограниченную поправку четырёх координат фактической рамки XYWHByteTrack по пикселям. '
               'Контроль получает текущую область дважды, временной вариант — прошлую и текущую области при одинаковой сети. '
               'Идентификаторы, классы, баллы, количество рамок, назначения и состояние исходного трекера не меняются. '
               'Новая информация — реальные пиксели и локальная структура области, а не сглаживание или выбор готовой детекторной рамки.',
        hypothesis='Изображение может позволить исправлять часть ошибок локализации; предыдущее изображение может добавить пользу. Обе возможности не установлены.',
        counterargument='Большой GT-резерв не доказывает обучаемость: разметка неоднозначна, границы могут быть невидимы. '
                        'Малые области 96×96 и обучение с нуля могут терять точность. Прошлое назначение трекера не гарантирует ту же реальную личность. '
                        'Прежние сглаживание и выбор детекторной рамки уже проверялись; их не повторяем под новым именем.',
        papers=[dict(title='Acquisition of Localization Confidence for Accurate Object Detection',url='https://arxiv.org/abs/1807.11590',role='Обоснование обучения локализации; наша реализация не IoU-Net'),
                dict(title='Cascade R-CNN: Delving into High Quality Object Detection',url='https://arxiv.org/abs/1712.00726',role='Учить на фактическом распределении выходных рамок; не воспроизведение каскада'),
                dict(title='Bounding Box Regression with Uncertainty for Accurate Object Detection',url='https://arxiv.org/abs/1809.08545',role='Возражение: неоднозначность границ разметки')],
        labels='Class-wise Hungarian IoU>=0.5, noncrowd. Только совпавшие рамки в регрессионной ошибке, unknown не zero/negative. Нормированные поправки углов ограничены ±0.25 размера рамки.',
        training_plan=dict(arms=['current','temporal'],seeds=[42,43,44],epochs=15,batch_size=128,lr=.001,weight_decay=.0001,loss='SmoothL1 beta0.05, known only',
                           checkpoint='minimum dev regression loss',alpha_grid=[0.,.5,1.],alpha_selection='maximum mean per-video dev HOTA, tie prefers0 then smaller',budget_seconds=7200),
        acceptance=['Раздельные130train/35dev, точные обучающие кадры YOLO исключены из потерь уточнения; признаки не используют GT или будущее',
                    'Одинаковая архитектура/параметры/сиды; нулевая поправка/выключение точно сохраняют исходную рамку; фиксированные веса и alpha до val',
                    'Все40историческихval, полныйHOTA/AssA/IDF1/FP/FN/IDSw; положительная нижняяграница95%парногоинтервала к базе во всех3сидах; добавка времени отдельно к current во всех3'],
        cost_gate='Первое полное видео:2*time*164+120с<остаток5400с; затем первые100реальных обучающих шагов, удвоенная стоимость оставшихся6×15эпох+оценка должна поместиться в7200с. Меньшие бюджеты не продлевать.',
        limitation='Собственная поправка, не авторские IoU-Net/Cascade и не Б1–Б3. Только изображения уже имеющихся видео, без новых данных/весов. '
                   'Детектор видел train-видео: это подготовка/подбор поправки, не проверка детектора. Итог толькоval; все40val имеют историю проекта. '
                   'GT-резерв не максимум и не обещание прибавки. Ядро/модуль FlyVis не используются. История по запросам, не непрерывная временная сеть30Гц.',
        sha256={f:sha(ROOT/f) for f in set(files)})
    write(folder/'protocol.json',p)
    path=ROOT/'experiments/project_history.json';before=path.read_bytes();cat=json.loads(before);assert not any(c['id']==p['id'] for c in cat['experiments'])
    cat['experiments'].append(dict(id=p['id'],title='Уточнить рамки по изображению — текущая область и её прошлое наблюдение',area='flyvis',state='ready',started='2026-10-27',
        question='Можно ли вернуть часть потерь локализации по реальным пикселям, и помогает ли прошлое изображение?',hypothesis=p['hypothesis'],expected=p['action'],summary=p['counterargument'],
        obtained='',interpretation='',decision='',cannot_claim=p['limitation'],acceptance=[dict(criterion=c,met=None) for c in p['acceptance']],metrics=[],commits=[],files=files[:2],
        evidence=[dict(kind='file',ref='runs/recovery_research_20261027/pixel_box_dataset/protocol.json',claim='Источники, контролируемая поправка, три сида и критерии до обучения')]))
    assert path.read_bytes()==before;write(path,cat)
    write(RUN/'pixel_box_dataset_job_prepared.json',dict(name='080_pixel_box_dataset',folder=folder.relative_to(ROOT).as_posix(),script=files[0],script_sha256=sha(ROOT/files[0]),protocol_sha256=sha(folder/'protocol.json')))
    journal(p['id'],'поставлена ограниченная поправка рамок по пикселям',
        '*Постановка.* '+p['action']+'\n\n*Возражение.* '+p['counterargument']+'\n\n*Проверка.* '+p['labels']+' '+p['cost_gate']+
        ' Все параметры обучения, выборов dev и полной HOTA зафиксированы до запуска в runs/recovery_research_20261027/pixel_box_dataset/protocol.json. '
        'Основание: отдельные вмешательства error_budget/recovery1/report.json, а не обещание забрать весь резерв. '
        'Первичные работы https://arxiv.org/abs/1807.11590, https://arxiv.org/abs/1712.00726, https://arxiv.org/abs/1809.08545 прочитаны; числа статей на наш трекер не переносятся. '
        'Google Scholar показал ограничение доступа, прежний Chrome-профиль недоступен; первоисточники доступны через arXiv. '
        'Код и первичный runs/gate_box/report.json сверены: прежний выбор детекторной рамки и сглаживание отличаются от новой поправки по пикселям. '
        'Карточка ready, публикация очереди только после обязательных проверок. Один GPU, сохранение28.10 07:30, остановка07:39:30/07:40МСК.')
    print('Prepared pixel-conditioned geometry experiment; not a result')

if __name__=='__main__':main()
