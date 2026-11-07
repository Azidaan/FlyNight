"""Зарегистрировать контроль дешёвой калибровки после малого пиксельного эффекта."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.recovery_research_service_20261027 import RUN,read,write,sha,now
from scripts.b123_safe_journal_20261027 import journal

def main():
    assert read(RUN/'pixel_box_validation/independent_audit.json')['passed'];folder=RUN/'pixel_box_numeric_control';assert not folder.exists()
    old=read(RUN/'pixel_box_validation/protocol.json');files=['scripts/pixel_box_numeric_control_20261027.py','tracking/pixel_box_refinement.py','runs/recovery_research_20261027/pixel_box_validation/report.json','runs/recovery_research_20261027/pixel_box_validation/independent_audit.json']
    p=dict(id='pixel_box_numeric_control_v1',created=now(),budget_seconds=1200,device='cpu',author_stop=read(RUN/'authorization.json')['save_at'],sources=old['sources'],groups=old['groups'],
        action='Сравнить уже зафиксированную пиксельную поправку с дешёвой калибровкой без изображений на тех же рамках. '
               'Первый кандидат — медианная поправка по классу, второй — линейная регрессия по тем же12числовымпризнакам с L2=1. '
               'Обучение только известныеtrain, unknownнеnegative;нормировкитолькоtrain. Выбор метода и alpha[0,.5,1] только35devHOTA,приравенствеменьшаясила. '
               'Потом40val,пиксельныемодели/выходы не меняются. Толькоbox2d;ID/класс/балл/число сохраняются.',
        hypothesis='Небольшой геометрический эффект может частично объясняться систематической калибровкой рамок, а не пиксельной информацией. Это контроль, а не новая заявка на большую прибавку.',
        counterargument='Линейный контроль имеет другую ёмкость и функцию ошибки; отсутствие преимущества над ним не доказывает бесполезность изображений. '
                        'Эта дополнительная проверка поставлена ПОСЛЕ наблюдения итоговой пиксельной HOTA;историческийvalне независимаяпроверка новой гипотезы.',
        acceptance=['Контроль без пикселей/GTприинференсе,фиксируетсянаtrain/devдоегоval,веса/пороги прежнейCNNне меняются',
                    'Все40,точныйбазис/тежеобласти,толькогеометрия,полныеHOTA/AssA/IDF1/FP/FN/IDSwи95%парныеинтервалы',
                    'Дляотдельнойпользыпиксельнойверсиинижняяграница к числовому контролю положительна во всех3сидах;частные сравнения не скрывают провал этого критерия'],
        cost_gate='Первое ПОЛНОЕdev-видео со всеми6калибровками:2*time*(34+40)+120<остаток1200с,иначестопбезпродления.',
        limitation='Последующая поисковая проверка на тех же историческихval;не новыйнезависимыйтест,неизолированнаяархитектурнаяабляция. '
                   'Это контроль систематической калибровки;ядро/модульFlyVisне используется. Основнойbaselineне меняется.',
        sha256={f:sha(ROOT/f) for f in files})
    # Замораживаем все используемые не-графические источники, включая обучающие цели.
    bound=[]
    for split in ['train','dev']:
        for item in read(RUN/'pixel_box_dataset/report.json')['videos']:
            if item['split']==split:bound.append(f'runs/recovery_research_20261027/pixel_box_dataset/{split}/{item["video"]}/targets.npz')
    for src in old['sources']:bound += [src['baseline'],f'runs/recovery_research_20261027/pixel_box_validation/offsets/{src["video"]}_records.json']
    p['sha256'].update({f:sha(ROOT/f) for f in bound});write(folder/'protocol.json',p)
    path=ROOT/'experiments/project_history.json';before=path.read_bytes();cat=json.loads(before);assert not any(c['id']==p['id'] for c in cat['experiments'])
    cat['experiments'].append(dict(id=p['id'],title='Отделить пользу изображения от систематической калибровки рамок',area='flyvis',state='ready',started='2026-10-27',
        question='Добавляют ли пиксельные поправки пользу сверх дешёвой числовой калибровки тех же рамок?',hypothesis=p['hypothesis'],expected=p['action'],summary=p['counterargument'],
        obtained='',interpretation='',decision='',cannot_claim=p['limitation'],acceptance=[dict(criterion=c,met=None) for c in p['acceptance']],metrics=[],commits=[],files=files[:2],
        evidence=[dict(kind='file',ref='runs/recovery_research_20261027/pixel_box_numeric_control/protocol.json',claim='Методы, выбор толькоdev и критерии до обучения калибровки')]))
    assert path.read_bytes()==before;write(path,cat)
    write(RUN/'pixel_box_numeric_job_prepared.json',dict(name='120_pixel_box_numeric_control',folder=folder.relative_to(ROOT).as_posix(),script=files[0],script_sha256=sha(ROOT/files[0]),protocol_sha256=sha(folder/'protocol.json')))
    journal(p['id'],'поставлен дешёвый контроль систематической калибровки рамок',
        '*Постановка.* '+p['action']+'\n\n*Возражение и ограничения.* '+p['counterargument']+' '+p['limitation']+
        '\n\n*Приёмка.* '+p['acceptance'][2]+' '+p['cost_gate']+' Источник pixel_box_numeric_control/protocol.json. '
        'Это одна ограниченная проверка объяснения уже полученной малой прибавки,не новая серия тюнингаCNN. Затем основнойузелсопоставления. Публикацияпослезелёныхпроверок.')
    print('Prepared cheap calibration control before its train/dev/val evaluation')

if __name__=='__main__':main()
