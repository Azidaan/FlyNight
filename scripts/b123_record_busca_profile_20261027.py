"""Записать технический итог, не подменяя им результат сопровождения."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
RUN=ROOT/'runs/b123_20261027';SOURCE='runs/b123_20261027/busca_runtime_profile/recovery1/report.json'
def main():
    report=json.loads((ROOT/SOURCE).read_text(encoding='utf-8'))
    end=json.loads((RUN/'busca_runtime_profile/recovery1/controller_result.json').read_text())
    assert report['complete'] and end['all_owned_ended'] and end['exit_code']==0
    assert report['weight_coverage']['all_state_keys_loaded_strictly']
    assert report['positional_equivalence']['full_positional_forward_exact']
    path=ROOT/'experiments/project_history.json';before=path.read_bytes();data=json.loads(before)
    card=next(x for x in data['experiments'] if x['id']=='busca_runtime_profile_v1');assert card['state']=='ready'
    card.update(state='infrastructure',done='2026-10-27',
        obtained='Строго загружены379ключей,изних322извлекателя; отдельный model_feats не нужен. '
            'Экономное позиционное кодирование побайтно совпало на125строках малой таблицы и полном позиционном проходе. '
            'Пакет4:0,01825с на объект,643471872байта GPU. Источник: '+SOURCE+' (weight_coverage,positional_equivalence,timings[2]).',
        interpretation='Авторская сеть совместима с текущим окружением. Это техническая выполнимость; качество восстановления и HOTA не измерены.',
        decision='Перейти к отдельно зарегистрированной выходной адаптации и стоимости первого полного видео; авторскую BUSCA не объявлять воспроизведённой.',
        cannot_claim='Нет HOTA или полного трекера. Профиль не доказывает пользу для BDD100K. Первоначальный отказ из-за менее пяти кандидатов сохранён; служебное дополнение повторило авторский механизм и уложилось в исходный срок03:34:51МСК.')
    for entry in card['acceptance']:entry.update(met=True,evidence=SOURCE)
    card['evidence'].append(dict(kind='file',ref=SOURCE,claim='Технический результат с ограничениями, не HOTA'))
    assert path.read_bytes()==before;path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    from scripts import z13_localization_labels_20261023 as J
    J.IDENT='busca_runtime_profile_v1'
    J.journal('BUSCA исполнилась с полными весами; подготовлено реальное вмешательство',
        '*Получено.* '+card['obtained']+'\n\n*Вывод.* '+card['interpretation']+'\n\n'
        '*Служебное восстановление.* В первом выбранном кадре было меньше пяти областей. '
        'Первая проверка остановилась после сверки весов и PE. Отдельная recovery1 добавила нулевые изображения и '
        'missing_candidate_bbox, как в исходном associate_embeddings. Старая ошибка сохранена, срок03:34:51не продлевался, '
        'код0 и конец дерева подтверждены. Ошибка декодирования русского текста в производном recovery-протоколе '
        'не меняет численные поля или исходный протокол; ограничения читать по исходному protocol.json и этой записи.\n\n'
        '*Решение.* '+card['decision']+' Подготовлен модуль tracking/busca_output_recovery.py: '
        'он проверяет только дополнительную выдачу и не меняет назначения или состояние XYWH. '
        'Память последних11уверенных реальных назначений, без камеры и фильтра плотности. '
        'Это собственная адаптация, не полное воспроизведение статьи. Две проверки строгого прошлого и отсутствия '
        'изменений трекера прошли. Отдельный полный протокол фиксируется до запуска. '
        'В процентах к baseline итог не выражается: измерена стоимость, HOTA нет.')
    with (RUN/'HANDOFF.md').open('a',encoding='utf-8') as f:
        f.write('\n\n## '+datetime.now(timezone(timedelta(hours=3))).strftime('%H:%M')+' МСК — реальный runtime BUSCA готов, выходная адаптация подготовлена\n\n'
            'Прочитать busca_runtime_profile/recovery1/report.json и controller_result.json. '
            'Строго379ключей/322ReID, отдельный feature checkpoint не нужен. '
            'PE хранит оси вместо огромной таблицы, точность проверена на125строках и полном позиционном forward. '
            'Первый отказ из-за<5кандидатов сохранён, recovery1 использовала штатные padding/missingbbox, '
            'исходный срок03:34:51сохранён, код0/all_owned_ended. '
            'GPUпакет4~0.01825с/объект,peak643471872байта; это НЕ HOTA. '
            'Подготовлены tracking/busca_output_recovery.py и scripts/b123_busca_output_pilot_20261027.py. '
            'Это собственная OUTPUT-ONLYадаптация: не менять назначения/состояние XYWH; последние11сильныхреальныхнаблюдений, '
            'безcamera/reliabilityгейта,пакет4,authorBUSCA5кандидатов+отказы. Не называть воспроизведением полной статьи. '
            'Два теста строгого прошлого/неизменного трекера passed. Полный протокол5400с, '
            'после первого полного видео2*стоимость*19+120должна помещаться в оставшийся бюджет идо07:20. '
            'Не повторять готовое первое видео, не подбирать пороги по нему. Далее LTR по исходному поручению. '
            'Граф только фундаментальный, файлы сохранять и скрывать фильтром; технических узлов не добавлять.\n')
    print('Technical result recorded; no HOTA claim')
if __name__=='__main__':main()
