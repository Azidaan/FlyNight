"""Append verified result; prior records and unrelated changes are preserved."""
from common import *

def main():
    p = verify()
    audit = read(OUT/'audit.json')
    s = read(OUT/'summary.json')
    pred = read(OUT/'predictions.json')
    assert audit['passed'] and audit['primary_joint_gate'] == s['primary_joint_gate']
    assert not (OUT/'recording.json').exists()
    names = dict(C34='FlyVis, 34 типа клеток', R134_spatial='Обычные пространственные признаки', R134_temporal='Обычные временные признаки')
    def number(x):
        if 0 < abs(x) < 1e-5:
            return f'{x:.3e}'.replace('.', ',')
        return f'{x:.6f}'.replace('.', ',')
    def counts(m):
        return str(m['own_accept'])+' своих / '+str(m['foreign_accept'])+' чужих'
    m = s['metrics']['C34__agreement']
    base = s['metrics']['C34__common_baseline']
    delta = (m['balanced_accuracy']/base['balanced_accuracy']-1)*100
    obtained = 'На 26 связях C34: '+counts(base)+' → '+counts(m)+'. На пяти рабочих случаях: '+counts(s['screen_metrics']['C34__agreement'])+'. Источник `research_agreement_readout_20261112/summary.json`: metrics, screen_metrics, cv_gates, screen_gates; покадровые оценки: predictions.json.'
    decision = 'Остановить эту фиксированную реализацию после двух допусков; пороги и штраф не подбирать, полный трекер не запускать.' if not s['primary_joint_gate'] else 'Разрешённый допуск завершён; полный трекер в этой задаче не запускать.'
    interpretation = 'Согласие значений не прошло совокупный критерий продолжения.' if not s['primary_joint_gate'] else 'Согласие значений прошло локальный разработческий допуск; сохранение публичного ID ещё не измерено.'
    interpretation += ' Проверен внешний считыватель полного модуля, ядро FlyVis не менялось. Результат относится к этому коэффициенту при прежнем сдвиге и фиксированном обучении.'
    cannot = 'Повторно используемые разработческие 26 связей и пять заранее известных случаев не являются независимым тестом. Не измерены назначение рамок, жизненный цикл публичного ID и HOTA (совместное качество обнаружения и связывания траекторий). Не доказана причина всех прежних ошибок и бесполезность средних/амплитуд вообще.'
    text = '1. Итог\n\n'+obtained+'\n\n'+interpretation+' '+decision+'\n\n'
    text += '2. Что проверено\n\nИспользованы только готовые карты. Вместо поканальной корреляции на прежнем выбранном сдвиге вычислено согласие: удвоенная ковариация, делённая на сумму двух дисперсий и квадрата разности средних. Моменты делятся на число ячеек. Отдельный контроль исключает только разность средних. Это изменение считывания, ядро и его веса не запускались и не менялись. Основание: коэффициент согласия Lin (1989), DOI 10.2307/2532051; перенос на принадлежность объектов является нашей проверяемой гипотезой. Подробное основание сохранено в `research_agreement_readout_20261109/ОСНОВАНИЕ.md`, математические проверки — в `math_checks.json` той же папки.\n\n'
    text += 'Нормировка вычислена только по обучающим видео каждой проверки. Штраф 1, порог 0, семь геометрических и 34 зрительных признака, всего 42 обучаемых параметра. Проверено исключение целого видео из обучения; для пяти случаев отдельно обучены заранее предусмотренные окончательные слои на всех 26 связях. Всего 120 проверочных и шесть окончательных обучений, источник `predictions.json`: fits. Подбора настроек нет.\n\n'
    text += '3. Сравнения на 26 связях\n\n'
    for arm in ARMS:
        b = s['metrics'][arm+'__common_baseline']
        full = s['metrics'][arm+'__agreement']
        var = s['metrics'][arm+'__variance_only']
        text += names[arm]+': прежняя общая нормировка — '+counts(b)+'; полное согласие — '+counts(full)+'; без разности средних — '+counts(var)+'. Сбалансированная точность (среднее долей верно принятых своих и отвергнутых чужих) полного согласия: '+number(full['balanced_accuracy']*100)+' %. Источник `summary.json`: metrics.\n\n'
    text += 'Для C34 требуется не менее 17 принятых своих, не более одной принятой чужой и хотя бы одно строгое улучшение. Допуск: '+('пройден' if s['cv_gates']['C34__agreement'] else 'не пройден')+'. Источник `protocol.json`: acceptance; `summary.json`: cv_gates.\n\n'
    text += '4. Пять рабочих случаев\n\n'
    old = {r['old_id']:r for r in read(SCALE/'deployment_screen.json')['records']}
    for row in pred['screen']:
        label = 'своё' if row['label'] == 'own' else 'чужое'
        n = row['old_id']
        z = row['scores']['C34__agreement']
        text += 'Связь '+str(n)+' ('+label+' продолжение): прежняя оценка '+number(old[n]['scores']['C34'])+', согласие '+number(z)+', '+('принята' if z > 0 else 'отвергнута')+'. Источник `predictions.json`: screen, old_id='+str(n)+'; прежнее значение — `research_signal_scaling_20261109/deployment_screen.json`.\n\n'
    text += 'Требовалось сохранить правильные 70 и 1102, принять 232 или 832 и отвергнуть чужую 574. Такой критерий проверяет изменение порядка своей и чужой связей, которое один общий сдвиг оценок объяснить не может. Допуск: '+('пройден' if s['screen_gates']['C34__agreement'] else 'не пройден')+'. Это сравнение сохранённых локальных оценок; общее назначение не выполнялось.\n\n'
    text += '5. Проверка и границы\n\n'
    for arm in ARMS[1:]:
        ci = s['intervals95']['C34__agreement minus '+arm+'__agreement']
        text += 'Разность сбалансированной точности C34 и контроля «'+names[arm].lower()+'»: описательный 95 % интервал ['+number(ci['low']*100)+'; '+number(ci['high']*100)+'] процентного пункта. Источник `summary.json`: intervals95.\n\n'
    text += 'В парной повторной выборке видео использованы 4000 выборок, пропущено '+str(s['omitted_bootstrap_draws'])+' без обоих классов. Это оценка разброса на изученной выборке, а не независимое доказательство обобщения. '+cannot+'\n\n'
    text += 'Независимо сверены интерполяция, формула через средний квадрат различий, 126 обученных слоёв, нормировки, градиенты и решения. Входы и научные исходники сохранили контрольные суммы. Источник `audit.json`; максимальное расхождение согласия '+number(audit['max_errors']['agreement'])+'. Проверка равного дублирования признаков сохраняет оценки и квадратичный штраф, новых обучений не требует.\n\n'
    text += '6. Решение и сохранение истории\n\n'+decision+' Отрицательный исход этой реализации не закрывает другие способы считывания. Не следует приписывать изменения обычных контролей ядру FlyVis.\n\n'
    text += 'Согласие C34 → сбалансированная точность → '+number(delta)+' % относительно сильнейшего готового варианта с общей нормировкой. Доля измеренного потолка не рассчитывается: потолок для этой постановки не измерен. Источник `summary.json`: metrics.\n\n'
    text += 'При подготовке списка готовых карт обнаружено, что часть снимков хранится только в завершённом наборе карт и отсутствует в прежнем списке запросов. До регистрации исправлено чтение этого списка; научные оценки и обучение до исправления не выполнялись. Старые контроллеры и очереди не запускались.\n'
    # Every paragraph has been checked for component/object distinction,
    # duplication, and agreement with current primary outputs; no future work
    # is phrased as an observed result.
    (OUT/'РЕЗУЛЬТАТ.md').write_text(text, encoding='utf8')
    evidence = [dict(kind='file', ref='research_agreement_readout_20261112/'+n,
                     claim=c) for n, c in [('summary.json','Измеренные сравнения и оба допуска'), ('predictions.json','Все оценки и обученные слои'), ('audit.json','Независимая сверка'), ('РЕЗУЛЬТАТ.md','Вывод, ограничения и решение')]]
    def update(d):
        card = next(c for c in d['experiments'] if c['id'] == ID)
        assert card['state'] == 'running'
        card.update(state='completed_positive' if s['primary_joint_gate'] else 'completed_negative', done=now().date().isoformat(),
                    obtained=obtained, interpretation=interpretation, decision=decision, cannot_claim=cannot)
        for a, met in zip(card['acceptance'], [s['cv_gates']['C34__agreement'], s['screen_gates']['C34__agreement'], s['primary_joint_gate'], s['component_advantage']]):
            a.update(met=met, evidence='research_agreement_readout_20261112/summary.json; research_agreement_readout_20261112/audit.json')
        card['evidence'] += evidence
    rewrite_json('experiments/project_history.json', update)
    def problem(d):
        target = next(x for x in d['problems'] if x['id'] == 'С1')
        target['cases'].append(dict(date=now().date().isoformat(), component='Считыватель полного модуля FlyVis; ядро неизменно', fact=obtained,
                                    hypothesis='Парные средние и амплитуды могут отражать освещение, фон и содержание; данный итог не устанавливает их причинную роль.',
                                    discriminating_check='Оба допуска и согласованные обычные контроли выполнены; дальнейший расчёт этой версии остановлен по решению автора.',
                                    limitation=cannot, evidence=['research_agreement_readout_20261112/summary.json','research_agreement_readout_20261112/audit.json'],
                                    source_sha256=sha(OUT/'summary.json')))
        target['cards'].append(ID)
        target['updated'] = now().date().isoformat()
    rewrite_json('experiments/system_problems.json', problem)
    journal('Согласие значений карт: завершён ограниченный допуск.\n\n*Получено (`'+ID+'`).* '+obtained+'\n\n*Вывод.* '+interpretation+' '+cannot+'\n\n*Решение.* '+decision+' Сводка: согласие C34 → сбалансированная точность → '+number(delta)+' % к готовому варианту с общей нормировкой; доля потолка не определена. Независимая сверка: `research_agreement_readout_20261112/audit.json`. Подготовительная ошибка списка снимков исправлена до регистрации, без научного расчёта. Все старые результаты сохранены.')
    save('recording.json', dict(passed=True, card=ID, primary_joint_gate=s['primary_joint_gate'], summary_sha256=sha(OUT/'summary.json'),
                               scientific_text_checks=dict(component_vs_object=True, no_adjacent_redundancy=True, matches_measured_work=True),
                               no_new_confirmed_result=s['component_advantage'] is False))
    print('RECORDED', s['primary_joint_gate'], counts(m), flush=True)

if __name__ == '__main__':
    main()
