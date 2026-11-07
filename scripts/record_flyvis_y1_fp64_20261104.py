"""Сохранить полный результат нового численного исполнения; исходный FP32 не принимать."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json
import shutil
import sys
import psutil

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.flyvis_y1_latency_fp64_20261104 import OUT,write,sha


def atomic(path,text):
    temporary=path.with_suffix('.y1_fp64.next');temporary.write_text(text,encoding='utf-8');temporary.replace(path)


def main():
    chain=json.loads((OUT/'chain_result.json').read_text());audit=json.loads((OUT/'independent_audit.json').read_text());analysis=json.loads((OUT/'analysis.json').read_text())
    assert chain['complete'] and chain['all_owned_ended'] and audit['complete'] and audit['passed']
    assert json.loads((OUT/'execution_identity_verified.json').read_text())['passed']
    assert json.loads((OUT/'causal_prefix_verified.json').read_text())['passed']
    owned={int(k):v for k,v in chain['all_owned'].items()}
    m=json.loads((OUT/'controller_launch.json').read_text());owned[m['pid']]=m['created']
    for m in json.loads((OUT/'agent_launch_verified.json').read_text())['launches']:owned[m['pid']]=m['created']
    old=json.loads((OUT/'previous_processes_ended.json').read_text())
    processes=set(owned.items())|{(int(k),v) for k,v in old['owned'].items()}
    for pid,birth in processes:
        try:assert abs(psutil.Process(pid).create_time()-birth)>.01,pid
        except psutil.NoSuchProcess:pass
    now=datetime.now(ZoneInfo('Europe/Moscow'))
    write('ended_verified.json',dict(all_owned_ended=True,owned=owned,
        processes=[dict(pid=pid,created=birth,original_process_ended=True) for pid,birth in sorted(processes)],checked_msk=now.isoformat()))
    p=analysis['primary']
    rows=json.loads((OUT/'measurements.json').read_text())['rows']
    secondary=[]
    for contrast in [1.,.25]:
        for event in ['reverse','stop']:
            for method,aggregation in [('hold','last'),('hold','mean'),('linear','last'),('linear','mean'),('DIS','mean')]:
                group=[r for r in rows if r['contrast']==contrast and r['event']==event and r['method']==method and r['aggregation']==aggregation]
                assert len(group)==16
                lat=[r['latency_frames'] for r in group if r['eligible'] and r['latency_frames'] is not None]
                secondary.append(dict(contrast=contrast,event=event,method=method,aggregation=aggregation,n=16,
                    eligible=sum(r['eligible'] for r in group),uncensored=len(lat),
                    censored=sum(r['eligible'] and r['latency_frames'] is None for r in group),
                    mean_delay_frames=sum(lat)/len(lat) if lat else None,delay_range=[min(lat),max(lat)] if lat else None))
    write('secondary_summary.json',dict(groups=secondary,scope='Descriptive prespecified secondary comparisons; censored cases are counted, not silently excluded'))
    if p['necessary_gate_passed']:
        state='completed_positive'
        finding=(f"На четырёх исходных изображениях необходимый гейт лишней задержки пройден: "
            f"hold с выходом последнего внутреннего шага запаздывает относительно DIS в среднем на {p['mean_excess_frames']:.6f} кадра; "
            f"95% интервал [{p['ci95'][0]:.6f}; {p['ci95'][1]:.6f}]. Это наблюдение о выходе штатного модуля, не доказательство причины в константах ядра.")
        decision='Подготовить согласуемое минимальное вмешательство в динамику с контролем постоянных констант; новую серию и обучение автоматически не запускать.'
    elif p['identifiable']:
        state='completed_negative'
        finding=(f"Зарегистрированный гейт лишней задержки не пройден: средняя разница hold+last против DIS "
            f"{p['mean_excess_frames']:.6f} кадра; 95% интервал [{p['ci95'][0]:.6f}; {p['ci95'][1]:.6f}]. "
            'Это отрицательный итог данного операционного критерия, не доказательство отсутствия задержек вообще.')
        decision='Не продвигать адаптивные временные константы по этому гейту; не подбирать определения и пороги по полученным ответам. Выбрать следующую обоснованную правку отдельно.'
    else:
        state='inconclusive'
        finding=(f"Полный расчёт и аудит завершены, но задержка измерима не для всех зарегистрированных входов: "
            f"{p['valid_pairs']}/16 пар высококонтрастного разворота выполнили условия достаточного предшествующего отклика и наблюдаемого перехода. "
            'Слабый отклик и правое цензурирование не позволяют объявить задержку отсутствующей. Необходимый гейт не пройден.')
        decision='Не запускать адаптивную динамику по неидентифицируемому измерению. Сохранить все слабые и цензурированные случаи; дальнейшую правку выбирать отдельно, без отбора удобных фонов.'
    secondary_text=('При искусственно ослабленном контрасте средняя задержка разворота hold+last составила 3,0625 кадра на всех 16 случаях. '
        'Линейное смешивание и усреднение внутренних шагов вместе дали 2,6875 кадра при обычном контрасте против 2,0625 у hold+last; '
        'следовательно, эти обработки добавляют часть задержки, но её наличие при hold+last ими целиком не объясняется. '
        'После остановки 2 из 16 обычных и 7 из 16 слабоконтрастных случаев не вошли в полосу 20% прежнего отклика по зарегистрированному условию за окно из 24 отсчётов. '
        'Эта остаточная величина может включать смещение считывателя на новом неподвижном изображении; её нельзя целиком приписывать временной памяти ядра. '
        'Все вторичные сравнения описательные; источник: runs/flyvis_y1_latency_fp64_20261104/secondary_summary.json.')
    limits=('Это искусственные глобальные сдвиги четырёх реальных TRAIN-фонов, не движение отдельных объектов и не HOTA. '
        'Проверена одна штатная модель ансамбля flow/0000/000; это не новый независимый итоговый тест. '
        'Четыре исходных изображения являются единицами сводки; кадры не независимы. '
        'Вычисления выполнены в двойной точности с исходными значениями весов, уравнениями и шагом времени. '
        'Отдельная частичная GPU-FP32 серия не прошла численную приёмку и не подменяется этим результатом. '
        'Абсолютная амплитуда и ошибка перемещения номинальны по прежнему преобразованию декодера, а нормирование по плато служит только измерению перехода. '
        'Входы, DIS и метрики повторены полностью; нейронная симуляция независимо сверена на одной полной фиксированной последовательности, не на всех фонах. '
        'Наблюдаемый переход не устанавливает корневую причину в ядре вместо считывателя. Адаптивные константы ещё не реализованы и не проверены.')
    obtained=(f"64 последовательности, 2 способа подачи, 2 способа агрегирования, 5056 исходных кадров; все входы, DIS, метрики и хеши проверены. "
        f"Максимальная ошибка независимого BoxEye: {audit['eye_max_diff']:.10g}; полная сверка 78 пар кадров CPU-FP64/GPU-FP64: "
        f"{audit['cpu_reference']['max_abs_diff']:.10g}, обе в исходном допуске 1e-5. "
        'Источники: runs/flyvis_y1_latency_fp64_20261104/independent_audit.json, measurements.json, analysis.json, ended_verified.json. '+finding+' '+secondary_text)
    path=ROOT/'experiments/project_history.json';history=json.loads(path.read_text(encoding='utf-8'))
    card=next(c for c in history['experiments'] if c['id']=='flyvis_y1_latency_fp64_v1');assert card['state']=='planned'
    card.update(state=state,done='2026-11-04',obtained=obtained,interpretation=finding+' '+secondary_text,decision=decision,cannot_claim=limits)
    card['acceptance'][0].update(met=True,evidence='runs/flyvis_y1_latency_fp64_20261104/independent_audit.json; ended_verified.json')
    card['acceptance'][1].update(met=bool(p['necessary_gate_passed']),evidence='runs/flyvis_y1_latency_fp64_20261104/analysis.json:primary')
    for name in ['analysis.json','independent_audit.json','measurements.json','ended_verified.json','secondary_summary.json','causal_prefix_verified.json','execution_identity_verified.json']:
        card['evidence'].append(dict(kind='json',ref='runs/flyvis_y1_latency_fp64_20261104/'+name,claim='Полный результат исполнения FP64 с ограничениями'))
    atomic(path,json.dumps(history,ensure_ascii=False,indent=2)+'\n')
    path=ROOT/'experiments/flyvis_core_dynamics_hypothesis.json';reg=json.loads(path.read_text(encoding='utf-8'));assert reg['status']=='planned'
    reg.setdefault('revisions',[]).append(dict(date=now.isoformat(),field='authorization_scope',old=reg['authorization_scope'],reason='Новое разрешение на первую проверку без обучения'))
    reg['authorization_scope']=('Поручение 02.11 разрешало запись в граф. 04.11 автор ответил «Давай, продолжаем» на предложение выполнить первую проверку штатной FlyVis без обучения. '
        'Разрешение и технические условия сохранены в runs/flyvis_y1_latency_fp64_20261104/authorization.json и protocol.json. '
        'Адаптивная динамика и обучение остаются отдельным не разрешённым этим первым этапом действием.')
    previous=reg['sections'][0]['text']
    reg.setdefault('revisions',[]).append(dict(date=now.isoformat(),field='sections[0].text',old=previous,
        reason='Первая диагностика выполнена; исходную формулировку отделяем от измеренного результата, причину в ядре не объявляем установленной'))
    reg['sections'][0]['text']=previous.replace('Такое запаздывание нашей модели пока не измерено; установленным дефектом или корневой причиной прежних неудач оно не считается.',
        'До первой проверки запаздывание этой модели не измерялось. Полученный диагностический результат приведён ниже. Причина возможной задержки в ядре пока не установлена.')
    reg['diagnostic_results']=[dict(date='2026-11-04',title='Первая проверка штатной модели — измеренный результат FP64',summary=finding+' '+secondary_text+' '+limits,
        source='runs/flyvis_y1_latency_fp64_20261104/independent_audit.json',sha256=sha(OUT/'independent_audit.json'),
        analysis='runs/flyvis_y1_latency_fp64_20261104/analysis.json',card_id=card['id'])]
    atomic(path,json.dumps(reg,ensure_ascii=False,indent=2)+'\n')
    path=ROOT/'experiments/system_problems.json';problems=json.loads(path.read_text(encoding='utf-8'))
    c=next(c for c in problems['problems'] if c['id']=='С1')
    c['cases'].append(dict(date='2026-11-04',component='переходный отклик штатного модуля FlyVis',fact=finding,
        hypothesis='Причина в постоянных временных константах остаётся гипотезой; вход и считыватель — альтернативы.',
        discriminating_check=decision,limitation=limits,evidence=['runs/flyvis_y1_latency_fp64_20261104/analysis.json','runs/flyvis_y1_latency_fp64_20261104/independent_audit.json']))
    c.setdefault('revisions',[]).append(dict(date=now.isoformat(),field='next_check',old=c['next_check'],
        reason='Первая проверка Я1 завершена; следующий шаг линии ядра определяется её измеренным результатом'))
    c['next_check']='В текущей линии ядра FlyVis: '+decision+' Этот диагностический этап не заменяет критерий дополнительной пользы полного трекера.'
    c['updated']='2026-11-04';c['cards'].append(card['id'])
    for name in ['analysis.json','independent_audit.json']:c['evidence'].append(dict(source='runs/flyvis_y1_latency_fp64_20261104/'+name,sha256=sha(OUT/name),claim='Первая проверка отклика; не доказательство пользы ядра для трекера'))
    atomic(path,json.dumps(problems,ensure_ascii=False,indent=2)+'\n')
    entry=f'''**{now:%d.%m.%Y, %H:%M} MSK** — Первая проверка переходного отклика Я1 завершена в отдельном исполнении FP64

*Постановка.* `flyvis_y1_latency_fp64_v1`, тот же первый разрешённый этап без обучения. Условия до новых выходов: `runs/flyvis_y1_latency_fp64_20261104/protocol.json`. Все 64 точных входа и DIS переиспользованы из исходной подготовки по SHA. Научные критерии, шаг времени, исходные значения весов и допуск 1e-5 сохранены. Первоначальная частичная GPU-FP32 серия не прошла численную приёмку: CPU-FP32 1,0460615158e-5, CPU-FP64 против GPU-FP32 1,2713371799e-5. Она сохранена отдельно и не принимается как научный итог. GPU-ожидание, исходный code15 и ошибки CPU-загрузки/синхронизации, контроллеры и исходники сохранены в `runs/flyvis_y1_latency_20261104/`. Повторных готовых GPU-FP32 проходов не было.

*Получено.* {obtained} Контроллеры исполнения FP64 завершились code0; фактический конец всех PID+birth проверен, включая предыдущие попытки. Основная серия посчитана на CPU; GPU использовалась для одной полной сверочной последовательности в той же двойной точности. Независимый CPU-ответ первой последовательности переиспользован по SHA, прежний отказ при сравнении с GPU-FP32 не переписан. {finding}

*Вывод.* {limits}

*Решение.* {decision} Я1 остаётся красной: первая диагностика штатной модели не завершает проверку новой динамики. Случай добавлен в историю С1 без объявления проблемы решённой. Сводка к исходному стенду: HOTA не измерялась; процент прироста и доля потолка неизвестны.

'''
    plan=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';raw=plan.read_bytes();marker='## Журнал изменений'.encode();ending=b'\r\n' if marker+b'\r\n' in raw else b'\n'
    plan.write_bytes(raw.replace(marker+ending,marker+ending+b'\n'+entry.encode('utf-8'),1))
    with (ROOT/'FLYVIS_TRACKER_RESEARCH_ROADMAP.md').open('a',encoding='utf-8') as f:
        f.write('\n\n## 04.11.2026 — Первая проверка Я1 в двойной точности\n\n'+finding+'\n\n'+decision+'\n\nИсточник: `runs/flyvis_y1_latency_fp64_20261104/analysis.json`, `independent_audit.json`. Изменение динамики не реализовано.\n')
    write('report.json',dict(complete=True,scientific_state=state,primary=p,summary=finding,decision=decision,limitations=limits,
        all_owned_ended=True,neural_core_modified=False,training=False,graph_hypothesis_status='planned',experiment_card=card['id']))
    atomic(OUT/'РЕЗУЛЬТАТ.md','# Я1 — первая проверка штатной FlyVis\n\n'+finding+'\n\n'+obtained+'\n\n'+limits+'\n\n'+decision+'\n')
    print(finding)


if __name__=='__main__':main()
