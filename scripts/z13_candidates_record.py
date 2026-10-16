"""Record preparation/launch/end without changing scientific criteria or prior work."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/z13_candidates_20261016'
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def main():
    stage='prepare' if '--prepare' in sys.argv else ('launch' if '--launch' in sys.argv else 'finish')
    marker=OUT/(stage+'_recorded.json'); assert not marker.exists()
    p=read(OUT/'protocol.json'); now=datetime.now(timezone(timedelta(hours=3)))
    path=ROOT/'experiments/project_history.json'; cat=read(path)
    specs={'foreground':('Цель относительно окружения','пространственный контраст цели и окружения'),
           'size':('Размеры объекта','остаточная поправка размеров при фиксированных центрах'),
           'adaptive':('Время накопления','адаптивное накопление против фиксированного с переносом координат')}
    report=read(OUT/'report.json') if (OUT/'report.json').exists() else None
    if stage=='finish': assert read(OUT/'controller_result.json')['own_worker_ended']
    for name,(title,meaning) in specs.items():
        ident='z13_candidate_'+name+'_v1'
        card=next((c for c in cat['experiments'] if c['id']==ident),None)
        if stage=='prepare':
            assert card is None
            cat['experiments'].append(dict(id=ident,title='З13: '+title+' — отдельный функциональный пилот',area='flyvis',state='planned',started=now.date().isoformat(),
                question='Добавляет ли '+meaning+' пользу относительно согласованного контроля?',
                hypothesis='Изменение считывания может улучшить функциональную локализацию; причина прежнего провала не установлена.',
                expected='Одна фиксированная реализация; B/C/D, ядра фиксированы, прежние train104/26; 500 шагов, без HOTA. Протокол runs/z13_candidates_20261016/protocol.json.',
                acceptance=[dict(criterion=c,met=None) for c in p['criteria'][name]],
                cannot_claim='Не проверены HOTA, независимая выборка, сходимость или польза совместного применения модулей.',
                evidence=[dict(kind='json',ref='runs/z13_candidates_20261016/protocol.json',claim='Фиксация до новых расчётов'),
                          dict(kind='file',ref='research_flyvis_china_2026-10-15/СВОДКА.md',claim='Источники и критика')]))
        elif stage=='launch': card['state']='running'
        else:
            card['done']=now.date().isoformat()
            if report:
                card['state']='completed_positive' if report['passed'][name] else 'completed_negative'
                for acceptance,met in zip(card['acceptance'],report['criteria'][name]):
                    acceptance.update(met=met,evidence='runs/z13_candidates_20261016/report.json → criteria.'+name)
                card['obtained']='runs/z13_candidates_20261016/report.json: '+json.dumps({'criteria':report['criteria'][name],'stats':{k:v for k,v in report['stats'].items() if k.endswith('_'+name)},'comparisons':report['comparisons']},ensure_ascii=False)
                card['interpretation']='Результат функционального пилота на многократно использованных train26, не улучшение трекера.'
                card['decision']='Сохранить итог; новый HOTA только отдельным согласованным этапом. Не менять параметры по результату.'
            else:
                card['state']='inconclusive'; card['obtained']='runs/z13_candidates_20261016/controller_result.json: '+json.dumps(read(OUT/'controller_result.json'))
                card['interpretation']='Весь набор сравнений не завершён; численная эффективность не объявляется.'
                card['decision']='Сохранить готовые задания и контрольные точки, продолжение только в новом разрешённом окне.'
    path.write_text(json.dumps(cat,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    plan=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md'; old=plan.read_text(encoding='utf-8'); anchor='## Журнал изменений\n\n'; assert old.count(anchor)==1
    text={'prepare':'Автор разрешил проверить все три предложения и указал абсолютный предел GPU 07:40 МСК 16.10.2026. Подготовлены отдельные реализации цель/окружение, размеры и адаптивное время, согласованные B/C/D и контрольные продолжения. Проверка структуры всех train104 без GPU пройдена: research_flyvis_china_2026-10-15/readiness.json. Параметры, критерии, источники и окно зафиксированы до обучения в runs/z13_candidates_20261016/protocol.json. Текущий запуск качества не меняется. Новые задания ждут его завершения; сохранение до 07:35, остановка до 07:40. Код новых узлов изолирован; старые источники не переписаны.',
          'launch':'Запущен независимый контроллер новых проб: runs/z13_candidates_20261016/controller.json. Он ожидает завершения текущего этапа и только затем открывает один GPU-процесс. Критерии и предел 07:40 не меняются.',
          'finish':'Завершение окна/расчёта: runs/z13_candidates_20261016/controller_result.json. '+('Все сравнения сохранены, итог: '+json.dumps(report['passed']) if report else 'Полного отчёта пока нет; сохранён только реальный остаток.')}[stage]
    entry=f'**{now:%d.%m.%Y, %H:%M} MSK** — {stage}: три собственных узла FlyVis\n\n{text}\n\nНовой HOTA нет; процент к стенду и доля измеренного запаса неизвестны.\n\n'
    plan.write_text(old.replace(anchor,anchor+entry,1),encoding='utf-8')
    marker.write_text(json.dumps(dict(recorded=now.isoformat(),stage=stage),indent=2),encoding='utf-8')
    print(stage,'recorded')

if __name__=='__main__': main()
