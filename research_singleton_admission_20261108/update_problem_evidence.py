"""Append audited identity evidence to existing system problems; preserve other problems."""
import sys,copy,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_identity_assignment_run_20261108.common import read,write,sha,datetime,MSK


def main():
    path=ROOT/'experiments/system_problems.json';data=read(path);original=copy.deepcopy(data);by={r['id']:r for r in data['problems']};stamp=datetime.now(MSK).isoformat();run=ROOT/'runs/singleton_admission_20261108'
    write(run/'system_problems_before_update.json',original,new=True)
    def revise(e,k,v):
        if e[k]==v:return
        e.setdefault('revisions',[]).append(dict(date=stamp,field=k,old=e[k],reason='Добавлены независимо проверенные полные результаты первой задачи и прямое поручение автора решать проблемы по одной.'));e[k]=v
    def source(e,n,claim):
        if not any(r['source']==n for r in e['evidence']):e['evidence'].append(dict(source=n,sha256=sha(ROOT/n),claim=claim))
    own='runs/singleton_verify_20261108/full/comparison_report.json';age='runs/singleton_age_20261108/full/comparison_report.json';quality='runs/singleton_admission_20261108/quality_report.json';parent='runs/identity_assignment_20261108/full_precision/comparison_report.json'
    e=by['С1'];e['updated']='2026-11-08'
    revise(e,'fact',e['fact']+' Последующие полные проверки стоимости назначения и собственной пространственной карты также не подтвердили принятую добавку FlyVis. Проверка собственной карты дала HOTA +0,011110 п.п. к B0 с 95% ДИ [−0,073633;0,102324] и FP +35; контроль по давности — −0,004454 п.п. с ДИ, содержащим ноль, и FP +29. Источники: runs/singleton_verify_20261108/full/comparison_report.json; runs/singleton_age_20261108/full/comparison_report.json. Локальная прибавка на запросах не стала решением полного трекера.')
    revise(e,'next_check','По прямому приоритету автора сейчас решать только отказ при единственной допустимой рамке. Проверить причинный возврат старой метки после фактического штатного подтверждения нового фрагмента, сохраняя ровно те же физические рамки. Геометрия и давность без дополнительных зрительных признаков, изображение, обычное движение и FlyVis получают одинаковое правило. Проблемы 2–4 и исследование разворота движения отложены. Зарегистрированный протокол: runs/native_confirm_policy_20261108/protocol.json. Ни готовые метрики, ни готовые карты не повторять.')
    e['cases'].append(dict(date='2026-11-08',component='Пространственное считывание и стоимость сопоставления полного авторского трекера',fact='Полный отрицательный результат собственной карты: HOTA +0,011110 п.п. к B0, 95% ДИ [−0,073633;0,102324], FP +35; у контроля давности FP +29. Правильных возвратов стало больше, но условия принятия не пройдены.',hypothesis='Полезный прошлый пространственный образ и геометрия могут влиять на выбор; дополнительная польза текущего сигнала FlyVis не установлена.',discriminating_check='Сохранять штатный выпуск рамок и менять только впервые выдаваемую метку после подтверждения нового фрагмента; те же правила без FlyVis.',limitation='Исторические выборки. В конкретном считывании текущий вход за два обновления не достигает T4/T5; это ограничение данного полного модуля, а не универсальный вывод о ядре. Проблемы 2–4 не проверялись.',evidence=[own,age,'runs/singleton_age_20261108/current_input_dependency.json']))
    for n,c in [(parent,'Полная первая версия назначения не прошла критерий'),(own,'Полная собственная карта не прошла критерий'),(age,'Контроль давности тоже не прошёл критерий'),('runs/singleton_verify_20261108/full/independent_audit.json','Независимая проверка полного пространственного пилота'),('runs/singleton_age_20261108/full/independent_audit.json','Независимая проверка давности'),('runs/singleton_age_20261108/current_input_dependency.json','Граница причинного объяснения текущего входа')]:source(e,n,c)
    for id in ['identity_assignment_v2_run_20261108','singleton_own_template_v1_20261108','singleton_recency_control_v1_20261108']:
        if id not in e['cards']:e['cards'].append(id)
    e=by['С2'];e['updated']='2026-11-08'
    revise(e,'fact',e['fact']+' В проверке единственной рамки снова уменьшились пропуски, но появились дополнительные ложные рамки: у FlyVis FP +35, у контроля давности +29. В прямой диагностике 22 добавленные рамки FlyVis на кадрах с разметкой не имеют допустимого объекта с IoU≥0,5 и не пересекают области исключения; Калман не испортил ни одной сопоставимой сырой рамки в этой проверке. Источники: runs/singleton_verify_20261108/full/comparison_report.json; runs/singleton_age_20261108/full/comparison_report.json; runs/singleton_admission_20261108/quality_report.json.')
    revise(e,'next_check','Разделить подтверждение присутствия и восстановление личности. При первом штатном подтверждении нового фрагмента пробовать причинно вернуть отсутствующую старую метку, сохраняя штатный набор рамок. Контроль без FlyVis обязателен. Исторические результаты оценивать по полной HOTA, ошибочным и устойчивым возвратам. Зарегистрированный протокол: runs/native_confirm_policy_20261108/protocol.json. Проблемы 2–4 отложены.')
    e['cases'].append(dict(date='2026-11-08',component='Допуск потерянного трека и штатное подтверждение нового объекта',fact='Все 335 дополнительных назначений FlyVis дошли до выхода. На размеченных кадрах найдены 22 дополнительные неподходящие рамки вне областей исключения. Ухудшение их положения после Калмана и удаление дубликатов не обнаружены.',hypothesis='Снижение стоимости Lost обходит обычное подтверждение нового объекта и тем самым выпускает рамки, которые исходный трекер ещё удерживал неподтверждёнными. Это рабочее объяснение.',discriminating_check='Тот же возврат старой личности, но только после фактического штатного подтверждения; физические рамки обязаны совпасть с B0. Геометрия, давность и зрительные сигналы сравниваются раздельно.',limitation='Неподходящая рамка не равна известной чужой личности. Локальные 22 случая не являются точным разложением глобальных +35 CLEAR FP. Доля неизвестных личностей велика; они не отрицательные примеры.',evidence=[own,age,quality,'runs/singleton_error_audit_20261108/report.json']))
    for n,c in [(own,'Полный пространственный пилот: лишние рамки'),(age,'Контроль по давности: лишние рамки'),(quality,'Прямой разбор качества дополнительных рамок'),('runs/singleton_error_audit_20261108/report.json','Разделение ошибки личности, рамки и удаления')]:source(e,n,c)
    for id in ['singleton_own_template_v1_20261108','singleton_recency_control_v1_20261108','singleton_error_source_audit_v1_20261108']:
        if id not in e['cards']:e['cards'].append(id)
    for pattern in data['patterns']:
        if pattern['id']=='transfer_pattern':
            pattern.setdefault('revisions',[]).append(dict(date=stamp,field='observation',old=pattern['observation'],reason='Новый подтверждённый повтор локальной прибавки без принятия полного трекера.'))
            pattern['observation']+=' Повтор 08.11: собственная карта и давность добавили правильные условные возвраты, но в полном пилоте выросли FP и условия принятия не пройдены. Это не доказывает общую причину прежних отказов.'
            for n,c in [(own,'Условная прибавка не стала принятой полной прибавкой'),(age,'Тот же повтор на контроле без дополнительных зрительных признаков')]:source(pattern,n,c)
    for a,b in zip(data['problems'][2:],original['problems'][2:]):assert a==b
    write(path,data);write(run/'system_problem_update_receipt.json',dict(passed=True,changed_only=['С1','С2','transfer_pattern'],before_sha256=sha(run/'system_problems_before_update.json'),after_sha256=sha(path),remaining_problem_objects_exact=True),new=True)


if __name__=='__main__':main()
