"""Append the completed matched full result; no change to unrelated problems."""
import sys,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_identity_assignment_run_20261108.common import *
RUN=ROOT/'runs/identity_evidence_20261108'


def main():
    full=ROOT/'runs/native_confirm_risk_20261108/full';assert read(full/'queue_result.json')['passed'];report=read(full/'comparison_report.json');assert not report['problem1_acceptance'];assert read(full/'independent_audit.json')['comparison_sha256']==sha(full/'comparison_report.json')
    path=ROOT/'experiments/system_problems.json';data=read(path);original=copy.deepcopy(data);write(RUN/'system_problems_before.json',original,new=True);by={r['id']:r for r in data['problems']};stamp=datetime.now(MSK).isoformat();source='runs/native_confirm_risk_20261108/full/comparison_report.json';audit='runs/native_confirm_risk_20261108/full/independent_audit.json'
    def revise(e,k,value):
        if e[k]==value:return
        e.setdefault('revisions',[]).append(dict(date=stamp,field=k,old=e[k],reason='Завершён независимо проверенный полный причинный вариант с точным сохранением штатных физических рамок.'));e[k]=value
    for key in ['С1','С2']:
        e=by[key];e['updated']='2026-11-08'
        for n,c in [(source,'Полный контроль подтверждения присутствия и отдельного возврата личности'),(audit,'Независимые траектории, метрики, возвраты и пространственные оценки')]:
            if not any(r['source']==n for r in e['evidence']):e['evidence'].append(dict(source=n,sha256=sha(ROOT/n),claim=c))
        if 'native_confirm_empirical_risk_v1_20261108' not in e['cards']:e['cards'].append('native_confirm_empirical_risk_v1_20261108')
    e=by['С1'];revise(e,'fact',e['fact']+' При сохранении точных штатных рамок и возврате номера только после подтверждения новая версия FlyVis дала HOTA+0,006806п.п.,95%ДИ[−0,003559;0,018089]. Геометрия с той же давностью дала+0,006532п.п., её ДИ тоже содержит0. Добавка FlyVis и принятый выигрыш всей системы не установлены (runs/native_confirm_risk_20261108/full/comparison_report.json).')
    e['cases'].append(dict(date='2026-11-08',component='Полная авторская система: причинный возврат метки после штатного подтверждения',fact='Физическая выдача всех вариантов совпала сB0; уFlyVis правильные возвраты+3,устойчивые+2,IDSw−6,FP/FN/известные чужие возвраты без изменений. HOTAДИ включает0; версия не принята.',hypothesis='Сохранённое из бинарной постановки ограничение ровно двумя поддержанными рамками могло сузить полезную область. Подсчёт подтвердил исключение320из454обычных одиночных геометрических возможностей на26видео, но не их правильность.',discriminating_check='Обычная геометрия без требования пары и малого числа ячеек; отдельный контроль величины поправки стоимости безFlyVis. Все научные результаты связывать с фактическим протоколом.',limitation='Исторические выборки; эмпирический ноль чужих на подборе не является гарантией обобщения. Первая проблема открыта; проблемы2–4отложены.',evidence=[source,audit,'runs/bbox_confirmation_inventory_20261108/inventory.json']))
    e=by['С2'];revise(e,'fact',e['fact']+' В следующей причинной версии дополнительный выпуск рамок предотвращён конструкцией: номер возвращается только после фактического штатного подтверждения; набор и положения рамок побайтно совпали сB0, FP/FNне выросли. Но HOTAДИ включает0, прирост личности пока не принят. Это частное устранение добавленного выпуска рамок, не закрытие всей проблемы восстановления (runs/native_confirm_risk_20261108/full/comparison_report.json).')
    e['cases'].append(dict(date='2026-11-08',component='Раздельное подтверждение присутствия и номера объекта',fact='Прежние дополнительные FP+35не воспроизвелись: во всех новых вариантах FP3331иFN13525, какуB0. Правильных возвратов+3,устойчивых+2,но принятый приростHOTAне доказан.',hypothesis='Обход штатного подтверждения был причиной добавленных физических рамок данной старой версии. Изменение политики изменило только публичные номера, поэтому лишних физических рамок добавить не могло.',discriminating_check='Дальше проверять правильность возвращаемого номера при той же штатной выдаче. Не принимать единственность геометрического кандидата за доказательство личности.',limitation='Сохранение FP/FNне доказывает, что все исходные рамки правильны, и не подтверждает восстановление после длительного перекрытия.',evidence=[source,audit]))
    latest=next(e for e in read(ROOT/'experiments/project_history.json')['experiments'] if e['id']=='bbox_singleton_normalized_maps_v1_20261108')
    next_check=latest.get('decision') or 'По прямому приоритету автора активна только первая проблема. Полная версия с точным сохранением штатных рамок не принята. Малая поправка стоимости не прошла критерий из-за одной чужой склейки. Проверяется относительное выравнивание наблюдаемых рамок при одинаковых контролях безFlyVis: runs/bbox_normalized_signals_20261108/protocol.json. При провале условного критерия полный20valне запускать. Проблемы2–4и разворот движения отложены. Готовые карты, детекции и метрики не повторять.'
    for k in ['С1','С2']:revise(by[k],'next_check',next_check)
    for before,after in zip(original['problems'][2:],data['problems'][2:]):assert before==after
    assert original['patterns']==data['patterns'];write(path,data);write(RUN/'system_problem_update_receipt.json',dict(passed=True,before_sha256=sha(RUN/'system_problems_before.json'),after_sha256=sha(path),changed_only=['С1','С2'],other_problem_objects_and_patterns_exact=True),new=True)


if __name__=='__main__':main()
