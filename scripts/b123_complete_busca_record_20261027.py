"""Дописать журнал после временного отказа открытия; научные результаты не повторять."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
def main():
    run=ROOT/'runs/b123_20261027/busca_output_pilot'
    audit=json.loads((run/'independent_audit.json').read_text(encoding='utf-8'));assert audit['passed']
    cat=json.loads((ROOT/'experiments/project_history.json').read_text(encoding='utf-8'))
    c=next(x for x in cat['experiments'] if x['id']=='busca_output_recovery_pilot_v1');assert c['state']=='completed_negative'
    from scripts.b123_safe_journal_20261027 import journal
    title='выходная адаптация BUSCA: вред меньше простого продолжения, но HOTA ниже базы'
    assert title.encode() not in (ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md').read_bytes()
    journal(c['id'],title,'*Получено.* '+c['obtained']+'\n\n*Вывод.* '+c['interpretation']+'\n\n*Решение.* '+c['decision']+'\n\n'
        '*Сводка к базе.* Выходная адаптация BUSCA → HOTA → −1,825865% к исходному XYWH; '
        'разность общей HOTA −1,057191п.п., средняя парная −0,966471п.п.[−1,322852;−0,548141]. '
        'Доля измеренного потолка неизвестна. Все20видео завершены, код0 и конец дерева проверены. '
        'Первый вызов записи журнала вернул OSError22 без изменения файла; карточка и независимый аудит уже сохранились. '
        'Дописан только журнал, расчёты не повторялись. Источники:busca_output_pilot/report.json и independent_audit.json.')
    with (ROOT/'runs/b123_20261027/HANDOFF.md').open('a',encoding='utf-8') as f:
        f.write('\n\n## '+datetime.now(timezone(timedelta(hours=3))).strftime('%H:%M')+' МСК — отрицательный итог выходной BUSCA проверен\n\n'
            'Все20видео завершены. HOTA57.900829→56.843638; средняя−0.966471[−1.322852;−0.548141],лучше3/20. '
            'FP3331→6304,FN13525→12488,IDSw1001→973. К безусловному продолжениюлучше20/20. '
            'independent_audit.json проверил20baselineJSON/метрики,append-only,хеши,другойсидДИ. '
            'Карточка completed_negative с HOTA/source/locator и revisions; журнал дописан отдельной служебной записью после OSError22,данные не пересчитывались. '
            'Не принимать в стенд и не объявлять опровержением всейBUSCA. Далее собственная масочная адаптация,точнуюLTRнеподменять. '
            'Нужны зелёные проверки каталога и обновление одной фундаментальной задачи4безновыхтехническихузлов. '
            'ОдинGPU,прежнийночнойсрок,.git/index.lockнетронут.\n')
    print('Recorded journal and handoff without recomputation')
if __name__=='__main__':main()
