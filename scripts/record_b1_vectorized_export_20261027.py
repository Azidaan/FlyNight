"""Регистрация новой CPU-реализации экспорта до расчёта."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import json

ROOT=Path(__file__).resolve().parents[1]


def main():
    protocol='runs/b123_20261027/b1_base_export_vectorized/protocol.json'
    assert (ROOT/protocol).exists()
    path=ROOT/'experiments/project_history.json';raw=path.read_bytes();data=json.loads(raw)
    identifier='b1_base_export_vectorized_v1'
    assert not any(c['id']==identifier for c in data['experiments'])
    data['experiments'].append(dict(id=identifier,title='Б1 — пакетное восстановление признаков базы для полного трекера',
        area='flyvis',state='running',started='2026-10-27',hypothesis='',
        question='Воспроизводит ли новая более дешёвая реализация наблюдаемые признаки базы и сохранённые прогнозы?',
        expected='Кэш однократного чтения NPZ и пакетное вычисление сравнений вместо вложенных циклов. Самостоятельная новая ступень до300с CPU, старые истёкшие контроллеры не возобновляются.',
        acceptance=[dict(criterion='Максимальная конечная разность признаков менее1e-5, одинаковые пропуски и точные сохранённые прогнозы личности и выбора действия; без новых обучений и GPU.',met=None)],
        summary='Тест эквивалентности на пропусках, равных значениях и одиночных кандидатах пройден; полный расчёт ещё не выполнен.',
        obtained='',interpretation='',decision='',
        cannot_claim='Проверка сохранённых признаков не подтверждает работу полного трекера, HOTA или научную гипотезу Б1.',
        files=['scripts/b1_base_export_vectorized_20261027.py','tracking/b1_appearance_fast.py','tests/test_b1_appearance_fast.py'],
        evidence=[dict(kind='file',ref=protocol,claim='Условия, бюджет и источники до расчёта')]))
    assert path.read_bytes()==raw
    path.write_bytes((json.dumps(data,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
    plan=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';raw=plan.read_bytes();text=raw.decode('utf-8')
    stamp=datetime.now(timezone(timedelta(hours=3))).strftime('%d.%m.%Y, %H:%M MSK')
    note=f'''**{stamp}** — Б1: новая пакетная реализация экспорта базы до расчёта

*Постановка (`{identifier}`).* `{protocol}`: кэш однократного чтения NPZ и пакетное сравнение кандидатов вместо вложенных циклов; прежняя формула и PCA сохраняются. Это отдельная реализация и новый ограниченный бюджет300с CPU внутри серии автора, а не продление истёкших контроллеров предыдущего кода. Тест равенства с прежним сравнением на равных значениях, пропусках и одиночных кандидатах прошёл: `tests/test_b1_appearance_fast.py`. Критерий до расчёта: конечная разность признаков<1e-5, одинаковые пропуски, точные прогнозы сохранённых моделей. Новые веса не обучаются; GPU не используется. Это подготовка включения в полный трекер, не измерение HOTA. Группа «Гипотезы» не меняется.

'''
    assert plan.read_bytes()==raw
    plan.write_bytes(text.replace('## Журнал изменений\n','## Журнал изменений\n\n'+note,1).encode('utf-8'))


if __name__=='__main__':main()
