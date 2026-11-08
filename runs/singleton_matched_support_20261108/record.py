import json
from pathlib import Path
from datetime import datetime, timezone, timedelta

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
identifier=OUT.name
catalog_path=ROOT/'experiments/project_history.json'
catalog=json.loads(catalog_path.read_text(encoding='utf8'))
assert not any(c['id']==identifier for c in catalog['experiments'])
catalog['experiments'].append(dict(id=identifier,
    title='Сравнение своих и чужих машин при одинаковых трёх ячейках карты',area='flyvis',
    state='running',started='2026-11-08',
    question='Как меняется сходство своей и чужой машины при одинаковой пространственной поддержке и поиске сдвига?',
    hypothesis='Высокое сходство чужой машины после поиска может определяться малым числом ячеек.',
    expected='Равное число ячеек позволит убрать одно из различий прежних двух примеров.',
    acceptance=[dict(criterion='Равная поддержка без подбора по оценкам, точное восстановление входа, неизменные веса и независимая сверка новых оценок.',met=None)],
    evidence=[dict(kind='file',ref=f'runs/{identifier}/protocol.json',claim='Условия зафиксированы до восстановления карт и расчёта')]))
catalog_path.write_text(json.dumps(catalog,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
plan_path=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md'
plan=plan_path.read_text(encoding='utf8')
now=datetime.now(timezone(timedelta(hours=3))).strftime('%d.%m.%Y, %H:%M')
entry=f'**{now} MSK** — Зарегистрировано сравнение карт при одинаковой поддержке.\n\n*Постановка.* По прямому поручению автора из сохранённого реестра выбраны все напрямую подтверждённые свои и чужие пары с одинаковой поддержкой единственной чужой пары. Получены чужая 390→448 и своя 3→273, по 3 ячейки. Восстанавливаются только две отсутствующие карты своей пары. Прежние оценки чужой пары читаются готовыми. Условия прежние — 19 сдвигов, общий участок для всех сдвигов и сигналов, яркость, градиенты, T4/T5 и перестановка. Порог не подбирается. Карточка `{identifier}`, протокол `runs/{identifier}/protocol.json`.\n\n'
plan_path.write_text(plan.replace('## Журнал изменений\n','## Журнал изменений\n\n'+entry,1),encoding='utf8')
