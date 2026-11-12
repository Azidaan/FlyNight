import json, re, hashlib
from pathlib import Path
OUT = Path(__file__).resolve().parent
path = OUT / 'criteria.json'
criteria = json.loads(path.read_text(encoding='utf8'))
translations = {
    'Correct additional recovery proposal for574': 'Правильное дополнительное предложение восстановления для574',
    'No foreign or unknown new proposal anywhere in all5complete pools': 'Нет чужих или неопределённых новых предложений во всех пяти полных наборах конкурентов',
    'Freeze every saved native public assignment and all occupied aliases': 'Все сохранённые штатные публичные назначения и занятые псевдонимы защищены от перехвата',
    'No target continuation in all5remove-all-own controls; no descriptor/permutation/missing-bank false success hidden': 'В пяти контролях без собственных кандидатов целевые строки не продолжаются; контроли перестановки и отсутствия описателей сохранены полностью',
    'Full107reference rows and222raw candidates retained for competition before NMS': 'Все107прошлых строк и222сырые детекции участвуют в конкуренции до подавления дублирующих рамок',
    'No evaluation labels or hardcoded identifier values in decision rule; candidate labels only outside policy': 'Правило не получает оценочную разметку и не содержит условий по известным значениям идентификаторов; разметка используется только во внешней оценке',
}
for variant in criteria['variants']:
    variant['scope'] = 'Только зарегистрированная инженерная проверка на пяти прежних случаях; независимое обобщение не измерено.'
    for condition in variant['acceptance']:
        if condition['criterion'] == 'Public-ID lifecycle/HOTA measured':
            condition['criterion'] = 'Не заявлены исправление публичного идентификатора и прирост HOTA без проверки полного трекера'
            condition['met'] = True
            condition['evidence'] = 'Полный трекер и жизненный цикл не запускались; граница вывода явно сохранена в итоговом отчёте.'
        elif condition['criterion'] in translations:
            condition['criterion'] = translations[condition['criterion']]
    variant['decision'] = 'Допустим как экспериментальный модуль ограниченного адаптера; полный исполняемый трекер не активируется.' if variant['state'] == 'completed_positive' else 'Широкая версия не принимается: сохранить три неопределённых предложения и локальный правильный результат574.'
    variant['cannot_claim'] = ['Прирост HOTA или качества полного трекера', 'Исправление публичного идентификатора и его жизненного цикла', 'Общая безопасность на новых кадрах', 'Правильность трёх неопределённых предложений широких версий', 'Польза, приписываемая ядру FlyVis']
path.write_bytes((json.dumps(criteria, ensure_ascii=False, indent=2).rstrip() + '\n').encode('utf8'))
path = OUT / 'ИТОГ_ВОССТАНОВЛЕНИЯ_574.md'
text = path.read_text(encoding='utf8').replace('; Детекция', '; детекция')
text = re.sub(r'(?<=[А-Яа-я]),(?=\d)', ', ', text)
text = re.sub(r'\)(?=\d)', ') ', text)
path.write_bytes((text.rstrip() + '\n').encode('utf8'))
review_path = OUT / 'text_review.json'
review = json.loads(review_path.read_text(encoding='utf8'))
review['report_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
review['final_formatting'] = 'Уточнены только пробелы, регистр после точки с запятой и русские формулировки критериев; научные условия и выводы сохранены.'
review_path.write_bytes((json.dumps(review, ensure_ascii=False, indent=2).rstrip() + '\n').encode('utf8'))
print('TEXT_FINALIZED')
