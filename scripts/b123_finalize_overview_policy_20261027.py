"""Сохранить новое правило обзора и восстановить точные исходные тексты гипотез."""
from pathlib import Path
import json,re,sys,time,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
VAULT=ROOT/'ObsidianFlyVis';RUN=ROOT/'runs/b123_20261027/graph_fundamental_rebuild'
from scripts.obsidian_hypothesis_tree import section,START,END
def atomic(path,data):
    temp=path.with_suffix('.policy-next');temp.write_bytes(data)
    for attempt in range(40):
        try:temp.replace(path);return
        except OSError:
            if attempt==39:raise
            time.sleep(.05)
def strip(raw):
    return re.sub(re.escape(START.encode())+rb'.*?'+re.escape(END.encode())+rb'\s*',b'',raw,flags=re.S).rstrip()
def main():
    history=json.loads((ROOT/'experiments/project_history.json').read_text(encoding='utf-8'))
    for card in history['experiments']:
        if not any(t in {'HY26-1','HY26-2','HY26-3'} for t in card.get('tags',[])):continue
        relative=Path('09 Запланировано')/(card['title']+'.md')
        original=(RUN/'before'/relative).read_bytes();current=(VAULT/relative).read_bytes()
        assert strip(current).replace(b'\r',b'')==strip(original).replace(b'\r',b''),relative
        atomic(VAULT/relative,original.rstrip()+b'\n\n'+section(card).encode('utf-8'))
        assert strip((VAULT/relative).read_bytes())==strip(original)
    manual=VAULT/'ИНСТРУКЦИЯ — как вести записи.md'
    before=(RUN/'vault-before/ИНСТРУКЦИЯ — как вести записи.md').read_bytes()
    assert manual.read_bytes()==before
    body='''---
tags: [инструкция]
---

# Как вести записи в проекте и в Obsidian

## 1. Единица записи в Obsidian — задача

В хранилище остаются только основные задачи, гипотезы, способы их проверки, важные научные итоги и решения о дальнейшей работе. Одна запись отвечает: чего хотели добиться, что содержательно попробовали, чем попытка закончилась и что делаем дальше. Несколько считывателей или исправлений внутри одной задачи не превращаются в десятки кружков.

## 2. Где остаются подробности

Полная хронология ведётся в E:/Dissertation/dissertation/ЕДИНЫЙ_ПЛАН.md. Каталог с критериями и всеми измерениями — E:/Dissertation/experiments/project_history.json. Первичные файлы — в runs/. Ошибки запуска, возобновления, производные, стоимость и промежуточные шаги записываются туда. Журнал, его индекс, копия всего плана, технические карточки и доска-каталог «Карта мухи.canvas» в Obsidian не создаются.

## 3. Отдельное дерево гипотез

В центре — «Гипотезы». От него три видимые ветви Б1–Б3. От каждой разрешены дочерние подходы: BUSCA, восстановление по маскам и другие обоснованные способы проверки. По прямому разрешению автора агент может добавлять такие дочерние узлы и внутренние связи самостоятельно только в этой форме. Основные научные формулировки и результаты произвольно не меняются.

## 4. Стелс-связь

Связь со статьёй, опорной задачей или другой группой — кликабельный переход obsidian://open без видимого ребра. По умолчанию он двусторонний. Внешних структурных рёбер у дерева гипотез нет. У каждой дочерней записи объясняется роль: реализация, кандидат на замену узла или альтернатива для сравнения. Статья не является доказательством нашей гипотезы.

## 5. Цвет

Зелёный означает, что конкретная проверка завершена и результат сохранён. Отрицательная проверка тоже зелёная. Это не означает успеха, пользы для HOTA или закрытия всего направления. Красный означает непроверенную гипотезу или подход. Исход и ограничения записываются словами. Техническая исправность не делает научную гипотезу проверенной.

## 6. Как обновлять

Генератор scripts/build_obsidian_vault.py строит только смысловой обзор через scripts/obsidian_fundamental_overview.py. Рабочая безопасная сборка выполняется обёрткой scripts/b123_sync_graph_preserving_hypotheses_20261027.py с обновлением разрешённых дочерних связей. Устаревшие производные заметки перемещаются в проверяемый архив вне хранилища; история и первичные результаты не удаляются. Перед публикацией проверяются реальные имена файлов, все цели ссылок, дерево гипотез и отсутствие случайных числовых узлов. Сначала проверяется копия хранилища, затем применяется миграция.

Эта инструкция отражает прямые уточнения автора от27.10.2026 и заменяет прежнее копирование подробной хроники в граф. Полные правила исследований остаются в AGENTS.md и Едином плане.
'''
    atomic(manual,body.encode('utf-8'))
    print('Exact hypothesis source text retained; concise current instruction saved. Old instruction remains in full snapshot.')
if __name__=='__main__':main()
