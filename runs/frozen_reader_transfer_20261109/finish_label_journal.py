"""Finish only the interrupted journal write; keep completed metadata and scores."""
import json,time,os
from pathlib import Path
from datetime import datetime,timedelta,timezone
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
p=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';original=p.read_text(encoding='utf8');note=json.loads((OUT/'label_grade.json').read_text(encoding='utf8'))['note'];assert len(original)>1000000
title='Уточнена степень подтверждения единственной чужой пары переноса';assert title not in original
now=datetime.now(timezone(timedelta(hours=3)));entry=f'**{now:%d.%m.%Y, %H:%M} MSK** — {title}.\n\n*Получено (`{OUT.name}`).* {note} Источники `runs/{OUT.name}/label_grade.json`, `audit.json`.\n\n*Вывод и решение.* Все оценки и исходные метки сохранены; прежнее ограничение карточки вrevisions[]. УсловнуюAUROCне выдавать за строгую независимую проверку личности. Технически записьJSONи карточки завершилась до временногоOSError22приоткрытииплана; исходныйплан сохранён. Сейчас завершается только недописанный журнал, расчёты не повторяются.\n\n'
updated=original.replace('## Журнал изменений\n','## Журнал изменений\n\n'+entry,1)
for attempt in range(5):
    try:
        assert p.read_text(encoding='utf8')==original
        p.write_text(updated,encoding='utf8');assert p.read_text(encoding='utf8')==updated;break
    except OSError:
        if attempt==4:raise
        time.sleep(.2)
print('Interrupted journal write completed; source scores retained')
