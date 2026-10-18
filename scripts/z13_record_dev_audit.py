"""Record the completed dev probe's selection limitation, preserving previous text."""
import json
import subprocess
from datetime import datetime,timedelta,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/z13_target_center_dev_20261018'


def main():
    marker=RUN/'audit_recorded.json'
    if marker.exists():raise RuntimeError('Already recorded')
    checked=json.loads((RUN/'final_verification.json').read_text())
    assert checked['paired_statistics_verified'] and checked['coverage_verified']
    now=datetime.now(timezone(timedelta(hours=3)))
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    source='runs/z13_target_center_dev_20261018/report.json'
    note=source+' (summary.*.coverage.0.01): одинаково отобраны94ответа; D_center — 2 полезных из85известных, 9неизвестных; D_prior — 14из54известных, 40неизвестных. Это описательные числа отбора, не измерение вреда HOTA и не сравнение точности на одном составе известных исходов.'
    path=ROOT/'experiments/project_history.json';cat=json.loads(path.read_text(encoding='utf-8'))
    card=next(c for c in cat['experiments'] if c['id']=='z13_target_center_dev_v1')
    updated=dict(obtained=card['obtained'].replace('п,п,','п.п.')+' '+note,
        interpretation=card['interpretation']+' Перенос улучшения локализации подтверждён при фиксированных параметрах. Надёжный отбор полезных вмешательств текущим баллом не получен. Преимущество D над штатной C не подтверждено.')
    for field,text in updated.items():
        card.setdefault('revisions',[]).append(dict(date=now.date().isoformat(),commit=head,field=field,old=card[field],reason='Проверка первичных строк: явно добавлен неблагоприятный итог отбора и исправлено обозначение п.п.'))
        card[field]=text
    card['evidence'].append(dict(kind='json',ref='runs/z13_target_center_dev_20261018/final_verification.json',claim='Повторная проверка парных статистик, исходных хешей и полноты'))
    path.write_text(json.dumps(cat,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    path=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';plan=path.read_text(encoding='utf-8');anchor='## Журнал изменений\n';assert anchor in plan
    entry=f'\n**{now:%d.%m.%Y, %H:%M} MSK** — проверка полного итога `z13_target_center_dev_v1`\n\n'
    entry+='*Получено.* runs/z13_target_center_dev_20261018/final_verification.json: проверены287замороженныхисточников, 35видео, 105заданийвариант×видео, 9314совпадающихстрок, воспроизведены парные статистики и таблицы отбора. Контроллер завершился без ошибки и превышения срока (controller_result.json). Локальный наблюдатель записал результат и выполнил проверки с кодом0, chat_calls=0 (postprocess_result.json).\n\n'
    entry+='*Ограничение.* '+note+' Прежние поля карточки сохранены в revisions.\n\n*Решение.* Улучшение локализации переносится, а действующая оценка уверенности остаётся ограничением применения. Новые обучение, GPU, HOTA и отложенные видео не запускались.\n'
    path.write_text(plan.replace(anchor,anchor+entry,1),encoding='utf-8')
    marker.write_text(json.dumps(dict(recorded_at=now.isoformat())),encoding='utf-8')


if __name__=='__main__':main()
