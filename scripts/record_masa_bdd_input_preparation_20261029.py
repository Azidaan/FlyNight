"""Append the verified input-contract correction; preserve all old checks."""
from pathlib import Path
import json
from datetime import datetime, timezone, timedelta
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/continuation_20261028'
OUT=RUN/'reid_candidates_preparation'

def main():
    report=json.loads((OUT/'bdd_input_check.json').read_text(encoding='utf-8'))
    assert report['passed'] and not report['hota_measured']
    queue=json.loads((RUN/'queue_status.json').read_text(encoding='utf-8'))
    assert queue['job']=='125_history_author_audit.json', 'Avoid catalogue checks'
    stamp=datetime.now(timezone(timedelta(hours=3))).strftime('%d.%m.%Y, %H:%M MSK')
    text=f'''**{stamp}** — MASA-Detic: уточнён официальный размер входа BDD

Постановка. Проверить фактическую подготовку изображений перед переносом признаков. В сохранённом официальном configs/datasets/bdd/bdd_dataset.py указан Resize(1333,800), keep_ratio=True; прежняя техническая проверка использовала общий путь MASA-R50 с длинной стороной1024. Поэтому она подтверждала загрузку/вычисление, но не официальный размер входа. Источник сохранён в reid_candidates_preparation/masa/reference_bdd_dataset.py, SHAвbdd_input_check.json.

Получено. Отдельная tracking/masa_detic_bdd_input.py реализует границы1333/800, положительное округление int(x+.5), реальные отдельные коэффициенты ширины/высоты для рамок, RGB/нормировку/дополнение32. Исторический путь1024непереписан. ДвеCPUпроверки геометрии/каналов/дополнения прошли за3,28с. Тот же реальныйtrain-кадр с7ROIдал7×256конечных признаков при входе1×3×768×1344 за14,547сCPU: reid_candidates_preparation/bdd_input_check.json. GTнечитался/CUDAнеинициализирована, прежний real_input_check.json сохранён.

Вывод. Локализовано различие подготовки входа до функции preprocess и конфигурации Resize; это исполнение кандидата Т10, не доказанная причина прежних научных результатов. Полное численное соответствие официальной цепочке признаков ещё не подтверждено. HOTAнеизмерена, научныйвердиктnull, Т10красная. Следующее действие — проверяемый путь признаков и отдельный согласованный протокол трекера; готовые старые головы с новым распределением признаков не переиспользовать автоматически. Текущая матрица/Б2сохраняют приоритет, новыхGPUзаданийнет.

'''
    marker='## Журнал изменений'
    plan=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md'
    body=plan.read_text(encoding='utf-8');assert body.count(marker)==1
    assert 'MASA-Detic: уточнён официальный размер входа BDD' not in body
    plan.write_text(body.replace(marker,marker+'\n\n'+text,1),encoding='utf-8')
    with (RUN/'HANDOFF.md').open('a',encoding='utf-8') as f:f.write('\n\n'+text)
    with (OUT/'ПОДГОТОВКА.md').open('a',encoding='utf-8') as f:f.write('\n\n## 7. Уточнение подготовки BDD\n\n'+text)
    print(stamp)

if __name__=='__main__':main()
