"""Record completed tree diagnostic and the independently bounded next pilot."""
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    source='runs/z13_tree_quality_probe_20261015/report.json'
    rep=json.loads((ROOT/source).read_text(encoding='utf-8'))
    path=ROOT/'experiments/project_history.json'
    catalog=json.loads(path.read_text(encoding='utf-8'))
    card=next(x for x in catalog['experiments'] if x['id']=='z13_tree_quality_probe_v1')
    assert card['state']=='planned'
    card.update(state='completed_negative',done='2026-10-15',
        obtained=f"{source}: arms.D.models: AP neural=0,1767363419, tree=0,1605320747, histgb=0,1581383858. "
            "diffs.D_histgb-D_neural: −0,0185979561 [−0,0324605059; −0,0046761213], парный бутстрэп по 26 видео. "
            "Максимум полезности при >=50: neural 0,2871287129, tree 0,2692307692, histgb 0,2250803859; tau=null у всех. "
            "Из 8326 естественных строк известны 5822, неизвестны 2504. Критерии false/true/false; sources и data_sha256 сохранены.",
        interpretation='Проверенная замена оценщика на деревья не помогает D на этих признаках и сохранённых центрах. '
            'AP D−B положительна, но цели успеха у вариантов различны из-за разных центров; это не чистое доказательство добавочной информации FlyVis. '
            'Train26 уже использованы в разработке, интервалы описательные; причина общего провала локализации не установлена.',
        decision='Не запускать HOTA или перебор деревьев. По разрешению автора продолжить следующий кандидат плана: пространственную цель обучения считывателя, отдельный парный пилот до 20:00 MSK.')
    for i,a in enumerate(card['acceptance']):
        a.update(met=rep['criteria'][i],evidence=f'{source}: criteria[{i}]')
    card['evidence'].extend([dict(kind='json',ref=source,claim='Полная CPU-проба B/C/D, парный разброс, хеши'),
        dict(kind='file',ref='research_flyvis_decision_trees_2026-10-15/СВОДКА.md',claim='Первичные работы, границы переноса, исследование и продолжение')])
    launch=ROOT/'runs/z13_distribution_pilot_20261015/controller.json'
    if launch.exists():
        other=next(x for x in catalog['experiments'] if x['id']=='z13_distribution_pilot_v1')
        other['state']='running'
        other['evidence'].append(dict(kind='json',ref=str(launch.relative_to(ROOT)),claim='Один собственный процесс и независимый предел 20:00'))
    path.write_text(json.dumps(catalog,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    t=datetime.now(timezone(timedelta(hours=3)))
    entry=f"""**{t:%d.%m.%Y, %H:%M} MSK** — деревья для оценки качества З13 проверены, реализация не прошла

*Постановка.* Одна CPU-проба `z13_tree_quality_probe_v1`: прежние центры и 12 признаков, обучение на внеобучающих прогнозах train104, проверка на прежних train26; B/C/D. Основной вариант HistGB, малое дерево поясняющее, настройки до расчёта.

*Получено.* `runs/z13_tree_quality_probe_20261015/report.json`: D AP нейросети 0,176736, дерева 0,160532, HistGB 0,158138. HistGB−нейросеть −0,018598 [−0,032461; −0,004676], 2000 повторов по 26 видео. Максимумы доли полезных при минимум 50: 28,71%, 26,92%, 22,51%; прежний допуск 90% не пройден. Неизвестные 2504 случая не превращались в отрицательные, оценка по 5822 известным. Все критерии вместе не выполнены. AP D и B относятся к разным меткам полезности их центров, поэтому межвариантное сравнение не изолирует вклад мухи.

*Вывод.* Этот оценщик на деревьях не оправдывает дорогой прогон. Простота вопросов дерева не гарантирует качество; HistGB уже использовался в прежних фильтрах. Исследование первоисточников, кодовая опора и ранжирование следующих узлов — `research_flyvis_decision_trees_2026-10-15/СВОДКА.md`. Новой HOTA нет, процент к стенду и доля оракула неизвестны.

*Решение.* Деревья не встраиваются в стенд, сетка настроек не запускается. По текущему разрешению автора начат ограниченный парный пилот пространственной цели `z13_distribution_pilot_v1`, ранее следующий кандидат обзора PrDiMP. Четыре CPU-проверки функции цели прошли: неизвестность, отсутствие, геометрия и близость ошибки. Запуск и предел — `runs/z13_distribution_pilot_20261015/controller.json`; один процесс GPU, сохранение до 19:55, завершение собственного процесса до 20:00. Результат пилота ещё не известен. Существующий `.git/index.lock` не удалялся; новые записи пока локальны.

"""
    plan=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md'; text=plan.read_text(encoding='utf-8')
    marker='## Журнал изменений\n\n'; assert text.count(marker)==1
    plan.write_text(text.replace(marker,marker+entry,1),encoding='utf-8')

if __name__=='__main__': main()
