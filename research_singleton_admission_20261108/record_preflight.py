import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_identity_assignment_run_20261108.common import read,datetime,MSK


def main():
    assert read(ROOT/'runs/native_confirm_policy_20261108/comparison_preflight_correction.json')['passed'];assert read(ROOT/'runs/singleton_admission_20261108/system_problem_update_receipt.json')['passed']
    path=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';text=path.read_text(encoding='utf-8');stamp=datetime.now(MSK).strftime('%d.%m.%Y, %H:%M MSK')
    entry=f'\n\n**{stamp}** — Первая проблема: проверка учёта результатов до нового сравнения и подготовка графа.\n\n*Проверка исполнения.* До появления исходов нового сравнения найдено: сложение `Counter` отбрасывает отрицательные значения J отдельных видео. Замороженный исходник сохранён; отдельная версия суммирует знаковые значения явно, проверка на +3 и −2 даёт +1. Остановлен только ожидавший служебный планировщик, дерево расчёта не затронуто. Данные и готовые исходы не повторялись (`runs/native_confirm_policy_20261108/comparison_preflight_correction.json`).\n\n*Записи.* Полные отрицательные итоги собственной карты и давности внесены в существующие С1/С2 и повтор переноса частного результата в полный трекер. Прежние формулировки сохранены; остальные пять объектов проблем, включая С7, не изменены (`runs/singleton_admission_20261108/system_problem_update_receipt.json`). Подготовлен текст существующей опорной задачи 3; основной граф ещё не пересобран. В текущем направлении первая проблема остаётся открытой, 2–4 отложены.\n';marker='## Журнал изменений';assert text.count(marker)==1;path.write_text(text.replace(marker,marker+entry,1),encoding='utf-8',newline='\n')


if __name__=='__main__':main()
