"""Correct overbroad wording: deferred does not mean banned or closed."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,now
from scripts.z13_margin_rule_audit_20261025 import revise
from scripts.z13_spatial_set_readout_20261025 import journal,handoff
OUT=ROOT/'runs/z13_spatial_set_readout_20261025/recovery2/training/result_audit/deferred_clarified_20261025'


def main():
    assert not (OUT/'decision.json').exists()
    cp=ROOT/'experiments/project_history.json';cat=read(cp);card=next(x for x in cat['experiments'] if x['id']=='z13_spatial_set_readout_v1')
    previous=card.get('deferred',{})
    reason='Автор уточнил25.10.2026: сказано отложить, а не запретить продолжение. Требование отдельного нового разрешения на возврат было лишним добавлением агента.'
    revise(card,'decision','Модуль временно ОТЛОЖЕН25.10.2026 по решению автора; к нему можно вернуться позже. Разработка не запрещена, направление не закрыто, дополнительных ограничений на возврат этим решением не введено. Сохранить исходники, веса, кэши, отрицательный итог75моделей и служебные ошибки для продолжения. Завершённая версия не прошла предварительный критерий, поэтому её15заключительных моделей, новая26оценка иHOTAне запускались. Проблема и выполненные проверки: dissertation/architecture/Пространственное_считывание_отложено_2026-10-25.md. Общие правила экспериментов остаются прежними.',reason)
    card['deferred']=dict(date='2026-10-25',by='author',module='spatial column activity-position readout',meaning='temporarily postponed; may return later; not closed or prohibited',source='dissertation/architecture/Пространственное_считывание_отложено_2026-10-25.md')
    write(cp,cat);write(OUT/'decision.json',dict(at=now().isoformat(),clarification=reason,previous_deferred_metadata=previous,current_deferred_metadata=card['deferred'],scientific_results_changed=False))
    journal('уточнение автора: отложить не означает запретить','*Исправление моей формулировки.* Автор просил временно отложить пространственный модуль. Формулировки агента «не продолжать разработку» и «возврат только по новому решению автора» были чрезмерными и отменены. В карточке/описании теперь: модуль временно отложен, к нему можно вернуться позже; направление не закрыто. Старый decision сохранён в revisions.\n\n*Результат.* Научный итог75моделей и непройденный критерий неизменны.15заключительных моделей/26/HOTA не проводились по критерию данного опыта, а не из-за нового бессрочного запрета. Граф/схема сохраняют пометку «отложено25.10.2026».')
    handoff('УТОЧНЕНИЕ АВТОРА25.10.2026: модуль z13_spatial_set_readout_v1 ВРЕМЕННО ОТЛОЖЕН; к нему можно вернуться позже. Это НЕ запрет дальнейшей разработки и НЕ закрытие направления. Предыдущие формулировки агента «не продолжать разработку», «возврат только по новому решению автора» и запрет возврата отменены. Дополнительного разрешения сверх общих правил не требуется. Научный итог75моделей completed_negative,5ДИвключают0,новой26/HOTAнет;всёсохранено. Текущаяпометка drawio/графа—отложено25.10.2026. Завершить публикацию собственных записей/схемы после зелёных проверок, чужие57правок/.gitignoreсохранить.')


if __name__=='__main__':main()
