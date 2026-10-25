"""Apply author's explicit deferral without erasing the measured negative outcome."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,now
from scripts.z13_margin_rule_audit_20261025 import revise
from scripts.z13_spatial_set_readout_20261025 import journal,handoff
OUT=ROOT/'runs/z13_spatial_set_readout_20261025/recovery2/training/result_audit/deferred_20261025'
NOTE='dissertation/architecture/Пространственное_считывание_отложено_2026-10-25.md'


def main():
    assert not (OUT/'decision.json').exists()
    cat=read(ROOT/'experiments/project_history.json');card=next(x for x in cat['experiments'] if x['id']=='z13_spatial_set_readout_v1')
    assert card['state']=='completed_negative'
    reason='Прямое решение автора25.10.2026: описать проблему/сделанные проверки и отложить данный модуль; прежние итоговые поля сохранены в revisions.'
    revise(card,'summary','На25.10.2026 пространственное считывание отдельных колонок не удалось успешно реализовать: критерий75моделей не пройден. Модуль отложен по решению автора.',reason)
    revise(card,'decision','ОТЛОЖЕНО25.10.2026 по прямому решению автора. Не продолжать разработку этого модуля, не запускать15заключительных моделей, новую26оценку илиHOTA этойверсии. Не подбирать другой порог/сид/длину по данному исходу. Сохранить все исходники, результаты и служебные ошибки. Возврат только по новому решению автора с отдельной постановкой, отличающейся от уже проверенной. Проблема, история попыток и ограничения: '+NOTE+'. НаправлениеFlyVis и чужие подтверждённые узлы этим решением не закрываются.',reason)
    card['deferred']=dict(date='2026-10-25',by='author',module='spatial column activity-position readout',resume_requires='new author decision and distinct preregistered proposal',source=NOTE)
    card['evidence'].append(dict(kind='file',ref=NOTE,claim='Суть проблемы, что сделано, все5сравнений и решение отложить'))
    write(ROOT/'experiments/project_history.json',cat)
    write(OUT/'decision.json',dict(at=now().isoformat(),author_request='Записать суть проблемы, сделанные исправления; вdrawio/графе неуспех на25.10.2026 и решение отложить модуль.',experiment=card['id'],state='completed_negative',deferred=True,source=NOTE,scientific_results_changed=False))
    journal('решение автора: пространственный модуль отложен','*Проблема.* Выбор реальной области и отказ от вредного вмешательства не дали подтверждённой пользы; проверялась потеря расположения активности при усреднении.\n\n*Сделано.* История описания цели, обучения полной рамки, явного отказа и75моделей нового пространственного считывания собрана в '+NOTE+'. Разные метрики/выборки не смешиваются.\n\n*Решение автора.* На25.10.2026 данная идея не была успешно реализована, модуль отложен на потом. Состояние completed_negative сохраняет факт завершённой проверки; decision/deferred фиксируют отложенную разработку. Возврат только по новому решению автора. Граф и draw.io показывают жёлтую отложенную ветвь.\n\n*Уточнение подписи.* Вектор цели —68коэффициентов наблюдаемой модели, а не простое среднее активности. Источники: scripts/z13_region_assignment_20261025.py и tracking/z13_filter_descriptor.py. Научные результаты/протоколы не менялись. Старые тексты карточки сохранены в revisions.')
    handoff('По прямому решению автора25.10.2026 модуль z13_spatial_set_readout_v1 ОТЛОЖЕН: не удалось успешно реализовать на25.10.2026. НЕпродолжать эту ветвь или её15заключительных моделей/26/HOTA. Отменена прежняя рекомендация продолжить данный модуль через пространственное сопоставление без нового решения автора. Все75моделей/1863блока сохранены; научный отрицательный результат независимо сверён. Проблема/история/числа/решение: '+NOTE+'. Каталог completed_negative плюс deferred, прежние поля в revisions. Доделать точную публикацию пометок графа/однолистовойdrawio и проверок. Иные обоснованные ветви общейFlyVis не закрыты, но данную не перезапускать. ОдинGPU, скрытыеокна,57чужихправок/.gitignore сохранить;меньшиеокна не продлевать.')


if __name__=='__main__':main()
