"""Подготовить аудит и обучение по уже зарегистрированному протоколу, без расширения окна."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.recovery_research_service_20261027 import RUN,read,write,sha,now
from scripts.b123_safe_journal_20261027 import journal

def main():
    original=read(RUN/'pixel_box_dataset/protocol.json')
    for job,folder,script,budget,extra in [
        ('085_pixel_box_audit','pixel_box_audit','scripts/audit_pixel_box_dataset_20261027.py',600,{}),
        ('090_pixel_box_learning','pixel_box_learning','scripts/train_pixel_box_refinement_20261027.py',7200,original['training_plan'])]:
        target=RUN/folder;assert not target.exists()
        p=dict(**extra);p.update(parent_experiment='pixel_box_refinement_v1',created=now(),budget_seconds=budget,
            author_stop=read(RUN/'authorization.json')['save_at'],original_protocol='runs/recovery_research_20261027/pixel_box_dataset/protocol.json',
            sha256={s:sha(ROOT/s) for s in [script,'tracking/pixel_box_refinement.py','runs/recovery_research_20261027/pixel_box_dataset/protocol.json']})
        write(target/'protocol.json',p)
        write(RUN/(job+'_prepared.json'),dict(name=job,folder=target.relative_to(ROOT).as_posix(),script=script,script_sha256=sha(ROOT/script),protocol_sha256=sha(target/'protocol.json')))
    journal('pixel_box_refinement_v1','подготовлены последовательные аудит и обучение геометрии',
        '*Сделано.* Подготовлены085аудитданных(600сCPU)и090обучение(7200с,одинGPU),параметрыизранее записанного pixel_box_dataset/protocol.json.training_plan не изменены. '
        'Аудит пересчитывает все целевые поправки, исключение точных обучающих кадров YOLO, строгое прошлое, хеши и реальные пиксели на выборке. '
        'Обучение само проверяет passed=true и конец аудитора доCUDA. Стоимость первых100реальных шагов не расходуется заново: при положительной цене продолжается тот же первый вариант. '
        'Шесть моделей,15эпох,выбор по dev-регрессии,затем отдельный devHOTAвыборalpha из[0,0.5,1] и полный40val;метрикаошибкинеуспехHOTA. '
        'Источники: pixel_box_audit/protocol.json и pixel_box_learning/protocol.json. Публикация заданий после зелёных проверок, без дублирования очереди.')
    print('Prepared guarded audit and actual geometry training')

if __name__=='__main__':main()
