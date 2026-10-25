"""Record independently audited negative result, preserving the failed controller."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
from scripts.z13_margin_rule_audit_20261025 import revise
from scripts.z13_spatial_set_readout_20261025 import journal,handoff
RUN=ROOT/'runs/z13_spatial_set_readout_20261025/recovery2'
OUT=RUN/'training/result_audit'


def main():
    assert not (OUT/'finish_recorded.json').exists()
    a=read(OUT/'audit.json');assert a['complete'] and a['primary_rows_labels_counts_means_intervals_verified']
    assert sha(RUN/'training/internal_report.json')==a['internal_report_sha256'] and not a['internal_pass']
    write(OUT/'previous_training_status.json',read(RUN/'training/status.json'))
    cat=read(ROOT/'experiments/project_history.json');card=next(x for x in cat['experiments'] if x['id']=='z13_spatial_set_readout_v1')
    source=(RUN/'training/internal_report.json').relative_to(ROOT).as_posix()
    reason='Завершены75моделей; независимая сверка строк, меток, хешей и интервалов; служебная ошибка после сохранения результата сохранена отдельно.'
    fields=dict(summary='Пространственное считывание отдельных колонок: внутренний критерий не пройден;75моделей на104видео, новых26/HOTAнет.',
        obtained=source+' → paired: '+str(a['paired'])+'. Все75моделей по1150шагов; audit.json в training/result_audit независимо проверил строки/метки/пороги/интервалы. Старый controller_result сохраняет exit1: после сохранения сравнений сработала оставшаяся проверка CPU-only; обучение не повторялось.',
        interpretation='Эта конечная модель пространственного считывания не дала подтверждённой чистой пользы или преимущества над равными D_mean/C/R1/геометрией. Все5интервалов включают0; это не доказательство нулевого истинного эффекта или отсутствия информации в активности. Оценка на повторно использованных104видео; интервалы условны на разбиении. Образец по-прежнему сжат до68признаков, поэтому полное пространственное сопоставление образца и области не проверено.',
        decision='Не принимать эту реализацию и не запускать15итоговых моделей, новую26оценку илиHOTA. Не менять порог/сид/длину обучения по этому итогу. Следующий кандидат по заранее записанному ранжированию — пространственное сопоставление образца и области; сначала дешёвая проверка доступности исходных наблюдений и отличий от уже испытанного внимания. Готового исполнителя следующего обучения нет. Служебный сбой завершения не скрывать и не повторять75готовыхмоделей.',
        cannot_claim='Нельзя заявлять улучшениеHOTA, исправление отказа, преимущество D надC/R1/контролем безмухи, доказанное отсутствие пространственной информации или итог по всей линииFlyVis. Конечные16скрытых признаков,1150шагов и сжатый образец ограничивают проверенный механизм. ОтдельнаяGPUреализация была допущена по стоимости;5одинаковыхблоковCPU и70GPU, старые ошибки и исходники сохранены.')
    for k,v in fields.items():revise(card,k,v,reason)
    card.update(state='completed_negative',done=now().strftime('%Y-%m-%d'))
    card['acceptance'][0].update(met=True,evidence=(OUT/'audit.json').relative_to(ROOT).as_posix()+';features/report.json complete;75точных сохранённых повторов предсказаний, хеши/метки/разбиения проверены')
    card['acceptance'][1].update(met=False,evidence=source+': mean<.02;все5основныхДИвключают0')
    card['acceptance'][2].update(met=None,evidence='Заключительная ступень не запускалась после непройденного внутреннего критерия;0новых26оценок')
    card.setdefault('evidence',[]).extend([dict(kind='file',ref=source,claim='Все5сравнений75моделей'),dict(kind='file',ref=(OUT/'audit.json').relative_to(ROOT).as_posix(),claim='Независимая сверка и сохранённая служебная ошибка')])
    write(ROOT/'experiments/project_history.json',cat)
    write(OUT/'confirmed_review.json',dict(eligible_new_confirmed_results=0,reason='All five paired intervals include zero.'))
    write(OUT/'finish_recorded.json',dict(at=now().isoformat(),complete=True,models=75,internal_pass=False,final_evaluated=False,hota_measured=False,original_worker_exit=1,completion='Saved scientific result independently verified; original failed controller unchanged.'))
    write(RUN/'training/status.json',dict(state='completed_after_independent_audit',updated=now().isoformat(),models_done=75,models_total=75,internal_pass=False,final_evaluated=False,original_worker_exit=1,evidence=(OUT/'audit.json').relative_to(ROOT).as_posix()))
    journal('итог пространственного считывания отдельных колонок','*Получено.* '+fields['obtained']+'\n\n*Вывод.* '+fields['interpretation']+'\n\n*Решение.* '+fields['decision']+'\n\n*Служебная ошибка.* GPU-обёртка сохранила CPU-only assertion после всех моделей. Первичные результаты независимо сверены; старый exit1/traceback не менялись. Неудачный исход не объясняется этой ошибкой завершения.\n\n*Архитектурная веха.* Новая отдельная жёлтая ветвь пространственного считывания; критерий не пройден. HOTA/проценткстенду/доляпотолканеизмерены.')
    handoff('z13_spatial_set_readout_v1 завершён75моделей; полный отрицательный внутренний итог104видео. D-baseline -0,01119п.п.[-0,07302;+0,05251],D-Dmean+0,06859[-0,01073;+0,15065];все5ДИвключают0. Никаких15итоговых/26/HOTA. После сохранения internal_report GPUworker exit1 на старом CPU-only assertion,старый код/лог/контроллер сохранены; result_audit/audit.json независимо сверил все75весов/строки/метки/разбиения/пороги/интервалы. Все свои PIDпоbirth завершены. Не повторять1863блока или75моделей. Следующийкандидат по ранжированию—пространственное сопоставление образца и области; сначала доступность образца/отличие от старойattention-головы,готового исполнителя нет. Завершить граф/архитектурнуювеху/проверки/точнуюпубликацию результата. Автор оспорил CPU: новыхCPUобучений вместоGPU без обоснования не начинать; ничего не менять в уже завершённом опыте. ОдинGPU,скрытыеокна,общееокно26Oct07:40;57чужихправок/.gitignore сохранить.')


if __name__=='__main__':main()
