"""Record the verified service repair without changing scientific results."""
from pathlib import Path
import sys,json,subprocess
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import z13_candidates_night as n
OUT=n.OUT

def main():
    marker=OUT/'recovery2_launch_recorded.json'
    assert not marker.exists()
    proof=n.read(OUT/'size_recovery_diagnostic.json')
    assert proof['finished'] and proof['shared_exact'] and proof['separate_pass_mismatch']
    protocol=n.read(OUT/'recovery2_protocol.json')
    for name,digest in protocol['preserved'].items(): assert n.sha(OUT/name)==digest
    # Use the same established recorder, explicitly bound to this attempt's result.
    original=ROOT/'scripts/z13_record_resume_20261016.py'
    target=ROOT/'scripts/z13_record_recovery2_20261016.py'
    assert not target.exists()
    source=original.read_text(encoding='utf-8')
    source=source.replace('resume_controller_result.json','recovery2_controller_result.json')
    source=source.replace('resume_finish_recorded.json','recovery2_finish_recorded.json')
    source=source.replace("    finish='--finish' in sys.argv", "    assert '--finish' in sys.argv, 'Only final recording is supported'\n    finish=True")
    target.write_text(source,encoding='utf-8')
    now=n.now(); commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    catpath=ROOT/'experiments/project_history.json'; catalog=n.read(catpath)
    for mode in ('foreground','size','adaptive'):
        card=next(c for c in catalog['experiments'] if c['id']=='z13_candidate_'+mode+'_v1')
        card.setdefault('evidence',[]).append(dict(kind='json',ref='runs/z13_candidates_20261016/recovery2_protocol.json',claim='Техническое восстановление проверки одинакового считывания; исходные веса, протокол и прежние результаты сохранены'))
        old=card.get('obtained','')
        card.setdefault('revisions',[]).append(dict(date=now.date().isoformat(),commit=commit,field='obtained',old=old,reason='Записана новая попытка после технической ошибки проверки; научного итога ещё нет'))
        card['obtained']='Продолжение после проверенного служебного исправления: runs/z13_candidates_20261016/recovery2_protocol.json. Семь готовых заданий сохранены с проверкой хешей; обучение C/size завершено ранее и не повторяется. Полного набора научных сравнений пока нет.'
    catpath.write_text(json.dumps(catalog,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    plan=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md'; text=plan.read_text(encoding='utf-8'); anchor='## Журнал изменений\n\n'; assert text.count(anchor)==1
    entry=f'''**{now:%d.%m.%Y, %H:%M} MSK** — восстановление проверки размеров на одинаковой активности

Прежняя попытка остановилась в 02:22 на C/size после обучения 500 шагов; сохранены семь готовых заданий и промежуточные предсказания (runs/z13_candidates_20261016/resume_controller_result.json, recovery2_protocol.json). Ошибка обнаружена при следующем срабатывании наблюдения около 02:53; фактическое расписание автоматизации оказалось 40 минут, хотя в инструкции стояли две. Расписание исправлено через инструмент автоматизации на две минуты, но своевременное срабатывание приложения этим не гарантируется.

Диагностика runs/z13_candidates_20261016/size_recovery_diagnostic.json: на 11 пакетах одинаковая активность дала точно одинаковые исходные и новые логиты; повторный независимый проход ядра на пакете 944 дал максимальную разность 0,0000209808 и не прошёл прежний допуск. Разность активности — до 0,00000143051. Это проверка исполнения, не эффективности. Сохранённые параметры исходного считывателя совпали с исходной моделью. Конкретная низкоуровневая операция, вызывающая численный разброс, не локализована.

Исправлен только диагностический родительский проход: он получает ту же активность ядра; научный проход и его результат не заменяются. Прежний допуск сохранён, добавлено требование точного равенства. Замороженные исходники не менялись. Служебные файлы и хеши — recovery2_protocol.json, текущий процесс — recovery2_controller.json, старые ошибки сохранены. Возобновление с контрольной точки, без повторения завершённого обучения. Независимое сохранение 07:35 и окончание до 07:40 сохранены. Консоль переведена на текущую попытку.

HOTA не измерялась; процент к стенду и доля оракульного запаса для технической проверки не определены.

'''
    plan.write_text(text.replace(anchor,anchor+entry,1),encoding='utf-8')
    n.write(marker,dict(at=now.isoformat(),record_script_sha256=n.sha(target),preserved_verified=True))

if __name__=='__main__': main()
