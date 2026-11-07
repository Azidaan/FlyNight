"""Publish the fully audited input intervention without revising earlier records."""
import json,sys,hashlib,subprocess
from pathlib import Path
from datetime import datetime,timezone,timedelta
import psutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.flyvis_input_separation_20261107 import OUT,IDENT,read,sha,write
def main():
 import numpy as np
 a=read(OUT/"independent_audit.json");p=read(OUT/"protocol.json");c=read(OUT/"recovery1/controller.json")
 assert a["passed"] and c["passed"] and a["pairs"]==172 and a["protocol_sha256"]==sha(OUT/"protocol.json")
 owned=dict(c["owned"]);launch=read(OUT/"recovery1/start_launcher.json");owned[str(launch["pid"])]=launch["birth"]
 inner=read(OUT/"recovery1/launch.json");owned[str(inner["pid"])]=inner["birth"]
 for pid,birth in owned.items():
  try:assert psutil.Process(int(pid)).create_time()!=birth,("owned_still_alive",pid)
  except psutil.NoSuchProcess:pass
 for fn,h in p["source_sha256"].items():assert sha(fn)==h,fn
 recovery=read(OUT/"recovery1/protocol.json")
 assert recovery["old_failure_sha256"]==sha(OUT/"chain_result.json")
 for fn,h in recovery["source_sha256"].items():assert sha(fn)==h,fn
 geometry=dict(cases=172,candidate_boxes=344,partly_outside_original_field=0,unchanged_separate_vs_union=0,neutral_only_sequences=0,empty_pixel_windows=0)
 for case in p["cases"]:
  with np.load(OUT/"inputs"/f"{case['uid']}.npz",allow_pickle=False) as z:
   boxes=z["boxes"];cr=z["crops"][-1];hx=z["inputs"]
   geometry["partly_outside_original_field"]+=int(np.any((boxes[:,:2]<cr[:2])|(boxes[:,2:]>cr[2:]),axis=1).sum())
   geometry["unchanged_separate_vs_union"]+=sum(np.array_equal(hx[k],hx[1]) for k in [2,3])
   geometry["neutral_only_sequences"]+=sum(bool(np.all(hx[k]==.5)) for k in [2,3])
   for crop in z["crops"]:
    W,H=crop[2:]-crop[:2];xx=(np.arange(W)+.5)/W;yy=(np.arange(H)+.5)/H
    geometry["empty_pixel_windows"]+=sum(not (np.any((xx>=w[0,0])&(xx<=w[1,0])) and np.any((yy>=w[0,1])&(yy<=w[1,1]))) for w in z["windows"])
 write(OUT/"input_effects.json",geometry)
 cmp={x["name"]:x for x in a["comparisons"]};conditions={x["name"]:x for x in a["conditions"]}
 fmt=lambda v:f"{v:.3f}".replace(".",",")
 desc=lambda r:f"{fmt(r['delta_pp'])} п.п.; 97,5% ДИ [{fmt(r['ci97_5_pp'][0])}; {fmt(r['ci97_5_pp'][1])}], исправлено {r['gains']}, испорчено {r['losses']}"
 primary=[x for x in a["comparisons"] if x["primary"]]
 state="completed_positive" if a["main"] else ("completed_negative" if all(x["ci97_5_pp"][1]<0 for x in primary) else "inconclusive")
 summary=("Раздельные входы прошли оба критерия." if a["main"] else "Раздельные входы не прошли общий критерий.")
 negatives=[x["name"] for x in primary if x["ci97_5_pp"][1]<0]
 if negatives:summary+=" Подтверждено ухудшение: "+", ".join(negatives)+"."
 result="# Раздельные входы FlyVis при прежнем масштабе\n\n"+summary+"\n\n"
 result+="На тех же172 исторических парах из23 видео использованы полные прежние префиксы кадров, прежняя сетка глаза, нормировка и веса образца цели. Каждое окно шире рамки в1,25 раза. Оно занимает прежнее место в поле; вне окна нейтральная яркость. Маски текущих детекций фиксированы в канонических координатах на всю предысторию; истинная прошлая траектория кандидата не используется.\n\n"
 result+="| Вход | Верных из172 | Доля верных |\n|---|---:|---:|\n"
 for name,title in [("common","Прежняя общая карта"),("union","Оба окна вместе"),("separate","Каждое окно отдельно")]:
  x=conditions[name];result+=f"| {title} | {x['correct']:g} | {fmt(x['accuracy']*100)}% |\n"
 result+="\nОсновные сравнения с поправкой на две проверки: отдельные входы минус общая карта — "+desc(cmp["separate-common"])+". Отдельные минус объединённые окна — "+desc(cmp["separate-union"])+".\n\n"
 result+="Контроль удаления внешнего фона: объединённые окна минус общая карта — "+desc(cmp["union-common"])+". Это описательное сравнение.\n\n"
 result+="Приёмка требует обеих положительных нижних границ97,5%. Проверены все172 сохранённых выхода; баллы независимо пересчитаны из карт. Источник всех чисел — independent_audit.json, поля conditions/comparisons. Вход общей версии восстановлен побитно в исходном float16. Тождественный контроль и сравнение с прежними картами выдержали заранее заданные допуски.\n\n"
 result+=f"Геометрический контроль: из344 кандидатных рамок {geometry['partly_outside_original_field']} частично выходят за прежнее поле глаза; {geometry['unchanged_separate_vs_union']} раздельных входов совпадают с объединённым окном на всём префиксе. Источник: input_effects.json. В этой пробе видимые части объектов остаются на прежних координатах.\n\n"
 result+="Эта проверка относится к ограничению входа перед ядром. Фронтальная обработка lamina остаётся общей; перекрывающаяся машина может оставаться в вырезке. Изменились также границы и фон окна. Не проверен отдельный центрированный вид каждого автомобиля целиком: он потребует контроля смены поля зрения. Результат не устанавливает качество полных траекторий, пользу сегментации или всех способов подачи объектов в FlyVis. Использованы прежние исследовательские видео, это не независимый итоговый тест.\n\n"
 result+="Технический первый отказ размерности BoxEye и его контроллер сохранены. Исправление recovery1 удалило только ось длиной1; ни научные условия, ни допуски не менялись.\n"
 dest=OUT/"РЕЗУЛЬТАТ.md";assert not dest.exists();dest.write_text(result,encoding="utf8")
 now=datetime.now(timezone(timedelta(hours=3)));cat=read("experiments/project_history.json");card=next(x for x in cat["experiments"] if x["id"]==IDENT)
 assert card["state"]=="ready" and not card.get("obtained")
 card.update(state=state,done=now.strftime("%Y-%m-%d"),summary=summary,
  obtained=f"172 пары/23 видео. Общая карта {conditions['common']['correct']:g}/172, объединённые окна {conditions['union']['correct']:g}/172, раздельные {conditions['separate']['correct']:g}/172. Separate-common: {desc(cmp['separate-common'])}. Separate-union: {desc(cmp['separate-union'])}. Источник: {OUT}/independent_audit.json, conditions/comparisons.",
  interpretation="Диагностический эффект ограниченных окон до ядра при прежнем масштабе. Отдельные входы сохраняют перекрывающиеся части объектов и используют исходный считыватель без переобучения.",
  decision="Сохранить полный результат и оба контроля; новый научный опыт требует отдельного решения автора.")
 card["acceptance"][0].update(met=True,evidence=f"{OUT}/inputs_completed.json; {OUT}/independent_audit.json; {OUT}/controller_verified.json")
 card["acceptance"][1].update(met=a["main"],evidence=f"{OUT}/independent_audit.json, comparisons/main")
 card["evidence"].append(dict(kind="file",ref=dest.as_posix(),claim="Полный итог, обе главные проверки, ограничения и прежняя ошибка"))
 write("experiments/project_history.json",cat)
 plan=Path("dissertation/ЕДИНЫЙ_ПЛАН.md");text=plan.read_text(encoding="utf8")
 entry=f"\n**{now:%d.%m.%Y, %H:%M} MSK** — Завершено {IDENT}. {summary} "+card["obtained"]+" Отказ первого исполнителя формы BoxEye сохранён; recovery1 не меняла значения и допуски. Полный аудит и фактический конец всех зарегистрированных PID+birth проверены. Это историческая диагностика входа, не HOTA; следующий опыт не запущен.\n"
 plan.write_text(text.replace("## Журнал изменений","## Журнал изменений\n"+entry,1),encoding="utf8")
 write(OUT/"controller_verified.json",dict(passed=True,all_owned_ended=True,owned=owned,audit_sha256=sha(OUT/"independent_audit.json"),record_source_sha256=sha(Path(__file__)),state=state))
 print(summary,conditions,flush=True)
if __name__=="__main__":main()
