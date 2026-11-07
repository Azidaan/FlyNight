"""Post-audit exposure check using saved arrays and the documented type graph; no inference."""
from pathlib import Path
import json,hashlib,subprocess
from datetime import datetime,timezone,timedelta
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"runs/flyvis_separate_inputs_20261107"
def read(p):return json.loads(p.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=read(OUT/"protocol.json");assert read(OUT/"controller_verified.json")["all_owned_ended"]
 stats={}
 for name,scope in [("whole_prefix",OUT),("same_history",OUT/"matched_history")]:
  changed=np.zeros(3,dtype=int);maxmap=np.zeros(3);maxscore=np.zeros(3);input_changed=np.zeros(3,dtype=int)
  done=read(scope/"inference_completed.json")
  for c in p["cases"]:
   fn=scope/"raw"/f"{c['uid']}.npz";assert sha(fn)==done["raw_sha256"][fn.name]
   with np.load(fn,allow_pickle=False) as z:
    d=np.max(np.abs(z["maps"][2:]-z["maps"][0]),axis=(1,2));maxmap=np.maximum(maxmap,d);changed+=d>0
    maxscore=np.maximum(maxscore,np.max(abs(z["worker_scores"][2:]-z["worker_scores"][0]),axis=1))
   with np.load(OUT/"inputs"/fn.name,allow_pickle=False) as z:
    x=z["inputs"];input_changed+=np.any(x[1:,-1]!=x[0,-1],axis=1)
  stats[name]=dict(order=["union","separate_a","separate_b"],changed_input_cases=input_changed.tolist(),changed_T4T5_map_cases=changed.tolist(),max_abs_map_delta=maxmap.tolist(),max_abs_worker_score_delta=maxscore.tolist())
 graphfile=ROOT/"ui/flyvis_atlas/dist/data/connectome.js";text=graphfile.read_text(encoding="utf8");g=json.JSONDecoder().raw_decode(text.split("=",1)[1].lstrip())[0]
 distances={t:0 for t in g["inputs"]};queue=list(g["inputs"])
 for node in queue:
  for a,b,*rest in g["edges"]:
   if a==node and b not in distances:distances[b]=distances[a]+1;queue.append(b)
 targets={k:distances[k] for k in g["types"] if k.startswith(("T4","T5"))}
 # Input first changes the photoreceptor's new state; every synaptic hop then needs another explicit update.
 explanation=dict(pairs=172,conditions=stats,type_graph_min_edges_from_input=targets,
  min_explicit_updates={k:v+1 for k,v in targets.items()},observed_internal_steps_per_frame=2,
  basis="FlyBackbone.step uses previous-state activity; original C uses two simultaneous explicit integration steps per frame. Type-graph distances are optimistic lower bounds, sufficient to rule out an effect within two updates.",
  conclusion="Current-only changed input did NOT reach the selected T4/T5 output in this readout step; unchanged decisions cannot assess whether that current input would be useful after propagation.",
  sources_sha256={str(x.relative_to(ROOT)):sha(x) for x in [graphfile,ROOT/"tracking/z13_readout.py",ROOT/"tracking/foveated_eyes.py",Path(__file__)]})
 dest=OUT/"activation_exposure.json";assert not dest.exists();dest.write_text(json.dumps(explanation,ensure_ascii=False,indent=2),encoding="utf8")
 assert all(x==0 for x in stats["same_history"]["max_abs_map_delta"])
 add="\n## Изменился ли исследуемый сигнал\n\nПри изменении только текущего входа все карты T4/T5 во всех172 случаях остались побитно прежними; это не только равенство ранжирования. При обрезании полного префикса карты отдельных кандидатов изменились во всех172 случаях. Источник: activation_exposure.json, conditions.\n\n"
 add+="По схеме типов от входных фоторецепторов до T4 требуется не менее двух синаптических переходов, до T5 — трёх. При явном одновременном обновлении текущий вход сначала меняет фоторецептор; до этих выходов нужны не менее трёх/четырёх внутренних обновлений. В прежнем режиме их два на видеокадр. Это объяснение конкретной реализации, не новая оценка задержки живой мухи.\n\n"
 add+="Поэтому контроль одного текущего кадра не проверил пользу нового сигнала T4/T5: вмешательство ещё не достигло выбранного выхода. Он не подтверждает равенство отдельных и общих видов. Другой режим — обрезание всей предыстории — действительно менял выход, но не дал подтверждённой прибавки. Следующая допустимая постановка должна сначала подтвердить достижение вмешательством считываемого сигнала; автоматически новый прогон не запускается.\n"
 with (OUT/"РЕЗУЛЬТАТ.md").open("a",encoding="utf8") as f:f.write(add)
 catpath=ROOT/"experiments/project_history.json";cat=read(catpath);card=next(c for c in cat["experiments"] if c["id"]==p["id"])
 now=datetime.now(timezone(timedelta(hours=3)));commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
 old=card["interpretation"];card.setdefault("revisions",[]).append(dict(date=now.date().isoformat(),commit=commit,field="interpretation",old=old,reason="По сохранённым массивам проверено достижение входного вмешательства выбранных карт; одинаковый выбор не трактуется как бесполезность текущего сигнала."))
 card["interpretation"]=old+" При изменении одного текущего входа T4/T5 не изменились побитно во всех172 случаях: вмешательство ещё не достигло этих выходов при двух внутренних шагах. Источник: runs/flyvis_separate_inputs_20261107/activation_exposure.json. Это контроль экспозиции, не отрицательный тест потенциальной пользы текущего отдельного вида."
 card["evidence"].append(dict(kind="file",ref="runs/flyvis_separate_inputs_20261107/activation_exposure.json",claim="Все сохранённые карты и минимальная длина пути сигнала по реализации"))
 catpath.write_text(json.dumps(cat,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
 plan=ROOT/"dissertation/ЕДИНЫЙ_ПЛАН.md";txt=plan.read_text(encoding="utf8")
 entry=f"\n**{now:%d.%m.%Y, %H:%M} MSK** — Проверка экспозиции раздельных входов после аудита: в контроле одного текущего входа T4/T5 побитно неизменны на172 парах (activation_exposure.json); в полном префиксе обе отдельные карты меняются во всех172. Два внутренних обновления не позволяют входу пройти минимум два/три синаптических перехода к T4/T5. Нулевые изменения ранжирования текущего контроля не означают бесполезность отдельного сигнала. Прежний текст интерпретации сохранён в revisions; новый опыт не запускался.\n"
 plan.write_text(txt.replace("## Журнал изменений","## Журнал изменений\n"+entry,1),encoding="utf8")
 print(json.dumps(explanation,ensure_ascii=False)[:1500])
if __name__=="__main__":main()
