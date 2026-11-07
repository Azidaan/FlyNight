"""Record the complete eight-group outcome without modifying earlier experiments."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json,hashlib,shutil,sys
import psutil
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"runs/flyvis_state_subgroups_20261107"
sys.path.insert(0,str(ROOT))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    from scripts.flyvis_state_subgroups_20261107 import journal
    p=json.loads((OUT/"protocol.json").read_text(encoding="utf-8"));a=json.loads((OUT/"independent_audit.json").read_text(encoding="utf-8"))
    end=json.loads((OUT/"controller_result.json").read_text());launch=json.loads((OUT/"controller.launch.json").read_text())
    assert a["passed"] and a["cases"]==64 and a["independent_scenes"]==8 and a["protocol_sha256"]==sha(OUT/"protocol.json")
    assert end["exit_code"]==0 and end["all_owned_ended"]
    owned={**end["owned"],str(launch["pid"]):launch["birth"]}
    for pid,birth in owned.items():
        try:assert abs(psutil.Process(int(pid)).create_time()-birth)>.01,f"Original process remains:{pid}"
        except psutil.NoSuchProcess:pass
    now=datetime.now(ZoneInfo("Europe/Moscow"))
    (OUT/"ended_verified.json").write_text(json.dumps(dict(checked_msk=now.isoformat(),all_owned_ended=True,owned=owned),indent=2)+"\n",encoding="utf-8")
    refs=[(OUT/f).relative_to(ROOT).as_posix() for f in ["protocol.json","inference_completed.json","independent_audit.json","controller_result.json","ended_verified.json"]]
    accepted=[g for g,v in a["candidates"].items() if v["passed"]];labels=p["labels"];groups=a["groups"];pairs=a["paired"]
    intro=("Проверены восемь заранее заданных непересекающихся подгрупп прежних 23 обозначений состояния и родительская группа как контроль повторяемости. "
           "Восемь новых для этой серии изображений,64 зависимых условия,24 варианта состояния, разворот и продолжение движения. "
           "Технические проверки и независимый пересчёт пройдены. "+
           ("Совместный критерий сокращения задержки и отсутствия среднего ухудшения контроля прошли: "+", ".join(labels[g] for g in accepted)+"."
            if accepted else "Ни одна подгруппа не прошла одновременно критерии сокращения задержки и сохранения продолжающегося движения."))
    if (OUT/"final_memory_usage.json").exists():
        memory=json.loads((OUT/"final_memory_usage.json").read_text())
        intro+=(" Исполнение восстановлено после ресурсного сбоя: входы и состояния считаются малыми порциями, "
                "начальный разгон хранит только текущее состояние. Полный разгон и продолжения сверены с сохранённым результатом. "
                f"ОС-лимит4ГиБ; пик выделенной памяти рабочего задания {memory['job']['peak_job_bytes']/1024**3:.3f}ГиБ. "
                "Три первоначально готовых условия переиспользованы по SHA; ошибки/прежние контроллеры сохранены.")
    paragraphs=[];brief=[]
    for g in p["groups"]:
        parts=[]
        for ct in ("1.0","0.25"):
            r=groups[ct]["reverse"];cont=groups[ct]["continue"];x=r["erase_"+g];base=r["moving"]
            key=f"{ct}:reverse:erase_{g}:capped_latency";pair=pairs[key]
            corrected=ct=="1.0" and g in p["primary_groups"];interval=pair["ci_family8" if corrected else "ci95"]
            delta=pairs[f"{ct}:continue:erase_{g}:early_epe"];full=pairs[f"{ct}:continue:erase_{g}:epe"]
            parts.append(f"Контраст {ct}: задержка {x['capped_latency']:.4f} против {base['capped_latency']:.4f} кадра; нет определения {x['not_found']}/32 против {base['not_found']}/32. "
                f"Разница {pair['mean']:+.4f}; {'99,375% интервал с поправкой на 8 проверок' if corrected else 'описательный 95% интервал'} [{interval[0]:+.4f}; {interval[1]:+.4f}]. "
                f"Ошибка разворота {x['epe']:.6f} против {base['epe']:.6f}. "
                f"Ошибка продолжающегося движения: первые 6 кадров {cont['erase_'+g]['early_epe']:.6f} против {cont['moving']['early_epe']:.6f} "
                f"({100*(cont['erase_'+g]['early_epe']/cont['moving']['early_epe']-1):+.3f}%; разница95%ДИ [{delta['ci95'][0]:+.6f}; {delta['ci95'][1]:+.6f}]); "
                f"все 24 кадра {cont['erase_'+g]['epe']:.6f} против {cont['moving']['epe']:.6f} "
                f"({100*(cont['erase_'+g]['epe']/cont['moving']['epe']-1):+.3f}%; разница95%ДИ [{full['ci95'][0]:+.6f}; {full['ci95'][1]:+.6f}]). "
                f"Доля верного направления продолжающегося движения {cont['erase_'+g]['correct_sign_fraction']:.6f} против {cont['moving']['correct_sign_fraction']:.6f}. "
                f"Обратный перенос дал задержку {r['insert_'+g]['capped_latency']:.4f} против {r['stationary']['capped_latency']:.4f}.")
        if g in a["candidates"]:
            fail=[k for k,b in a["candidates"][g]["checks"].items() if not b]
            conclusion="Критерий пройден." if not fail else "Критерий не пройден: "+", ".join(fail)+"."
        else:conclusion="Заранее заданный описательный контроль всей родительской группы, вне восьми первичных сравнений."
        paragraphs.append(labels[g]+" ("+", ".join(p["groups"][g])+"). "+" ".join(parts)+" "+conclusion)
        x=groups["1.0"]["reverse"]["erase_"+g];cont=groups["1.0"]["continue"];base=groups["1.0"]["reverse"]["moving"]
        brief.append(f"{labels[g]}: задержка {x['capped_latency']:.4f} вместо {base['capped_latency']:.4f}; ошибка первых 6 кадров продолжающегося движения {100*(cont['erase_'+g]['early_epe']/cont['moving']['early_epe']-1):+.3f}% к исходной")
    parent=pairs["1.0:reverse:erase_parent:capped_latency"]
    interpretation=(f"На новых изображениях родительская группа изменила задержку на {parent['mean']:+.4f} кадра; описательный 95%ДИ [{parent['ci95'][0]:+.4f};{parent['ci95'][1]:+.4f}]. "
        "Это вмешательство в рекуррентное состояние ядра, не только во вход считывателя. Отдельные эффекты подгрупп не складываются. "
        "Обратный перенос помогает установить действие данного состояния, но не доказывает единственный источник его формирования. "
        "Отсутствие сокращения после подмены одной группы не исключает её участия в распределённой динамике. "
        "Искусственное сочетание состояний может нарушить согласованность сети; группы имеют разный размер и величину изменения активности.")
    limits=("Одна штатная модель flow/0000/000, восемь изображений BDD TRAIN, ранее не перечисленных в протоколах и манифестах изображений этой серии FlyVis. "
        "Это64 зависимых условия, а не64независимые сцены; интервалы рассчитаны по8 сценам. Искусственное движение, не естественное видео и не HOTA. "
        "Названия23 элементов относятся к коду: CT1 представлен двумя отделами. Контроль продолжающегося движения измеряет одну функцию памяти. "
        "Отсутствие среднего ухудшения не доказывает эквивалентность или сохранение всей полезной памяти. Стационарный донор создан специально; это не готовая причинная реализация сброса. Доля измеренного потолка не определена.")
    decision=("Сохранить прошедшие подгруппы только как диагностических кандидатов. До внедрения нужен доступный без будущих кадров механизм изменения состояния и отдельная проверка; веса и штатный путь не менять."
              if accepted else "Не внедрять подмену ни одной проверенной подгруппы. Завершить эту ограниченную серию локализации; не подбирать новые объединения или силу сброса по этим же изображениям. Направление FlyVis и память в целом этим не закрываются.")
    literature=("Полезная литература: Shin-ya Takemura и соавторы (2017), The comprehensive connectome of a neural substrate for ON motion detection in Drosophila, "
        "DOI10.7554/eLife.24394, https://elifesciences.org/articles/24394. Различные места входов Mi1/Mi4/Mi9 в T4 послужили основанием разделить эти клетки ДО нового результата. "
        "Это литературное основание проверки, не опубликованное решение нашего запаздывания. Lappalainen и соавторы (2024), Connectome-constrained networks predict neural activity across the fly visual system, "
        "DOI10.1038/s41586-024-07939-3, https://doi.org/10.1038/s41586-024-07939-3: рекуррентное ядро и отдельное считывание определяют место вмешательства. "
        "Точные названия/индексы взяты из установленного официального коннектома, его SHA зафиксирован в протоколе.")
    full=intro+"\n\n"+"\n\n".join(paragraphs)+"\n\n"+interpretation+"\n\n"+limits+"\n\n"+decision+"\n\n"+literature
    (OUT/"РЕЗУЛЬТАТ.md").write_text("# Узкие группы состояния FlyVis на новых изображениях\n\n"+full+"\n\nИсточник всех чисел: "+refs[2]+", groups/paired/candidates/rows.\n",encoding="utf-8")
    path=ROOT/"experiments/project_history.json";cat=json.loads(path.read_text(encoding="utf-8"));card=next(c for c in cat["experiments"] if c["id"]==p["id"]);assert card["state"]=="planned"
    card.update(state="completed_positive" if accepted else "completed_negative",done=now.date().isoformat(),
        obtained=intro+" "+"; ".join(brief)+". Источник: "+refs[2]+", groups; полный отчёт "+(OUT/"РЕЗУЛЬТАТ.md").relative_to(ROOT).as_posix(),
        interpretation=interpretation,decision=decision,cannot_claim=limits,confirmed_results=[])
    card["acceptance"][0].update(met=bool(accepted),evidence=refs[2]+": candidates; технический passed=true")
    card["evidence"] += [dict(kind="json",ref=r,claim="Завершённый опыт и независимая проверка") for r in refs[1:]]
    for filename in ("recovery_manifest.json","batch_equivalence.json","cached_input_equivalence.json","final_memory_usage.json"):
        if (OUT/filename).exists():
            card["evidence"].append(dict(kind="json",ref=(OUT/filename).relative_to(ROOT).as_posix(),claim="Проверенное восстановление исполнения и предел RAM"))
    partials=dict(source=refs[2],source_sha256=sha(OUT/"independent_audit.json"),effects={})
    for g in p["primary_groups"]:
        pair=pairs[f"1.0:reverse:erase_{g}:capped_latency"]
        if pair["ci_family8"][1]<0:
            cont=groups["1.0"]["continue"]
            partials["effects"][g]=dict(mean_per_video=-pair["mean"],ci95=[-pair["ci95"][1],-pair["ci95"][0]],
                ci_corrected=[-pair["ci_family8"][1],-pair["ci_family8"][0]],n=8,baseline=groups["1.0"]["reverse"]["moving"]["capped_latency"],
                early_continuation_error_percent=100*(cont["erase_"+g]["early_epe"]/cont["moving"]["early_epe"]-1),preservation_criterion_met=a["candidates"][g]["passed"])
    partial_path=OUT/"partial_effects.json";partial_path.write_text(json.dumps(partials,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    for g,e in partials["effects"].items():
        card["confirmed_results"].append(dict(id="state_subgroup_"+g+"_delay",title=labels[g]+" — частное сокращение задержки",category="representation",
            metric="flyvis_transition_delay_frames",unit="points",comparison="Исходная задержка минус задержка после подмены этой подгруппы неподвижным донором; пункт равен кадру.",
            subset="Восемь новых для этой серии изображений BDD TRAIN;32 условия обычного контраста, группировка по исходной сцене.",
            conditions="Штатная flow/0000/000 CPU FP64, одинаковый суффикс. Положительна также нижняя граница с поправкой на 8 первичных сравнений.",
            limitations="Диагностическая искусственная подмена, не готовое исправление. Совместный критерий сохранения ответа: "+str(e["preservation_criterion_met"])+
                "; ошибка первых 6 кадров продолжающегося движения "+f"{e['early_continuation_error_percent']:+.3f}% к исходной. Все неблагоприятные исходы в первичном отчёте. Не HOTA.",
            evidence=dict(source=partial_path.relative_to(ROOT).as_posix(),sha256=sha(partial_path),locator=["effects",g])))
    path.write_text(json.dumps(cat,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    problem_path=ROOT/"experiments/system_problems.json";problems=json.loads(problem_path.read_text(encoding="utf-8"));problem=next(x for x in problems["problems"] if x["id"]=="С7")
    fact=card["obtained"]+" "+interpretation+" "+limits+" "+decision+" "+literature
    problem["cases"].append(dict(date=now.date().isoformat(),fact=fact,evidence=refs));problem["cards"].append(card["id"])
    problem["evidence"] += [dict(source=r,sha256=sha(ROOT/r),claim="Узкие подгруппы состояния и полезный ответ на новых изображениях") for r in refs]
    problem_path.write_text(json.dumps(problems,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    journal(full+" Источник чисел: "+refs[2]+".")
    with (ROOT/"FLYVIS_TRACKER_RESEARCH_ROADMAP.md").open("a",encoding="utf-8") as f:f.write("\n\n## "+now.strftime("%d.%m.%Y")+" — подгруппы памяти на новых изображениях\n\n"+fact+"\n")
    from scripts.obsidian_system_problems import load,problem_path as note_path
    from scripts.obsidian_fundamental_overview import build
    from scripts.audit_fundamental_obsidian import audit
    vault=ROOT/"ObsidianFlyVis";stage=OUT/"vault_checked";backup=OUT/"vault_before";assert not stage.exists() and not backup.exists()
    shutil.copytree(vault,stage);shutil.copytree(vault,backup)
    before={f.relative_to(vault).as_posix():sha(f) for f in vault.rglob("*") if f.is_file()}
    name=note_path(next(x for x in load()["problems"] if x["id"]=="С7"))+".md"
    assert build(stage,[f for f in before if f.endswith(".md") and f!=name])==0
    for f in (".obsidian/graph.json",".graph-visible.json",".generated.json"):shutil.copy2(backup/f,stage/f)
    check=audit(stage);assert not check["external_hypothesis_edges"]
    assert set(before)=={f.relative_to(stage).as_posix() for f in stage.rglob("*") if f.is_file()}
    changed=[f for f,h in before.items() if sha(stage/f)!=h];assert changed==[name] and all(sha(vault/f)==h for f,h in before.items())
    shutil.copy2(stage/name,vault/name);result=audit(vault);assert result["hashes"]==check["hashes"]
    result.update(changed_files=changed,deleted_files=[],protected_hypotheses_byte_exact=True)
    (OUT/"graph_audit.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(intro)
if __name__=="__main__":main()
