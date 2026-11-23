from collections import Counter
from lab import OUT, OLD, read, write, sha, now
import numpy as np


def score_gaps():
    complete=read(OUT/"gap_complete.json")
    selected=read(OUT/"gap_selection.json")["cases"]
    assert complete["inputs_unchanged"] and complete["cases"]==len(selected)
    summary={}
    events=[]
    changed=[]
    matrices=0
    archive={(r["case"],r["frame"]):r for r in read(OLD/"actions.json")["traces"]}
    for c in selected:
        item=read(OUT/"gap_rows"/(c["id"]+".json"))
        assert len(item["rows"])==len(c["frames"])-1
        for mode in ("ordinary","precut"):
            seq=[]
            rival_wrong=0
            for r in item["rows"]:
                scores=np.asarray(r["methods"][mode]["scores"])
                observed=dict(r["methods"][mode]["pairs"])
                ids=r["ids"]
                both={}
                for i,line in enumerate(scores):
                    for j,val in enumerate(line):
                        if val>.5 and all(val>row[j] for k,row in enumerate(scores) if i!=k) and all(val>x for k,x in enumerate(line) if j!=k):
                            both[i]=ids[j]
                assert observed==both
                matrices+=1
                if c["cohort"]=="known" and mode=="ordinary":
                    baseline=archive[c["id"],r["frame"]]
                    assert dict(baseline["actions"][mode])==both and baseline["rids"]==ids
                    assert np.allclose(scores,baseline["scores"][mode],rtol=0,atol=1e-7)
                def status(i):
                    if r["ignored"][i]:
                        return "unknown"
                    if i not in both:
                        return "wait"
                    return "correct" if r["oracle"][i][ids.index(both[i])]>.5 else "wrong"
                seq.append(status(0))
                rival_wrong+=int(status(1)=="wrong")
                if mode=="precut" and (dict(r["methods"]["ordinary"]["pairs"])!=both):
                    changed.append(dict(case=c["id"],cohort=c["cohort"],frame=r["frame"],last=r["last"],
                                        before=r["methods"]["ordinary"]["pairs"],after=r["methods"]["precut"]["pairs"],
                                        verified_target=status(0),truth_overlap=r["oracle"],ids=ids))
            e=dict(case=c["id"],video=c["video"],cohort=c["cohort"],mode=mode,sequence=seq,
                   full=all(x=="wait" for x in seq[:-1]) and seq[-1]=="correct",rival_wrong=rival_wrong)
            events.append(e)
    for cohort in ("known","new_video","new_episode"):
        metrics={}
        wins=losses=0
        for mode in ("ordinary","precut"):
            rows=[e for e in events if e["cohort"]==cohort and e["mode"]==mode]
            metrics[mode]=dict(episodes=len(rows),full=sum(e["full"] for e in rows),
                   return_correct=sum(e["sequence"][-1]=="correct" for e in rows),
                   return_wrong=sum(e["sequence"][-1]=="wrong" for e in rows),
                   gap_wrong=sum(e["sequence"][:-1].count("wrong") for e in rows),
                   gap_unknown=sum(e["sequence"][:-1].count("unknown") for e in rows),
                   wrong_rival=sum(e["rival_wrong"] for e in rows))
        before=[e for e in events if e["cohort"]==cohort and e["mode"]=="ordinary"]
        after={e["case"]:e for e in events if e["cohort"]==cohort and e["mode"]=="precut"}
        delta=np.array([int(after[e["case"]]["full"])-int(e["full"]) for e in before])
        ci=np.quantile(np.random.default_rng(42).choice(delta,(10000,len(delta))).mean(1),[.025,.975]).tolist() if len(delta) else [None,None]
        summary[cohort]=dict(**metrics,wins=int((delta>0).sum()),losses=int((delta<0).sum()),interval=ci)
    write(OUT/"gap_result.json",dict(cohorts=summary,changes=changed,events=events,
          independent_matrices=matrices,ended=now(),source="gap_rows/",scanned=read(OUT/"gap_selection.json")["scanned"]))
    return summary


def audit_transformer():
    q=read(OUT/"transformer_result.json")
    rows=read(OUT/"transformer_rows.json")
    times=read(OUT/"transformer_timings.json")
    assert len(set((r["video"],r["frame"],r["owner"]) for r in rows))==len(rows)
    for method in ("geometry","local","attention"):
        actual=q["stats"][method]
        assert actual["correct_competing"]==sum(r["truth"]>=0 and r["competition"] and r[method]==r["truth"] for r in rows)
        assert actual["false_on_missing"]==sum(r["truth"]<0 and r[method]>=0 for r in rows)
        assert float(np.quantile([r[method] for r in times],.95))==q["timings"][method]["p95"]
    missing=[r for r in rows if r["truth"]<0]
    wrong_pair=Counter(("wait" if r["geometry"]<0 else "assigned","wait" if r["attention"]<0 else "assigned") for r in missing)
    return {"quality_rows":len(rows),"timed_calls":len(times),"unavailable_transitions":{"/".join(k):v for k,v in wrong_pair.items()}}


if __name__=="__main__":
    g=score_gaps()
    t=audit_transformer()
    artifacts=["gap_result.json","transformer_result.json","transformer_rows.json","transformer_timings.json",
               "gap_complete.json","gap_inputs_v2.json","transformer_inputs_v2.json","attention.pt","local.pt"]
    write(OUT/"audit.json",dict(passed=True,transformer=t,sha256={f:sha(OUT/f) for f in artifacts}))
    print(g)
    print(t)
