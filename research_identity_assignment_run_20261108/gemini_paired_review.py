"""Read existing per-query outcomes to check claims made from marginal totals."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from collections import Counter
from research_identity_assignment_run_20261108.common import *


def main():
    paths={a:RUN/'outcomes'/('evaluate26_'+a+'.json') for a in ['B0','G','G_FLYVIS']}
    rows={a:{r['uid']:r for r in read(p)['rows'] if r['primary_known']} for a,p in paths.items()}
    assert all(len(r)==641 for r in rows.values()) and set(rows['G'])==set(rows['G_FLYVIS'])==set(rows['B0'])
    transitions=Counter();changed=[];assigned_sets={a:set() for a in rows};correct_sets={a:set() for a in rows};wrong_sets={a:set() for a in rows}
    for uid,g in rows['G'].items():
        f=rows['G_FLYVIS'][uid];b=rows['B0'][uid]
        transitions[g['outcome']['kind']+' -> '+f['outcome']['kind']]+=1
        for a in rows:
            k=rows[a][uid]['outcome']['kind']
            if k in ['correct','wrong']:assigned_sets[a].add(uid)
            if k=='correct':correct_sets[a].add(uid)
            if k=='wrong':wrong_sets[a].add(uid)
        if g['outcome']!=f['outcome']:
            changed.append(dict(uid=uid,video=g['video'],frame=g['frame'],tid=g['tid'],baseline=b['outcome'],geometry=g['outcome'],flyvis=f['outcome'],
                geometric_candidates=g['geometric_candidates'],geometry_duplicate_events=g['duplicate_events'],flyvis_duplicate_events=f['duplicate_events']))
    result=dict(passed=True,registered=datetime.now(MSK).isoformat(),sources={str(p.relative_to(ROOT)):sha(p) for p in paths.values()},
        transitions=dict(transitions),changes=changed,geometry_only_assignments=sorted(assigned_sets['G']-assigned_sets['G_FLYVIS']),
        flyvis_only_assignments=sorted(assigned_sets['G_FLYVIS']-assigned_sets['G']),
        lost_correct=sorted(correct_sets['G']-correct_sets['G_FLYVIS']),gained_correct=sorted(correct_sets['G_FLYVIS']-correct_sets['G']),
        removed_wrong=sorted(wrong_sets['G']-wrong_sets['G_FLYVIS']),added_wrong=sorted(wrong_sets['G_FLYVIS']-wrong_sets['G']),
        method='Paired UID ledger only; no model, tracker, old cost or metric rerun.',
        caveat='Conditional one-frame archived-state queries. These transitions do not by themselves identify a visual or age mechanism.')
    write(RUN/'gemini_paired_review.json',result,new=True)
    print(json.dumps({k:v for k,v in result.items() if k not in ['changes','sources']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
