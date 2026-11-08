"""Correct only evaluator-side candidate labels; reuse all completed inventory geometry."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from collections import Counter,defaultdict
from research_identity_assignment_run_20261108.common import *


def main():
    path=RUN/'singleton_inventory_fit20.json';old=read(path);rows=old['rows'];labels={};sources={};groups=defaultdict(Counter);checks=0
    for row in rows:
        video=row['video']
        if video not in labels:
            source=ROOT/f'runs/z13_temporal_pair_readiness_20261025/supervision/{video}.json'
            labels[video]={r['uid']:r for r in read(source)['rows']};sources[source.relative_to(ROOT).as_posix()]=sha(source)
        label=labels[video][row['uid']];index=row['candidate_high_ids'].index(row['candidate_high_index'])
        target=label['target_identity'];current=label['current_identity'][index]
        role='unknown' if target is None or current is None else ('target' if target==current else 'foreign')
        row['candidate_role']=role
        if row['primary_known']:assert role!='unknown'
        group=('primary' if row['primary_known'] else 'shadow')+'|'+row['group'];c=groups[group]
        c['N']+=1;c['candidate_'+role]+=1;c['native_cost_admissible']+=int(row['native_cost']<=.8)
        for arm,outcome in row['outcomes'].items():
            c[arm+'_'+outcome['kind']]+=1;c[arm+'_J']+=outcome['J']
            if outcome['raw_id']==row['raw_id'] and outcome['kind'] in ['correct','wrong']:
                assert (outcome['kind']=='correct')==(role=='target');checks+=1
        if row['primary_known']:
            c['G_deltaJ']+=row['outcomes']['G']['J']-row['outcomes']['B0']['J']
            c['F_deltaJ']+=row['outcomes']['G_FLYVIS']['J']-row['outcomes']['B0']['J']
    result=dict(old,rows=rows,groups={k:dict(v) for k,v in groups.items()},
        evaluator_label_sources=sources,completed_inventory_sha256=sha(path),
        recovery='The old inventory incorrectly treated target_side as a candidate ordinal. This recovery reads actual current_identity and target_identity; geometry, ownership, costs and outcomes are reused exactly.',
        independent_label_binding_checks=checks)
    write(RUN/'singleton_inventory_fit20_recovery1.json',result,new=True)
    print(json.dumps({k:v for k,v in result.items() if k not in ['rows','sources','evaluator_label_sources']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
