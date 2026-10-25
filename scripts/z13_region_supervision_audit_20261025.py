"""One preregistered descriptive label audit before a semantic-target proposal."""
from pathlib import Path
import sys
import time
from collections import Counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
OUT=ROOT/'runs/z13_region_supervision_audit_20261025'
BASE=ROOT/'runs/z13_region_assignment_20261025/labels'


def main():
    assert not OUT.exists()
    manifest=read(BASE/'report.json');p=read(ROOT/'runs/z13_region_assignment_20261025/protocol.json')
    write(OUT/'protocol.json',dict(at=now().isoformat(),budget_seconds=60,device='cpu',no_training=True,
        question='Does the current utility-only positive label omit a substantially larger set of own-correct nonharmful actions?',
        data='All saved130 region-label files, original known/unknown definition and outcome booleans unchanged; split summaries, no new GT interpretation or model evaluation.',
        decision='A semantic-target trial may be worth one bounded comparison only if fit84 safe-own-correct actions are>=2x beneficial actions and there are>=1000 positive and>=1000 negative known actions. This is a data-availability gate, not efficacy evidence or proven failure cause.',
        counterargument='Correct neutral actions are not useful interventions; adding them can dilute useful positives and increase pointless actions. A proposed marginal semantic likelihood must include unchanged as acceptable when it is already correct. A same-loss utility-only control is mandatory.',
        sources={str((BASE/'report.json').relative_to(ROOT)):sha(BASE/'report.json'),'scripts/z13_assignment_effect_20261024.py':sha(ROOT/'scripts/z13_assignment_effect_20261024.py')}))
    began=time.monotonic();counts={s:Counter() for s in ('fit84','fit20','train26')}
    for video,h in manifest['video_sha256'].items():
        assert time.monotonic()-began<60
        path=BASE/'videos'/(video+'.json');assert sha(path)==h
        split='fit20' if video in p['fit20'] else 'fit84' if video in p['fit84'] else 'train26'
        for row in read(path)['rows']:
            c=counts[split];c['rows']+=1
            if row['kind']<0:c['unknown']+=1;continue
            c['known']+=1;outcomes=row['arms']['regions'];c['baseline_correct']+=bool(outcomes[0]['own_before'])
            assert all(o['own_before']==outcomes[0]['own_before'] for o in outcomes)
            actions=outcomes[1:];c['actions']+=len(actions)
            good=[bool(o['own_after']) and not bool(o['harmful']) for o in actions]
            c['safe_correct_actions']+=sum(good);c['incorrect_or_harmful_actions']+=len(actions)-sum(good)
            c['beneficial_actions']+=sum(bool(o['beneficial']) for o in actions)
            c['safe_correct_queries']+=any(good);c['beneficial_queries']+=any(o['beneficial'] for o in actions)
            c['no_safe_correct_action']+=not any(good)
    c=counts['fit84'];passed=c['safe_correct_actions']>=2*c['beneficial_actions'] and c['safe_correct_actions']>=1000 and c['incorrect_or_harmful_actions']>=1000
    write(OUT/'report.json',dict(at=now().isoformat(),complete=True,counts={s:dict(c) for s,c in counts.items()},availability_gate=passed,
        seconds=time.monotonic()-began,claim='Counts describe supervision availability only. They do not establish the cause, show a representation improvement, or authorize HOTA.'))
    print({s:dict(c) for s,c in counts.items()});print('availability_gate',passed)


if __name__=='__main__':main()
