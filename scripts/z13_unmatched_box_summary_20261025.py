"""Descriptive summary of the frozen guard outcomes; no new policy selection."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import z13_unmatched_box_20261025 as m
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now


def main():
    run=m.RUN;r=read(run/'report.json');p=read(run/'protocol.json')
    assert read(run/'finish_recorded.json')['complete']
    assert read(run/'unconditional_reproduction.json')['all4unconditional_full_arms_exact']
    rows=[x for v in p['evaluate26'] for x in read(run/'videos'/(v+'.json'))['rows']]
    known=[x for x in rows if x['kind']>=0];assert len(rows)==8326 and len(known)==5822
    counts={}
    for arm in m.ARMS:
        a,mode=arm.rsplit('_',1)
        chosen=[x for x in known if (x['proposal_choices'][a]['index']>=0 if mode=='full' else x['proposal_choices'][a][mode+'_admitted'])]
        counts[arm]=dict(known=len(known),unknown=len(rows)-len(known),admitted_known=len(chosen),
            beneficial=sum(bool(x['arms'][arm][1]['beneficial']) for x in known),
            harmful=sum(bool(x['arms'][arm][1]['harmful']) for x in known))
        assert counts[arm]['beneficial']==r['counts'][arm]['beneficial']
        assert counts[arm]['harmful']==r['counts'][arm]['harmful']
    write(run/'final_summary.json',dict(at=now().isoformat(),report_sha256=sha(run/'report.json'),
        preregistered_result=m.passed_modes(r['paired']),counts=counts,
        paired_pp={k:dict(mean=100*v['mean_per_video'],ci975=[100*x for x in v['ci975']]) for k,v in r['paired'].items()},
        scope='Mean-video difference of (beneficial-harmful)/known query situations. Same reused26, no closed-loop HOTA, no independent confirmation. Admission counts descriptive; no new thresholds or post-hoc policy.',
        architecture='These are application guards only. No internal-node milestone; existing one-sheet architecture remains current. No new confirmed_results unless the preregistered positive-control interval is satisfied.'))


if __name__=='__main__':main()
