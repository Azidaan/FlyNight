"""Recover only JSON summary from completed coverage rows; no geometry or model rerun."""
from collections import defaultdict,Counter
from research_identity_assignment_run_20261108.common import *
from research_identity_assignment_run_20261108.coverage_followup import summarize


def main():
    verify_phase('coverage_phase.json');rows=read(RUN/'coverage_rows.json');assert len(rows)==1991 and len({r['uid'] for r in rows})==1991
    old={r['uid']:r for r in read(ROOT/'runs/spatial_identity_maps_20261107/bottleneck_rows.json')}
    groups=defaultdict(list);flags=defaultdict(Counter);missing=Counter()
    for r in rows:
        split=r['split'];primary='primary' if r['primary_known'] else 'shadow'
        if 'reference' in r:groups[split+'|reference'].append(r['reference'])
        for candidate in r['candidates']:groups[split+'|'+primary+'|'+candidate['role']].append(candidate)
        if r['current_GT_target'] is None:missing[split]+=1
        else:groups[split+'|'+primary+'|GT_target'].append(r['current_GT_target'])
        if r['uid'] in old:
            p=old[r['uid']];c=flags[split];idx=p['true_candidate'];c['pairs']+=1;c['reference_clipped']+=int(p['reference_field_clipped'])
            c['any_candidate_clipped']+=int(any(p['current_field_clipped']));c['true_candidate_clipped']+=int(p['current_field_clipped'][idx]);c['foreign_candidate_clipped']+=int(p['current_field_clipped'][1-idx])
    report=dict(passed=True,queries=len(rows),old_clipping_flags_reused={k:dict(v) for k,v in flags.items()},
        summaries={k:summarize(v) for k,v in groups.items()},GT_target_bbox_unavailable=dict(missing),rows_sha256=sha(RUN/'coverage_rows.json'),
        recovery=dict(failed_summary_preserved='coverage_followup.failed_partial.json',reason='NumPy bool summation produced int64 in the summary; decoded completed JSON rows provide native Python numeric types.',
            new_geometry_evaluations=0,new_core_evaluations=0,raw_rows_reused=True,
            prior_geometry_checks='The completed worker passed100fixed-box checks against OpenCV polygon area and brute receptor integration before saving all1991rows. Summary serialization failed afterwards.'),
        causal_limit='Area coverage does not establish recognition benefit. Whole-frame lamina preprocessing and recurrent receptive fields prevent interpreting a clipped hull of centers as complete loss of information. No larger field or input intervention was tested.',
        runtime_leakage='All GT geometry belongs only to the evaluator; no tracker consumes these rows or labels.')
    write(RUN/'coverage_followup_recovery1.json',report,new=True)


if __name__=='__main__':main()
