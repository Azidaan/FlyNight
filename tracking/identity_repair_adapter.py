"""Dormant bounded proposal composer; no labels, tracker or model access."""
import numpy as np
from tracking.appearance_recovery_policy import candidate_groups,overlap,confirm_recovery,observable_episode_tids
from tracking.correspondence_repair_guard import choose_correspondence

VARIANTS=('baseline','universal_census_negative_control','guarded_census_appearance_low_native_domain','targetdecision_without_core_appearance_low_native_domain')

def complete_support(rows,candidates,references,current,target_row,target_raw_id,frozen_pairs):
    groups=candidate_groups(candidates);representatives=[g['representative_index'] for g in groups]
    a=np.asarray(references,float);b=np.asarray(current,float)
    if not(len(rows)==len(a) and len(candidates)==len(b) and np.isfinite(a).all() and np.isfinite(b).all()):
        return dict(supported=None,owned_by_other=True,reason='incomplete_competition',groups=groups)
    a=a/np.linalg.norm(a,axis=1)[:,None];b=b/np.linalg.norm(b,axis=1)[:,None]
    scores=np.clip(a@b[representatives].T,-1,1);i=target_row;row=rows[i]
    selected=[j for j,g in enumerate(groups) if target_raw_id in g['alias_raw_ids']]
    if len(selected)!=1:return dict(supported=False,owned_by_other=False,reason='target_not_in_unique_group',groups=groups)
    j=selected[0];same=np.array([candidates[k]['cls']==row['cls'] for k in representatives]);values=np.where(same,scores[i],-np.inf);best=np.flatnonzero(values==values.max())
    reverse_values=np.where([r['cls']==candidates[representatives[j]]['cls'] for r in rows],scores[:,j],-np.inf);reverse=np.flatnonzero(reverse_values==reverse_values.max())
    proximity=overlap(row['predicted_xyxy'],candidates[representatives[j]]['box']);distance=(1-scores[i,j])/2
    owner=[dict(tid=int(t),raw_id=int(r)) for t,r in frozen_pairs if r in groups[j]['alias_raw_ids'] and t!=row['tid']]
    supported=bool(same[j] and len(best)==1 and int(best[0])==j and len(reverse)==1 and int(reverse[0])==i and distance<=.2 and proximity>=.5)
    return dict(supported=supported,owned_by_other=bool(owner),owners=owner,reason='strict_mutual_fixed_gate' if supported else 'appearance_or_proximity_contradiction',representative_raw_id=groups[j]['representative_raw_id'],alias_raw_ids=groups[j]['alias_raw_ids'],cosine=float(scores[i,j]),appearance_cost=float(distance),proximity_iou=proximity,groups=groups)

def compose(snapshot,rows,candidates,references,current,original_score,census_score,variant):
    """Change only the preselected native target decision; all other votes fixed.

    Target is an existing native proposal, not selected using evaluation labels.
    Safe variants freeze every actual_first pair AND the complete saved assigned
    ownership map. Evaluation fields must not be passed to this pure function.
    """
    assert variant in VARIANTS
    rowids=[r['tid'] for r in rows];rawids=list(snapshot['high_raw_ids']);i=snapshot['target_row'];j=snapshot['target_column'];tid=rowids[i];target_raw=rawids[j]
    assert tid==snapshot['target_tid'] and target_raw==snapshot['target_raw_id']
    first=[(rowids[ii],rawids[jj]) for ii,jj in snapshot['actual_first']];assigned=[(int(t),int(r)) for t,r in snapshot['assigned'].items()];frozen=sorted(set(first+assigned));frozen_tids={t for t,r in frozen}
    record=dict(variant=variant,video=snapshot['video'],frame=snapshot['frame'],base='corrected_cost',edge_updates=[],append_columns=[],freeze='none' if variant in ('baseline','universal_census_negative_control') else 'all',defer_tids=[],alias_groups=[])
    note=dict(original_score=float(original_score),census_score=float(census_score),target_tid=tid,target_raw_id=target_raw)
    if variant=='baseline':return record,dict(**note,action='unchanged',reason='literal_saved_matrix')
    if variant=='universal_census_negative_control':
        cost=np.float32(snapshot['base_cost'][i][j]);cost=np.float32(cost-np.float32(.2)) if census_score>0 else cost
        record['edge_updates']=[dict(tid=tid,raw_id=target_raw,cost=float(cost))]
        return record,dict(**note,action='universal_census_vote',accepted=bool(census_score>0),reason='negative_control_without_ownership_protection')
    appearance=complete_support(rows,candidates,references,current,i,target_raw,frozen);record['alias_groups']=[g['alias_raw_ids'] for g in appearance['groups']]
    if variant=='guarded_census_appearance_low_native_domain':
        choice=choose_correspondence(float(original_score),float(census_score),appearance['supported'],appearance['owned_by_other'])
    else:
        accepted=appearance['supported'] is True and not appearance['owned_by_other'];choice=dict(accepted=accepted,use_repair=False,coordinate=None,selected_score=None,action='appearance_proposal' if accepted else 'defer',reason='targetdecision_without_core_fixed_appearance_ownership')
    # Native assignments remain authoritative even when a shadow gate disagrees.
    if tid not in frozen_tids:
        cost=np.float32(snapshot['base_cost'][i][j]);cost=np.float32(cost-np.float32(.2)) if choice['accepted'] else cost
        record['edge_updates'].append(dict(tid=tid,raw_id=target_raw,cost=float(cost)))
    eligible=observable_episode_tids(snapshot['native_proposals'])
    record['native_eligible_tids']=sorted(eligible)
    recovery=confirm_recovery(rows,candidates,np.asarray(references).copy(),np.asarray(current).copy(),frozen,eligible_tids=eligible,competition_complete=True,enabled=True)
    assert all(next(r['state'] for r in rows if r['tid']==p['tid'])==2 for p in recovery['proposals'])
    for proposal in recovery['proposals']:
        assert proposal['tid'] not in frozen_tids and proposal['raw_id'] not in rawids
        costs=[1e6]*len(rows);costs[rowids.index(proposal['tid'])]=proposal['appearance_cost'];record['append_columns'].append(dict(raw_id=proposal['raw_id'],costs=costs))
    low_target=next((p for p in recovery['proposals'] if p['tid']==tid),None)
    if tid not in frozen_tids and not choice['accepted']:
        # Reject all original high edges for a deferred target; a separately
        # supported new low observation may still recover it in the same solve.
        record['edge_updates']=[u for u in record['edge_updates'] if u['tid']!=tid]
        record['edge_updates'].extend(dict(tid=tid,raw_id=rid,cost=1e6) for rid in rawids)
        if low_target is None:record['defer_tids'].append(tid)
    action='preserve_native_assignment' if tid in frozen_tids else 'recover_excluded_low' if low_target is not None else choice['action']
    return record,dict(**note,action=action,reason=choice['reason'],high_choice=choice,appearance=appearance,low_target_proposal=low_target,low_proposals=recovery['proposals'],low_decisions=recovery['decisions'],native_eligible_tids=sorted(eligible),frozen_native_pairs=[list(p) for p in frozen],scope='Onlytargetdecisionnewcorevotecontrolled; other originalC34votes and historicallyconditionedstatesfixed. Nativeepisodeadmissionrestrictsemissionforallrows, notcompetitorpool; noGTavailabilityfilter. This is not wholetrackerwithoutFlyVis.')
