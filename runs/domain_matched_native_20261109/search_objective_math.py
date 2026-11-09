"""Algebraic prescreen: weighted shift search alone cannot reject an accepted edge."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,hashlib,math
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
model=json.loads((OUT/'deployment.json').read_text(encoding='utf8'));weights=model['models']['C34']['theta'][7:-1]
assert len(weights)==34 and min(weights)>0 and max(weights)>min(weights)
lo=min(range(34),key=weights.__getitem__);hi=max(range(34),key=weights.__getitem__)
# Every channel correlation is within [-1,1]. Reference-active denominator=34.
a=[0.]*34;b=[0.]*34;a[lo]=1.;b[lo]=.4;b[hi]=.5
mean_a=sum(a)/34;mean_b=sum(b)/34;score_a=sum(x*w for x,w in zip(a,weights));score_b=sum(x*w for x,w in zip(b,weights))
assert mean_a>mean_b and score_b>score_a
report=dict(passed=True,recorded=datetime.now(timezone(timedelta(hours=3))).isoformat(),
    source_sha256=sha(Path(__file__)),sources={p.relative_to(ROOT).as_posix():sha(p) for p in [OUT/'deployment.json',ROOT/'tracking/channel_identity_head.py']},
    actual_weights=dict(positive=34,negative=0,min=min(weights),max=max(weights),low_channel=lo,high_channel=hi),
    constructed_counterexample=dict(mean_a=mean_a,mean_b=mean_b,weighted_a=score_a,weighted_b=score_b,all_correlations_bounded=True,not_an_observed_video=True),
    theorem='Let s0 maximize equal-channel mean over the same valid shift set, z_s=g_score+w dot rho_s+b. Then max_s z_s>=z_s0. At fixed threshold0 and fixed validity,fallback,support,head and geometry, every previously accepted edge remains accepted. Changing only the selected shift to maximize the head cannot reject any previously accepted foreign edge. It can change global assignments, so it does not prove public foreign IDs remain unchanged.',
    distinction='Unequal positive weights suffice for mean-max and weighted-max to differ; negative weights are not required. This establishes possible objective mismatch, not its causal role in either own refusal.',
    decision='Do not present maximize-the-new-head alone as a solution to both own refusal and foreign permission. Any retrained joint spatial classifier is a separate intervention with matched controls and prior old-target-model comparison, not a proven fix.',
    no_inference=True,no_fit=True,no_tracker_or_core=True,goal_achieved=False)
with (OUT/'search_objective_math.json').open('x',encoding='utf8') as f:json.dump(report,f,ensure_ascii=False,indent=2);f.write('\n')
print('SEARCH_OBJECTIVE_COUNTEREXAMPLE_AND_MONOTONICITY_CHECKED',flush=True)
