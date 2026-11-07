from pathlib import Path
import json
from scripts.summarize_history_tracker_20261028 import summarize


def test_main_criterion_requires_each_seed_and_each_strong_control():
    root=Path(__file__).resolve().parents[1]
    videos=json.loads((root/'runs/recovery_research_20261027/lazy_appearance_validation/protocol.json').read_text(encoding='utf-8'))['sources']
    variants=['0','1','3','4','13','14','34','134','mean','1mean']
    def metric(h):return dict(HOTA=h,AssA=h,IDF1=h,CLR_FP=10.,CLR_FN=20.,IDSw=2.)
    per={'native':{s['video']:metric(10.) for s in videos}}
    for seed in [42,43,44]:
        for v in variants:
            value={'0':10.5,'mean':10.6,'1mean':10.7,'134':12.}.get(v,11.)
            per[f'v{v}_seed{seed}']={s['video']:metric(value) for s in videos}
    result=summarize(per,{},dict(sources=videos,variants=variants))
    assert result['main_acceptance'] and len(result['primary'])==12
    assert abs(result['interactions']['134_seed42']['mean_per_video']-1.5)<1e-10
    assert result['primary']['v134_seed42-v1mean_seed42']['adjusted_ci'][0]>0
    for s in videos:per['v134_seed44'][s['video']]=metric(10.6)
    assert not summarize(per,{},dict(sources=videos,variants=variants))['main_acceptance']
