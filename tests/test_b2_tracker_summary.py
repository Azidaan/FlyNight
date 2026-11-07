from scripts.summarize_b2_tracker_20261029 import summarize


def test_primary_requires_every_matched_control_not_just_native():
    names=['native','zero','C','D','R1','B','last','mean']
    keys=['HOTA','DetA','AssA','IDF1','CLR_FP','CLR_FN','IDSw']
    sources=[dict(video=str(i)) for i in range(40)]
    per={n:{str(i):{m:50.+(2 if n=='D' else 0) for m in keys} for i in range(40)} for n in names}
    assert summarize(per,{},sources)['main_acceptance']
    for v in per['mean']:per['mean'][v]['HOTA']=per['D'][v]['HOTA']
    result=summarize(per,{},sources)
    assert not result['main_acceptance'] and result['primary_D_vs_controls']['native']['adjusted_ci'][0]>0
