"""Парныеразностипооднимвидео;взаимодействияневыдаютсязасуммуприбавок."""
from pathlib import Path
import json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]


def interval(differences,adjusted=False):
    d=np.asarray(differences,np.float64);rng=np.random.default_rng(1)
    boot=d[rng.integers(0,len(d),(20000,len(d)))].mean(1)
    q=.05/(2*12) if adjusted else .025
    return dict(mean_per_video=float(d.mean()),ci95=[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))],
        adjusted_ci=[float(np.quantile(boot,q)),float(np.quantile(boot,1-q))] if adjusted else None,n=len(d))


def summarize(per,pooled,p):
    videos=sorted(s['video'] for s in p['sources']);assert len(videos)==40
    paired={};primary={};interactions={}
    metrics=['HOTA','AssA','IDF1','CLR_FP','CLR_FN','IDSw']
    for seed in [42,43,44]:
        target='v134_seed'+str(seed)
        for control in ['native','v0_seed'+str(seed),'vmean_seed'+str(seed),'v1mean_seed'+str(seed)]:
            key=target+'-'+control;primary[key]=interval([per[target][v]['HOTA']-per[control][v]['HOTA'] for v in videos],True)
        for variant in p['variants']:
            name=f'v{variant}_seed{seed}'
            paired[name+'-native']={m:interval([per[name][v][m]-per['native'][v][m] for v in videos]) for m in metrics}
        for label,terms in [('13',[('13',1),('1',-1),('3',-1),('0',1)]),('14',[('14',1),('1',-1),('4',-1),('0',1)]),
            ('34',[('34',1),('3',-1),('4',-1),('0',1)]),('134',[('134',1),('13',-1),('14',-1),('34',-1),('1',1),('3',1),('4',1),('0',-1)])]:
            interactions[f'{label}_seed{seed}']=interval([sum(coef*per[f'v{variant}_seed{seed}'][v]['HOTA'] for variant,coef in terms) for v in videos])
    strong=json.loads((ROOT/'runs/z13_val20_matched_20261024/native_masa_recovery2/report.json').read_text(encoding='utf-8'))
    common=sorted(strong['videos']);assert len(common)==20 and set(common).issubset(videos)
    reference={name:interval([per[name][v]['HOTA']-strong['per_video']['xywh_masa'][v]['HOTA'] for v in common]) for name in per if name!='zero'}
    return dict(primary=primary,paired_vs_native=paired,interactions=interactions,
        main_acceptance=all(c['adjusted_ci'][0]>0 for c in primary.values()),
        multiplicity='12 primary HOTAcomparisons;Bonferroni simultaneous intervals;20k pairedvideo bootstrap;descriptive intervals nominal95%',
        separate_strong_reference20=reference,all40_never_pooled_with20=True,
        limitations='Two-epoch pilot,not completed six-epoch study;3seeds/40historicalval,not new independent final test. NoFlyViscore. Lastrealidentitytarget/6observations60frames/translationonly;not arbitrary long memory. Nominal parameter count equal but masked active state updates differ from repeated last/mean control;not a pure order-only intervention. Factorialinteractions are contrasts,not sums of achievable improvements.')
