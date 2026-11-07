"""Полный парный результат Б2; отдельный вклад ядра и польза системы."""
import numpy as np


def summarize(per,pooled,sources):
    videos=sorted(x['video'] for x in sources);assert len(videos)==40
    def interval(values,adjusted=False):
        d=np.asarray(values,np.float64);rng=np.random.default_rng(1)
        boot=d[rng.integers(0,len(d),(20000,len(d)))].mean(1)
        q=.05/(2*6) if adjusted else .025
        return dict(mean_per_video=float(d.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),
                    adjusted_ci=np.quantile(boot,[q,1-q]).tolist() if adjusted else None,n=len(d))
    primary={control:interval([per['D'][v]['HOTA']-per[control][v]['HOTA'] for v in videos],True)
             for control in ['native','C','R1','B','last','mean']}
    paired={name:{metric:interval([per[name][v][metric]-per['native'][v][metric] for v in videos])
                  for metric in ['HOTA','DetA','AssA','IDF1','CLR_FP','CLR_FN','IDSw']}
            for name in per if name not in ('native','zero')}
    return dict(primary_D_vs_controls=primary,paired_vs_native=paired,
                main_acceptance=all(x['adjusted_ci'][0]>0 for x in primary.values()),
                multiplicity='Six primary D HOTAcomparisons;Bonferroni intervals;nominal95%also shown;20kpairedvideo bootstrap',
                limitations='Oneepoch/seed26/sparse10targets perhistoricaltrain/devvideo;40historicalvalnotnewindependentfinaltest. '
                'Full detector and tracker system compared separately from D-Ctrainedcore contribution. '
                'C/D/R1/Bparametercounts differ;last/mean trainable strongcontrols notpureorderonly. FixedP3positions are not object-aligned. '
                'Fivequerywindow reset perprediction,not unlimited streaming memory;no extrapolation to wholeB2.')
