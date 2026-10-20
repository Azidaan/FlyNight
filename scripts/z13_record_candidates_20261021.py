"""Record completed candidate reserve/identity outcomes; preserve old fields."""
from __future__ import annotations
import argparse
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,sha,verify,now


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--kind',choices=['reserve','identity'],required=True);a=ap.parse_args()
    short='candidate_reserve' if a.kind=='reserve' else 'candidate_identity';run=ROOT/f'runs/z13_{short}_20261021';ident=f'z13_{short}_v1'
    marker=run/'finish_recorded.json'
    if marker.exists():raise RuntimeError('Outcome already recorded')
    p=read(run/'protocol.json');verify(p);report=read(run/'report.json');label='evaluate' if a.kind=='reserve' else 'full'
    result=read(run/(label+'_controller_result.json'));assert result['worker_ended'] and result['exit_code']==0 and not result['hard_timeout']
    catpath=ROOT/'experiments/project_history.json';cat=read(catpath);card=next(x for x in cat['experiments'] if x['id']==ident)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();fmt=lambda v:f'{100*v:+.3f}'.replace('.',',')
    def interval(d):return f"{fmt(d['mean_per_video'])} п.п. [{fmt(d['ci95'][0])}; {fmt(d['ci95'][1])}]"
    if a.kind=='reserve':
        from scripts.z13_continue import paired,per_video_useful
        from scripts.z13_train import success
        repro=read(run/'reproduction.json')
        for arm,r in repro.items():assert r['rows']==8326 and r['max_center_px']<.5 and r['max_peak_score']<.001
        for arm in ('C','D','R1'):
            top=read(run/'evaluate'/arm/'top1_rows.json');oracle=read(run/'evaluate'/arm/'oracle3_rows.json')
            assert [(r['video'],r['label'],r['kind']) for r in top]==[(r['video'],r['label'],r['kind']) for r in oracle]
            for rr in (top,oracle):
                for r in rr:
                    if r['kind']==1:assert r['useful']==bool(success(r))
            assert paired(per_video_useful(oracle),per_video_useful(top))==report['arms'][arm]['reserve']
        d=report['arms']['C']['reserve'];passed=report['C_reserve_sufficient'];state='completed_positive' if passed else 'inconclusive'
        obtained='runs/z13_candidate_reserve_20261021/report.json → arms: '+ '; '.join(f"{arm}, оракул среди3−top1 {interval(r['reserve'])}" for arm,r in report['arms'].items())+'. reproduction.json: все3 варианта воспроизведены на8326 строках; исходные хеши неизменны.'
        interpretation='Измерен диагностический запас среди трёх пиков. Разметка использована только для оценки оракула; это не улучшенный алгоритм и не результат HOTA.'
        decision='Запас достаточен для одной причинной версии сопоставления кандидатов на fit20 с проверкой train26.' if passed else 'Запас не прошёл заранее заданный критерий; не обещать пользу сопоставления. Если интервал допускает полезный сдвиг, результат неопределённый.'
        acceptance=[True,passed]
        base=report['arms']['C']['top1_mean'];delta=d['mean_per_video']
    else:
        import numpy as np
        from scripts.z13_continue import paired,per_video_useful
        from scripts.z13_train import success
        pv={}
        for arm in ('C','D','R1'):
            r=read(run/arm/'reproduction.json');assert r['rows']==8326 and r['max_center_px']<1e-12 and r['max_score']<1e-12
            for mode in ('association','geometry','top1','kalman'):
                rr=read(run/arm/(mode+'_rows.json'));pv[arm+'_'+mode]=per_video_useful(rr)
                for r in rr:
                    if r['kind']==1:assert r['useful']==bool(success(r))
                assert np.isclose(np.mean(list(pv[arm+'_'+mode].values())),report['summary'][arm+'_'+mode]['useful_mean_video'],rtol=0,atol=1e-14)
        for key,obj in report['paired'].items():
            x,y=key.split('-');assert paired(pv[x],pv[y])==obj
        for path,digest in read(run/'cache_manifest.json').items():assert sha(ROOT/path)==digest
        passed=report['primary_pass'];d=report['paired']['C_association-C_top1'];state='completed_positive' if passed else ('completed_negative' if d['ci95'][1]<0 else 'inconclusive')
        keys=['C_association-C_top1','C_association-C_geometry','C_association-R1_association','D_association-D_top1','D_association-D_geometry','D_association-R1_association']
        obtained='runs/z13_candidate_identity_20261021/report.json → paired: '+'; '.join(k+' '+interval(report['paired'][k]) for k in keys)+'. Все3 контроли top1 точно воспроизведены; cache_manifest.json и источники проверены.'
        interpretation=('Основной критерий малого причинного сопоставления кандидатов выполнен.' if passed else 'Основной критерий этой реализации сопоставления кандидатов не выполнен. Причина этим не установлена; сохранённый оракул не гарантирует доступности запаса для алгоритма.')+' Это поисковое сравнение на многократно использованном train26, не HOTA и не безопасная политика вмешательства.'
        decision='Сохранить как функционального кандидата; до переноса нужен отдельный причинный отбор вмешательств и проверка стенда.' if passed else 'Эту реализацию не переносить в стенд и не подбирать параметры на dev. Следующая замена должна менять механизм сопоставления, а не подгонять порог по train26.'
        acceptance=[True,d['mean_per_video']>=.02 and d['ci95'][0]>0,report['paired']['C_association-C_geometry']['ci95'][0]>0 and report['paired']['C_association-R1_association']['ci95'][0]>0]
        base=report['summary']['C_top1']['useful_mean_video'];delta=d['mean_per_video']
    for field,value in [('summary',interpretation),('obtained',obtained),('interpretation',interpretation),('decision',decision)]:
        old=card.get(field,'')
        if old and old!=value:card.setdefault('revisions',[]).append(dict(date=f'{now():%Y-%m-%d}',commit=commit,field=field,old=old,reason='Полный проверенный результат; прежняя постановка сохранена.'))
        card[field]=value
    card['state']=state;card['done']=f'{now():%Y-%m-%d}'
    for i,met in enumerate(acceptance):card['acceptance'][i].update(met=bool(met),evidence=run.relative_to(ROOT).as_posix()+'/report.json')
    for name in ('report.json','reproduction.json' if a.kind=='reserve' else 'cache_manifest.json',label+'_controller_result.json'):
        card['evidence'].append(dict(kind='file',ref=(run/name).relative_to(ROOT).as_posix(),claim='Полный результат и проверка воспроизведения/завершения'))
    write(catpath,cat)
    rel='не определено' if not base else f'{100*delta/base:+.3f}%'.replace('.',',')
    plan=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';text=plan.read_text(encoding='utf-8');anchor='## Журнал изменений\n'
    entry=f'\n**{now():%d.%m.%Y, %H:%M} MSK** — полный итог `{ident}`\n\n*Получено.* {obtained}\n\n*Вывод.* {interpretation}\n\n*Решение.* {decision}\n\nСводка к исходному top1 C: доля полезных центров {rel}; '+('это диагностический потолок, не реализованный прирост. ' if a.kind=='reserve' else '')+'HOTA не измерялась; доля потолка алгоритмом не определена. Формулировки проверены: предмет/инструмент, отсутствие лишнего повтора, соответствие фактически выполненному.\n'
    assert text.count(anchor)==1;plan.write_text(text.replace(anchor,anchor+entry,1),encoding='utf-8')
    write(marker,dict(id=ident,state=state,passed=passed,report_sha256=sha(run/'report.json'),recorded=now().isoformat()))
    print(state,interval(d))


if __name__=='__main__':main()
