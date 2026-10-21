"""Record one frozen spatial-window outcome without evaluating new data."""
from pathlib import Path
import subprocess
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_motion_window_20261021 import RUN,SOURCE,BCE,IDENT,read,write,sha,verify,now,journal
from scripts.z13_continue import paired,per_video_useful
from scripts.z13_train import success


def main():
    if (RUN/'finish_recorded.json').exists():raise RuntimeError('Already recorded')
    p=read(RUN/'protocol.json');verify(p);end=read(RUN/'full_controller_result.json');assert end['worker_ended'] and end['exit_code']==0 and not end['hard_timeout']
    report=read(RUN/'report.json');pv={};keys=None;expected={}
    for arm in ('C','D','R1'):
        expected.update(read(SOURCE/arm/'cache_manifest.json'))
        for mode in ('top1','window'):
            rows=read(RUN/arm/(mode+'_rows.json'));kk=[(r['video'],r['label'],r['kind']) for r in rows];assert len(rows)==8326
            if keys is None:keys=kk
            else:assert keys==kk
            for r in rows:
                if r['kind']==1:assert bool(r['useful'])==bool(success(r))
            pv[arm+'_'+mode]=per_video_useful(rows)
        pv[arm+'_nearest']=per_video_useful(read(ROOT/f'runs/z13_candidate_identity_20261021/{arm}/kalman_rows.json'));pv[arm+'_BCE']=per_video_useful(read(BCE/arm/'geometry/ensemble_rows.json'))
        choice=read(RUN/arm/'choice.json');assert choice==report['choices'][arm] and choice['best']==max(choice['grid'],key=lambda r:r['fit_mean_video'])
        rep=read(RUN/'reproduction.json')[arm];assert rep['rows']==8326 and rep['max_center_px']<1e-6 and rep['max_score']<1e-9
    for rel,digest in read(RUN/'cache_manifest.json').items():assert sha(ROOT/rel)==digest==expected[rel]
    for key,obj in report['paired'].items():
        if key=='C_gain-R1_gain':
            a={v:pv['C_window'][v]-pv['C_top1'][v] for v in pv['C_top1']};b={v:pv['R1_window'][v]-pv['R1_top1'][v] for v in pv['R1_top1']}
        else:a,b=[pv[n] for n in key.split('-')]
        assert paired(a,b)==obj
    d=report['paired']['C_window-C_top1'];passed=d['mean_per_video']>=.02 and d['ci95'][0]>0 and all(report['paired'][k]['ci95'][0]>0 for k in ('C_window-C_nearest','C_window-C_BCE','C_window-R1_window'));assert passed==report['primary_pass']
    state='completed_positive' if passed else ('completed_negative' if d['ci95'][1]<.02 else 'inconclusive');fmt=lambda x:f'{100*x:+.3f}'.replace('.',',')
    def interval(o):return f"{fmt(o['mean_per_video'])} п.п. [{fmt(o['ci95'][0])}; {fmt(o['ci95'][1])}]"
    obtained='runs/z13_motion_window_20261021/report.json → paired: '+'; '.join(k+' '+interval(v) for k,v in report['paired'].items())+'. choices: '+str({k:v['best']['influence'] for k,v in report['choices'].items()})+'.8326 строк всех3 вариантов, исходные центры/баллы и хеши проверены.'
    interpretation=('Основные критерии окна выполнены.' if passed else 'Основные критерии окна не выполнены.')+' R1 с тем же окном получает большой общий эффект, который не приписывается мухе. Подтверждённый сдвиг против максимума не равен узнаванию объекта или улучшению HOTA.'
    decision='Зафиксировать веса и отдельно зарегистрировать перенос до новых dev расчётов.' if passed else 'Не запускать dev этой версии и не менять веса по26. Сохранить частные сравнения; рабочим остаётся прежнее центрированное считывание. Его проверка стенда — самостоятельная задача, не перенос проваленного окна.'
    path=ROOT/'experiments/project_history.json';cat=read(path);card=next(c for c in cat['experiments'] if c['id']==IDENT);head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    for field,v in [('summary',interpretation),('obtained',obtained),('interpretation',interpretation),('decision',decision)]:
        old=card.get(field,'')
        if old and old!=v:card.setdefault('revisions',[]).append(dict(date=f'{now():%Y-%m-%d}',commit=head,field=field,old=old,reason='Полный проверенный исход неизменной версии окна.'))
        card[field]=v
    card.update(state=state,done=f'{now():%Y-%m-%d}');card['acceptance'][0].update(met=True,evidence='runs/z13_motion_window_20261021/reproduction.json');card['acceptance'][1].update(met=bool(passed),evidence='runs/z13_motion_window_20261021/report.json')
    card['confirmed_results']=[];src='runs/z13_motion_window_20261021/report.json';digest=sha(RUN/'report.json')
    for arm in ('C','D'):
        key=arm+'_window-'+arm+'_top1'
        if report['paired'][key]['ci95'][0]>0:
            card['confirmed_results'].append(dict(id=arm+'_window_vs_top1',title='Пространственное окно против максимума карты — '+arm,category='readout',metric='useful_center_fraction',unit='fraction',evidence=dict(source=src,locator=['paired',key],sha256=digest),baseline=dict(source=src,locator=['summary',arm+'_top1','useful_mean_video'],sha256=digest),subset='train26,26 многократно использованных видео',comparison='Те же карты, локальные веса и балл; окно влияет на выбор колонки. Вес выбран только на20fit.',conditions='Общий критерий не пройден; все источники/строки и интервалы проверены.',limitations='R1 получает+20,21п.п.; весь эффект нельзя приписать мухе. Преимущество C над R1 и обученным выбором карты не подтверждено. Нет dev/HOTA/безопасного отказа; старый3-пиковый оракул не потолок.'))
    for name in ('report.json','reproduction.json','result_audit.json','full_controller_result.json'):card['evidence'].append(dict(kind='file',ref='runs/z13_motion_window_20261021/'+name,claim='Полный исход и проверка'))
    write(RUN/'result_audit.json',dict(all_8326_keys_and_intervals_recomputed=True,frozen_source_cache_and_grid_verified=True,full_tree_ended_exit0=True,report_sha256=digest));write(path,cat)
    rel=100*d['mean_per_video']/report['summary']['C_top1']['useful_mean_video']
    journal('полный итог пространственного окна карты','*Получено.* '+obtained+'\n\n*Вывод.* '+interpretation+'\n\n*Решение.* '+decision+'\n\nСводка к C top1: доля полезных центров '+f'{rel:+.3f}%'.replace('.',',')+'; доля потолка не определена — выбор по всей карте создаёт другие координаты, потолок прежних3пиков неприменим. HOTA не измерялась. Формулировки проверены на предмет/инструмент, повторы и соответствие выполненному.')
    write(RUN/'finish_recorded.json',dict(state=state,primary_pass=passed,recorded=now().isoformat()));write(RUN/'night_finished.json',dict(compute_ended=True,result_recorded=True,publication_pending=True,at=now().isoformat()));print(state,interval(d))


if __name__=='__main__':main()
