"""Audit and record the frozen pairwise-objective experiment, including stop gate."""
from pathlib import Path
import subprocess
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_pairwise_map_20261021 import RUN,SOURCE,DEV,IDENT,read,write,sha,verify,now,journal,summarize
from scripts.z13_train import success


def main():
    if (RUN/'finish_recorded.json').exists():raise RuntimeError('Already recorded')
    p=read(RUN/'protocol.json');verify(p);end=read(RUN/'train_controller_result.json');assert end['worker_ended'] and end['exit_code']==0 and not end['hard_timeout']
    train=read(RUN/'report_train26.json');transfer=train['primary_pass'];reports={};models={}
    if transfer:
        end=read(RUN/'transfer_controller_result.json');assert end['worker_ended'] and end['exit_code']==0 and not end['hard_timeout']
    else:assert not (RUN/'report_dev35.json').exists(),'Stop gate must not be bypassed'
    for arm in ('C','D','R1'):
        for seed in (0,1,2):
            sd=RUN/arm/f'seed{seed}';tr=read(sd/'training.json');assert tr['seed']==seed and tr['steps']==1150 and tr['presented_rows']==145950 and len(tr['loss'])==1150 and tr['observed_training_pairs']>0 and np.isfinite(tr['loss']).all()
            assert (sd/'done.json').exists() and sha(sd/'model.pt')==tr['model_sha256'];models[sd.relative_to(ROOT).as_posix()]=tr['model_sha256']
    for split in (['train26','dev35'] if transfer else ['train26']):
        rows={};keys=None;nr=8326 if split=='train26' else 9314;nv=26 if split=='train26' else 35
        for arm in ('C','D','R1'):
            rr=read(RUN/split/arm/'rows.json');kk=[(r['video'],r['label'],r['kind']) for r in rr];assert len(rr)==nr and len(set(kk))==nr and len({r['video'] for r in rr})==nv
            if keys is None:keys=kk
            else:assert keys==kk
            for r in rr:
                if r['kind']==1:assert bool(r['useful'])==bool(success(r))
            rows[arm+'_rank']=rr;rows[arm+'_BCE']=read(SOURCE/arm/'geometry/ensemble_rows.json') if split=='train26' else read(DEV/arm/'map_rows.json')
            for mode,saved in [('top1','top1'),('nearest','kalman')]:rows[arm+'_'+mode]=read(ROOT/f'runs/z13_candidate_identity_20261021/{arm}/{saved}_rows.json') if split=='train26' else read(DEV/arm/(mode+'_rows.json'))
            assert all([(r['video'],r['label'],r['kind']) for r in rows[arm+'_'+mode]]==kk for mode in ('BCE','top1','nearest'))
        seeds=read(RUN/('seeds_'+split+'.json'));report=read(RUN/('report_'+split+'.json'));assert summarize(rows,seeds,split)==report
        for rel,digest in read(RUN/('cache_manifest_'+split+'.json')).items():assert sha(ROOT/rel)==digest
        reports[split]=report
    final=reports['dev35'] if transfer else train;d=final['paired']['C_rank-C_BCE'];passed=transfer and final['primary_pass'];state='completed_positive' if passed else ('completed_negative' if d['ci95'][1]<.02 else 'inconclusive')
    fmt=lambda v:f'{100*v:+.3f}'.replace('.',',')
    def interval(o):return f"{fmt(o['mean_per_video'])} п.п. [{fmt(o['ci95'][0])}; {fmt(o['ci95'][1])}]"
    obtained='; '.join('runs/z13_pairwise_map_20261021/report_'+s+'.json → paired: '+', '.join(k+' '+interval(v) for k,v in r['paired'].items()) for s,r in reports.items())+'.9 моделей,1150 шагов/145950 предъявлений, исходные хеши, строки и интервалы перепроверены.'
    interpretation=('Основные критерии на train26 и фиксированном dev35 выполнены.' if passed else 'Основной критерий парного обучения этой реализации не выполнен.')+' Поменялась только функция потерь считывателя; причинная роль BCE не установлена. Нет HOTA и безопасного отказа.'
    decision='Сохранить фиксированную версию для технического подключения и оценки стоимости стенда с R1 и без поправки.' if passed else ('Dev не запускался по заранее заданной остановке; эту функцию потерь не переносить в стенд. Прежний центрированный считыватель сохранён.' if not transfer else 'После фиксированного dev не подбирать параметры; эту версию не переносить в стенд, исходный центрированный считыватель сохранён.')
    path=ROOT/'experiments/project_history.json';cat=read(path);card=next(c for c in cat['experiments'] if c['id']==IDENT);head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    for field,value in [('summary',interpretation),('obtained',obtained),('interpretation',interpretation),('decision',decision)]:
        old=card.get(field,'')
        if old and old!=value:card.setdefault('revisions',[]).append(dict(date=f'{now():%Y-%m-%d}',commit=head,field=field,old=old,reason='Полный проверенный исход и выполнение заранее заданной остановки.'))
        card[field]=value
    card.update(state=state,done=f'{now():%Y-%m-%d}')
    card['acceptance'][0].update(met=True,evidence='runs/z13_pairwise_map_20261021/result_audit.json');card['acceptance'][1].update(met=bool(train['primary_pass']),evidence='runs/z13_pairwise_map_20261021/report_train26.json');card['acceptance'][2].update(met=bool(final['primary_pass']) if transfer else None,evidence='runs/z13_pairwise_map_20261021/report_dev35.json' if transfer else 'Не запускался: основной train26 гейт не пройден.')
    for name in ('report_train26.json','result_audit.json','train_controller_result.json')+ (('report_dev35.json','transfer_controller_result.json') if transfer else ()):
        card['evidence'].append(dict(kind='file',ref='runs/z13_pairwise_map_20261021/'+name,claim='Проверенный исход и исполнение ограничения'))
    write(RUN/'result_audit.json',dict(source_and_cache_sha256_unchanged=True,model_sha256=models,matched_optimizer_and_frame_budgets=True,all_primary_rows_and_intervals_recomputed=True,transfer_ran_only_if_train_primary_pass=transfer,report_sha256={s:sha(RUN/('report_'+s+'.json')) for s in reports}));write(path,cat)
    rel=100*d['mean_per_video']/final['summary']['C_BCE']['useful_mean_video']
    journal('полный итог парного обучения пиков','*Получено.* '+obtained+'\n\n*Вывод.* '+interpretation+'\n\n*Решение.* '+decision+'\n\nСводка к совпадающему C BCE104 на '+final['split']+': доля полезных центров '+f'{rel:+.3f}%'.replace('.',',')+'; доля потолка не пересчитана — здесь сравнение с обучаемым BCE, а не с исходным максимумом, отдельный остаточный потолок не записан. HOTA не измерялась. Формулировки проверены: предмет/инструмент, отсутствие лишнего повтора и соответствие выполненному.')
    write(RUN/'finish_recorded.json',dict(state=state,primary_pass=passed,transfer_ran=transfer,recorded=now().isoformat()));write(RUN/'night_finished.json',dict(compute_ended=True,result_recorded=True,publication_pending=True,at=now().isoformat()));print(state,final['split'],interval(d))


if __name__=='__main__':main()
