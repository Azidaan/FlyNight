"""Reuse exact selected-box outcomes; fit a small internal response-value decoder."""
from pathlib import Path
from datetime import datetime,timedelta
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']=os.environ['OPENBLAS_NUM_THREADS']='2'
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import z13_region_context_20261025 as context
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
from scripts.z13_continue import paired
from scripts.z13_localization_labels_20261023 import verify_sources
from scripts.z13_hota_deadline_guard import WindowsProcesses
from tracking.z13_selected_box_value import fit_value,predict_value
RUN=ROOT/'runs/z13_selected_box_value_20261025'
BOX=ROOT/'runs/z13_box_proposal_20261025'
IDENT='z13_selected_box_value_v1'
ARMS=('D','C','R1','geometry')


def journal(title,text):
    from scripts import z13_candidate_reserve_20261021 as m
    old=m.IDENT
    try:m.IDENT=IDENT;m.journal(title,text)
    finally:m.IDENT=old


def comparisons(values,old):
    zero={v:0. for v in values['D']};c={'Dvalue-baseline':paired(values['D'],zero),'Dvalue-Dprior':paired(values['D'],old)}
    for a in ('C','R1','geometry'):c['Dvalue-'+a]=paired(values['D'],values[a])
    return c


def primary(c):return c['Dvalue-baseline']['mean_per_video']>=.02 and all(x['ci95'][0]>0 for x in c.values())


def prepare():
    assert not RUN.exists();p=read(context.RUN/'protocol.json');verify_sources(p)
    assert read(ROOT/'runs/z13_activity_amplitude_20261025/finish_recorded.json')['complete']
    names=[Path(__file__).relative_to(ROOT).as_posix(),'tracking/z13_selected_box_value.py','tests/test_z13_selected_box_value.py',
        'runs/z13_box_proposal_20261025/report.json','runs/z13_box_proposal_20261025/final_summary.json','runs/z13_region_context_20261025/cache/report.json']
    names+=list(read(context.BASE/'features/report.json')['manifest'])
    for v in p['fit20']+p['evaluate26']:
        names += [(BOX/'videos'/(v+'.json')).relative_to(ROOT).as_posix(),(context.RUN/'cache/videos'/(v+'.json')).relative_to(ROOT).as_posix()]
    start=now();stop=min(start+timedelta(seconds=600),datetime.fromisoformat('2026-10-26T07:35:00+03:00'))
    q=dict(p,id=IDENT,created=start.isoformat(),device='cpu',budget_seconds=600,save_at=stop.isoformat(),hard_stop=(stop+timedelta(seconds=20)).isoformat(),stop_request_path=(RUN/'stop.request').relative_to(ROOT).as_posix(),
        action='Internal final response decoder predicts signed value of its one frozen target-model-selected actual box. If predicted value>0 transfer the complete box with alpha1; else unchanged. The selected box is fixed before this decoder. NULL or unavailable exact pooled region stays unchanged. No manual unmatched/free guards.',
        features='150numbers:68raw pooled normalized-by-dataset activity,68componentwise products of that activity and observed target ridge coefficients,8original geometry and6observed original assignment context. Original per-vector LayerNorm is not part of this new analytic decoder. Geometry control zeros both68feature blocks, keeps same150coefficients. D/C/R1 same pipeline, own frozen selected region; all query UIDs retained.',
        training='Only20fit videos fit the ridge value model against exact saved alpha1 beneficial-harmful outcomes of that selected box. 84other videos are not used because their full-box outcomes are absent. Standardize each feature using only supported known fit20 rows; sd floor1e-6; intercept=training mean; ridge lambda1 on mean squared error, fixed before run. No grid, calibration, seed selection, coverage quota or threshold fitting. Evaluate26 only after all4models saved. This is a narrower single-proposal action-value test, not resumption of the130video candidate-choice experiment.',
        technical='Exact UID/kind/real-index bindings to original pooled features/context and saved full-box rows. All46baseline replays and C/D/R1 maps were verified in parent BOX; reused here without recomputation. Persist every supported/unsupported/NULL decision including unknowns. Independently recompute all model predictions/outcomes/video intervals and verify hashes; no GPU. CPU600s+20, never extend other attempt budgets.',
        primary='Dvalue-baseline mean>=2pp and all5paired95%video lower>0: baseline,prior Dbox alpha0.25 selected on20fit,D/C/R1 matched no-new-core controls (C,R1,geometry). Full criteria required; no HOTA from a partial contrast. Values are net one-step assignment effects, not center accuracy.',
        counterargument='20videos may be too few for rare beneficial actions; zero-dominated labels may suppress all actions. The fixed target-conditioned proposal may already be wrong; value decoder cannot replace it. Geometry may explain any gain. No proper held-out confirmation:26is repeatedly reused and fit20 has informed development. This changes both representation/readout family and fitting target relative to the earlier neural categorical head, so it cannot isolate one cause.',
        literature='SiamMOT https://arxiv.org/html/2105.11595 section3.1 learns target visibility jointly with its box output; supports learning refusal for the actual geometric response. Our signed ridge value and150inputs are a separate adaptation, not their model or demonstrated efficacy.',
        next='Pass -> separate frozen matched closed-loop parity/price/HOTA with all controls. Fail -> preserve this exact single-proposal value decoder; no lambda/threshold/feature/normalizer grid on26. Full all-region supervision remains untested, with4videos preserved.',sha256={**p['sha256'],**{n:sha(ROOT/n) for n in names}})
    write(RUN/'protocol.json',q);write(RUN/'status.json',dict(state='ready',stage='Ценность готового ответа полной рамки',updated=start.isoformat()))
    cat=read(ROOT/'experiments/project_history.json');assert not any(c['id']==IDENT for c in cat['experiments'])
    cat['experiments'].append(dict(id=IDENT,title='Внутренний отказ по ценности готовой полной рамки FlyVis',area='flyvis',state='ready',started=start.strftime('%Y-%m-%d'),tags=['fly-problem-P8','fly-problem-P9'],question='Можно ли использовать готовый ответ полной рамки, обучив отказ его фактическим последствиям?',hypothesis='Рабочая гипотеза: готовые метки полного ответа позволяют изучить пользу против исходного решения без дорогостоящей разметки всех альтернатив.',expected=q['action']+' '+q['features'],acceptance=[dict(criterion=q['technical'],met=None),dict(criterion=q['primary'],met=None)],obtained='',interpretation='',decision='',cannot_claim=q['counterargument'],evidence=[dict(kind='file',ref=(RUN/'protocol.json').relative_to(ROOT).as_posix(),claim='Точный более узкий ответ и условия до расчёта')]))
    write(ROOT/'experiments/project_history.json',cat)
    journal('постановка ценности готового ответа полной рамки','*Действие.* '+q['action']+'\n\n*Признаки.* '+q['features']+'\n\n*Обучение и отличие.* '+q['training']+'\n\n*Критерии.* '+q['primary']+'\n\n*Возражение.* '+q['counterargument']+'\n\n*Источник.* '+q['literature']+'\n\n'+q['next'])


def data():
    import torch
    torch.set_num_threads(2);torch.set_default_device('cpu')
    p=read(RUN/'protocol.json');verify_sources(p);source=context.original.namespace();loader=source['load_split'];byuid={};contexts={}
    for v in p['fit20']+p['evaluate26']:
        for row in read(BOX/'videos'/(v+'.json'))['rows']:assert row['uid'] not in byuid;byuid[row['uid']]=row
        contexts.update({int(u):e for u,e in read(context.RUN/'cache/videos'/(v+'.json'))['entries'].items()})
    result={}
    for a in ARMS:
        arm='D' if a=='geometry' else a
        for split,videos in (('fit20',p['fit20']),('train26',p['evaluate26'])):
            d=loader(torch,arm,split);rows=[];xx=[]
            for i,uid in enumerate(d['uid'].tolist()):
                row=byuid[uid];meta=contexts[uid];selected=row['proposal_choices'][a]['index']
                assert row['kind']==int(d['kind'][i]);n=int(d['mask'][i].sum());assert len(meta['choices'])==n
                supported=selected>=0 and selected in meta['choices']
                feature=np.zeros(150,np.float64)
                if supported:
                    k=meta['choices'].index(selected);x=d['x'][i,k].numpy().astype(np.float64);w=d['seed'][i].numpy().astype(np.float64)
                    feature=np.r_[x,x*w,d['geo'][i,k].numpy(),meta['context'][k]]
                    if a=='geometry':feature[:136]=0
                assert feature.shape==(150,) and np.isfinite(feature).all()
                outcome=row['arms'][a+'_box'][3]
                rows.append(dict(uid=uid,video=row['video'],kind=row['kind'],selected=selected,supported=supported,full_outcome=outcome))
                xx.append(feature)
            result[a,split]=(np.asarray(xx),rows)
    assert not torch.cuda.is_initialized()
    return result


def outcomes(rows,predictions):
    return [dict(uid=r['uid'],video=r['video'],kind=r['kind'],supported=r['supported'],selected=r['selected'],prediction=float(s),apply=bool(r['supported'] and s>0),beneficial=bool(r['kind']>=0 and r['supported'] and s>0 and r['full_outcome']['beneficial']),harmful=bool(r['kind']>=0 and r['supported'] and s>0 and r['full_outcome']['harmful'])) for r,s in zip(rows,predictions)]


def aggregate(rows,videos):
    return {v:sum(int(x['beneficial'])-int(x['harmful']) for x in rows if x['video']==v)/sum(x['video']==v and x['kind']>=0 for x in rows) for v in videos}


def worker():
    p=read(RUN/'protocol.json');d=data();models={};began=time.monotonic()
    for a in ARMS:
        x,rows=d[a,'fit20'];valid=np.array([r['supported'] and r['kind']>=0 for r in rows])
        y=np.array([int(bool(r['full_outcome']['beneficial']))-int(bool(r['full_outcome']['harmful'])) for r in rows])
        models[a]=fit_value(x[valid],y[valid]);write(RUN/a/'model.json',dict(model=models[a],train_rows=int(valid.sum()),fit_videos=p['fit20'],fixed_lambda=1,threshold=0))
    write(RUN/'models_frozen.json',dict(at=now().isoformat(),before_evaluation=True,model_sha256={a:sha(RUN/a/'model.json') for a in ARMS}))
    values={};counts={};hashes={}
    for a in ARMS:
        if now()>=datetime.fromisoformat(p['save_at']) or (RUN/'stop.request').exists():raise TimeoutError('Original selected-box deadline')
        x,rows=d[a,'train26'];out=outcomes(rows,predict_value(models[a],x));values[a]=aggregate(out,p['evaluate26'])
        counts[a]=dict(rows=len(out),known=sum(x['kind']>=0 for x in out),supported=sum(x['supported'] for x in out),applied=sum(x['apply'] for x in out),beneficial=sum(x['beneficial'] for x in out),harmful=sum(x['harmful'] for x in out),unknown=sum(x['kind']<0 for x in out))
        write(RUN/a/'rows.json',out);hashes[a]=sha(RUN/a/'rows.json')
    old=read(BOX/'report.json')['per_video']['D_box'];c=comparisons(values,old);verify_sources(p)
    write(RUN/'report.json',dict(complete=True,primary_pass=primary(c),paired=c,per_video=values,counts=counts,row_sha256=hashes,seconds=time.monotonic()-began,models_frozen_sha256=sha(RUN/'models_frozen.json')))
    write(RUN/'status.json',dict(state='completed',stage='Все4внутренних декодера готового ответа сохранены',updated=now().isoformat()))


def finish():
    p=read(RUN/'protocol.json');e=read(RUN/'controller_result.json');ps=WindowsProcesses()
    assert e['exit_code']==0 and e['all_owned_ended'] and not e['remaining'] and all(ps.creation(int(pid))!=b for pid,b in e['owned'].items())
    d=data();r=read(RUN/'report.json');frozen=read(RUN/'models_frozen.json');values={}
    for a in ARMS:
        model=read(RUN/a/'model.json');assert sha(RUN/a/'model.json')==frozen['model_sha256'][a]
        x,rows=d[a,'fit20'];valid=np.array([z['supported'] and z['kind']>=0 for z in rows]);y=np.array([int(bool(z['full_outcome']['beneficial']))-int(bool(z['full_outcome']['harmful'])) for z in rows])
        assert model['model']==fit_value(x[valid],y[valid])
        x,rows=d[a,'train26'];out=outcomes(rows,predict_value(model['model'],x))
        assert out==read(RUN/a/'rows.json') and sha(RUN/a/'rows.json')==r['row_sha256'][a]
        values[a]=aggregate(out,p['evaluate26'])
    c=comparisons(values,read(BOX/'report.json')['per_video']['D_box']);assert c==r['paired'] and values==r['per_video'] and primary(c)==r['primary_pass']
    write(RUN/'independent_audit.json',dict(at=now().isoformat(),all_models_predictions_actual_action_outcomes_and_intervals_verified=True,report_sha256=sha(RUN/'report.json')))
    cat=read(ROOT/'experiments/project_history.json');card=next(x for x in cat['experiments'] if x['id']==IDENT)
    card.update(state='completed_positive' if r['primary_pass'] else 'completed_negative',done=now().strftime('%Y-%m-%d'),summary='Ценность готовой полной рамки: критерий '+('пройден.' if r['primary_pass'] else 'не пройден.'),obtained=(RUN/'report.json').relative_to(ROOT).as_posix()+' → paired/counts: '+str(dict(paired=c,counts=r['counts'])),interpretation='Проверен более узкий внутренний отказ для одного готового ответа. Все четыре модели и строки независимо пересчитаны из старых меток фактического действия. Это не завершение выбора среди всех рамок и не HOTA;26видео повторно используются.',decision=p['next']+' Основной критерий: '+str(r['primary_pass']))
    for i,met in enumerate((True,r['primary_pass'])):card['acceptance'][i].update(met=met,evidence=(RUN/('independent_audit.json' if i==0 else 'report.json')).relative_to(ROOT).as_posix())
    write(ROOT/'experiments/project_history.json',cat);write(RUN/'finish_recorded.json',dict(at=now().isoformat(),complete=True,primary_pass=r['primary_pass']))
    journal('итог внутреннего отказа готовой полной рамки','*Получено.* '+card['obtained']+'\n\n*Вывод.* '+card['interpretation']+'\n\n*Решение.* '+card['decision']+'\n\nHOTA и процент к стенду не измерены,доляпотолканеустановлена.')


if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    elif '--worker' in sys.argv:
        try:worker()
        except Exception as error:write(RUN/'failure.json',dict(at=now().isoformat(),error=repr(error)));raise
    elif '--finish' in sys.argv:finish()
