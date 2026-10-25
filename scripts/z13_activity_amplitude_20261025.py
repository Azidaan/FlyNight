"""Matched normalization/affine CPU readouts with unchanged action labels."""
from pathlib import Path
from datetime import datetime,timedelta
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']=os.environ['OPENBLAS_NUM_THREADS']='2'
import sys
import inspect
import hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import z13_region_context_20261025 as context
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
from scripts.z13_localization_labels_20261023 import verify_sources
from scripts.z13_hota_deadline_guard import WindowsProcesses
from tracking.z13_activity_amplitude import make_amplitude_ranker
RUN=ROOT/'runs/z13_activity_amplitude_20261025'
CONTEXT=context.RUN
IDENT='z13_activity_amplitude_v1'
VARIANTS=tuple(a+'_'+mode for a in ('D','C','R1','geometry') for mode in ('norm','affine'))


def journal(title,text):
    from scripts import z13_candidate_reserve_20261021 as m
    old=m.IDENT
    try:m.IDENT=IDENT;m.journal(title,text)
    finally:m.IDENT=old


def comparisons(means,videos):
    from scripts.z13_continue import paired
    out={'Daffine-baseline':paired(means['D_affine'],{v:0. for v in videos})}
    for a in ('D_norm','C_affine','R1_affine','geometry_affine'):
        out['Daffine-'+a]=paired(means['D_affine'],means[a])
    return out


def primary(c):
    return c['Daffine-baseline']['mean_per_video']>=.02 and c['Daffine-D_norm']['mean_per_video']>=.02 and all(x['ci95'][0]>0 for x in c.values())


def training_source():
    s=context.training_source();old="model=make_ranker(torch,activity=base_variant!='geometry')"
    assert s.count(old)==1
    return s.replace(old,"model=make_amplitude_ranker(torch,activity=base_variant!='geometry',preserve=(mode=='affine'))")


def loader_source():
    s=inspect.getsource(context.train_worker)
    s=s.replace("RUN/'cache/report.json'","CONTEXT/'cache/report.json'").replace("RUN/'cache/videos'","CONTEXT/'cache/videos'")
    assert s.count("if mode=='context':")==1
    s=s.replace("if mode=='context':","if mode in ('norm','affine'):")
    assert s.count("ns={**ns,'RUN':RUN")==1
    return s.replace("ns={**ns,'RUN':RUN","ns={**ns,'make_amplitude_ranker':make_amplitude_ranker,'RUN':RUN")


def status(phase,stage,**kw):
    p=read(RUN/'training/protocol.json')
    if (RUN/'training/stop.request').exists() or now()>=datetime.fromisoformat(p['save_at']):raise TimeoutError('Original amplitude budget')
    write(RUN/'training/status.json',dict(state='running',stage=stage,updated=now().isoformat(),**kw))


def prepare():
    assert not RUN.exists();p=read(CONTEXT/'protocol.json');verify_sources(p)
    assert read(ROOT/'runs/z13_full_box_supervision_20261025/partial_recorded.json')['scientific_verdict'] is None
    names=[Path(__file__).relative_to(ROOT).as_posix(),'tracking/z13_activity_amplitude.py','tests/test_z13_activity_amplitude.py','tracking/z13_filter_descriptor.py',
        'tracking/z13_region_assignment_context.py','scripts/z13_region_context_20261025.py',
        'runs/z13_region_context_20261025/cache/report.json','runs/z13_region_context_20261025/training/report.json']
    names+=list(read(context.BASE/'features/report.json')['manifest'])
    for v in p['fit84']+p['fit20']+p['evaluate26']:names.append((CONTEXT/'cache/videos'/(v+'.json')).relative_to(ROOT).as_posix())
    start=now();stop=min(start+timedelta(seconds=1800),datetime.fromisoformat('2026-10-26T07:35:00+03:00'))
    q=dict(p,id=IDENT,created=start.isoformat(),budget_seconds=1800,device='cpu',save_at=stop.isoformat(),hard_stop=(stop+timedelta(seconds=20)).isoformat(),stop_request_path=(RUN/'training/stop.request').relative_to(ROOT).as_posix(),variants=list(VARIANTS),
        action='Change only candidate activity embedding LayerNorm68 to a per-channel learnable affine transform (68gain+68bias, same initialization/capacity). Leave target-coefficient LayerNorm and fixed dataset-channel normalization intact. Same actual regional categorical action and signed good-harm labels as prior context readout.',
        fixed='4representations x2norm/affine x3seeds=24models, same initial parameter tensors, optimizer,1150steps,batch128,UID batches, old6observed assignment-context features, same84train/20fit/26evaluation, original frozen fit20 threshold grid. Geometry zeros activity. No GPU, new video replay, features, full-box labels or core training.',
        technical='All4normalization controls reproduce their prior context models tensor-by-tensor, selected tau and every8326saved row exactly. Equal parameter counts and initial tensors. All feature/context hashes and24model outputs audited against unchanged action labels; real CPU gradient test finite, CUDA uninitialized. First completed model2*time*23+60 fits remaining1800s; otherwise preserve and stop.',
        primary='Daffine-baseline and Daffine-Dnorm eachmean>=2pp; all5paired95%video lower>0: baseline,Dnorm,Caffine,R1affine,geometryaffine. Every seed/count/threshold retained. Daffine fixed primary, no choosing seed/mode on26.',
        reason='Source tracking/z13_filter_descriptor.py applies per-candidate LayerNorm before query embedding; this discards common shifts and approximately positive rescaling (finite epsilon) of its68channels. Current heads may lose response magnitude needed for refusal. This is a hypothesis about utility, not an established cause. Previous map-standard-deviation repair acted on a scalar response map, not raw68-vector normalization inside learned embedding.',
        counterargument='Normalization may be essential for stable cross-channel matching. Dataset-level scales and relative channel patterns may already contain enough evidence; retaining magnitude can saturate Tanh and make generalization worse. A useful effect may occur equally in eye/geometry controls. Reused26 is exploratory, not independent confirmation.',
        literature='Ba,Kiros,Hinton2016 Layer Normalization https://arxiv.org/html/1607.06450 section3 equation3 and section5.1/Table1 support the invariance mechanism, not a tracking improvement. The proposed equal-capacity affine ablation is our test of its relevance, not their recommended replacement.',
        next='Pass -> separate frozen closed-loop gate, including native/C/R1/no-FlyVis. Fail -> preserve this exact normalization ablation and do not claim input amplitude is universally useless; no epsilon/gain/normalizer sweep on26.',
        executed_training_sha256=hashlib.sha256(training_source().encode()).hexdigest(),executed_loader_sha256=hashlib.sha256(loader_source().encode()).hexdigest(),sha256={**p['sha256'],**{n:sha(ROOT/n) for n in names}})
    write(RUN/'protocol.json',q);write(RUN/'training/protocol.json',q);write(RUN/'training/status.json',dict(state='ready',stage='Сохранение амплитуды внутри считывателя',updated=start.isoformat()))
    cat=read(ROOT/'experiments/project_history.json');assert not any(c['id']==IDENT for c in cat['experiments'])
    cat['experiments'].append(dict(id=IDENT,title='Амплитуда активности внутри регионального считывателя FlyVis',area='flyvis',state='ready',started=start.strftime('%Y-%m-%d'),tags=['fly-problem-P8','fly-problem-P9'],question='Мешает ли отдельная нормировка каждого вектора68признаков обучаемому выбору и отказу?',hypothesis=q['reason'],expected=q['action']+' '+q['fixed'],acceptance=[dict(criterion=q['technical'],met=None),dict(criterion=q['primary'],met=None)],obtained='',interpretation='',decision='',cannot_claim=q['counterargument'],evidence=[dict(kind='file',ref=(RUN/'protocol.json').relative_to(ROOT).as_posix(),claim='Один внутренний преобразователь и равные контроли до расчёта')]))
    write(ROOT/'experiments/project_history.json',cat)
    journal('постановка сохранения уровня активности внутри считывателя','*Действие.* '+q['action']+'\n\n*Основание и отличие.* '+q['reason']+' '+q['literature']+'\n\n*Контроли.* '+q['fixed']+'\n\n*Критерии.* '+q['primary']+' '+q['technical']+'\n\n*Возражение.* '+q['counterargument']+'\n\nCPU1800с+20с безпрогонавидео. '+q['next'])


def worker():
    p=read(RUN/'protocol.json');s=loader_source();assert hashlib.sha256(s.encode()).hexdigest()==p['executed_loader_sha256']
    ns={**vars(context),**globals(),'original':context.original,'signed_action_loss':context.signed_action_loss}
    exec(compile(s,__file__,'exec'),ns);ns['train_worker']()


def finish():
    import torch
    torch.set_num_threads(2);p=read(RUN/'protocol.json');verify_sources(p);ps=WindowsProcesses();e=read(RUN/'training/controller_result.json')
    assert e['exit_code']==0 and e['all_owned_ended'] and not e['remaining'] and all(ps.creation(int(pid))!=b for pid,b in e['owned'].items())
    r=read(RUN/'training/report.json');assert r['complete'] and r['models']==24 and not r['cuda_initialized'];means={};exact=0
    for a in VARIANTS:
        part=[];representation,mode=a.rsplit('_',1)
        for seed in p['seeds']:
            d=RUN/'training'/a/f'seed{seed}';rec=read(d/'report.json');rows=read(d/'rows.json');done=read(d/'done.json')
            assert sha(d/'model.pt')==done['model_sha256'] and sha(d/'report.json')==done['report_sha256'] and sha(d/'rows.json')==rec['rows_sha256']
            assert len(rows)==8326 and sum(x['kind']>=0 for x in rows)==5822 and rec['steps']==1150
            per={v:sum(int(x['beneficial'])-int(x['harmful']) for x in rows if x['video']==v)/sum(x['video']==v and x['kind']>=0 for x in rows) for v in p['evaluate26']};assert per==rec['per_video'];part.append(per)
            if mode=='norm':
                old=CONTEXT/'training'/(representation+'_context')/f'seed{seed}'
                assert rows==read(old/'rows.json') and rec['tau']==read(old/'report.json')['tau']
                one=torch.load(d/'model.pt',map_location='cpu',weights_only=True);two=torch.load(old/'model.pt',map_location='cpu',weights_only=True)
                assert one.keys()==two.keys() and all(torch.equal(x,two[k]) for k,x in one.items());exact+=1
        means[a]={v:float(np.mean([x[v] for x in part])) for v in p['evaluate26']}
    c=comparisons(means,p['evaluate26']);assert c==r['paired'] and means==r['mean_seed_per_video'] and primary(c)==r['primary_pass'] and exact==12
    write(RUN/'independent_audit.json',dict(at=now().isoformat(),all24_models_rows_intervals_verified=True,original_norm_models_tensor_exact=12,all_original_norm_tau_rows_exact=True,report_sha256=sha(RUN/'training/report.json')))
    cat=read(ROOT/'experiments/project_history.json');card=next(c for c in cat['experiments'] if c['id']==IDENT)
    card.update(state='completed_positive' if r['primary_pass'] else 'completed_negative',done=now().strftime('%Y-%m-%d'),summary='Сохранение амплитуды: критерий '+('пройден.' if r['primary_pass'] else 'не пройден.'),obtained=(RUN/'training/report.json').relative_to(ROOT).as_posix()+' → paired: '+str(c),interpretation='Проверено одно преобразование68входов при равной ёмкости, одинаковых начальных весах и точном повторе12старых моделей. Это действие регионального считывателя на повторных26видео, не HOTA и не полное объяснение отказа.',decision=p['next']+' Основной критерий: '+str(r['primary_pass']))
    for i,met in enumerate((True,r['primary_pass'])):card['acceptance'][i].update(met=met,evidence=(RUN/('independent_audit.json' if i==0 else 'training/report.json')).relative_to(ROOT).as_posix())
    write(ROOT/'experiments/project_history.json',cat);write(RUN/'finish_recorded.json',dict(at=now().isoformat(),complete=True,primary_pass=r['primary_pass']))
    journal('итог сохранения уровня активности','*Получено.* '+card['obtained']+'\n\n*Вывод.* '+card['interpretation']+'\n\n*Решение.* '+card['decision']+'\n\nHOTA и процент к стенду не измерены; доля потолка не установлена.')
    from scripts.z13_full_box_supervision_20261025 import handoff
    handoff(IDENT+' завершён. '+card['obtained']+' '+card['decision']+' Требуются проверки/граф/архитектурнаявеха/коммит. Скрытые процессы;57чужихправок/.gitignoreсохранить; общееокно26.10до07:40МСК.')


if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    elif '--worker' in sys.argv:
        try:worker()
        except Exception as error:write(RUN/'failure.json',dict(at=now().isoformat(),error=repr(error)));raise
    elif '--finish' in sys.argv:finish()
