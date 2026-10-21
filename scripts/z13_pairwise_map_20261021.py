"""One-factor replacement of candidate BCE by within-frame ranking loss."""
from __future__ import annotations
import argparse
from datetime import datetime
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,sha,verify,now,MSK
RUN=ROOT/'runs/z13_pairwise_map_20261021';SOURCE=ROOT/'runs/z13_candidate_data104_20261021';CACHE=ROOT/'runs/z13_candidate_reserve_20261021';DEV=ROOT/'runs/z13_map_transfer104_20261021'
IDENT='z13_pairwise_map_v1';ATTEMPT='train';LIMIT=900;T0=time.monotonic()


def check():
    if time.monotonic()-T0>LIMIT or (RUN/(ATTEMPT+'.stop')).exists() or now()>=datetime(2026,10,21,7,35,tzinfo=MSK):raise TimeoutError('Budget/save deadline')
def status(stage,**kw):
    check();write(RUN/'status.json',dict(state='running',phase=ATTEMPT,stage=stage,updated=now().isoformat(),elapsed_seconds=int(time.monotonic()-T0),**kw))
def journal(title,text):
    path=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';s=path.read_text(encoding='utf-8');a='## Журнал изменений\n';assert s.count(a)==1
    path.write_text(s.replace(a,a+f'\n**{now():%d.%m.%Y, %H:%M} MSK** — {title} `{IDENT}`\n\n'+text+'\n',1),encoding='utf-8')
def prepare():
    if RUN.exists():raise RuntimeError('Existing run; do not duplicate')
    assert read(DEV/'finish_recorded.json')['state']=='inconclusive'
    p=read(SOURCE/'protocol.json');verify(p);names=['tracking/z13_pairwise_candidate_loss.py','tests/test_z13_pairwise_candidate_loss.py','scripts/z13_pairwise_map_20261021.py','runs/z13_map_transfer104_20261021/report.json','runs/z13_map_transfer104_20261021/protocol.json','runs/z13_candidate_data104_20261021/cache_manifest.json']
    names.extend(['runs/z13_map_transfer104_20261021/cache_manifest.json','runs/z13_map_transfer104_20261021/evaluate/labels.json','runs/z13_map_transfer104_20261021/seed_metrics.json','runs/z13_candidate_data104_20261021/seed_metrics.json'])
    for arm in ('C','D','R1'):
        names.append(f'runs/z13_candidate_data104_20261021/{arm}/geometry/ensemble_rows.json')
        names.extend(f'runs/z13_map_transfer104_20261021/{arm}/{mode}_rows.json' for mode in ('map','top1','nearest'))
        for seed in (0,1,2):names.append(f'runs/z13_candidate_data104_20261021/{arm}/geometry/seed{seed}/training.json')
    protocol=dict(id=IDENT,created=now().isoformat(),authorization='Explicit author autonomous necessary runs/repairs/publication until07:40 MSK21.10.2026.',
        action='Change only the training objective inside the map-reading candidate selector: independent weighted BCE to pairwise logistic ordering. Same fixed104 train examples, geometry-only model, initialization seeds0/1/2, optimizer and sample/update budgets. No new core training or extraction.',
        loss='For valid candidate i useful=1 and j useful=0 in the same known frame: softplus(score_j-score_i), averaged over all such pairs in the batch. Tied labels and padding ignored. Empty-pair batch: differentiable zero loss; AdamW step still performed and counted, recorded separately. The BCE positive class weight is part of the replaced objective and is not used. No presence probability or refusal claim.',
        training='Identical train104 input order and matched_batches(known104,reference_known20,50,128,seed),1150updates/145950presentedframes; same geometry factory, AdamW lr.001 wd.01,gradientclip10,all3seeds mean logits, no best checkpoint/seed. Source train20 normalizer and all candidate/core caches unchanged.',
        primary='C_rank-C_BCE104: mean>=0.02 and paired-video95%lower>0; C_rank-C_nearest, C_rank-C_top1 and C_rank-R1_rank: all lower>0. All3 matched C seed point gains over BCE104 nonnegative. C primary, D secondary cannot rescue.',
        transfer='Only if all train26 main criteria pass, apply the same9 models without fitting to the existing full35dev cache. Exactly the same main comparisons/gates on dev35, including >=2pp over BCE104. No model/objective/seed selection on dev. All old dev outcomes already viewed; this remains exploratory reuse, not independent confirmation.',
        counterargument='Ranking may discard useful absolute information or have too few positive-negative pairs. No new information is added, and nearest-to-prediction may remain as good. The old failure does not prove BCE caused it.',
        decision='If train26 main fails, no new dev evaluation and stop this objective variant. If dev fails, no deployment/retuning. Record partial positive contrasts without replacing the primary gate. Original centered readout retained until full stand proof.',
        budget_seconds=900,transfer_budget_seconds=120,save_at='2026-10-21T07:35:00+03:00',hard_stop='2026-10-21T07:39:30+03:00',
        references=[dict(url='https://www.microsoft.com/en-us/research/wp-content/uploads/2005/08/icml_ranking.pdf',section='3, eq1-3',supports='Pairwise logistic ranking of scores; not a tracking or FlyVis effect.'),dict(url='https://arxiv.org/abs/2205.11761',section='abstract',supports='Ranking-based Siamese tracking motivates relative candidate training. Our simple RankNet loss is not a reproduction of its two losses.')],
        prohibited='No new GPU extraction,core training,downloads,protected75,HOTA,dev fitting or tuning.',sha256={**p['sha256'],**{n:sha(ROOT/n) for n in names}})
    write(RUN/'protocol.json',protocol);cp=ROOT/'experiments/project_history.json';cat=read(cp);assert not any(c['id']==IDENT for c in cat['experiments'])
    cat['experiments'].append(dict(id=IDENT,title='З13: порядок пиков вместо независимой оценки',area='flyvis',state='ready',started=f'{now():%Y-%m-%d}',tags=['fly-problem-P8','engineering-future'],
        question='Даст ли обучение относительному порядку тех же пиков выигрыш сверх BCE-считывателя карты и ближайшего к прогнозу кандидата?',hypothesis='Парное обучение может лучше соответствовать выбору одного кандидата; причинная роль прежней функции потерь не установлена.',expected='Одна замена функции потерь, те же104train/кэши/модели/3сида/бюджет; CPU до900с. Дев35 — только при полном прохождении train26, неизменные модели до120с. Данных и расчётов ядра не добавляется.',
        acceptance=[dict(criterion='Исходные кэши/хеши, равные1150 шагов и145950 предъявлений,3сида, отсутствие оценки в обучении проверены.',met=None),dict(criterion=protocol['primary'],met=None),dict(criterion='Условный фиксированный перенос: те же основные критерии на всех35dev без подбора; если train-гейт не пройден, этот критерий остаётся невыполненным, dev не считается.',met=None)],summary='Зарегистрирована одна замена цели обучения считывателя; исход не получен.',done='',obtained='',interpretation='',decision='',cannot_claim='Нет доказательства, что BCE была причиной провала. Парный балл не калиброванная вероятность присутствия или полезности. Повторно использованные26/35 — не слепое подтверждение; нет HOTA и анатомической причинности.',metrics=[],commits=[],files=['tracking/z13_pairwise_candidate_loss.py','scripts/z13_pairwise_map_20261021.py'],evidence=[dict(kind='file',ref='runs/z13_pairwise_map_20261021/protocol.json',claim='Однофакторные критерии до новых моделей/исходов'),dict(kind='file',ref='runs/z13_map_transfer104_20261021/report.json',claim='Предыдущая карта переносится относительно максимума, но не подтверждена сверх прогноза')]))
    write(cp,cat);journal('постановка парного порядка пиков','*Постановка.* '+protocol['action']+'\n\n'+protocol['loss']+'\n\nКритерии до исходов: '+protocol['primary']+'\n\n'+protocol['transfer']+'\n\nОпора механизма: Burges et al.2005, раздел3/формулы1–3 (парная логистическая потеря), https://www.microsoft.com/en-us/research/wp-content/uploads/2005/08/icml_ranking.pdf ; Tang & Ling2022, https://arxiv.org/abs/2205.11761 (ранжирование кандидатов в сиамском трекере). Их численные эффекты к FlyVis не переносятся, это не воспроизведение полной статьи. Одна CPU-реализация, не новый перебор гиперпараметров; до900с плюс условный перенос120с. Исходная версия сохраняется, ядро не обучается.')


def arrays(labels,parts):
    import numpy as np
    from scripts.z13_candidate_ranker_20261021 import pack
    return tuple(np.concatenate(v) for v in zip(*(pack(l,s) for l,s in zip(labels,parts))))


def summarize(results,seeds,split):
    import numpy as np
    from scripts.z13_continue import paired,per_video_useful,coverage_table
    pv={k:per_video_useful(v) for k,v in results.items()};comp={}
    for arm in ('C','D'):
        for base in (arm+'_BCE',arm+'_top1',arm+'_nearest','R1_rank'):comp[arm+'_rank-'+base]=paired(pv[arm+'_rank'],pv[base])
    c=comp['C_rank-C_BCE'];old=read(SOURCE/'seed_metrics.json')['per_video']['C_geometry'] if split=='train26' else read(DEV/'seed_metrics.json')['C']
    seedgain={s:paired(v,old[s]) for s,v in seeds['C'].items()};passed=c['mean_per_video']>=.02 and c['ci95'][0]>0 and all(v['mean_per_video']>=0 for v in seedgain.values()) and all(comp[k]['ci95'][0]>0 for k in ('C_rank-C_top1','C_rank-C_nearest','C_rank-R1_rank'))
    return dict(split=split,primary_pass=passed,paired=comp,C_seed_gain=seedgain,summary={k:dict(rows=len(v),useful_mean_video=float(np.mean(list(pv[k].values()))),coverage=coverage_table(v)) for k,v in results.items()},limitations='Only objective changed; fixed3 seeds, exploratory reused data; no calibrated confidence or HOTA.')


def worker(transfer=False):
    import gc
    import numpy as np
    import torch
    torch.set_default_device('cpu');torch.set_default_dtype(torch.float32);torch.set_num_threads(2)
    import scripts.z13_candidate_identity_20261021 as reader
    from scripts.z13_candidate_ranker_20261021 import pack,predict,make_rows
    from tracking.z13_candidate_ranker import make_ranker
    from tracking.z13_matched_data_budget import matched_batches
    from tracking.z13_pairwise_candidate_loss import pairwise_candidate_loss
    from scripts.z13_continue import per_video_useful
    p=read(RUN/'protocol.json');verify(p);results={};seeds={};manifest={}
    if transfer:
        end=read(RUN/'train_controller_result.json');assert end['worker_ended'] and end['exit_code']==0 and not end['hard_timeout'] and read(RUN/'report_train26.json')['primary_pass']
        labels=read(DEV/'evaluate/labels.json')
    else:
        ol=read(CACHE/'fit/labels.json');el=read(SOURCE/'evaluate/labels.json');tl=ol+el;ids=np.asarray([i for i,l in enumerate(tl) if l['kind']>=0]);nref=sum(l['kind']>=0 for l in ol);assert len(ids)==20778 and nref==2919
        labels=read(CACHE/'evaluate/labels.json')
    for ai,arm in enumerate(('C','D','R1')):
        status('Фиксированные кандидаты и парные примеры',variant=arm,done=ai,total=3)
        root=reader.CACHE
        if not transfer:
            a,m=reader.load('fit',arm);manifest.update(m)
            try:reader.CACHE=SOURCE;b,m=reader.load('evaluate',arm)
            finally:reader.CACHE=root
            manifest.update(m);ta=arrays([ol,el],[a,b]);del a,b
            train=[torch.as_tensor(v[ids],device='cpu') for v in ta[:5]];del ta
        try:
            reader.CACHE=DEV if transfer else CACHE;seq,m=reader.load('evaluate',arm)
        finally:reader.CACHE=root
        manifest.update(m);ea=pack(labels,seq);del seq;preds=[];seeds[arm]={}
        for seed in (0,1,2):
            sd=RUN/arm/f'seed{seed}';torch.manual_seed(seed);model=make_ranker(torch,False)
            if (sd/'done.json').exists():assert sha(sd/'model.pt')==read(sd/'training.json')['model_sha256'];model.load_state_dict(torch.load(sd/'model.pt',map_location='cpu',weights_only=True))
            elif transfer:raise RuntimeError('Transfer cannot train a model')
            else:
                opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01);losses=[];presented=empty=pairs=0
                for step,ix in enumerate(matched_batches(len(ids),nref,50,128,seed)):
                    if step%23==0:status('Парное обучение только считывателя',variant=arm+' seed'+str(seed),done=step,total=1150)
                    check();x,s,g,mask,target=[v[torch.as_tensor(ix,device='cpu')] for v in train];score=model(x,s,g,mask);loss,count=pairwise_candidate_loss(torch,score,target,mask);assert torch.isfinite(loss)
                    opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.);opt.step();losses.append(float(loss.detach()));presented+=len(ix);empty+=int(count==0);pairs+=int(count)
                assert len(losses)==1150 and presented==145950 and pairs>0
                sd.mkdir(parents=True,exist_ok=True);tmp=sd/'model.tmp.pt';torch.save(model.state_dict(),tmp);tmp.replace(sd/'model.pt')
                write(sd/'training.json',dict(seed=seed,steps=len(losses),presented_rows=presented,known104=len(ids),reference_known20=nref,empty_pair_batches=empty,observed_training_pairs=pairs,loss=losses,model_sha256=sha(sd/'model.pt')));write(sd/'done.json',dict(ended=now().isoformat()))
            pred=predict(torch,model,ea);preds.append(pred);seeds[arm][str(seed)]=per_video_useful(make_rows(labels,ea,pred))
        rr=make_rows(labels,ea,np.mean(preds,axis=0));results[arm+'_rank']=rr;write(RUN/('dev35' if transfer else 'train26')/arm/'rows.json',rr)
        results[arm+'_BCE']=read(DEV/arm/'map_rows.json') if transfer else read(SOURCE/arm/'geometry/ensemble_rows.json')
        for name,source in [('top1','top1'),('nearest','kalman')]:results[arm+'_'+name]=read(DEV/arm/(name+'_rows.json')) if transfer else read(ROOT/f'runs/z13_candidate_identity_20261021/{arm}/{source}_rows.json')
        if not transfer:del train
        del ea;gc.collect()
    expected=read(DEV/'cache_manifest.json') if transfer else read(SOURCE/'cache_manifest.json')
    assert all(expected[rel]==digest for rel,digest in manifest.items())
    verify(p);split='dev35' if transfer else 'train26';write(RUN/('cache_manifest_'+split+'.json'),manifest);write(RUN/('seeds_'+split+'.json'),seeds);write(RUN/('report_'+split+'.json'),summarize(results,seeds,split))


def main():
    global ATTEMPT,LIMIT
    a=argparse.ArgumentParser();a.add_argument('--prepare',action='store_true');a.add_argument('--worker',action='store_true');a.add_argument('--attempt',choices=['train','transfer'],default='train');a.add_argument('--budget',type=int,default=900);x=a.parse_args()
    if x.prepare:prepare();return
    if not x.worker:raise ValueError('Choose action')
    ATTEMPT=x.attempt;LIMIT=x.budget;os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
    try:worker(ATTEMPT=='transfer');write(RUN/'status.json',dict(state='completed',phase=ATTEMPT,stage='Парное считывание сохранено',updated=now().isoformat()))
    except Exception as exc:write(RUN/'status.json',dict(state='failed',phase=ATTEMPT,stage=str(exc),updated=now().isoformat()));raise


if __name__=='__main__':main()
