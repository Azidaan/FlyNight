"""Fixed three-seed learned candidate selection. CPU, frozen core/cache."""
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
RUN=ROOT/'runs/z13_candidate_ranker_20261021';CACHE=ROOT/'runs/z13_candidate_reserve_20261021'
IDENT='z13_candidate_ranker_v1';ATTEMPT='full';LIMIT=3600;T0=time.monotonic()


def check():
    if time.monotonic()-T0>LIMIT or (RUN/(ATTEMPT+'.stop')).exists() or now()>=datetime(2026,10,21,7,35,tzinfo=MSK):raise TimeoutError('Budget/save deadline')
def status(stage,**kw):
    check();write(RUN/'status.json',dict(state='running',phase=ATTEMPT,stage=stage,updated=now().isoformat(),elapsed_seconds=int(time.monotonic()-T0),**kw))
def journal(title,text):
    p=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';s=p.read_text(encoding='utf-8');a='## Журнал изменений\n';assert s.count(a)==1
    p.write_text(s.replace(a,a+f'\n**{now():%d.%m.%Y, %H:%M} MSK** — {title} `{IDENT}`\n\n'+text+'\n',1),encoding='utf-8')


def prepare():
    if RUN.exists():raise RuntimeError('Existing pilot must not be duplicated')
    previous=read(ROOT/'runs/z13_candidate_identity_20261021/report.json');assert not previous['primary_pass']
    base=read(ROOT/'runs/z13_candidate_identity_20261021/protocol.json');verify(base)
    names=['tracking/z13_candidate_ranker.py','scripts/z13_candidate_ranker_20261021.py','tests/test_z13_candidate_ranker.py','runs/z13_candidate_identity_20261021/report.json']
    p=dict(id=IDENT,created=now().isoformat(),authorization='Author authorized autonomous necessary runs/repairs/publication until07:40 MSK21.10.2026.',
        action='Replace fixed cosine/geometric candidate matching with a small learned choice over the same3 current peaks, observed seed activity and current tracker prediction. This changes the selected center, unlike prior confidence-only heads.',
        method='Shared LayerNorm68-linear16-tanh activity embedding, seed/query/difference/product/mean-candidate-context and8 geometric features; MLP88-32-1. Geometry ablation identical parameters but zero activity. This is not the full KeepTrack GNN and does not propagate explicit object IDs.',
        features='Geometry: candidate minus current tracker prediction /crop width; candidate minus last observed center /width; response height; difference from top1 response; nearest competing peak distance /width clipped1; search position/30. Unknown outcomes and GT are excluded from feature API.',
        training='All known fit20 rows only; BCE labels each candidate useful iff known-positive, err<=0.25*target-scale and err<current-prediction-error. Known absence rows all negative; unknown rows excluded. Class weight=min(20,negative/positive), training only. Fixed50 epochs, AdamW lr0.001,weight_decay0.01,batch128. No checkpoint selection. Fixed seeds0,1,2; mean logits ensemble, never select best seed. Always choose one candidate; no calibrated refusal is claimed.',
        primary='C ensemble-C top1: mean gain>=0.02 and paired-video95%lower>0. C ensemble-C geometry_ablation, C ensemble-C nearest_Kalman and C ensemble-R1 ensemble: all lower>0. All3 individual C seed gains against top1 must be nonnegative points. All conditions required.',
        secondary='D same contrasts; all individual seeds, full per-video results, fixed1/2/5% known nonuseful/unknown separately. No secondary rescue. Fit20/train26 repeatedly used, exploratory.',
        counterargument='Only801 positive fit20 frames; a learned selector can overfit or merely reproduce geometric preference. Strong controls and fixed3 seeds test these limitations; failure does not establish an intrinsic representation limit.',
        decision='No train26 tuning, no dev or HOTA unless a later separately specified stage is justified. If criteria fail, retain original centered readout; any further change needs a different measurable mechanism.',
        budget_seconds=3600,save_at='2026-10-21T07:35:00+03:00',hard_stop='2026-10-21T07:39:30+03:00',
        paper='https://arxiv.org/html/2103.16556v2 sections3.3-3.6: learn candidate matching from appearance/location/response, rather than fixed distances. Our supervised utility ranker differs and is not a reproduction.',
        sha256={**base['sha256'],**{n:sha(ROOT/n) for n in names}})
    write(RUN/'protocol.json',p)
    cp=ROOT/'experiments/project_history.json';cat=read(cp);assert not any(x['id']==IDENT for x in cat['experiments'])
    cat['experiments'].append(dict(id=IDENT,title='З13: обучаемый выбор центра среди трёх пиков',area='flyvis',state='planned',started=f'{now():%Y-%m-%d}',tags=['fly-problem-P8','KeepTrack','engineering-future'],
        question='Использует ли обучаемый узел выбора пиков активность FlyVis сверх геометрии, прежнего максимума и такого же узла на входе глаза?',
        hypothesis='Обучение сопоставлению текущих кандидатов с наблюдаемым образцом может использовать часть запаса, недоступного испытанному фиксированному расстоянию.',
        expected='Те же3 пика, фиксированные C/D/R1, активность и её обнуление при одинаковой архитектуре. Только fit20,50 эпох, три фиксированных сида без выбора лучшего; оценка на всех train26. CPU до3600с.',
        acceptance=[dict(criterion='Хеши/кэши/исходный top1 и8326 совпадающих строк всех вариантов проверены.',met=None),dict(criterion='C ансамбль−C top1: средний сдвиг>=2п.п.,нижняя95%повидеоДИ>0; точки всех3 сидов неотрицательны.',met=None),dict(criterion='C ансамбль лучше C без активности, ближайшего к прогнозу кандидата и такого же R1: все3 нижние95%повидеоДИ>0.',met=None)],
        summary='Зафиксирован один обучаемый узел выбора кандидатов; исход не получен.',done='',obtained='',interpretation='',decision='',cannot_claim='Нет HOTA, безопасного отказа и доказательства анатомии. Взвешенная BCE не калибрует вероятность. Ансамбль трёх моделей не доказывает независимость от любого сида. Это не полная реализация KeepTrack.',metrics=[],commits=[],files=names[:2],evidence=[dict(kind='file',ref='runs/z13_candidate_ranker_20261021/protocol.json',claim='Постановка и критерии до исходов'),dict(kind='file',ref='runs/z13_candidate_identity_20261021/report.json',claim='Предыдущая фиксированная версия не прошла критерии')]))
    write(cp,cat);journal('постановка обучаемого выбора кандидатов','*Постановка.* '+p['action']+'\n\n'+p['training']+'\n\nКритерии до исходов: '+p['primary']+'\n\nМеняется сам выбранный центр внутри считывания; это отличается от прежних внешних оценок качества фиксированного центра. Причина неудачи фиксированного сопоставления не установлена. Ядро не обучается; CPU до3600с, сохранение07:35, конец дерева до07:39:30.')


def pack(labels,sequences):
    import numpy as np
    from tracking.z13_candidate_ranker import candidate_inputs
    lookup={(e,f['t']):f for e,frames in sequences.items() for f in frames};X=[];S=[];G=[];M=[];Y=[];centers=[]
    for l in labels:
        f=lookup[(l['episode'],l['pos']+4)];x,s,g,m=candidate_inputs(f['centers'],f['features'],f['scores'],f['seed_center'],f['seed_features'],l['pred'],f['width'],l['pos'])
        y=np.zeros(3,np.float32);n=len(f['centers']);cs=np.zeros((3,2));cs[:n]=f['centers']
        if l['kind']==1:
            err=np.linalg.norm(f['centers']-l['gt'],axis=1);ep=np.linalg.norm(np.asarray(l['pred'])-l['gt']);y[:n]=(err<=.25*l['scale'])&(err<ep)
        X.append(x);S.append(s);G.append(g);M.append(m);Y.append(y);centers.append(cs)
    return tuple(np.asarray(v) for v in (X,S,G,M,Y,centers))


def predict(torch,model,arrays):
    import numpy as np
    out=[];model.eval()
    with torch.no_grad():
        for left in range(0,len(arrays[0]),512):
            t=[torch.as_tensor(a[left:left+512],device='cpu') for a in arrays[:4]];out.append(model(*t).cpu().numpy())
    return np.concatenate(out)


def make_rows(labels,arrays,logits):
    import numpy as np
    out=[];js=np.argmax(logits,axis=1)
    for i,l in enumerate(labels):
        j=int(js[i]);center=arrays[5][i,j];r={k:l[k] for k in ('video','label','kind','pos','pred')}
        r.update(center=center.tolist(),p_present=float(logits[i,j]),candidate=j,intervene=True)
        if l['kind']==1:
            err=float(np.linalg.norm(center-l['gt']));ep=float(np.linalg.norm(np.asarray(l['pred'])-l['gt']))
            r.update(err=err,err_pred=ep,scale=l['scale'],useful=bool(err<=.25*l['scale'] and err<ep))
        elif l['kind']==0:r['useful']=False
        out.append(r)
    return out


def worker():
    import gc
    import numpy as np
    import torch
    torch.set_default_device('cpu');torch.set_num_threads(2)
    from tracking.z13_candidate_ranker import make_ranker
    from scripts.z13_candidate_identity_20261021 import load
    from scripts.z13_continue import paired,per_video_useful,coverage_table
    p=read(RUN/'protocol.json');verify(p);results={};seedpv={};manifest={};trainlabels=read(CACHE/'fit/labels.json');evlabels=read(CACHE/'evaluate/labels.json')
    for ai,arm in enumerate(('C','D','R1')):
        status('Чтение проверенных кандидатов',variant=arm,done=0,total=1,arms_done=ai,arms_total=3)
        seq,m=load('fit',arm);manifest.update(m);ta=pack(trainlabels,seq)
        ids=np.asarray([i for i,l in enumerate(trainlabels) if l['kind']>=0]);valid=ta[3][ids];y=ta[4][ids]
        positives=float((y*valid).sum());negatives=float(valid.sum()-positives);weight=min(20.,negatives/max(positives,1.))
        seq,m=load('evaluate',arm);manifest.update(m);ea=pack(evlabels,seq);del seq
        for mode,activity in [('activity',True),('geometry',False)]:
            folder=RUN/arm/mode;logits=[];seedpv[arm+'_'+mode]={}
            for seed in (0,1,2):
                sd=folder/f'seed{seed}';done=sd/'done.json'
                torch.manual_seed(seed);model=make_ranker(torch,activity)
                if done.exists():model.load_state_dict(torch.load(sd/'model.pt',map_location='cpu',weights_only=True))
                else:
                    rng=np.random.default_rng(seed);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01);history=[]
                    train=[torch.as_tensor(a[ids],device='cpu') for a in ta[:5]]
                    for epoch in range(50):
                        status('Обучение только нового считывателя на fit20',variant=arm+' '+mode+' seed'+str(seed),done=epoch,total=50,arms_done=ai,arms_total=3)
                        model.train();losses=[]
                        for start in range(0,len(ids),128):
                            if start==0:order=rng.permutation(len(ids))
                            ix=torch.as_tensor(order[start:start+128],device='cpu');x,s,g,mask,target=[v[ix] for v in train]
                            score=model(x,s,g,mask);loss=torch.nn.functional.binary_cross_entropy_with_logits(score,target,pos_weight=torch.tensor(weight),reduction='none')
                            loss=(loss*mask).sum()/mask.sum();assert torch.isfinite(loss)
                            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.);opt.step();losses.append(float(loss.detach()))
                        history.append(float(np.mean(losses)))
                    sd.mkdir(parents=True,exist_ok=True);temp=sd/'model.tmp.pt';torch.save(model.state_dict(),temp);temp.replace(sd/'model.pt')
                    write(sd/'training.json',dict(seed=seed,epochs=50,loss=history,positive_candidates=positives,negative_candidates=negatives,pos_weight=weight,known_rows=len(ids),model_sha256=sha(sd/'model.pt')))
                    write(done,dict(ended=now().isoformat()))
                pred=predict(torch,model,ea);logits.append(pred);rr=make_rows(evlabels,ea,pred);seedpv[arm+'_'+mode][str(seed)]=per_video_useful(rr)
            rr=make_rows(evlabels,ea,np.mean(logits,axis=0));results[arm+'_'+mode]=rr;write(folder/'ensemble_rows.json',rr)
        del ta,ea;gc.collect()
        for baseline,source in [('top1','top1'),('kalman','kalman')]:results[arm+'_'+baseline]=read(ROOT/f'runs/z13_candidate_identity_20261021/{arm}/{source}_rows.json')
    pv={k:per_video_useful(v) for k,v in results.items()};pairs=[]
    for arm in ('C','D'):pairs.extend([(arm+'_activity',arm+'_'+b) for b in ('top1','geometry','kalman')]);pairs.append((arm+'_activity','R1_activity'))
    pairs.append(('D_activity','C_activity'));comp={a+'-'+b:paired(pv[a],pv[b]) for a,b in pairs}
    seedcontrast={arm:{s:paired(v,pv[arm+'_top1']) for s,v in seedpv[arm+'_activity'].items()} for arm in ('C','D','R1')}
    c=comp['C_activity-C_top1'];primary=c['mean_per_video']>=.02 and c['ci95'][0]>0 and all(comp[k]['ci95'][0]>0 for k in ('C_activity-C_geometry','C_activity-C_kalman','C_activity-R1_activity')) and all(v['mean_per_video']>=0 for v in seedcontrast['C'].values())
    verify(p);write(RUN/'cache_manifest.json',manifest);write(RUN/'seed_metrics.json',dict(per_video=seedpv,against_top1=seedcontrast))
    write(RUN/'report.json',dict(primary_pass=primary,paired=comp,seeds_against_top1=seedcontrast,summary={k:dict(rows=len(v),useful_mean_video=float(np.mean(list(pv[k].values()))),coverage=coverage_table(v)) for k,v in results.items()},limitations='Exploratory repeatedly used train26; fixed3-seed ensemble, no best seed selection; no calibrated rejection or HOTA.'))


def main():
    global ATTEMPT,LIMIT
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--worker',action='store_true');ap.add_argument('--attempt',default='full');ap.add_argument('--budget',type=int,default=3600);a=ap.parse_args()
    if a.prepare:prepare();return
    ATTEMPT=a.attempt;LIMIT=a.budget;os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
    if not a.worker:raise ValueError('Choose prepare/worker')
    try:worker();write(RUN/'status.json',dict(state='completed',phase=ATTEMPT,stage='Обучаемый выбор кандидатов сохранён',updated=now().isoformat()))
    except Exception as exc:write(RUN/'status.json',dict(state='failed',phase=ATTEMPT,stage=str(exc),updated=now().isoformat()));raise


if __name__=='__main__':main()
