"""Template-supported confidence: frozen features -> paired ten-arm CPU replay."""
from pathlib import Path
from datetime import datetime,timedelta
import os
PHASE=os.environ.get('Z13_UNCERTAINTY_PHASE','')
if PHASE!='features':os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']=os.environ['OPENBLAS_NUM_THREADS']='2'
import sys,inspect,hashlib,subprocess,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
from scripts import z13_history_clock_20261025 as clock
from scripts import z13_history_clock_recovery1_20261025 as clock_recovery
from scripts import z13_region_readout_20261025 as decoder
from scripts import z13_region_readout_recovery1_20261025 as decoder_recovery
from scripts.z13_localization_labels_20261023 import verify_sources
from scripts.z13_hota_deadline_guard import WindowsProcesses
from tracking.z13_ridge_uncertainty import ridge_posterior
RUN=ROOT/'runs/z13_ridge_uncertainty_20261025'
IDENT='z13_ridge_uncertainty_v1'
ARMS=tuple(a+'_'+mode for a in ('D','C','R1') for mode in ('old','posterior','constant'))+('geometry',)
ALPHAS=(0.,.25)
BUDGETS={'features':1800,'replay':900}

def journal(title,text):
    from scripts import z13_localization_labels_20261023 as notes
    notes.IDENT=IDENT;notes.journal(title,text)

def status(stage,**kw):
    p=read(RUN/PHASE/'protocol.json')
    if now()>=datetime.fromisoformat(p['save_at']) or (RUN/PHASE/'stop.request').exists():raise TimeoutError('Original uncertainty phase deadline')
    write(RUN/PHASE/'status.json',dict(state='running',stage=stage,updated=now().isoformat(),**kw))

def feature_source():
    s=clock_recovery.source()
    changes=[
        ("reference=old.episode_features(torch,arm,bb,hx,geom.neighbors,need) if bi<2 else None","reference=None"),
        ("F=clock_features(torch,arm,bb,hx,geom.neighbors,need,ff)","F=old.episode_features(torch,arm,bb,hx,geom.neighbors,need);reference=F"),
        ('maps=[];ids=[];rows=[]','maps=[];stds=[];ids=[];rows=[]'),
        ('scores=ridge_scores(xe,target,xs,[lam])[0]','scores,uncertainty,posterior_metadata=ridge_posterior(xe,target,xs,lam)'),
        ("rows.append(row);maps.append(score);ids.append(l['label'])","rows.append(row);maps.append(score);stds.append(uncertainty[k]);ids.append(l['label'])"),
        ('scores=np.asarray(maps),labels=np.asarray(ids,np.int64)','scores=np.asarray(maps),stds=np.asarray(stds),labels=np.asarray(ids,np.int64)'),
        ("status('Frozen recurrent clock at actual frame intervals'","status('Template-supported uncertainty; original unit-clock features'"),
        ("clock_rule='Actual integer frame gaps; linear stimulus interpolation; zero gap no state evolution; original nominal30Hz/substep unchanged'","feature_rule='Original unit-clock features and unchanged ridge mean; conditional weight posterior with template residual noise only'")]
    for a,b in changes:assert s.count(a)==1,a;s=s.replace(a,b)
    return s

def comparisons(values):
    from scripts.z13_continue import paired
    videos=list(values['D_posterior']);out={'Dposterior-baseline':paired(values['D_posterior'],{v:0. for v in videos})}
    for a in ('D_old','D_constant','C_posterior','R1_posterior','geometry'):out['Dposterior-'+a]=paired(values['D_posterior'],values[a])
    return out

def primary(c):
    return all(c[k]['mean_per_video']>=.02 for k in ('Dposterior-baseline','Dposterior-D_old','Dposterior-D_constant')) and all(x['ci95'][0]>0 for x in c.values())

def load_uncertainty_maps():
    r=read(RUN/'features/report.json');assert r['complete'];out={a:{} for a in ('D','C','R1')}
    for name,h in r['manifest'].items():
        p=ROOT/name;assert sha(p)==h and sha(p.with_suffix('.npz'))==read(p)['sha256']
        with np.load(p.with_suffix('.npz')) as z:
            assert z['stds'].shape==z['scores'].shape and np.isfinite(z['stds']).all() and np.all(z['stds']>=0)
            for label,sd in zip(z['labels'],z['stds']):assert int(label) not in out[p.parent.name];out[p.parent.name][int(label)]=sd
    assert all(len(x)==8326 for x in out.values());return out

def replay_source():
    s=decoder_recovery.worker_source()
    changes=[
        ('from tracking.z13_region_readout import region_probabilities,cost_row','from tracking.z13_ridge_uncertainty import region_probabilities\n    from tracking.z13_region_readout import cost_row'),
        ("p=read(RUN/'protocol.json');verify_sources(p)","p=read(RUN/'protocol.json');verify_sources(p);uncertainty_maps=load_uncertainty_maps()"),
        ("for split in ('fit','evaluate'):","for split in ('evaluate',):"),
        ("field=geometry if arm=='geometry' else maps[split,arm][l['label']]","if arm=='geometry':field=geometry;mode='old';u=None\n            else:\n                name,mode=arm.rsplit('_',1);field=maps[split,name][l['label']];u=uncertainty_maps[name][l['label']]"),
        ('region_probabilities(field,geom.points,projected,supported)','region_probabilities(field,geom.points,projected,supported,u,mode)'),
        ("choices={a:max(range(len(ALPHAS)),key=lambda j:np.mean(list(fit[a][j].values()))) for a in ARMS}","choices={a:1 for a in ARMS} # Fixed prioralpha0.25, no retuning"),
        ("contrasts={'D-baseline':paired(chosen['D'],{v:0. for v in p['evaluate26']})}\n    for arm in ('C','R1','geometry'):contrasts['D-'+arm]=paired(chosen['D'],chosen[arm])\n    passed=contrasts['D-baseline']['mean_per_video']>=.02 and all(v['ci95'][0]>0 for v in contrasts.values())",'contrasts=comparisons(chosen)\n    passed=primary(contrasts)'),
        ('exact_baseline_videos=46','exact_baseline_videos=26')]
    for a,b in changes:assert s.count(a)==1,a;s=s.replace(a,b)
    return s.replace('total=46','total=26')

def worker():
    phase=os.environ['Z13_UNCERTAINTY_PHASE'];p=read(RUN/phase/'protocol.json');verify_sources(p)
    if phase=='features':
        code=feature_source();ns={**vars(clock),'RUN':RUN,'PHASE':phase,'status':status,'ridge_posterior':ridge_posterior}
        assert hashlib.sha256(code.encode()).hexdigest()==p['executed_features_sha256'];fn='feature_worker'
    else:
        code=replay_source();ns={**vars(decoder),'RUN':RUN/'replay','ARMS':ARMS,'ALPHAS':ALPHAS,'load_uncertainty_maps':load_uncertainty_maps,'comparisons':comparisons,'primary':primary}
        assert hashlib.sha256(code.encode()).hexdigest()==p['executed_worker_sha256'];fn='worker'
    exec(compile(code,str(Path(__file__)),'exec'),ns);ns[fn]()

def prepare():
    assert not RUN.exists();ps=WindowsProcesses();prior=ROOT/'runs/z13_spatial_bins_gate_20261025'
    assert read(prior/'publication.json')['push_completed'] and not read(prior/'report.json')['admission_pass']
    c=read(prior/'recovery1/controller_result.json');assert c['exit_code']==0 and c['all_owned_ended'] and not c['remaining'] and all(ps.creation(int(pid))!=b for pid,b in c['owned'].items())
    old=read(clock.RUN/'protocol.json');verify_sources(old);start=now();stop=min(start+timedelta(seconds=2820),datetime.fromisoformat('2026-10-26T07:35:00+03:00'))
    names=[Path(__file__).relative_to(ROOT).as_posix(),'tracking/z13_ridge_uncertainty.py','tests/test_z13_ridge_uncertainty.py','scripts/z13_history_clock_recovery1_20261025.py','runs/z13_spatial_bins_gate_20261025/report.json']
    p=dict(old,id=IDENT,created=start.isoformat(),timezone='Europe/Moscow',pipeline_save_at=stop.isoformat(),pipeline_hard_stop=(stop+timedelta(seconds=20)).isoformat(),phase_budgets=BUDGETS,videos=old['evaluate26'],fit20=[],
        action='Change only region readout confidence: subtract1.96times the template-only conditional posterior standard deviation from the old standardized region response. Mean target model and original unit-clock core/history unchanged. Per-query constant-penalty control uses the same full-field mean standard deviation. Old/variable/constant forC/D/R1 plusunchanged geometry,10arms; same fixedalpha0.25 and null0.',
        formula='n=721;G=Xc.TXc/n;w=(G+lambdaI)^-1 Xc.Tyc/n, exactly originalridge. df=trace(G(G+lambdaI)^-1). noise=sum((Xcw-yc)^2)/max(n-df-1,1). Sigma_w=noise/n*(G+lambdaI)^-1. Latentstd(x)=sqrt((x-mu).TSigma_w(x-mu)); fixed empirical intercept. Region logit=mean((score-score.mean)/score.std)-1.96*mean(latentstd/score.std). No query truth, normalization refit, newlambda, alpha/threshold fitting or core training.',
        primary='Means Dposterior-baseline,Dposterior-Dold,Dposterior-Dconstant each>=2pp. All6paired-video95%lower>0 includingCposterior,R1posterior,geometry. All conditions required; no favorable arm/subset selection. Mean-score/center parity is technical only, not a positive result.',
        technical='All8326old feature-derived centers<0.5px/scores<0.001 for eachC/D/R1; all stored means/stds finite with exact label bindings and original8326rows. Posterior mean equals oldridge in direct-matrix unit checks. All26raw tracker outputs and everyold arm/geometry action exactly equal prior mean-region prototype. First2Dgroups conservative price fits original1800remaining; each phase independently bounded; completed chunks not repeated.',
        literature='Rasmussen and Williams,Gaussian Processes for Machine Learning,MITPress2006,section2.1 eq2.8-2.9 https://gaussianprocess.org/gpml/chapters/RW2.pdf : conditional parameter covariance/predictive distribution for linear Gaussian regression. Our residual plug-in, fixed empirical intercept, correlated hexagonal features and1.96penalty are an engineering prototype, not calibrated95%coverage or posterior probability of useful tracking action.',
        counterargument='The model can be confidently wrong about identity; spatially correlated721columns violate iidnoise assumptions and underestimate uncertainty. A simple global penalty may explain any benefit, so the matched constant branch is mandatory. This changes an internal confidence readout, not measured feature quality or biological dynamics. Repeatedtrain26 exploratory; noHOTAclaim.',
        next='Pass -> separate bounded runtime parity/price and frozen-policy matched HOTA. Fail -> preserve this conditional-covariance confidence implementation; do not retune1.96, lambda, alpha or thresholds on26. Select a genuinely different target/state mechanism with measured prerequisite, not another scalar penalty.',
        ranking=[
          '1 Conditional ridge support confidence: analytic current target model; no core training; unlike old learned14features/height; paired constant-control separates stricter refusal. Sources old normalizations/clock parity and GPML2.1.',
          '2 Target-specific hard negative weighting: DiMP/DaSiamRPN analogy; requires reliable causal neighbor masks at observation, not future truth; not yet implemented.',
          '3 KYS-like spatial correspondence in readout memory: new planned state node; old efference_eye_v1/v2 failures and coordinate ambiguity require prerequisite.',
          '4 Structured four-bin regions: inconclusive exposure2.28%[1.64;2.93] in runs/z13_spatial_bins_gate_20261025/report.json; not admitted, not resumed.',
          '5 Longer online sample memory: existing observed_memory criterion failed; confidence-based reliable write rule needed before another variant.',
          '6 End-to-end core via applied target loss: prior aligned_core failed; expensive without a demonstrated readout gate; deferred.',
          '7 More core epochs: core_continue did not improve; not repeat same setup.',
          '8 More readout data: candidate_data104 criterion failed; not repeat unchanged head.',
          '9 Actual-frame clock: runs/z13_history_clock_20261025/replay/report.json negative; not retune timing.',
          '10 Semantic/balanced/utility scalar loss changes: region_semantic/balanced/value primary gates failed; action stopped.',
          '11 Scalar mean->max: runs/z13_region_pooling_20261025/report.json negative; fixed pooling branch stopped.',
          '12 More static context/channel products: region_context/product primary gates failed; not repeat same descriptors.'],
        limits='New2820s pipeline, feature1800s -> CPUreplay900s, oneGPU mutex and separate independent phase controllers. Hidden, no downloads/dev/HOTA/75/coretraining. Sunday25Oct no07:40restriction; author cutoffMonday26Oct07:40, save07:35/hard07:39:30; smaller per-stage deadlines never extend.',
        executed_features_sha256=hashlib.sha256(feature_source().encode()).hexdigest(),executed_worker_sha256=hashlib.sha256(replay_source().encode()).hexdigest(),sha256={**old['sha256'],**{n:sha(ROOT/n) for n in names}})
    compile(feature_source(),__file__,'exec');compile(replay_source(),__file__,'exec');write(RUN/'protocol.json',p);write(RUN/'status.json',dict(state='ready',stage='Template-only uncertainty, ten matched conditions',updated=now().isoformat()))
    cat=read(ROOT/'experiments/project_history.json');assert not any(x['id']==IDENT for x in cat['experiments'])
    cat['experiments'].append(dict(id=IDENT,title='Уверенность считывателя по опоре признаков в наблюдённом образце',area='flyvis',state='ready',started=start.strftime('%Y-%m-%d'),tags=['fly-problem-P8'],question='Улучшит ли штраф за неопределённость модели цели полезные назначения сверх прежней D и одинакового штрафа всем областям?',hypothesis='Рабочая гипотеза: часть высоких неверных ответов может быть экстраполяцией вне признаков образца. Это не установленная причина; уверенная ошибка идентичности может остаться.',expected=p['action'],acceptance=[dict(criterion=p['technical'],met=None),dict(criterion=p['primary'],met=None)],obtained='',interpretation='',decision='',cannot_claim=p['counterargument'],evidence=[dict(kind='file',ref=(RUN/'protocol.json').relative_to(ROOT).as_posix(),claim='Ранжирование12вариантов, формула, равные контроли и критерии до расчёта')]))
    write(ROOT/'experiments/project_history.json',cat)
    journal('постановка опоры уверенности в наблюдённом образце','*Постановка.* '+p['action']+'\n\n*Формула.* '+p['formula']+'\n\n*Критерии.* '+p['primary']+' '+p['technical']+'\n\n*Возражение.* '+p['counterargument']+'\n\n*Отбор.* '+' '.join(p['ranking'])+'\n\n*Бюджет.* '+p['limits']+' Общий save '+p['pipeline_save_at']+' / hard '+p['pipeline_hard_stop']+'. Источник runs/z13_ridge_uncertainty_20261025/protocol.json. '+p['literature'])

def phase_protocol(phase):
    p=read(RUN/'protocol.json');stop=min(now()+timedelta(seconds=BUDGETS[phase]),datetime.fromisoformat(p['pipeline_save_at']));extra={}
    if phase=='replay':extra[(RUN/'features/report.json').relative_to(ROOT).as_posix()]=sha(RUN/'features/report.json')
    q=dict(p,device='cuda' if phase=='features' else 'cpu',budget_seconds=BUDGETS[phase],save_at=stop.isoformat(),hard_stop=min(stop+timedelta(seconds=20),datetime.fromisoformat(p['pipeline_hard_stop'])).isoformat(),stop_request_path=(RUN/phase/'stop.request').relative_to(ROOT).as_posix(),sha256={**p['sha256'],**extra})
    write(RUN/phase/'protocol.json',q);write(RUN/phase/'status.json',dict(state='ready',stage=phase,updated=now().isoformat()))

def finish():
    p=read(RUN/'protocol.json');verify_sources(p);r=read(RUN/'replay/report.json');assert r['complete'] and r['exact_baseline_videos']==26
    values={a:{} for a in ARMS}
    for v in p['videos']:
        path=RUN/'replay/videos'/(v+'.json');assert sha(path)==r['video_sha256'][v];record=read(path);assert record['checks']['raw_replay_exact']
        prior={x['uid']:x for x in read(decoder_recovery.RUN/'videos'/(v+'.json'))['rows']}
        for row in record['rows']:
            old=prior[row['uid']];assert all(row[k]==old[k] for k in ('uid','video','frame','tid','kind','label','split'))
            for a in ('D','C','R1'):assert row['arms'][a+'_old'][1]==old['arms'][a][1]
            assert row['arms']['geometry'][1]==old['arms']['geometry'][1]
        known=[x for x in record['rows'] if x['kind']>=0]
        for a in ARMS:values[a][v]=float(np.mean([int(x['arms'][a][1]['beneficial'])-int(x['arms'][a][1]['harmful']) for x in known]))
    comp=comparisons(values);assert comp==r['paired'] and values==r['per_video'] and primary(comp)==r['primary_pass']
    write(RUN/'independent_audit.json',dict(at=now().isoformat(),all26_old_actions_and_raw_baselines_exact=True,all10_conditions_and_intervals_verified=True,report_sha256=sha(RUN/'replay/report.json')))
    cat=read(ROOT/'experiments/project_history.json');card=next(x for x in cat['experiments'] if x['id']==IDENT);passed=r['primary_pass']
    card.update(state='completed_positive' if passed else 'completed_negative',done=now().strftime('%Y-%m-%d'),summary='Условная неопределённость: критерий '+('пройден.' if passed else 'не пройден.'),obtained=(RUN/'replay/report.json').relative_to(ROOT).as_posix()+' → paired: '+str(comp),interpretation='Измерены первые назначения10парных условий при прежних средних откликах/весах/времени. Постоянный штраф отделяет общее ужесточение отказа от пространственной неопределённости. Это не калиброванная вероятность полезности и не HOTA.',decision=p['next']+' Primary: '+str(passed))
    for i,met in enumerate((True,passed)):card['acceptance'][i].update(met=met,evidence=(RUN/('independent_audit.json' if i==0 else 'replay/report.json')).relative_to(ROOT).as_posix())
    write(ROOT/'experiments/project_history.json',cat);write(RUN/'finish_recorded.json',dict(at=now().isoformat(),complete=True,primary_pass=passed));journal('итог опоры уверенности в образце','*Получено.* '+card['obtained']+'\n\n*Вывод.* '+card['interpretation']+'\n\n*Решение.* '+card['decision'])
    from scripts import z13_region_assignment_recovery2_20261025 as notes
    notes.namespace()['handoff']('Полный z13_ridge_uncertainty_v1 завершён. '+card['obtained']+' '+card['decision']+' Публикация runs/z13_ridge_uncertainty_20261025/publication.json; проверить отдельные confirmed_results и архитектурную веху. Не возобновлять четырёхчастное представление без допуска. Всё скрыто;57чужихправок/.gitignore сохранять.')
    publish('final')

def publish(stage):
    s=inspect.getsource(clock.publish).replace("'tracking/z13_history_clock.py','tests/test_z13_history_clock.py'","'tracking/z13_ridge_uncertainty.py','tests/test_z13_ridge_uncertainty.py'")
    s=s.replace("'tests/test_z13_history_clock.py'","'tests/test_z13_ridge_uncertainty.py'").replace('physical-frame temporal resampling inside frozen FlyVis','template-supported uncertainty in frozen FlyVis readout')
    ns=dict(globals());exec(compile(s,__file__,'exec'),ns);ns['publish'](stage)

def pipeline():
    ps=WindowsProcesses();write(RUN/'pipeline.json',dict(pid=os.getpid(),birth=ps.creation(os.getpid()),at=now().isoformat()))
    try:
        for phase in BUDGETS:
            folder=RUN/phase;phase_protocol(phase)
            with (folder/'controller.stdout.log').open('w',encoding='utf-8') as out,(folder/'controller.stderr.log').open('w',encoding='utf-8') as err:
                child=subprocess.Popen([str(ROOT/'.venv/Scripts/python.exe'),'-u','scripts/z13_bounded_recovery_controller_20261022.py','--run',str(folder),'--script',__file__],cwd=ROOT,env={**os.environ,'Z13_UNCERTAINTY_PHASE':phase},stdout=out,stderr=err,creationflags=subprocess.IDLE_PRIORITY_CLASS|subprocess.CREATE_NO_WINDOW)
                while child.poll() is None:
                    if (folder/'status.json').exists():
                        try:write(RUN/'status.json',dict(read(folder/'status.json'),phase=phase))
                        except (OSError,ValueError):pass
                    time.sleep(2)
                child.wait()
            c=read(folder/'controller_result.json');assert c['exit_code']==0 and c['all_owned_ended'] and not c['remaining'] and all(ps.creation(int(pid))!=b for pid,b in c['owned'].items())
            assert read(folder/'report.json')['complete']
        finish()
    except Exception as e:write(RUN/'pipeline_failure.json',dict(at=now().isoformat(),phase=phase,error=repr(e)));raise

if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    elif '--pipeline' in sys.argv:pipeline()
    elif '--worker' in sys.argv:worker()
    elif '--publish-final' in sys.argv:publish('final')
    else:raise RuntimeError('Explicit mode required')
