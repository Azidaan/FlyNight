"""Frozen full-history FlyVis with a two-direction spatial readout extension."""
from pathlib import Path
from datetime import datetime,timedelta
import os,sys,inspect,hashlib,subprocess,time
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
from scripts import z13_history_clock_20261025 as clock
from scripts import z13_history_clock_recovery1_20261025 as recovery
from scripts import z13_anchor_state_20261025 as anchor
from scripts import z13_region_readout_20261025 as decoder
from scripts import z13_region_readout_recovery1_20261025 as oldreplay
from scripts.z13_localization_labels_20261023 import verify_sources
from scripts.z13_hota_deadline_guard import WindowsProcesses
RUN=ROOT/'runs/z13_spatial_derivatives_20261025';IDENT='z13_spatial_derivatives_v1'
PHASE=os.environ.get('Z13_HISTORY_CLOCK_PHASE','')
BUDGETS={'features':1800,'replay':900}
ARMS=tuple(a+'_'+mode for a in ('D','C','R1') for mode in ('old','gradient'))+('geometry',)
ALPHAS=(0.,.25)

def journal(title,text):
    from scripts import z13_localization_labels_20261023 as notes
    notes.IDENT=IDENT;notes.journal(title,text)

def comparisons(v):
    from scripts.z13_continue import paired
    videos=list(v['D_gradient']);out={'Dgradient-baseline':paired(v['D_gradient'],{k:0. for k in videos}),'Dgradient-Dold':paired(v['D_gradient'],v['D_old'])}
    for a in ('C_gradient','R1_gradient','geometry'):out['Dgradient-'+a]=paired(v['D_gradient'],v[a])
    for a in ('R1','C'):out['gradient_effect_D-'+a]=paired({k:v['D_gradient'][k]-v['D_old'][k] for k in videos},{k:v[a+'_gradient'][k]-v[a+'_old'][k] for k in videos})
    return out

def primary(c):return all(c[k]['mean_per_video']>=.02 for k in ('Dgradient-baseline','Dgradient-Dold')) and all(v['ci95'][0]>0 for v in c.values())

def feature_source():
    s=recovery.source()
    changes=[
      ('from tracking.z13_history_clock import clock_features','from tracking.z13_spatial_derivatives import derivative_projector,spatial_descriptor,zero_extension'),
      ("frames=[]","projector=derivative_projector(geom.points,geom.neighbors,geom.spacing)\n    frames=[]"),
      ("reference=old.episode_features(torch,arm,bb,hx,geom.neighbors,need) if bi<2 else None","reference=None"),
      ('F=clock_features(torch,arm,bb,hx,geom.neighbors,need,ff)','F=old.episode_features(torch,arm,bb,hx,geom.neighbors,need);reference=F'),
      ("group=byep[e];xe=(F[3][j].T-mu)/sd;xs=np.stack([(F[4+l['pos']][j].T-mu)/sd for l in group])","group=byep[e];xe=spatial_descriptor(F[3][j],mu,sd,geom.neighbors,projector);xs=np.stack([spatial_descriptor(F[4+l['pos']][j],mu,sd,geom.neighbors,projector) for l in group])"),
      ("old_scores=ridge_scores(a,target,b,[lam])[0]","old_scores=ridge_scores(a,target,b,[lam])[0]\n                        zeros=ridge_scores(zero_extension(a),target,zero_extension(b),[lam])[0]\n                        assert np.allclose(zeros,old_scores,atol=1e-8,rtol=1e-7),'Equal136dimensional zero control must reproduce68dimensional ridge'\n                        assert np.isfinite(scores).all()"),
      ("status('Frozen recurrent clock at actual frame intervals'","status('Original recurrent features plus local horizontal/vertical derivatives'"),
      ("clock_rule='Actual integer frame gaps; linear stimulus interpolation; zero gap no state evolution; original nominal30Hz/substep unchanged'","descriptor_rule='Original68standardized channels plus34x/34y first-ring least-squares slopes; original full unit-clock state unchanged;136dimensional zero control verified'")]
    for a,b in changes:assert s.count(a)==1,a;s=s.replace(a,b)
    return s

def load_clock_maps():
    ns={**vars(clock),'RUN':RUN};exec(compile(inspect.getsource(clock.load_clock_maps),__file__,'exec'),ns);return ns['load_clock_maps']()

def status(stage,**kw):
    p=read(RUN/PHASE/'protocol.json')
    if now()>=datetime.fromisoformat(p['save_at']) or (RUN/PHASE/'stop.request').exists():raise TimeoutError('Original spatial-descriptor deadline')
    write(RUN/PHASE/'status.json',dict(state='running',stage=stage,updated=now().isoformat(),**kw))

def worker():
    p=read(RUN/PHASE/'protocol.json');verify_sources(p)
    if PHASE=='features':s=feature_source();ns={**vars(clock),'RUN':RUN,'PHASE':PHASE,'status':status};fn='feature_worker';key='executed_feature_worker_sha256'
    else:s=clock.replay_worker_source();ns={**vars(decoder),'RUN':RUN/'replay','ARMS':ARMS,'ALPHAS':ALPHAS,'load_clock_maps':load_clock_maps,'comparisons':comparisons,'primary':primary};fn='worker';key='executed_worker_sha256'
    assert hashlib.sha256(s.encode()).hexdigest()==p[key];exec(compile(s,__file__,'exec'),ns);ns[fn]()

def prepare():
    assert not RUN.exists();parent=anchor.RUN
    assert read(parent/'publication.json')['push_completed'] and not read(parent/'replay/report.json')['primary_pass']
    ps=WindowsProcesses()
    for phase in BUDGETS:
        c=read(parent/phase/'controller_result.json');assert c['exit_code']==0 and c['all_owned_ended'] and not c['remaining'] and all(ps.creation(int(pid))!=b for pid,b in c['owned'].items())
    old=read(clock.RUN/'protocol.json');verify_sources(old)
    reserve=ROOT/'runs/z13_region_choice_gate_20261025/recovery1/report.json';r=read(reserve);assert r['complete']
    from tracking.z13_readout import HexGeometry
    from tracking.z13_spatial_derivatives import derivative_projector,spatial_descriptor
    g=read(ROOT/'runs/z13_local_peak_20261021/geometry.json');geom=HexGeometry(g['points'],g['eye_size']);P=derivative_projector(geom.points,geom.neighbors,geom.spacing)
    center=geom.nearest([geom.eye_size/2]*2);x=(geom.points[:,0].astype(np.float64)-float(geom.points[center,0]))/geom.spacing
    def describe(v):
        raw=np.stack((v,v[geom.neighbors[:,1:]].mean(-1)))
        return spatial_descriptor(raw,np.zeros(2),np.ones(2),geom.neighbors,P)[center]
    a,b=describe(x),describe(-x);assert np.allclose(a[:2],b[:2],atol=1e-12) and np.linalg.norm(a[2:]-b[2:])>1
    write(RUN/'information_loss_check.json',dict(at=now().isoformat(),old_center_and_mean_equal=True,new_direction_features_differ=True,projector_columns=721,all_local_offsets_rank2=True,synthetic_positive=a.tolist(),synthetic_negative=b.tolist(),claim='Structural alias at one central column confirmed; this is not a measured cause of natural tracking errors. Existing full region-action oracle reserve is the prerequisite for this output; four-bin support and observed-negative gates are not reused or relaxed.'))
    start=now();stop=min(start+timedelta(seconds=2820),datetime.fromisoformat('2026-10-26T07:35:00+03:00'))
    names=[Path(__file__).relative_to(ROOT).as_posix(),'tracking/z13_spatial_derivatives.py','tests/test_z13_spatial_derivatives.py','scripts/z13_anchor_state_20261025.py','scripts/z13_history_clock_recovery1_20261025.py',reserve.relative_to(ROOT).as_posix(),(RUN/'information_loss_check.json').relative_to(ROOT).as_posix(),'runs/z13_spatial_derivatives_preflight_20261025/test_fixture_precision.json']
    p=dict(old,id=IDENT,created=start.isoformat(),timezone='Europe/Moscow',pipeline_save_at=stop.isoformat(),pipeline_hard_stop=(stop+timedelta(seconds=20)).isoformat(),phase_budgets=BUDGETS,videos=old['evaluate26'],fit20=[],
      action='Change only the local spatial descriptor entering the target readout: retain original68standardized features (34activity+34neighbor means), append34horizontal+34vertical first-ring least-squares slopes. Fit identical observed-center ridge with unchanged savedlambda. Frozen C/D/R1 core/full unit-clock/history/input/norms/target/regionmean/null0/alpha0.25 unchanged. Seven paired conditions old/gradient forC/D/R1 plusgeometry;136dimensional zero-extension control reproduces old68ridge.',
      formula='Offsets D_i=(p_neighbor-p_i)/spacing, missing neighbors retain original self padding. P_i=pinv(D_i) has2x6. For each raw activity channel, g_i=P_i*(neighbor_activity-center_activity). Append gx/sd_original_channel and gy/sd_original_channel; old normalized68vector retained exactly. New136weights adapt only to the same observed exemplar, no offline training or new normalization/calibration.',
      distinction='This changes the receptive field of column features, not mean/max pooling inside detection rectangles or four bins over detections. The latter support gate remains inconclusive and is not advanced. Six-neighbor mean in scripts/z13_target_model.py discards their arrangement at each local descriptor; two orientations can share center/mean. The derivative descriptor restores only first spatial moments; its usefulness on natural cases is a hypothesis. No neural-state reset/clock or failed NULL/uncertainty is combined.',
      primary='Dgradient-baseline and Dgradient-Dold means>=2pp; all7paired-video95%lower>0, including Cgradient,R1gradient,geometry and the intervention effects D-R1 and D-C. No favorable arm/subset/threshold/lambda selection. All mandatory; absolute improvement and FlyVis-specific addition both required.',
      technical='Three tests: affine-field slopes including boundaries; orientation alias broken at center while old values retained; neighbor-order invariance and direct-matrix zero-padding ridge equivalence. All721offset matrices rank2. All8326old C/D/R1 center<0.5px/score<0.001, original raw feature arrays shared; all136zero-control maps reproduce68within1e-8absolute+1e-7relative. All new maps finite, exact source/label/crop binding; all26old action/baseline outputs exact. First2Dgroups conservative price fits original1800s or preservepartial/stop; completed chunks never repeated.',
      counterargument='FlyVis channels already include directional neurons, so spatial slopes may be redundant. Adding features can overfit the721observed columns; unchanged lambda/norms may not suit136dimensions. Equal136zero extension isolates mere empty dimensions, not the additional useful degrees of freedom; equal C/R1 and D-specific interaction controls are essential. A synthetic local alias is not evidence that it caused actual errors. No HOTA or global cause from one-step train26.',
      ranking=['1 Local spatial derivatives: new136descriptor; no core training, direct structural alias test and equal-dimension zero control; current action.','2 KYS learned correspondence state: planned; anchor removal changed D-old only-0.009pp with CIcross0 in runs/z13_anchor_state_20261025/replay/report.json, so no large beneficial reset; learned transport still untested and needs a distinct prerequisite.','3 Candidate identity memory KeepTrack: requires reliable causal identity for neighbors, not merely another sample bank.','4 Chromatic input: no cachedRGBfeatures; ordinary color controls and extraction scope/cost must be established first.','5 Observed-negative weighting: gateCIcrosses2%, not admitted; do not relax.','6 Four-bin region features: gateCIcrosses2%, not admitted; local derivatives are a distinct column-level receptive field.','7 Sample/query memory: prior criteria failed; no unchanged repetition.','8 More core epochs or same aligned loss: failed, not repeated.','9 Actual frame clock and anchor reset: completed negative; no frequency/time-constant search.','10 More scalar losses/thresholds: regionsemantic/value/balanced failed; stopped.','11 Static context/channel products: failed same representation; stopped.','12 Scalar mean/max: negative; fixed region-pooling branch stopped.'],
      literature='Bhat etal,DiMP,ICCV2019 https://arxiv.org/html/1904.07220 section3.1 eq1: target response is a spatial convolution x*f. Danelljan etal,SRDCF,ICCV2015 https://openaccess.thecvf.com/content_iccv_2015/papers/Danelljan_Learning_Spatially_Regularized_ICCV_2015_paper.pdf section3 eq1: separate spatial convolution filters perfeaturechannel. We transfer the spatial-filter principle only. Two local derivatives on a hexagonal FlyVis field are our reduced engineering prototype, not either method or their published gain; no circular FFT/boundary fix is claimed.',
      next='Pass -> separate frozen-runtime parity/price and matched closed-loop gate before HOTA. Fail -> preserve this exact two-direction first-ring descriptor; no derivative scale/radius/lambda sweep on26 and no core training rescue. Distinguish representation change from proven natural error cause; choose another node only with measured prerequisite.',
      limits='OneGPUfeatures1800s thenCPUreplay900s, independent controller20s shutdown, global2820s not extended. Hidden Idle/no focus. No downloads/core training/dev/HOTA/heldout75. AuthorMonday26Octsave07:35/hard07:39:30; Sunday no07:40limit.',executed_feature_worker_sha256=hashlib.sha256(feature_source().encode()).hexdigest(),executed_worker_sha256=hashlib.sha256(clock.replay_worker_source().encode()).hexdigest(),sha256={**old['sha256'],**{n:sha(ROOT/n) for n in names}})
    write(RUN/'protocol.json',p);write(RUN/'status.json',dict(state='ready',stage='Local spatial first moments; structural check passed',updated=now().isoformat()))
    cat=read(ROOT/'experiments/project_history.json');assert not any(c['id']==IDENT for c in cat['experiments'])
    cat['experiments'].append(dict(id=IDENT,title='Считывание FlyVis: пространственные производные вместо потери расположения соседей',area='flyvis',state='ready',started=start.strftime('%Y-%m-%d'),tags=['fly-problem-P8'],question='Помогут ли два направления изменения активности по соседним колонкам при том же состоянии и модели цели?',hypothesis='Рабочая гипотеза: среднее соседей скрывает локальное расположение признаков. Потеря направления подтверждается формулой и синтетическим примером; причинность реальных ошибок и польза новых признаков не установлены.',expected=p['action'],acceptance=[dict(criterion=p['technical'],met=None),dict(criterion=p['primary'],met=None)],obtained='',interpretation='',decision='',cannot_claim=p['counterargument'],evidence=[dict(kind='file',ref=(RUN/'protocol.json').relative_to(ROOT).as_posix(),claim='Новый локальный узел, ранжирование12, равные контроли и критерии до GPU')]))
    write(ROOT/'experiments/project_history.json',cat)
    journal('постановка локальной пространственной структуры считывателя','*Действие.* '+p['action']+'\n\n*Формула и отличие.* '+p['formula']+' '+p['distinction']+'\n\n*Критерии до GPU.* '+p['primary']+' '+p['technical']+'\n\n*Возражение.* '+p['counterargument']+'\n\n*Отбор.* '+' '.join(p['ranking'])+'\n\n*Источник.* '+p['literature']+'\n\n*Бюджет.* '+p['limits']+' Save '+p['pipeline_save_at']+' / hard '+p['pipeline_hard_stop']+'. Источники runs/z13_spatial_derivatives_20261025/protocol.json и information_loss_check.json.')

def phase_protocol(phase):
    ns={**vars(clock),**globals()};exec(compile(inspect.getsource(clock.phase_protocol),__file__,'exec'),ns);return ns['phase_protocol'](phase)

def finish():
    s=inspect.getsource(anchor.finish).replace('Независимые запросы от образца: критерий','Локальные пространственные производные: критерий').replace('итог причинной проверки поисковой памяти','итог локальных пространственных признаков').replace('z13_anchor_state_v1','z13_spatial_derivatives_v1').replace('не повторять готовое и не подбирать частоту сброса','не повторять готовое, не подбирать масштаб/радиус производной')
    s=s.replace('Парное вмешательство меняет только накопление поискового состояния, сохраняя образец/текущий запрос и считыватель. Это поисковая проверка первых назначений на многократно использованном train26, не HOTA. Неопределённый эффект не доказывает ни вредность всей памяти, ни невозможность пространственного переноса.','Парное вмешательство меняет только локальные входы модели цели: прежние68каналов дополнены двумя пространственными производными. Состояние/образец/нормы/регуляризация/политика прежние. Это первые назначения на многократно использованном train26, не HOTA; структурная потеря направления не доказывает причину реальных ошибок.')
    ns={**vars(anchor),**globals()};exec(compile(s,__file__,'exec'),ns);ns['finish']()

def publish(stage):
    s=inspect.getsource(clock.publish).replace('tracking/z13_history_clock.py','tracking/z13_spatial_derivatives.py').replace('tests/test_z13_history_clock.py','tests/test_z13_spatial_derivatives.py').replace('physical-frame temporal resampling inside frozen FlyVis','local spatial derivatives in frozen FlyVis readout')
    ns={**vars(clock),**globals()};exec(compile(s,__file__,'exec'),ns);ns['publish'](stage)

def pipeline():
    ns={**vars(clock),**globals()};exec(compile(inspect.getsource(clock.pipeline),__file__,'exec'),ns);ns['pipeline']()

if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    elif '--pipeline' in sys.argv:pipeline()
    elif '--worker' in sys.argv:worker()
    else:raise RuntimeError('Explicit mode required')
