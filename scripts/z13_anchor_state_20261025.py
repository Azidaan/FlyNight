"""Bounded matched test of intervening search history in frozen FlyVis state."""
from pathlib import Path
from datetime import datetime,timedelta
import os,sys,inspect,hashlib,subprocess,time
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
from scripts import z13_history_clock_20261025 as clock
from scripts import z13_history_clock_recovery1_20261025 as recovery
from scripts import z13_region_readout_20261025 as decoder
from scripts import z13_region_readout_recovery1_20261025 as oldreplay
from scripts.z13_localization_labels_20261023 import verify_sources
from scripts.z13_hota_deadline_guard import WindowsProcesses
RUN=ROOT/'runs/z13_anchor_state_20261025'
IDENT='z13_anchor_state_v1'
PHASE=os.environ.get('Z13_HISTORY_CLOCK_PHASE','')
BUDGETS={'features':1800,'replay':900}
ARMS=tuple(a+'_'+mode for a in ('D','C','R1') for mode in ('old','anchor'))+('geometry',)
ALPHAS=(0.,.25)

def journal(title,text):
    from scripts import z13_localization_labels_20261023 as notes
    notes.IDENT=IDENT;notes.journal(title,text)

def comparisons(v):
    from scripts.z13_continue import paired
    videos=list(v['D_anchor']);out={'Danchor-baseline':paired(v['D_anchor'],{k:0. for k in videos}),'Danchor-Dold':paired(v['D_anchor'],v['D_old'])}
    for a in ('C_anchor','R1_anchor','geometry'):out['Danchor-'+a]=paired(v['D_anchor'],v[a])
    out['anchor_effect_D-R1']=paired({k:v['D_anchor'][k]-v['D_old'][k] for k in videos},{k:v['R1_anchor'][k]-v['R1_old'][k] for k in videos})
    return out

def primary(c):
    return all(c[k]['mean_per_video']>=.02 for k in ('Danchor-baseline','Danchor-Dold')) and all(v['ci95'][0]>0 for v in c.values())

def feature_source():
    s=recovery.source()
    changes=[
      ('from tracking.z13_history_clock import clock_features','from tracking.z13_anchor_state import anchor_features'),
      ("reference=old.episode_features(torch,arm,bb,hx,geom.neighbors,need) if bi<2 else None","reference=old.episode_features(torch,arm,bb,hx,geom.neighbors,need)"),
      ('F=clock_features(torch,arm,bb,hx,geom.neighbors,need,ff)','F=anchor_features(torch,arm,bb,hx,geom.neighbors,need,old.episode_features,prefix=reference)'),
      ("score=scores[k];center,pres=map_center(score,geom)","score=scores[k]\n                        if l['pos']==0:assert np.array_equal(score,old_scores[k]),'Shared first-search prefix must be exact'\n                        center,pres=map_center(score,geom)"),
      ('weights.append(int(np.diff(ff,axis=1).max(0).sum()))',"weights.append(4+5*len({l['pos'] for e in batch for l in byep[e]}))"),
      ("status('Frozen recurrent clock at actual frame intervals'","status('Independent search query from unchanged observed exemplar'"),
      ("clock_rule='Actual integer frame gaps; linear stimulus interpolation; zero gap no state evolution; original nominal30Hz/substep unchanged'","state_rule='Each query is original four observed frames plus that query only; old full recurrence separately reproduced on every row'")]
    for a,b in changes:assert s.count(a)==1,a;s=s.replace(a,b)
    return s

def load_clock_maps():
    ns={**vars(clock),'RUN':RUN};exec(compile(inspect.getsource(clock.load_clock_maps),__file__,'exec'),ns);return ns['load_clock_maps']()

def status(stage,**kw):
    p=read(RUN/PHASE/'protocol.json')
    if now()>=datetime.fromisoformat(p['save_at']) or (RUN/PHASE/'stop.request').exists():raise TimeoutError('Original anchor-state deadline')
    write(RUN/PHASE/'status.json',dict(state='running',stage=stage,updated=now().isoformat(),**kw))

def worker():
    p=read(RUN/PHASE/'protocol.json');verify_sources(p)
    if PHASE=='features':
        s=feature_source();ns={**vars(clock),'RUN':RUN,'PHASE':PHASE,'status':status};fn='feature_worker';key='executed_feature_worker_sha256'
    else:
        s=clock.replay_worker_source();ns={**vars(decoder),'RUN':RUN/'replay','ARMS':ARMS,'ALPHAS':ALPHAS,'load_clock_maps':load_clock_maps,'comparisons':comparisons,'primary':primary};fn='worker';key='executed_worker_sha256'
    assert hashlib.sha256(s.encode()).hexdigest()==p[key]
    exec(compile(s,__file__,'exec'),ns);ns[fn]()

def prepare():
    assert not RUN.exists()
    parent=ROOT/'runs/z13_null_mixture_20261025'
    assert read(parent/'publication.json')['push_completed'] and not read(parent/'report.json')['primary_pass']
    assert read(parent/'architecture_milestone/publication.json')['push_completed']
    ps=WindowsProcesses();c=read(parent/'controller_result.json')
    assert c['exit_code']==0 and c['all_owned_ended'] and not c['remaining'] and all(ps.creation(int(pid))!=b for pid,b in c['owned'].items())
    old=read(clock.RUN/'protocol.json');verify_sources(old);start=now();stop=min(start+timedelta(seconds=2820),datetime.fromisoformat('2026-10-26T07:35:00+03:00'))
    from scripts.z13_continue import paired
    labels=read(ROOT/'runs/z13_candidate_reserve_20261021/evaluate/labels.json')
    per={v:np.mean([l['pos']>0 for l in labels if l['video']==v and l['kind']>=0]).item() for v in old['evaluate26']}
    exposure=paired(per,{v:0. for v in per});admit=exposure['ci95'][0]>.02
    write(RUN/'exposure.json',dict(at=now().isoformat(),per_video=per,paired=exposure,admission_pass=admit,criterion='Lower95%fraction of known later-than-first queries>2%, necessary exposure for a2pp main effect. This is not evidence that recurrence is harmful. First search must remain numerically identical.'))
    names=[Path(__file__).relative_to(ROOT).as_posix(),'tracking/z13_anchor_state.py','tests/test_z13_anchor_state.py','scripts/z13_history_clock_recovery1_20261025.py',(RUN/'exposure.json').relative_to(ROOT).as_posix()]
    p=dict(old,id=IDENT,created=start.isoformat(),timezone='Europe/Moscow',pipeline_save_at=stop.isoformat(),pipeline_hard_stop=(stop+timedelta(seconds=20)).isoformat(),phase_budgets=BUDGETS,videos=old['evaluate26'],fit20=[],
      action='One internal state intervention: suppress updates from intervening search frames. For each query, execute original frozen core on exactly its original four exemplar frames followed by this query. Keep the exemplar itself, all34features/neighbors, original norms/lambda/Gaussian target/mean region readout/null0/alpha0.25. Not combined with failed uncertainty or NULL mixture. Seven conditions C/D/R1 old/anchor plus geometry.',
      distinction='Actual-clock experiment changed integration time, observed/query memory changed the target-model sample bank, efference_eye warped neural potentials, clean-readout had no actual input difference. None tested removing only intervening search-state updates while keeping the four-frame exemplar and matched current query. This state ablation is not KYS implementation: no learned correspondence, state warp or trained predictor.',
      primary='Danchor-baseline and Danchor-Dold means>=2pp, all6paired-video95%lower>0, includingC/R1/geometry and D-R1 intervention interaction. Reused train26 exploratory. No alpha/normalization/lambda fitting or favorable history/subset selection.',
      technical='Three nonlinear-state tests: first-search/exemplar exact; middle-history invariance; exemplar remains influential. Exposure lower95%>2% before extraction. Every8326old C/D/R1 center<0.5px/score<0.001 against retained original. Identical causal exemplar/first-search prefix arrays reused from the old producer; first-search score maps exactly equal on every retained query, no separate-GPU-pass numerical comparison. Full source/row bindings, finite maps, exact alpha0/baseline and all old actions. First2Dgroup conservative price must fit original1800s; completed chunks reused.',
      counterargument='Intervening search frames may contain useful target motion or disambiguation. Removing them changes available information intentionally and could hurt D even if learned spatial transport would help. Frozen norms/weights were learned for full recurrence; a failed ablation does not disprove KYS. Positive first-assignment effect is not HOTA or proof of all confidence failures. Original teacher/tracker errors in exemplar are retained, not corrected by truth.',
      literature='KYS,Bhat etal,ECCV2020 https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123680205.pdf sections3.1-3.3 eq1-5: localized state, learned correspondence, entropy reliability; section3.3 warns state leakage into occluders. This motivates checking state influence, not this reset rule or a claimed gain. Three candidate replacements ranked: 1 anchored independent queries (cheap causal ablation now);2 learned correspondence/reliability on readout state (only if data/cost and training gate justify);3 candidate-identity memory KeepTrack (requires reliable observed identity). Existing12-way ranking retained; no automatic extra experiments.',
      next='Pass -> distinct runtime parity/price and frozen matched closed-loop gate; no HOTA until that protocol. Fail -> keep this precise state ablation; do not tune reset frequency or retrain core to rescue. Use primary outcome to distinguish harmful-history evidence from absent evidence; separately design spatial correspondence only with a measurable cheap prerequisite.',
      limits='OneGPU features1800s plusCPUreplay900s; independent controllers20s shutdown; global2820s not extended. Hidden Idle, no focus/windows. No downloads/dev/heldout75/HOTA/core training. AuthorMonday26Octsave07:35/hard07:39:30; Sunday no07:40stop.',
      executed_feature_worker_sha256=hashlib.sha256(feature_source().encode()).hexdigest(),executed_worker_sha256=hashlib.sha256(clock.replay_worker_source().encode()).hexdigest(),sha256={**old['sha256'],**{n:sha(ROOT/n) for n in names}})
    write(RUN/'protocol.json',p);write(RUN/'status.json',dict(state='ready' if admit else 'inconclusive',stage='Anchor-state exposure checked; no duplicated completed run',updated=now().isoformat()))
    cat=read(ROOT/'experiments/project_history.json');assert not any(x['id']==IDENT for x in cat['experiments'])
    cat['experiments'].append(dict(id=IDENT,title='Состояние FlyVis: отдельный запрос от сохранённого образца',area='flyvis',state='ready' if admit else 'inconclusive',started=start.strftime('%Y-%m-%d'),tags=['fly-problem-P3','fly-problem-P8'],question='Помогает ли накопленная между образцом и запросом активность или её исключение улучшит назначения?',hypothesis='Рабочая гипотеза: история поисковых кадров может сохранять мешающий отклик; это не установленная причина. Удаление истории может также лишить муху полезного движения.',expected=p['action'],acceptance=[dict(criterion=p['technical'],met=None),dict(criterion=p['primary'],met=None)],obtained='',interpretation='',decision='',cannot_claim=p['counterargument'],evidence=[dict(kind='file',ref=(RUN/'protocol.json').relative_to(ROOT).as_posix(),claim='Одно изменение состояния, равные контроли и критерии до GPU')]))
    write(ROOT/'experiments/project_history.json',cat)
    journal('постановка причинной проверки накопленной поисковой памяти','*Действие.* '+p['action']+'\n\n*Отличие.* '+p['distinction']+'\n\n*Критерии до GPU.* '+p['primary']+' '+p['technical']+' Частота допустимого изменения '+str(exposure)+'; runs/z13_anchor_state_20261025/exposure.json, это наличие мишени, не польза.\n\n*Возражение.* '+p['counterargument']+'\n\n*Основание и замены.* '+p['literature']+'\n\n*Бюджет.* '+p['limits']+' Save '+p['pipeline_save_at']+' / hard '+p['pipeline_hard_stop']+'.')
    assert admit,'Exposure gate did not admit GPU; preserve result'

def phase_protocol(phase):
    ns={**vars(clock),**globals()};exec(compile(inspect.getsource(clock.phase_protocol),__file__,'exec'),ns);return ns['phase_protocol'](phase)

def finish():
    p=read(RUN/'protocol.json');verify_sources(p);r=read(RUN/'replay/report.json');assert r['complete'] and r['exact_baseline_videos']==26
    values={a:{} for a in ARMS}
    for v in p['videos']:
        path=RUN/'replay/videos'/(v+'.json');assert sha(path)==r['video_sha256'][v];record=read(path);assert record['checks']['raw_replay_exact']
        prior={x['uid']:x for x in read(oldreplay.RUN/'videos'/(v+'.json'))['rows']}
        for row in record['rows']:
            ref=prior[row['uid']];assert all(row[k]==ref[k] for k in ('uid','video','frame','tid','kind','label','split'))
            for a in ('D','C','R1'):assert row['arms'][a+'_old'][1]==ref['arms'][a][1]
            assert row['arms']['geometry'][1]==ref['arms']['geometry'][1]
        known=[x for x in record['rows'] if x['kind']>=0]
        for a in ARMS:values[a][v]=float(np.mean([int(x['arms'][a][1]['beneficial'])-int(x['arms'][a][1]['harmful']) for x in known]))
    comp=comparisons(values);assert comp==r['paired'] and values==r['per_video'] and primary(comp)==r['primary_pass']
    write(RUN/'independent_audit.json',dict(at=now().isoformat(),all26_old_actions_and_raw_baselines_exact=True,all7_conditions_and_paired_intervals_verified=True,report_sha256=sha(RUN/'replay/report.json')))
    cat=read(ROOT/'experiments/project_history.json');card=next(x for x in cat['experiments'] if x['id']==IDENT);passed=r['primary_pass']
    card.update(state='completed_positive' if passed else 'completed_negative',done=now().strftime('%Y-%m-%d'),summary='Независимые запросы от образца: критерий '+('пройден.' if passed else 'не пройден.'),obtained=(RUN/'replay/report.json').relative_to(ROOT).as_posix()+' → paired: '+str(comp),interpretation='Парное вмешательство меняет только накопление поискового состояния, сохраняя образец/текущий запрос и считыватель. Это поисковая проверка первых назначений на многократно использованном train26, не HOTA. Неопределённый эффект не доказывает ни вредность всей памяти, ни невозможность пространственного переноса.',decision=p['next']+' Primary: '+str(passed))
    for i,met in enumerate((True,passed)):card['acceptance'][i].update(met=met,evidence=(RUN/('independent_audit.json' if i==0 else 'replay/report.json')).relative_to(ROOT).as_posix())
    write(ROOT/'experiments/project_history.json',cat);write(RUN/'finish_recorded.json',dict(at=now().isoformat(),complete=True,primary_pass=passed))
    journal('итог причинной проверки поисковой памяти','*Получено.* '+card['obtained']+'\n\n*Вывод.* '+card['interpretation']+'\n\n*Решение.* '+card['decision'])
    from scripts import z13_region_assignment_recovery2_20261025 as notes
    notes.namespace()['handoff']('Полный z13_anchor_state_v1 окончен. '+card['obtained']+' '+card['decision']+' Проверить частные confirmed_results и архитектурную веху; не повторять готовое и не подбирать частоту сброса. Всё скрыто;57чужихправок/.gitignore сохранять.')
    publish('final')

def publish(stage):
    s=inspect.getsource(clock.publish).replace('tracking/z13_history_clock.py','tracking/z13_anchor_state.py').replace('tests/test_z13_history_clock.py','tests/test_z13_anchor_state.py').replace('physical-frame temporal resampling inside frozen FlyVis','anchored search-state ablation inside frozen FlyVis')
    ns={**vars(clock),**globals()};exec(compile(s,__file__,'exec'),ns);ns['publish'](stage)

def pipeline():
    ns={**vars(clock),**globals()};exec(compile(inspect.getsource(clock.pipeline),__file__,'exec'),ns);ns['pipeline']()

if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    elif '--pipeline' in sys.argv:pipeline()
    elif '--worker' in sys.argv:worker()
    elif '--publish-start' in sys.argv:publish('start')
    else:raise RuntimeError('Explicit mode required')
