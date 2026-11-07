"""Prepare documents and input manifests only. No model, tracker or scoring run."""
import csv
import gzip
import hashlib
import json
import subprocess
from collections import Counter,defaultdict
from datetime import datetime,timedelta,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=Path(__file__).resolve().parent
MSK=timezone(timedelta(hours=3))


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def write(p,value):
    with Path(p).open('x',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def main():
    stamp=datetime.now(MSK)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    assert head=='01815dc9253108c81737c6723077ab603336d7c2'
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
    assert branch=='codex/system-integrity-20261106-e8f674ce'
    plan=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';tracker_plan=ROOT/'dissertation/План целей по Трекеру.md'
    protected=['experiments/project_history.json','experiments/metrics_glossary.json','experiments/system_problems.json']
    protected_hashes={n:sha(ROOT/n) for n in protected}
    tracker_hash=sha(tracker_plan)
    source=ROOT/'runs/flyvis_exclusive_t4t5_20261107/choices.csv.gz'
    counts=defaultdict(Counter);queries=[];strata=[];videos=defaultdict(set)
    with gzip.open(source,'rt',encoding='ascii') as f:
        for r in csv.DictReader(f):
            if r['exactly2']!='1':continue
            counts[r['split']][r['label_reason']]+=1;videos[r['split']].add(r['video'])
            if r['split'] not in ['fit20','evaluate26']:continue
            uid=int(r['uid'])
            queries.append(dict(uid=uid,video=r['video'],split=r['split'],candidate_high_ids=[int(r['j']),int(r['k'])],
                source=f"runs/z13_temporal_pair_readiness_20261025/inputs/{r['video']}.json"))
            strata.append(dict(uid=uid,split=r['split'],reason=r['label_reason'],target_side=int(r['target_side']),
                primary_known=r['label_reason'] in ['ok','neither_target']))
    assert len(queries)==1991 and len({q['uid'] for q in queries})==1991
    source_pins={source.relative_to(ROOT).as_posix():sha(source)}
    for video in sorted({q['video'] for q in queries}):
        path=ROOT/f'runs/z13_temporal_pair_readiness_20261025/inputs/{video}.json'
        records={r['uid']:r for r in read(path)['rows']};source_pins[path.relative_to(ROOT).as_posix()]=sha(path)
        for q in (x for x in queries if x['video']==video):
            r=records[q['uid']];assert r['current_choices']==q['candidate_high_ids']
            q.update(frame=r['current_frame'],tid=r['tid'],episode=r['episode_local'])
    write(OUT/'queries.json',dict(scope='Archived causal candidate bindings, no identity labels in runtime manifest.',rows=queries))
    write(OUT/'evaluation_strata.json',dict(scope='Evaluator only; never supplied to proposal, feature or assignment code.',rows=strata))
    evidence=[
      'AGENTS.md','runs/spatial_identity_maps_20261107/comparison_report.json',
      'runs/spatial_identity_maps_20261107/bottleneck_report.json','runs/spatial_identity_maps_20261107/native_diagnosis.json',
      'runs/spatial_identity_maps_20261107/competitor_histories.json','runs/spatial_identity_maps_20261107/execution_protocol.json',
      'runs/spatial_identity_maps_20261107/independent_audit.json','runs/spatial_identity_maps_20261107/image_normalization.json',
      'runs/continuation_20261029/cost_use_val_audit/report.json','runs/continuation_20261029/cost_use_val/protocol.json',
      'runs/stitch_v2/gap150_dist5/report.json','tracking/xywh_bytetrack.py',
      '.venv/Lib/site-packages/ultralytics/trackers/byte_tracker.py',
      '.venv/Lib/site-packages/ultralytics/trackers/utils/stracks.py',
      '.venv/Lib/site-packages/ultralytics/trackers/utils/matching.py',
      'research_spatial_identity_20261107/comparison_protocol.json',
      'runs/z13_local_peak_20261021/geometry.json','runs/z13_target_center_dev_20261018/C/normalization.json',
      'runs/z13_target_center_dev_20261018/R1/normalization.json','scripts/run_bdd_video_rate_experiment.py',
      'runs/train_cache/0001542f-ec815219/motion/dis_quarter.json']
    source_pins.update({n:sha(ROOT/n) for n in evidence})
    cached=read(ROOT/'runs/continuation_20261029/cost_use_val/protocol.json')
    inventory={split:dict(total=sum(c.values()),by_reason=dict(c),contributing_videos=len(videos[split])) for split,c in counts.items()}
    arms=['B0','G','G_SHIFT','G_IMAGE','G_R1','G_DIS','G_FLYVIS','G_PERM']
    comparisons=[['G','B0']]+[['G_FLYVIS',x] for x in ['G','G_SHIFT','G_IMAGE','G_R1','G_DIS','G_PERM']]
    protocol=dict(
      id='identity_geometry_competition_protocol_v1_20261108',status='prepared_not_executed_not_launch_ready',
      prepared=stamp.isoformat(),base_commit=head,branch=branch,
      authorization='Prepare a concrete protocol for singleton refusal and old-ID competition, with matched no-FlyVis controls. No scientific execution.',
      system_level='External spatial readout and first-stage association of the author system; frozen FlyVis core.',
      problems=['singleton_readout_fallback','old_track_loses_global_assignment'],
      native_config=cached['config'],
      geometry=dict(source='research_spatial_identity_20261107/comparison_protocol.json/geometry',
          eligibility='Same causal exactly-two high-confidence search and nominal 2 marginal SD size/edge ranges; nonempty position grid; >=3 current and reference geometric cells.',
          singleton='Return the unique admissible candidate; do not require evaluating the invalid candidate.',
          proposal_cost='C_G[i,j]=C[i,j]-alpha*g[i,j]; retain other native edges.',
          alignment='One feature-independent minimum standardized-edge-residual transform; tie W,H,tx,ty.',
          common_support='Exact receptor geometry intersected with DIS interpolation domain; >=3 unique reference cells for feature scores. No outside fill.',
          no_observation_change=True),
      competition=dict(
          timing='After prediction, before any state update. One solve of the full first-stage matrix.',
          native_ids='Preserve through native re_activate(new_id=False), no output relabeling or historical stitching.',
          rivals='R(j): native C<=0.8 or g=1. At least2 same-class rows; missing geometry or mandatory score disables only feature correction of the column.',
          correction='C_Z[i,j]=C_G[i,j]-beta*(a[i,j]-mean_R(a[:,j])); center across TRACKS, never across a singleton row.',
          observer='One current eye per affected column: minimum position-covariance trace, then start time, then ID as a computational tie.',
          complete_lifecycle='Evaluate second stage, unconfirmed tracks, births, duplicate removal and actual output; a first-stage win is not yet ID preservation.'),
      arms=arms,
      representations=dict(G='No image or core features.',G_SHIFT='Uniform -beta over competing edges; no feature content.',
          G_IMAGE='Processed receptor brightness and2 spatial derivatives, old fixed fit84 normalization.',
          G_R1='17 fixed causal EMA+17 transient direct channels, old R1 normalization.',
          G_DIS='Cached dis_quarter gain1.0; vector cosine without spatial centering; validate units, axes, times and interpolation.',
          G_FLYVIS='8 direct standardized T4/T5 maps, frozen flow/0000/000; centered spatial correlation.',
          G_PERM='Same8 maps,32 fixed joint-vector spatial permutations; mean score, no best permutation.'),
      undefined='Zero feature correction for entire affected column of that arm; preserve G. Never treat missing as bad identity.',
      calibration=dict(source='fit20',initial_known_queries=323,initial_shadow_queries=300,
          alpha_grid=[0,.1,.2,.4,.8],alpha_for='G only, then fixed shared alpha',beta_grid=[0,.1,.2,.4,.8],
          beta_for='Each additional arm, equal5-value budget, includes zero.',
          criterion='J=correct old-ID assignments-wrong old-ID assignments on fixed known queries, full assigned detection including outside pair; unassigned0; unknown masked.',
          tie='Fewer changes of native assignments, then smaller parameter.',freeze_before='evaluate26'),
      cohorts=dict(inventory=inventory,queries='queries.json',evaluator_only='evaluation_strata.json',
          known_filter='ok or neither_target; frozen before selection; neither means neither SEARCH candidate, not absence from frame.',
          historical_limit='Reused train-source queries and selected historical archive; no independent or unconditional tracker claim.',
          future_val20=dict(source='runs/continuation_20261029/cost_use_val/protocol.json/sources',
             videos=[s['video'] for s in cached['sources']],detector_unseen=True,researcher_unseen=False,
             readiness='Verify cached detector/checkpoint provenance and disjointness before launch; no silent substitution.')),
      readiness=dict(launch_ready=False,
          available='935 recovered positive-case map/state packages; native full matrices for those935 queries; source candidate metadata; one DIS cache inspected.',
          missing=['Complete negative/unknown query export','All rival preassociation mean/covariance and real observation histories',
                   'Rival reference maps and full column evaluator identities','Validated DIS bindings/coverage for every cohort',
                   'Runtime implementation and zero-intervention parity','Prospective run card, actual source hashes and explicit run instruction']),
      stages=['Technical readiness and first fixed fit84 cost/parity gate','Fit20 calibration on complete known positives and negatives',
              'Single evaluate26 fixed-state diagnostic, retain unknown shadow cases','Separately authorized full causal historical val20 pilot only after gate'],
      screening='Prepare full pilot if lower95% paired-video J improvement vsB0 positive for G orG_FLYVIS; exploratory gate, not tracker evidence.',
      full_metrics=['HOTA','AssA','IDF1','IDSw','FP','FN','old_ID_reactivation','wrong_target_takeover','duplicate_ID','sustained_ID_return'],
      sustained='Current plus next2 available annotated observations; censored if unavailable; never used for runtime decision.',
      inference=dict(unit='video',samples=20000,seed=42,primary_comparisons=comparisons,
          family_size=7,ci_levels_percent=[95,100*(1-.05/7)],
          geometry_acceptance='Corrected lower HOTA bound G-B0>0.',
          flyvis_acceptance='All6 corrected lower HOTA bounds of G_FLYVIS vscontrols>0.',
          harm_guard='For accepted variant vs its main control: nonnegative mean AssA/IDF1 changes; no increase in total IDSw,FP,FN.',
          no_success_from='Conditional pair accuracy, selected subgroup, pure cost shift, first-stage win before lifecycle, or reused-history pseudo rollout.'),
      literature=[
          dict(title='ByteTrack',year=2022,url='https://arxiv.org/abs/2110.06864v3',
               code='https://github.com/FoundationVision/ByteTrack/blob/main/yolox/tracker/byte_tracker.py',
               used='Global assignment and native same-ID reactivation are existing baseline mechanisms, not our novelty.'),
          dict(title='Tracking without bells and whistles',year=2019,url='https://arxiv.org/html/1903.05625v3',
               code='https://github.com/phil-bergmann/tracking_wo_bnw/blob/master/src/tracktor/tracker.py',
               used='Inactive-track references and geometry-gated reidentification; not evidence for our active-rival correction.'),
          dict(title='Fast Optical Flow using Dense Inverse Search',year=2016,url='https://arxiv.org/abs/1603.03590',
               documentation='https://docs.opencv.org/4.13.0/de/d4f/classcv_1_1DISOpticalFlow.html',used='Ordinary no-core motion control.')],
      failed_links=[dict(url='https://openaccess.thecvf.com/content_ICCV_2019/papers/Bergmann_Tracking_Without_Bells_and_Whistles_ICCV_2019_paper.pdf',result='403, replaced by checked arXiv and official code'),
          dict(url='https://docs.opencv.org/4.x/da/d06/classcv_1_1DISOpticalFlow.html',result='404, replaced by checked official4.13 documentation')],
      execution_policy='No execution now. At future launch: explicit MSK deadline, one CUDA mutex, hidden Idle workers, sequential system reserve max(6GiB,20%RAM)+2GiB at start. Do not repeat finished work.',
      source_sha256=source_pins)
    write(OUT/'protocol.json',protocol)
    assert len(protocol['cohorts']['future_val20']['videos'])==20
    journal=f"**{stamp.strftime('%d.%m.%Y, %H:%M MSK')}** — По поручению автора подготовлен протокол исправления единственного допустимого кандидата и конкуренции треков: `research_identity_assignment_protocol_20261108/ПРОТОКОЛ.md`, `protocol.json`. Уточнено: первый отказ относится к пробному считывателю; штатный решатель уже общий. Предложены геометрическая поправка существующего ребра и центрирование сходства между треками до единственного решения полной матрицы, со штатным сохранением ID и проверкой после всех стадий. Контроли: G, общий сдвиг, изображение, R1, DIS и перестановка карт. Из готового архива выписаны 623/1368 запросов fit20/evaluate26, включая 233/463 случая чужих обеих поисковых рамок; это не отсутствие цели во всём кадре. Источники и хеши — в `protocol.json`. Полные истории конкурентов и отрицательные запросы ещё требуют экспорта; исполняемая правка не реализована. Опыт не зарегистрирован и не запускался; каталог, очередь, защищённые гипотезы и план целей не менялись. Подготовка не является подтверждением пользы.\n\n"
    write(OUT/'journal_entry.json',dict(entry=journal))
    before=plan.read_bytes();text=before.decode('utf-8').replace('\r\n','\n');anchor='## Журнал изменений\n\n';assert text.count(anchor)==1
    plan.write_text(text.replace(anchor,anchor+journal,1),encoding='utf-8')
    assert sha(tracker_plan)==tracker_hash
    assert all(sha(ROOT/n)==h for n,h in protected_hashes.items())
    write(OUT/'preparation_checks.json',dict(passed=True,scientific_experiments_started=0,
        runtime_files_changed=False,catalog_and_glossary_unchanged=True,tracker_plan_unchanged=True,
        counts=inventory,queries=len(queries),primary_family_size=len(comparisons),
        protected_sha256=protected_hashes,tracker_plan_sha256=tracker_hash,
        working_plan_sha256_before=hashlib.sha256(before).hexdigest(),working_plan_sha256_after=sha(plan)))
    manifest={p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='checksums.json'}
    write(OUT/'checksums.json',dict(prepared=stamp.isoformat(),files=manifest))
    print(json.dumps(dict(status=protocol['status'],inventory=inventory,scientific_experiments_started=0),ensure_ascii=False))


if __name__=='__main__':main()
