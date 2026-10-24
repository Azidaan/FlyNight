"""Frozen-policy decomposition of FlyVis refusal: diagnostic, never a new policy."""
from pathlib import Path
from datetime import datetime, timedelta, timezone
from collections import Counter
import json
import os
import sys
import time
import subprocess
os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts.z13_candidate_reserve_20261021 import read, write, sha

RUN = ROOT/'runs/z13_refusal_causes_20261025'
OLD = ROOT/'runs/z13_assignment_effect_20261024/recovery1'
CACHE = ROOT/'runs/z13_candidate_reserve_20261021'
BENCH = ROOT/'runs/z13_val20_matched_20261024'
IDENT = 'z13_refusal_causes_v1'
ARMS = ('D', 'C', 'R1')
MSK = timezone(timedelta(hours=3))


def now():
    return datetime.now(MSK)


def journal(title, text):
    from scripts import z13_val20_common_20261024 as m
    saved = m.IDENT
    try:
        m.IDENT = IDENT
        m.journal(title, text)
    finally:
        m.IDENT = saved


def bucket(before, choices):
    if before:
        return 'baseline_already_correct'
    if choices[0]['beneficial']:
        return 'top1_can_help'
    if any(c['beneficial'] for c in choices[1:]):
        return 'only_other_candidate_can_help'
    return 'no_current_candidate_can_help'


def curve_stats(scores, targets):
    from sklearn.metrics import roc_auc_score, average_precision_score
    targets = np.asarray(targets, bool)
    return dict(n=len(targets), positives=int(targets.sum()), prevalence=float(targets.mean()),
        roc_auc=float(roc_auc_score(targets, scores)) if len(set(targets)) == 2 else None,
        average_precision=float(average_precision_score(targets, scores)) if targets.any() else None)


def runtime_meta(arm, video):
    path=BENCH/('center'+arm)/'signatures'/video/'signatures.json'
    if path.exists():
        return path, []
    marker=BENCH/'common_authorized_recovery2/jobs'/f'signature_center{arm}_{video}.json'
    job=read(marker)
    source=ROOT/job['reused_instrumented_diagnostic']/('center'+arm)/'signatures'/video
    source.resolve().relative_to(BENCH.resolve())
    expected=job['result']['signature_sha256']
    target=path.with_suffix('.npz')
    assert sha(source/'signatures.npz') == sha(target) == expected
    return source/'signatures.json', [marker,source/'signatures.npz',target]


def prepare():
    assert not RUN.exists(), 'No duplicate or overwrite'
    oldp = read(OLD/'protocol.json')
    assert read(OLD/'controller_result.json')['all_owned_ended']
    names = [Path(__file__).relative_to(ROOT).as_posix(), 'scripts/z13_refusal_causes_console.ps1',
        'tests/test_z13_refusal_causes.py', 'scripts/z13_bounded_stage_20261022.py',
        'tracking/z13_spatial_localizer.py', 'tracking/z13_ridge_localizer.py',
        'tracking/z13_spatial.py', 'tracking/z13_readout.py', 'scripts/build_fly_signatures.py',
        'scripts/z13_spatial_train.py', 'scripts/z13_spatial_hota_protocol.py',
        'scripts/z13_center_hota_20261021.py',
        (OLD/'protocol.json').relative_to(ROOT).as_posix(),
        (OLD/'report.json').relative_to(ROOT).as_posix(),
        (OLD/'controller_result.json').relative_to(ROOT).as_posix(),
        (OLD/'independent_audit_20261025.json').relative_to(ROOT).as_posix(),
        'runs/z13_candidate_reserve_20261021/evaluate/labels.json',
        'runs/z13_candidate_reserve_20261021/fit/labels.json',
        'runs/z13_candidate_data104_20261021/evaluate/labels.json']
    for v in oldp['evaluate26']:
        names.append((OLD/'videos'/(v+'.json')).relative_to(ROOT).as_posix())
    benchmark = read(BENCH/'common_authorized_recovery2/report.json')
    for arm in ARMS:
        names += [f'runs/z13_target_center_20261018/{arm}/center_rows.json',
                  f'runs/z13_center_hota_20261021/{arm}/config.json']
        names += [x.relative_to(ROOT).as_posix() for x in (CACHE/'evaluate'/arm).glob('cache_*.npz')]
        for video in benchmark['videos']:
            meta, bindings=runtime_meta(arm,video)
            names += [x.relative_to(ROOT).as_posix() for x in [meta,*bindings]]
    p = dict(id=IDENT, created=now().isoformat(), device='cpu', budget_seconds=300,
        save_at=(now()+timedelta(seconds=300)).isoformat(), hard_stop=(now()+timedelta(seconds=320)).isoformat(),
        authorization='Author explicitly requested causes of mistimed FlyVis intervention; standing autonomous weekend authorization until26.10 07:40MSK. Read-only cached diagnosis, no training.',
        action='Locate whether the FlyVis readout needs a utility-relative-to-baseline refusal output, a better candidate choice, or candidate generation. Do not choose a repair before separating these failures.',
        question='With the existing fixed top1% policy, do high map peaks select useful first-association corrections? Is lack of benefit caused by unnecessary intervention, missed useful top1, wrong candidate rank or no useful proposal?',
        hypotheses=['Center-useful training labels differ from association benefit (already observed).',
            'Centered-map confidence measures peak distinctiveness rather than utility relative to unchanged tracker.',
            'A useful candidate can exist but be rejected or ranked below top1.',
            'Current three proposals may lack a useful intervention.',
            'Runtime intervention rate may differ from annotated-time training rate; descriptive comparison cannot isolate why.'],
        method='All8326 cached natural rows for eachC/D/R1; UID/video/frame/track binding to exact assignment-effect rows. Verify all saved top1 centers within0.5px and peak scores within0.001 before joining effects. Keep saved center tau unchanged, no optimization. All known5822rows: partition baseline correct/top1help/onlyotherhelp/nonehelp, split by fixed decision. Unknown2504reported separately. Rank metrics descriptive only, no fit. Compare aggregate runtime metadata on existing20val without using outcomes or tuning thresholds.',
        technical='All26videos and8326rows each; known/unknown exhaustive, all four categories partition known, selected and refused exhaustive. Exact reproduction of original assignment counts and SHA. No legacyD center is joined to centeredD effects: distinct outputs. Runtime metadata frozen.',
        decision_rule='Selected harmful/neutral outcomes demonstrate failure of fixed refusal for this one-step label. Helpful top1 rejected indicates a gate miss; helpful alternative but nottop1 indicates candidate choice; no beneficial candidate indicates proposal/action limitation. These classifications are descriptive and not causal attribution to weights, lighting or biological wiring. No performance acceptance or new threshold is introduced.',
        next='At most one further exact CPU trace diagnosis of selected harmful/missed cases to identify missing detections, competition or geometry; then register a concrete readout repair or explicitly stop that repair. Do not train the previously failed assignment-supervision version.',
        preparation_note='First metadata inventory found one R1 JSON absent beside the reused NPZ. Resolve only through the successful signature job reused_instrumented_diagnostic field and require both source/destination NPZ SHA equal job SHA. No inference repeated and no scientific tolerance changed.',
        literature=[dict(url='https://arxiv.org/html/2103.16556v2', section='3.2-3.7', use='Candidate association and explicit unmatched option motivate separating identity, presence and decision; not a reproduction or transfer of its gains.')],
        limits='Episodes on unchanged baseline states and sparse original labels, reused train26; no closed-loop counterfactual or HOTA. Top1selection here is only centered readout; legacy quality head remains separate. Statistics over known labels are not unconditional safety; unknown outcomes stay unknown.',
        sha256={n:sha(ROOT/n) for n in sorted(set(names))})
    write(RUN/'protocol.json', p)
    write(RUN/'status.json', dict(state='ready', stage='Проверка причин решения вмешаться', done=0, total=3, updated=now().isoformat()))
    cp = ROOT/'experiments/project_history.json'
    cat = read(cp)
    assert not any(c['id'] == IDENT for c in cat['experiments'])
    cat['experiments'].append(dict(id=IDENT, title='З13: причины неверного вмешательства — фиксированное правило и запас кандидатов',
        area='flyvis', state='ready', started=now().strftime('%Y-%m-%d'), tags=['fly-problem-P8'],
        question=p['question'], hypothesis='Неверное вмешательство может возникать на разных узлах; высота пика не равна ожидаемой пользе относительно исходного назначения.',
        expected=p['method'], acceptance=[dict(criterion=p['technical'], met=None)],
        summary='Диагностика существующей политики; без нового обучения и изменения порога.',
        obtained='', interpretation='', decision='', cannot_claim=p['limits'], metrics=[], commits=[],
        files=[Path(__file__).relative_to(ROOT).as_posix()],
        evidence=[dict(kind='file', ref=(RUN/'protocol.json').relative_to(ROOT).as_posix(), claim='Причины и правила разбора записаны до расчёта')]))
    write(cp,cat)
    journal('постановка разбора причин вмешательства', '*Постановка.* '+p['question']+'\n\n*Действие.* '+p['action']+
        '\n\n*Проверка.* '+p['method']+'\n\n*Критерий полноты.* '+p['technical']+'\n\n*Решение по результату.* '+p['decision_rule']+
        '\n\nДо300сCPU, независимый контроллер, без GPU/обучения/новойHOTA/отложенных75. Разрешение автора25.10:вплотную найти причины неверного включения мухи. Старый критерий2%не меняется.')


def worker():
    from scripts.z13_continue import paired
    p = read(RUN/'protocol.json')
    for rel, h in p['sha256'].items():
        assert sha(ROOT/rel) == h, rel
    oldp, oldr = read(OLD/'protocol.json'), read(OLD/'report.json')
    uid = len(read(CACHE/'fit/labels.json')) + len(read(ROOT/'runs/z13_candidate_data104_20261021/evaluate/labels.json'))
    labels = read(CACHE/'evaluate/labels.json')
    binding = []
    for video in oldp['evaluate26']:
        outcomes = {r['uid']:r for r in read(OLD/'videos'/(video+'.json'))['rows']}
        for label in [l for l in labels if l['video'] == video]:
            r = outcomes[uid]
            assert r['label'] == label['label'] and r['video'] == video and r['kind'] == label['kind']
            binding.append((label,r));uid += 1
    assert len(binding) == 8326
    results, allrows = {}, []
    for ai, arm in enumerate(ARMS):
        if (RUN/'stop.request').exists() or now() >= datetime.fromisoformat(p['save_at']):
            raise TimeoutError('Diagnostic stage budget')
        write(RUN/'status.json', dict(state='running', stage='Связь фиксированного отбора с последствиями', variant=arm, done=ai, total=3, updated=now().isoformat()))
        saved = {r['label']:r for r in read(ROOT/f'runs/z13_target_center_20261018/{arm}/center_rows.json')}
        cache = {}
        for path in sorted((CACHE/'evaluate'/arm).glob('cache_*.npz')):
            with np.load(path) as z:
                for i in range(len(z['episode'])):
                    cache[(int(z['episode'][i]),int(z['t'][i]))] = (z['centers'][i,0].copy(),float(z['scores'][i,0]))
        config = read(ROOT/f'runs/z13_center_hota_20261021/{arm}/config.json')
        tau = config['tau']
        assert sum(r['p_present'] >= tau for r in saved.values()) == config['policy']['selected']
        decisions, categories, strata = Counter(), Counter(), Counter()
        max_center, max_score = 0., 0.
        known_scores, beneficial, center_labels = [], [], []
        pv = {}
        for label, outcome in binding:
            s = saved[label['label']]
            assert s['video'] == label['video'] and s['kind'] == label['kind'] and s['pos'] == label['pos']
            center, score = cache[(label['episode'],label['pos']+4)]
            max_center = max(max_center,float(np.max(np.abs(center-s['center']))))
            max_score = max(max_score,abs(score-s['p_present']))
            chosen = s['p_present'] >= tau
            side = 'selected' if chosen else 'refused'
            decisions[side] += 1
            row = dict(arm=arm, uid=outcome['uid'], label=label['label'], video=label['video'],
                frame=outcome['frame'], tid=outcome['tid'], kind=label['kind'], score=s['p_present'],
                tau=tau, selected=chosen, candidate=0)
            if label['kind'] < 0:
                strata[side+'_unknown'] += 1
                row['outcome'] = 'unknown'
            else:
                options = outcome['arms'][arm]
                old = options[0]
                category = bucket(old['own_before'],options)
                name = 'beneficial' if old['beneficial'] else 'harmful' if old['harmful'] else 'neutral'
                categories[category+';'+side] += 1
                strata[side+'_known'] += 1
                strata[side+'_'+name] += 1
                row.update(outcome=name, category=category, own_before=old['own_before'],
                    own_after=old['own_after'], any_beneficial=any(x['beneficial'] for x in options),
                    old_center_useful=old['old_center_useful'])
                known_scores.append(s['p_present']);beneficial.append(old['beneficial']);center_labels.append(old['old_center_useful'])
                v=pv.setdefault(label['video'],dict(known=0,selected=0,helpful=0,harmful=0))
                v['known']+=1;v['selected']+=chosen;v['helpful']+=chosen and old['beneficial'];v['harmful']+=chosen and old['harmful']
            allrows.append(row)
        assert max_center < .5 and max_score < .001, (arm,max_center,max_score)
        assert sum(categories.values()) == oldr['candidate_counts'][arm]['known'] == 5822
        assert sum(decisions.values()) == 8326
        assert decisions['selected'] == config['policy']['selected']
        assert strata['selected_known'] + strata['selected_unknown'] == decisions['selected']
        benchmark=read(BENCH/'common_authorized_recovery2/report.json')
        runtime={}
        for video in benchmark['videos']:
            meta_path,_=runtime_meta(arm,video)
            meta=read(meta_path)
            assert meta['z13_tau'] == tau and meta['z13_arm'] == arm
            e=meta['z13_events'];assert e['steps'] == e['interventions']+e['refusals']
            runtime[video] = e
        totals={k:sum(x[k] for x in runtime.values()) for k in ('steps','interventions','refusals','births','nonfinite_refusals')}
        gain={v:(x['helpful']-x['harmful'])/x['known'] for v,x in pv.items()}
        results[arm]=dict(tau=tau, reproduction=dict(rows=8326,max_center_px=max_center,max_score=max_score),
            decisions=dict(decisions), known_unknown_and_outcomes=dict(strata), failure_partition=dict(categories),
            score_ranking_for_association_benefit=curve_stats(known_scores,beneficial),
            score_ranking_for_old_center_utility=curve_stats(known_scores,center_labels),
            per_video=pv, paired_net_action_frequency=paired(gain,{v:0. for v in gain}),
            runtime20_totals=totals, runtime20=runtime,
            runtime20_intervention_fraction=totals['interventions']/totals['steps'])
    for rel,h in p['sha256'].items():
        assert sha(ROOT/rel)==h,rel
    write(RUN/'rows.json',allrows)
    write(RUN/'report.json',dict(complete=True, diagnostic_only=True, not_hota=True, arms=results,
        rows_sha256=sha(RUN/'rows.json'), protocol_sha256=sha(RUN/'protocol.json'),
        limits=p['limits'], gate_unchanged=True, no_training=True))
    write(RUN/'status.json',dict(state='completed',stage='Разбор фиксированного решения сохранён;не HOTA',done=3,total=3,updated=now().isoformat()))


if __name__ == '__main__':
    if '--prepare' in sys.argv:
        prepare()
    elif '--worker' in sys.argv:
        try:
            worker()
        except Exception as error:
            write(RUN/'status.json',dict(state='failed',stage=repr(error),updated=now().isoformat()))
            raise
    else:
        raise ValueError('Choose --prepare or --worker')
