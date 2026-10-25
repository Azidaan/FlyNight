"""Rebind every saved candidate identity to primary annotations and assess saved cost evidence."""
from pathlib import Path
from collections import Counter
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts import z13_temporal_pair_readiness_20261025 as S
from scripts.association_log import label


def main():
    p = S.read(S.RUN/'protocol.json'); report = S.read(S.RUN/'report.json')
    count = Counter(); common_reference = 0
    other = ROOT/'runs/z13_temporal_observation_gate_20261025'
    for video in p['videos']:
        with np.load(ROOT/f'runs/train_cache/{video}/detections/raw_video.npz') as z:
            detections = {k:z[k] for k in ('xyxy','conf','cls','frame_idx')}
        frames = S.read(ROOT/f'data/bdd100k/labels/box_track_20/train/{video}.json')
        aligned = S.read(ROOT/f'runs/train_cache/{video}/alignment.json')['matches']
        truth = {}
        for frame, a in zip(frames, aligned):
            objects = []
            for obj in frame.get('labels') or []:
                if obj['category'] not in S.CLASSES or (obj.get('attributes') or {}).get('crowd',False):
                    continue
                b = obj['box2d']
                objects.append((str(obj['id']),obj['category'],[b[k] for k in ('x1','y1','x2','y2')]))
            truth[a['video_frame_index']] = objects
        cache = {}
        def identities(f):
            if f not in cache:
                ix=np.flatnonzero((detections['frame_idx']==f)&(detections['conf']>=.25))
                cache[f]=label(list(detections['xyxy'][ix]),[S.CLASSES[int(c)] for c in detections['cls'][ix]],truth[f])
            return cache[f]
        rows = S.read(S.RUN/'inputs'/(video+'.json'))['rows']
        labs = S.read(S.RUN/'supervision'/(video+'.json'))['rows']
        other_rows = {x['uid']:x for x in S.read(other/'videos'/(video+'.json'))['rows']}
        for r,l in zip(rows,labs):
            assert l['current_identity']==[identities(r['current_frame'])[j] for j in r['current_choices']]
            prev=r['previous']
            if prev:
                assert l['previous_identity']==[identities(prev['frame'])[j] for j in prev['choices']]
                o=other_rows[r['uid']]['previous']
                common_reference += int(bool(o and o['frame_array_index']==prev['array_index'] and o['frame']==prev['frame']))
            else:
                assert l['previous_identity']==[]
            count['queries']+=1
            count['candidate_identities']+=len(l['current_identity'])+len(l['previous_identity'])
            count['unknown_candidate_identities']+=sum(x is None for x in l['current_identity']+l['previous_identity'])
    assert count['queries']==37296
    previous = report['oracle26']['oracle_old']['mean_per_video']
    restored = report['oracle26']['oracle_restored']['mean_per_video']
    values={v:(report['per_video'][v]['oracle_restored']-report['per_video'][v]['oracle_old'])/report['per_video'][v]['known']
        for v in p['splits']['evaluate26']}
    other_report=other/'report.json'
    price=ROOT/'runs/z13_temporal_region_features_20261025/price.json'
    cost=S.read(price)
    receipt=dict(at=S.now().isoformat(),all_candidate_identity_labels_rebound_to_primary_annotations=True,
        counts=dict(count),same_previous_reference_as_unrestricted_gate=common_reference,
        labeled_reference_policy_differs_from_unrestricted_gate=True,
        unrestricted_gate_source=other_report.relative_to(ROOT).as_posix(),unrestricted_gate_sha256=S.sha(other_report),
        reserve_increment_over_old=S.paired(values),
        structural_reserve_relative_change_percent=100*(restored/previous-1),
        structural_reserve_fraction_of_all=restored/report['oracle26']['oracle_all']['mean_per_video'],
        oracle_required_recovery_without_harm=.02/restored,
        oracle_required_recovery_with_one_percent_harm=(.02+.01*(1-restored))/restored,
        neural_cost_evidence=dict(source=price.relative_to(ROOT).as_posix(),sha256=S.sha(price),saved_measurement=cost,
            interpretation='Existing distinct D/C/R1 preparation price estimate, not a newly measured cost for this annotated-reference subset. Do not launch a duplicate GPU job; compare exact UID/frame/region bindings before any reuse. Model training cost remains unmeasured.'),
        cpu_preparation_seconds=report['seconds'],new_gpu_measurements=0,model_training_cost_measured=False)
    S.write(S.RUN/'label_provenance_and_cost_audit.json',receipt)
    S.journal('дополнительная сверка происхождения меток и цены',
        '*Получено.* `runs/z13_temporal_pair_readiness_20261025/label_provenance_and_cost_audit.json`: '+str(dict(count))+
        '; все метки кандидатов заново привязаны к исходным реальным кадрам. Совпадающих прежних ссылок с отдельной подготовкой без требования разметки: '+str(common_reference)+
        '.\n\n*Различие.* Размеченная и наиболее свежая предыдущая область — разные правила входа; автоматически подменять одно другим нельзя. Существующая оценка цены отдельной подготовки D/C/R1 сохранена с хешем источника; это не новый замер нашей выборки и не цена обучения. Новый GPU-процесс не запускался.\n\n*Сводка.* '+str(receipt['reserve_increment_over_old'])+
        '. HOTA не измерялась; доля потолка относится к доступным действиям, не к предсказаниям модели.')
    print(receipt)


if __name__=='__main__':
    main()
