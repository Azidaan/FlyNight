"""Existing matrix/box accounting only; no new assignment intervention."""
from common import ROOT, OUT, read, sha, write, now
import gzip
import numpy as np
video = 'b1ca2e5d-84cf9134'
archive_path = f'runs/domain_matched_native_20261109/rollouts/C34/{video}.events.json.gz'
with gzip.open(ROOT / archive_path, 'rt', encoding='utf8') as stream:
    event = next(event for event in __import__('json').load(stream)['events'] if event['frame'] == 1160)
native = read(ROOT / 'runs/domain_matched_native_20261109/protocol.json')
cache_path = native['videos'][video]['detections']
with np.load(ROOT / cache_path, allow_pickle=False) as stream:
    raw = {key: stream[key].copy() for key in ['xyxy', 'frame_idx', 'conf', 'cls']}
columns = list(event['high_raw_ids'])
target_column = columns.index(109699)
rows = event['rows']

def box(row):
    x, y, width, height = row['mean'][:4]
    left, top = x - width / 2, y - height / 2
    return [left, top, left + width, top + height]

def detection(raw_id):
    return dict(raw_id=int(raw_id), cls=int(raw['cls'][raw_id]), confidence=float(raw['conf'][raw_id]), raw_xyxy=raw['xyxy'][raw_id].astype(float).tolist())

ranking = []
for index in sorted(range(len(rows)), key=lambda index: (event['C'][index][target_column], rows[index]['tid'])):
    row = rows[index]
    ranking.append(dict(rank=len(ranking) + 1, row=index, tid=row['tid'], cls=row['cls'], state=row['state'], start_frame=row['start_frame'],
                        predicted_xyxy=box(row), last_history_frame=row['history'][-1]['frame'] if row['history'] else None,
                        base_cost=event['C'][index][target_column], corrected_cost=event['corrected'][index][target_column],
                        effective_iou_from_saved_cost=(1 - event['C'][index][target_column]) / float(raw['conf'][109699]),
                        saved_actual_first=[dict(raw_id=columns[j], column=j) for i, j in event['actual_first'] if i == index]))
target_row = next(i for i, row in enumerate(rows) if row['tid'] == 2179)
nearest = [dict(**detection(columns[j]), column=j, base_cost=event['C'][target_row][j], corrected_cost=event['corrected'][target_row][j]) for j in sorted(range(len(columns)), key=lambda column: (event['C'][target_row][column], columns[column]))]
low = [detection(int(i)) for i in np.flatnonzero((raw['frame_idx'] == 1160) & (raw['conf'] > .1) & (raw['conf'] < .25))]
target = next(row for row in ranking if row['tid'] == 2179)
same_class = [row for row in ranking if row['cls'] == int(raw['cls'][109699])]
write('column109699_metadata.json', dict(created=now(), video=video, frame=1160, candidate=detection(109699), all_row_ranking=ranking,
                                        target2179_rank_all=target['rank'], target2179_rank_same_class=1 + next(i for i, row in enumerate(same_class) if row['tid'] == 2179),
                                        target2179_sorted_high_edges=nearest, target2179_high_edges_within_native_limit=[edge for edge in nearest if edge['base_cost'] <= .8],
                                        raw_native_second_candidates=low, low_costmatrix_saved=False, low_nearest_not_inferred=True,
                                        existing_archive_fields=sorted(event), no_new_assignment_interventions=True, policy_calls=0, model_calls=0,
                                        inputs={archive_path: sha(ROOT / archive_path), cache_path: sha(ROOT / cache_path), 'datasets/bdd100k_loader.py': sha(ROOT / 'datasets/bdd100k_loader.py')},
                                        scope='cls0=pedestrian/cls2=car are detector labels; physical identity at1160 is not inferred. Native IoU/confidence matrix contains cross-class edges. Ranking is not a new class-gated intervention.'))
print('COLUMN_METADATA_READY', len(ranking), len(low))
