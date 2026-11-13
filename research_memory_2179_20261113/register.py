from common import *
import shutil, subprocess
import numpy as np
assert not (OUT / 'protocol.json').exists()
before = OUT / 'before'; before.mkdir()
for n in ['experiments/project_history.json', 'experiments/system_problems.json', 'experiments/obsidian_experiment_topics.json', 'experiments/useful_literature.json', 'dissertation/ЕДИНЫЙ_ПЛАН.md']:
    shutil.copyfile(ROOT / n, before / Path(n).name)
runtime = read(PREV / 'runtime.json')
plan = read(PREV / 'descriptor_plan.json')
frames = {}; observations = []
with np.load(ROOT / read(PREV / 'evaluation_plan.json')['detector_file'], allow_pickle=False) as z:
    raw = z['xyxy'].copy(); rawframes = z['frame_idx'].copy()
for e in runtime['events']:
    for r in e['rows']:
        if r['tid'] not in [2179, 2445]: continue
        for h in r['history']:
            rid = h['raw_id']; frame = h['frame']
            assert frame < e['frame'] and int(rawframes[rid]) == frame
            entry = dict(tid=r['tid'], frame=frame, raw_id=rid, filtered=h['box'], raw=raw[rid].astype(float).tolist())
            if entry not in observations: observations.append(entry)
def add(frame, box):
    p = frames.setdefault(str(frame), dict(frame=frame, boxes=[]))
    if box not in p['boxes']: p['boxes'].append(box)
    return [str(frame), p['boxes'].index(box)]
for o in observations:
    for kind in ['filtered', 'raw']: o[kind + '_slot'] = add(o['frame'], o[kind])
save('descriptor_plan.json', dict(video_path=plan['video_path'], frames=frames, observations=observations, scope='Only already observed history boxes of two implicated tracks; no future or GT selection.'))
criteria = [
    'Reproduce all previous native and appearance matrices at fixed alpha .25 and 1 within 1e-6; retain every row and candidate, threshold .8 and archived neural delta.',
    'Change only descriptors of rows 2179/2445; compare latest filtered, latest raw, earliest retained filtered/raw, minimum distance over all retained raw history, and isolated 2445 freeze at1158/1159. No tuned weights or GT in runtime.',
    'Independent scalar cosine and independent assignment solver objective agree; exact labels are evaluation-only, unannotated frames remain unknown.',
    'Any loss of baseline known-correct assignments or new known-wrong assignment rejects promotion; unknown added links cannot establish benefit.',
    'Record trajectory/box provenance and exact-anchor raw-versus-filtered ownership separately from descriptor evidence; do not claim first physical ID switch without exact labels.',
    'All source inputs and production settings unchanged; no core/full-tracker/training/download calls; at most one frozen MASA feature extraction for missing observations, all temporary outputs on E.'
]
inputs = {}
paths = [PREV / n for n in ['runtime.json','descriptor_plan.json','descriptors.npz','evaluation_plan.json','summary.json','result.json']]
paths += [ROOT / n for n in read(PREV / 'initial_state.json')['inputs']]
paths += [ROOT / n for n in ['weights/masa/masa_r50.pth','tracking/masa_r50.py','tracking/gpu_lock.py','data/bdd100k/videos/val/b1ca2e5d-84cf9134.mov']]
paths += list(OUT.glob('*.py')) + [OUT / 'descriptor_plan.json']
for p in paths: inputs[p.relative_to(ROOT).as_posix()] = sha(p)
policies = ['native', 'latest', 'latest_raw', 'earliest_filtered', 'earliest_raw', 'gallery_raw', 'freeze2445_1158', 'freeze2445_1159', 'swap_latest_negative']
save('protocol.json', dict(registered=now(), head=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(), server_head='bf565df4fc02720e790cf6fe55d733d47baa82a0', question='Does history representation/update cause or reinforce the 2179 competition failure?', hypothesis='Competing track history may already contain a broad mixed observation; later updates may reinforce this. Alternatives: initial geometry dominance or insufficient descriptor separation even with earlier/raw observations.', action='Copy-only history selector replacing latest filtered descriptor by earliest retained raw or raw-history gallery; no production promotion without exact non-harm evidence.', policies=policies, alphas=[.25,1.0], frames=[1159,1160,1165], acceptance=criteria, inputs=inputs, literature=[dict(url='https://arxiv.org/html/1703.07402', role='Deep SORT stores a gallery of associated appearance descriptors and uses nearest cosine distance; motivates raw-history gallery control, not claimed unchanged algorithm.'),dict(url='https://github.com/nwojke/deep_sort/blob/master/deep_sort/nn_matching.py', role='Official implementation of bounded gallery updates; no claimed guarantee against contaminated associations.')], scope='Three frozen full assignment matrices of one known case. Two-row descriptor interventions. Not lifecycle replay, unseen validation or HOTA.', deadline='2026-11-16T07:35:00+03:00'))
cat = read(ROOT / 'experiments/project_history.json')
identity = 'history_representation_2179_20261113'
assert not any(c['id'] == identity for c in cat['experiments'])
cat['experiments'].append(dict(id=identity,title='Память объекта2179 — исходная рамка и обновление образцов',area='own_tracker',state='running',started='2026-11-13',question='В какой части памяти возникает поддержка конкурирующего номера2445?',hypothesis='Сглаженная рамка может захватывать соседа; обновления способны закрепить ошибку. Проверяется заменой только исторических образцов двух конкурентов при прежней полной матрице.',expected='Разделить влияние исходной рамки, возраста образца и обновления памяти без полного трекера.',acceptance=[dict(criterion=c,met=None) for c in criteria],evidence=[dict(kind='file',ref='research_memory_2179_20261113/protocol.json')]))
(ROOT / 'experiments/project_history.json').write_bytes((json.dumps(cat,ensure_ascii=False,indent=2)+'\n').encode())
tp=ROOT/'experiments/obsidian_experiment_topics.json'; topics=read(tp);topics['topics'][identity]=['Сохранение личности','Память объектов','Рамка и обновление образцов'];tp.write_bytes((json.dumps(topics,ensure_ascii=False,indent=2)+'\n').encode())
journal('Постановка (`history_representation_2179_20261113`). Продолжение после отрицательной пробы весов: на копии проверяются исходная/сглаженная рамка памяти, ранний образец и набор прошлых образцов только для2179и2445. Все строки, кандидаты, порог0,8 и прежние веса0,25/1 сохраняются; точная разметка отделена от расчёта. Ядро и полный трекер запрещены. Критерии до расчёта: research_memory_2179_20261113/protocol.json. Локальный и проверенный GitHubHEADbf565df4fc02720e790cf6fe55d733d47baa82a0 совпадают; чужие правки сохраняются.')
print('REGISTERED', len(observations), 'history observations', len(frames), 'frames', sum(len(x['boxes']) for x in frames.values()), 'ROIs')
