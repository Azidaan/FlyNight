"""Freeze inputs and admission rules, without computing new scientific scores."""
from common import *
import gzip, subprocess, shutil

def register():
    assert not (OUT / 'protocol.json').exists()
    snapshots = {}
    inputs = {}
    sources = {}
    def bind(path, target=sources):
        path = Path(path)
        if not path.is_absolute():
            path = ROOT / path
        n = path.relative_to(ROOT).as_posix()
        target[n] = sha(path)
        return n
    for name in ['common.py', 'register.py', 'experiment.py', 'audit.py', 'preflight.py']:
        bind(OUT / name)
    for name in ['tracking/channel_identity_head.py', 'tracking/channel_agreement_correspondence.py',
                 'tracking/channel_balanced_correspondence.py', 'tracking/full_template_correspondence.py',
                 'tracking/hex_linear_interpolation.py', 'tracking/observed_spatial_reader.py',
                 'tracking/z13_readout.py', 'runs/z13_local_peak_20261021/geometry.json',
                 'runs/flyvis_repairs_20261106/unification_20261107/system_memory_guard.py',
                 'research_agreement_readout_20261109/math_checks.json',
                 'research_agreement_readout_20261109/ОСНОВАНИЕ.md']:
        bind(name)
    for base, names in [(TRAIN, ['protocol_v2.json', 'execution_protocol.json', 'maps_complete.json', 'features.json', 'predictions.json']),
                        (SCALE, ['summary.json', 'predictions.json', 'deployment.json', 'deployment_screen.json']),
                        (NATIVE, ['execution_protocol.json', 'deployment.json']),
                        (STABLE, ['inventory.json', 'comparison.json'])]:
        for name in names:
            bind(base / name)
    proof = read(ROOT / 'research_agreement_readout_20261109/math_checks.json')
    assert proof['passed']
    for n, h in proof['sources'].items():
        assert sha(ROOT / n) == h
        bind(n)
    phase = read(TRAIN / 'execution_protocol.json')
    edges = read(TRAIN / 'protocol_v2.json')['edges']
    oldrows = read(TRAIN / 'features.json')['records']
    train = []
    import numpy as np
    for e, old in zip(edges, oldrows):
        assert all(e[k] == old[k] for k in ['video', 'frame', 'old_id', 'raw_id'])
        assert e['kind'] == old['label']
        keys = []
        for seq in [e['sequence'][:4], e['sequence']]:
            k = key(e['video'], seq)
            files = list(phase['inventory_requests'].get(k, {}).get('ready_files', []))
            extra = TRAIN / 'maps' / (k + '.npz')
            if extra.exists():
                meta = extra.with_suffix('.json')
                assert sha(extra) == read(meta)['sha256']
                bind(meta, inputs)
                with np.load(extra, allow_pickle=False) as z:
                    files += [dict(path=extra.relative_to(ROOT).as_posix(), array=n, slot=n) for n in z.files]
            for item in files:
                bind(item['path'], inputs)
            snapshots[k] = dict(video=e['video'], sequence=seq, files=files)
            keys.append(k)
        train.append(dict(**{k: old[k] for k in ['video', 'frame', 'old_id', 'raw_id', 'label', 'geometry', 'full_cells']},
                          **{k: e[k] for k in ['reference_box', 'reference_crop', 'current_box', 'current_crop', 'offsets']},
                          keys=keys, old_variants=old['variants']))
    screen = []
    comparisons = read(STABLE / 'comparison.json')['cases']
    inventory = read(STABLE / 'inventory.json')
    assert inventory['ready']
    for c, comp in zip(inventory['records'], comparisons):
        assert all(c[k] == comp[k] for k in ['video', 'frame', 'old_id', 'raw_id'])
        eventfile = NATIVE / 'rollouts/C34' / (c['video'] + '.events.json.gz')
        bind(eventfile, inputs)
        events = json.loads(gzip.decompress(eventfile.read_bytes()))['events']
        ev = next(e for e in events if e['frame'] == c['frame'])
        matches = [x for x in ev['proposals'] if x['old_id'] == c['old_id'] and x['raw_id'] == c['raw_id']]
        assert len(matches) == 1
        proposal = matches[0]
        for name in ['reference_box', 'reference_crop', 'current_box', 'current_crop', 'geometry']:
            assert proposal[name] == c[name]
        assert proposal['channels'] == c['channels'] == comp['ready_C34_channels']
        assert [[h['frame'], h['crop']] for h in proposal['history']] == c['history']
        assert proposal['search'] == c['search']
        keys = []
        for snap, seq in zip(c['maps'], [c['history'], c['history'] + c['search']]):
            k = key(c['video'], seq)
            assert k == snap['key'] and seq == snap['sequence']
            cpath = NATIVE / 'features' / c['video'] / (k + '.C34.npy')
            meta = cpath.with_suffix('.json')
            m = read(meta)
            assert m['sequence'] == seq and m['video'] == c['video']
            assert sha(cpath) == m['sha256'] and m['execution_sha256'] == sha(NATIVE / 'execution_protocol.json')
            r = snap['matches'][0]
            assert sha(ROOT / r['path']) == r['sha256']
            bind(meta, inputs)
            files = [dict(path=bind(cpath, inputs), slot='C34', array=None),
                     dict(path=bind(r['path'], inputs), slot='R1', array=None)]
            snapshots[k] = dict(video=c['video'], sequence=seq, files=files)
            keys.append(k)
        eye = read(ROOT / 'runs/z13_local_peak_20261021/geometry.json')['eye_size']
        axes = np.array([(u, v) for u in range(-2, 3) for v in range(-2, 3) if max(abs(u), abs(v), abs(u+v)) <= 2]) @ np.array([[13., 6.5], [0., 13.]])
        rc, cc = np.array(c['reference_crop']), np.array(c['current_crop'])
        offsets = axes * np.maximum((rc[2:]-rc[:2])/eye, (cc[2:]-cc[:2])/eye)
        variants = dict(C34=dict(winner=proposal['winner'], channels=proposal['channels']), **comp['new_image_features'])
        screen.append(dict(**{k: c[k] for k in ['video', 'frame', 'old_id', 'raw_id', 'geometry', 'reference_box', 'reference_crop', 'current_box', 'current_crop']},
                           label=c['kind'], full_cells=proposal['full_cells'], offsets=offsets, keys=keys, old_variants=variants))
    assert len(train) == 26 and len(screen) == 5 and len({r['video'] for r in train}) == 20
    assert not ({r['video'] for r in train} & {r['video'] for r in screen})
    p = dict(registered=now().isoformat(), head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
             authorization='12.11: готовые карты, без подбора порогов; сначала 26 связей и пять рабочих случаев; при провале остановиться; полный трекер запрещён.',
             stages='Both named screens are one bounded admission package, frozen together before scores. No next scientific stage in this task.',
             sources=sources, inputs=inputs, snapshots=snapshots, train=train, screen=screen,
             arms=ARMS, transforms=KINDS, primary='C34__agreement',
             formula='2 population covariance / (population variance_a + population variance_b + squared mean difference); moments divide by m.',
             ablation='variance_only removes only squared paired mean difference; diagnostic, never post-hoc replacement of primary.',
             fixed='Saved old per-arm winning shift and full template, seven old geometry inputs, 34 visual inputs, 42 fitted parameters including intercept.',
             fit=dict(threshold=0, regularization=1, class_balance='n/(2*n_class)', folds='leave whole video out, 20 folds',
                      geometry='saved G7 fold normalizations; saved final normalization for 6 final fits',
                      signal='training-only channel means and common RMS standard deviation over nonconstant channels; constant scale=1',
                      cv_fits=120, final_fits=6, new_core_calls=0, new_maps=0, gradient_tolerance=1e-7),
             refusal='Fewer than three cells, fully constant reference/current or missing input: stop admission without imputation; no visual evidence. No runtime integration.',
             acceptance=dict(cv='C34 agreement own>=17, foreign<=1 with at least one strict improvement over ready common-scale baseline.',
                             screen='Preserve correct own70 and own1102; accept at least one of own232/own832 AND reject foreign574. All at threshold0.',
                             joint='Both primary gates and independent audit must pass. Otherwise stop this fixed version; no tuning, no tracker.',
                             component='No unique FlyVis advantage without positive paired-video interval versus both matched ordinary agreement controls.'),
             uncertainty=dict(draws=4000, seed=20261109, unit='video', method='paired percentile95; omit and count samples lacking a class', scope='descriptive development resampling, not independent generalization'),
             deadline=STOP.isoformat(), limitations='26 reused development links, only6 foreign; five preselected diagnostic cases. Not HOTA, not actual assignment/public-ID survival.')
    save('protocol.json', p)
    backup = OUT / 'before'
    backup.mkdir()
    for n in ['experiments/project_history.json', 'experiments/obsidian_experiment_topics.json', 'experiments/system_problems.json', 'dissertation/ЕДИНЫЙ_ПЛАН.md']:
        shutil.copy2(ROOT / n, backup / Path(n).name)
    save('worktree_before.json', dict(status=subprocess.check_output(['git', '-c', 'core.quotepath=false', 'status', '--short', '--untracked-files=no'], cwd=ROOT, text=True, encoding='utf8'), head=p['head']))
    card = dict(id=ID, title='Согласие значений карт — средние и амплитуды при прежнем сдвиге', area='flyvis', state='running', started=now().date().isoformat(),
                question='Помогают ли парные различия средних и амплитуд отличать своё продолжение от чужого?',
                hypothesis='Корреляция удаляет парные средние и амплитуды; их сохранение коэффициентом согласия может изменить неверное зрительное решение.',
                expected='Пройти оба заранее фиксированных допуска на готовых 26 связях и пяти рабочих случаях.',
                acceptance=[dict(criterion=v, met=None) for v in p['acceptance'].values()],
                evidence=[dict(kind='file', ref=(OUT/'protocol.json').relative_to(ROOT).as_posix(), claim='Условия, входы и критерии до расчёта')],
                tags=['fly-problem-P1'])
    def add(d):
        assert not any(e['id'] == ID for e in d['experiments'])
        d['experiments'].append(card)
    rewrite_json('experiments/project_history.json', add)
    rewrite_json('experiments/obsidian_experiment_topics.json', lambda d: d['topics'].update({ID: ['Сохранение личности', 'Выбор продолжения среди рамок', 'Согласие значений карт']}))
    journal('Согласие значений карт: постановка ограниченной проверки.\n\n*Постановка.* По прямому разрешению автора зарегистрирован `' + ID + '`: готовые 26 связей и пять рабочих случаев; полное согласие C34 — основной вариант, согласие амплитуд без разности средних — контроль. Порог 0, штраф 1, прежние сдвиги и состав данных фиксированы; новые карты и полный трекер исключены. Критерии и контрольные суммы до расчёта: `research_agreement_readout_20261112/protocol.json`. На 26 связях требуется не менее 17 своих, не более 1 чужой и строгое улучшение; на пяти случаях — сохранить 70 и 1102, принять 232 или 832 и отвергнуть чужую 574. Оба набора — единый разрешённый допуск, без продолжения после его проверки.')
    status('registered', pairs=26, cases=5)
    print('REGISTERED', len(snapshots), len(inputs))

if __name__ == '__main__':
    register()
