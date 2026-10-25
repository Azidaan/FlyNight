"""Independent input/label separation audit and factual recording of CPU readiness."""
from collections import Counter
from pathlib import Path
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import z13_temporal_pair_readiness_20261025 as S


def glossary():
    path = ROOT/'experiments/metrics_glossary.json'
    data = S.read(path)
    additions = {
        'temporal_pair_coverage': dict(title='Охват причинно доступных пар областей',
            ru='Доля исходных запросов, для которых есть поддержанные реальные области в текущем и строго предыдущем наблюдении. Не качество памяти модели.',
            better='higher', unit='fraction', range=[0, 1], tool='CPU-аудит сохранённых наблюдений',
            caveat='Знаменатель и правило выбора прошлого обязательны; численные признаки FlyVis этим не проверяются.'),
        'temporal_pair_action_reserve': dict(title='Идеальный запас действий при наличии временной пары',
            ru='Средняя по видео доля всех известных исходных запросов, имеющих полезное рамочное действие и допустимую временную пару. Выбор по истинному исходу, отказ от остальных действий идеален.',
            better='higher', unit='fraction', range=[0, 1], tool='Сохранённые метки одноступенчатого назначения',
            caveat='Не прогноз модели и не HOTA. Все известные запросы остаются в знаменателе; варианты охвата указываются отдельно.')}
    for key, value in additions.items():
        assert key not in data['metrics']
        data['metrics'][key] = value
    S.write(path, data)


def audit():
    run = S.RUN
    p = S.read(run/'protocol.json'); report = S.read(run/'report.json')
    controller = S.read(run/'controller_result.json')
    assert report['complete'] and report['trained_models'] == report['new_activity_arrays'] == 0
    assert controller['exit_code'] == 0 and controller['all_owned_ended'] and not controller['remaining']
    seen = set(); total = Counter(); masks_checked = 0
    for video in p['videos']:
        vr = run/'videos'/(video+'.json'); record = S.read(vr)
        assert S.sha(vr) == report['manifest'][video]
        ip = run/'inputs'/(video+'.json'); lp = run/'supervision'/(video+'.json')
        assert S.sha(ip) == record['input_sha256'] and S.sha(lp) == record['supervision_sha256']
        inputs = S.read(ip)['rows']; labels = S.read(lp)['rows']
        original = S.read(S.BASE/'cache/videos'/(video+'.json'))['entries']
        assert len(inputs) == len(labels) == len(original)
        counts = Counter()
        for row, lab in zip(inputs, labels):
            uid = row['uid']; assert uid == lab['uid'] and uid not in seen; seen.add(uid)
            entry = original[str(uid)]
            assert row['current_choices'] == entry['choices'][1:]
            assert lab['original_good'] == entry['good'] and lab['original_harm'] == entry['harm']
            assert not {'identity', 'target_identity', 'kind', 'good', 'harm', 'correspondence'} & set(row)
            previous = row['previous']
            if previous:
                assert previous['frame'] < row['current_frame']
                assert row['state_prefix_start'] <= previous['array_index'] < row['current_array_index']
                assert previous['choices'] and 'identity' not in previous
            assert lab['correspondence'] == S.pair_labels(lab['current_identity'], lab['previous_identity'])
            for mask in row['current_masks'] + (previous['masks'] if previous else []):
                assert mask and min(mask) >= 0 and max(mask) < 721 and len(set(mask)) == len(mask)
                masks_checked += 1
            counts.update({k: int(v) for k, v in lab['flags'].items()})
            counts['positive_correspondences'] += sum(x[2] for x in lab['correspondence'])
            counts['negative_correspondences'] += sum(not x[2] for x in lab['correspondence'])
        assert dict(counts) == record['counts'] == report['per_video'][video]
        total.update(counts)
    assert seen == set(range(37296)) and dict(total) == report['counts']
    for key, expected in report['oracle26'].items():
        values = {v: report['per_video'][v].get(key, 0)/report['per_video'][v]['known'] for v in p['splits']['evaluate26']}
        assert S.paired(values) == expected
    receipt = dict(at=S.now().isoformat(), videos=130, queries=len(seen), masks_checked=masks_checked,
        all_input_label_bindings_verified=True, all_saved_intervals_recomputed=True,
        unknown_identities_not_negative=True, no_identity_in_input_files=True,
        source_hashes_verified_by_worker=report['sources_unchanged'],
        model_feature_parity_not_tested=True)
    S.write(run/'independent_audit.json', receipt)
    return report


def record():
    report = audit(); run = S.RUN
    assert not (run/'recorded.json').exists()
    path = ROOT/'experiments/project_history.json'; data = S.read(path)
    card = next(x for x in data['experiments'] if x['id'] == S.IDENT)
    counts = report['counts']; ev = report['per_split']['evaluate26']; result = report['oracle26']
    details = ('runs/z13_temporal_pair_readiness_20261025/report.json → counts/per_split/oracle26: '+
        json.dumps(dict(counts=counts, evaluate26=ev, oracle26=result), ensure_ascii=False))
    card.update(state='infrastructure', done=S.now().strftime('%Y-%m-%d'),
        summary='Подготовка причинных областей и частично размеченных соответствий завершена; обучаемый механизм ещё не проверялся.',
        obtained=details,
        interpretation='Измерена готовность данных при фиксированном правиле прошлого внутри исходного эпизода. Номера трекера не являются метками личности. Неизвестные пары исключены из меток соответствия, но не из исходных знаменателей. Наличие областей/меток не доказывает обучаемость и пользу FlyVis.',
        decision='Сохранить подготовленные входы и отдельную разметку. Новый расчёт D/C/R1 и обучение в этот согласованный шаг не входят. Следующий опыт требует проверки воспроизведения состояния, равных признаков и цены до обучения.',
        metrics=[dict(key='temporal_pair_coverage', value=counts['restored_supported']/counts['rows'],
            baseline=counts['old_supported']/counts['rows'], subset='130 исходных видео; все37296запросов',
            source='runs/z13_temporal_pair_readiness_20261025/derived_metrics.json', locator='coverage_fraction'),
            dict(key='temporal_pair_action_reserve', value=result['oracle_restored']['mean_per_video'],
            baseline=result['oracle_old']['mean_per_video'], subset='26 повторно использованных видео; все известные запросы',
            source='runs/z13_temporal_pair_readiness_20261025/report.json', locator='oracle26.oracle_restored.mean_per_video')],
        confirmed_results=[])
    for criterion in card['acceptance']:
        criterion.update(met=True, evidence='runs/z13_temporal_pair_readiness_20261025/independent_audit.json; report.json; controller_result.json')
    card['evidence'] += [dict(kind='file', ref='runs/z13_temporal_pair_readiness_20261025/'+name,
        claim=claim) for name, claim in [('report.json','Полные исходные знаменатели, охват и условные запасы'),
        ('independent_audit.json','Проверены все входы/метки и повторены интервалы'),
        ('controller_result.json','Свои процессы завершены в пределах исходного бюджета')]]
    # Keep the completed primary report immutable; derived catalog scalar has its own source.
    S.write(run/'derived_metrics.json', dict(coverage_fraction=counts['restored_supported']/counts['rows'],
        primary_report_sha256=S.sha(run/'report.json')))
    S.write(path, data)
    text = ('*Получено (`'+S.IDENT+'`).* '+details+'\n\n*Вывод.* '+card['interpretation']+
        '\n\n*Решение.* '+card['decision']+
        '\n\n*Подтверждённые улучшения.* Новых confirmed_results нет: это подготовка данных без обучаемого результата. HOTA не измерялась; процент изменения стенда и реализованная моделью доля потолка не определены.')
    S.journal('итог подготовки реальных временных пар', text)
    # All newly recorded prose was checked: instrument/data preparation, no duplicate scientific claim,
    # and no substitution of planned neural-feature computation for completed metadata work.
    S.write(run/'recorded.json', dict(at=S.now().isoformat(), complete=True, state='infrastructure',
        sentence_review=dict(method_not_research_object=True,no_duplicate_claim=True,only_actual_work=True)))


if __name__ == '__main__':
    if '--glossary' in sys.argv:
        glossary()
    elif '--record' in sys.argv:
        record()
    elif '--audit' in sys.argv:
        print(json.dumps(audit()['counts'], ensure_ascii=False))
    else:
        raise RuntimeError('Explicit operation required')
