"""Audit saved full results and record their limits without rerunning science."""
from pathlib import Path
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.z13_center_hota_20261021 import RUN, LEGACY, read, write, sha, now, journal
from scripts.z13_record_center_hota_20261021 import revise
from scripts.z13_center_hota_resume_20261022 import require_unchanged_markers, verify_stitched_frames
from scripts.z13_hota_deadline_guard import WindowsProcesses


def main():
    import numpy as np
    from scripts.z13_eval import paired
    from datasets.subset_filter import filter_videos_by_subset, load_video_attributes
    receipt = RUN/'result_review_20261022.json'
    if receipt.exists():
        raise RuntimeError('Review already recorded; preserve it')
    result = read(RUN/'resume1_controller_result.json')
    assert result['state'] == 'completed' and result['completed_jobs'] == result['total_jobs'] == 176
    assert result['all_own_workers_ended'] and not result['remaining_owned_processes'] and not result['remaining']
    processes = WindowsProcesses()
    assert not {pid: born for pid, born in result['known_owned_processes'].items() if processes.creation(int(pid)) == born}
    control = read(RUN/'resume1_controller.json')
    finalizer = read(RUN/'resume1_finalize_launch.json')
    assert processes.creation(control['pid']) != finalizer['controller']['birth']
    assert read(RUN/'resume1_finalize_audit.json')['checks_passed']
    assert read(RUN/'resume1_finalize_result.json')['state'] == 'records_ready_for_agent_review'
    protocol = read(RUN/'protocol.json')
    identity = read(RUN/'execution_identity.json')
    assert sha(RUN/'protocol.json') == identity['protocol_sha256']
    assert sha(RUN/'technical.json') == identity['technical_sha256']
    assert sha(RUN/'masa_head.joblib') == identity['masa_head_sha256']
    auth = read(RUN/'resume1_authorization.json')
    assert sha(RUN/'resume1_authorization.json') == read(RUN/'resume1_preflight.json')['authorization_sha256']
    frozen = {**protocol['sha256'], **auth['sha256'], **auth['saved_sha256']}
    for name, digest in frozen.items():
        assert sha(ROOT/name) == digest, name
    markers = [read(p) for p in (RUN/'completed_jobs').glob('*.json')]
    assert len(markers) == 176 and not require_unchanged_markers(read(RUN/'jobs.json'), markers, identity)
    report = read(RUN/'eval/report.json')
    old = read(LEGACY/'eval/report.json')
    videos = protocol['videos']
    assert report['videos'] == len(videos) == 35
    for current, previous in [('A_masa', 'A_masa'), ('legacyC', 'C_masa'), ('legacyD', 'D_masa')]:
        assert report['pooled'][current+'|all']['HOTA'] == old['pooled'][previous+'|all']['HOTA']
    pairs = [('D_masa','A_masa'), ('D_masa','R1_masa'), ('D_masa','legacyD'), ('D_masa','C_masa'), ('C_masa','A_masa'), ('C_masa','R1_masa'), ('C_masa','legacyC')]
    night = list(filter_videos_by_subset(load_video_attributes(str(ROOT/'data/bdd100k/labels/bdd100k_labels_images_train.json')), videos, 'night'))
    rng = np.random.default_rng(1)
    for subset, vv in [('all', videos), ('night', night)]:
        for left, right in pairs:
            calculated = paired(report['per_video'], left, right, vv, rng)
            saved = report['diff'][left+'-'+right+'|'+subset]
            assert all(saved[k] == v for k, v in calculated.items())
            assert saved['pooled_gain'] == report['pooled'][left+'|'+subset]['HOTA']-report['pooled'][right+'|'+subset]['HOTA']
    events, equality, sources = {}, {}, {}
    for arm in protocol['arms']:
        total = {}
        for video in videos:
            directory = RUN/arm/'signatures'/video
            meta = read(directory/'signatures.json')
            with np.load(directory/'signatures.npz') as actual, np.load(ROOT/'runs/fly_signatures'/video/'signatures.npz') as baseline:
                assert int(actual['frame_count']) == meta['frames'] == int(baseline['frame_count'])
                assert np.isfinite(actual['outputs']).all()
                n = len(actual['track_id'])
            with np.load(RUN/arm/'masa'/(video+'.npz')) as embeddings:
                assert embeddings['first'].shape == embeddings['last'].shape == (n, 256)
                assert np.isfinite(embeddings['first']).all() and np.isfinite(embeddings['last']).all()
            path = RUN/'eval'/(arm+'_masa')/(video+'.json')
            frames = read(path)
            verify_stitched_frames(frames, read(ROOT/'runs/train_cache'/video/'alignment.json')['matches'])
            sources[path.relative_to(ROOT).as_posix()] = sha(path)
            for key, value in meta['z13_events'].items():
                total[key] = total.get(key, 0)+int(value)
        assert total['nonfinite_refusals'] == 0
        events[arm] = total
        d = [report['per_video'][arm+'_masa'][v]['HOTA']-report['per_video']['A_masa'][v]['HOTA'] for v in videos]
        equality[arm] = dict(better=sum(x>0 for x in d), equal=sum(x==0 for x in d), worse=sum(x<0 for x in d))
    digest = sha(RUN/'eval/report.json')
    gain = report['diff']['D_masa-A_masa|all']['pooled_gain']
    base = report['pooled']['A_masa|all']['HOTA']
    audit = dict(verified=now().isoformat(), report_sha256=digest, scientific_frozen_files_verified=len(frozen), complete_markers=176,
                 full_videos=35, all_owned_processes_ended=True, baseline_reproduced_exactly=True, paired_intervals_recomputed=14,
                 all_signature_and_embedding_arrays_finite=True, stitched_source_sha256=sources, events=events, video_hota_vs_stand=equality,
                 relative_hota_percent=100*gain/base, ceiling_fraction=None,
                 interpretation='D meets the R1 comparison, but fails the predeclared practical +0.10 HOTA gate and superiority to legacyD. No D superiority to C; no night subgroup rescue. No new scientific calculations or threshold changes.')
    write(receipt, audit)
    catalog = read(ROOT/'experiments/project_history.json')
    card = next(c for c in catalog['experiments'] if c['id']=='z13_center_hota_v1')
    assert card['state']=='inconclusive' and [a['met'] for a in card['acceptance']]==[True,False,True,False]
    head = subprocess.check_output(['git','--no-optional-locks','rev-parse','HEAD'], cwd=ROOT, text=True).strip()
    common = 'runs/z13_center_hota_20261021/eval/report.json'
    text = ('Все35 dev-видео,176 заданий завершены03:44 МСК22.10; старые HOTA воспроизведены точно. '+common+
            ' → pooled/*|all/HOTA: стенд61,65477894; прежняяD61,67269039; новаяD61,65836372; новаяC61,65416664; R1 61,46076348. '
            'diff/D_masa-A_masa|all: общий сдвиг+0,00358479 пункта; средняя разность по видео+0,00575549,95%ДИ[+0,00011446;+0,01492941]. '
            'diff/D_masa-R1_masa|all: средняя+0,16417953[+0,05053902;+0,33450360], общий+0,19760024. '
            'diff/D_masa-legacyD|all: средняя−0,00292435[−0,04896864;+0,03521940], общий−0,01432667. '
            'diff/D_masa-C_masa|all: +0,00652475[−0,00232496;+0,01805046]. '
            'C−R1: +0,15765479[+0,04031664;+0,32217769]; C−стенд и все7ночных интервалов включают0. '
            'Все исходы/разрезы сохранены в diff; 14 интервалов пересчитаны из первичных per_video без повторного TrackEval. '
            'Проверки полноты/хешей/событий и конца дерева: runs/z13_center_hota_20261021/result_review_20261022.json.')
    interpretation = ('Есть малый положительный сдвиг D относительно стенда и преимущество C/D над применением R1. '
                      'Практический критерий+0,10 HOTA не пройден, превосходство новой D над прежней D и штатной C не подтверждено. '
                      'Большая часть разрыва с R1 связана с ухудшением R1 относительно стенда; это не крупный выигрыш мухи над исходной системой. Общий итог неопределённый, политика не принята.')
    for field, value in [('summary','Полное сравнение HOTA завершено: 1 из3 научных критериев выполнен; политика не принята.'), ('obtained',text), ('interpretation',interpretation),
                         ('decision','Сохранить функционально полезный центрированный считыватель, но не заменять им прежнюю D в стенде при этой политике1%. Не подбирать порог по dev/HOTA. Перед следующим изменением проверить на сохранённых данных, где поправки перестают менять итоговые траектории; причинный узел пока не установлен.')]:
        revise(card, field, value, head, 'Проверка первичных полного результата и практического смысла без изменения критериев.')
    card['confirmed_results'] = []
    for rid, title, key, comparison in [
        ('D_center_hota_vs_R1_dev35','Новая D лучше применения входа глаза R1 по HOTA','D_masa-R1_masa|all','Новая D против R1 с той же политикой1%'),
        ('C_center_hota_vs_R1_dev35','Штатная C лучше применения входа глаза R1 по HOTA','C_masa-R1_masa|all','Штатная C против R1 с той же политикой1%'),
        ('D_center_hota_vs_stand_dev35','Микроэффект D относительно стенда; практический критерий не пройден','D_masa-A_masa|all','Новая D против стенда без поправки')]:
        assert report['diff'][key]['ci95'][0]>0
        card['confirmed_results'].append(dict(id=rid,title=title,category='tracker',metric='HOTA',unit='points',comparison=comparison,
            subset='Полные35 ранее использованных dev-видео BDD100K MOT',
            conditions='Один фиксированный train26-порог верхнего1% для каждого варианта; одинаковые детектор,ByteTrack,MASA; замороженные модели; парный повидео бутстрэп5000,сид1.',
            limitations='Общий итог inconclusive: не пройдены+0,10 HOTA и превосходство над прежнейD. Интервал относится к средней разности по видео, не к общей HOTA. R1 ухудшает стенд; разрыв с R1 не равен пользе относительно отсутствия поправки. D−C не подтверждён, night не подтверждён; dev35 не независим, вклад схемы связей не выделен.',
            evidence=dict(source=common,sha256=digest,locator=['diff',key])))
    card['metrics'] = [dict(key='HOTA',value=report['pooled'][a+'|all']['HOTA'],baseline=base,subset='dev35; '+a,source=common,locator='pooled.'+a+'|all.HOTA') for a in ('A_masa','legacyD','C_masa','D_masa','R1_masa')]
    card['commits'] = list(dict.fromkeys(card.get('commits',[])+[head]))
    card['evidence'].append(dict(kind='file',ref=receipt.relative_to(ROOT).as_posix(),claim='Проверка всех сохранённых вариантов,14парных ДИ,неизменных источников и конца дерева'))
    write(ROOT/'experiments/project_history.json',catalog)
    journal('проверен полный результат HOTA и границы вывода', '*Получено.* '+text+'\n\n*Вывод.* '+interpretation+
            '\n\n*Решение.* '+card['decision']+' Три положительных частных сравнения добавлены в confirmed_results со ссылками/хешем; непройденные критерии сохранены. P8 остаётся частичной. На схеме обновляется только статус завершённой ветви применения. Формулировки проверены: объект — FlyVis, HOTA — инструмент проверки; новых причинных выводов и повторов прежнего результата нет.\n\nОтносительно стенда: '+f'{100*gain/base:+.5f}%'.replace('.',',')+' HOTA; доля измеренного потолка не определена — сопоставимый потолок этой политики не измерялся.')
    print('Verified full result, immutable sources,35 videos and14 intervals; records updated.')
    print(events, equality)


def publication_scope():
    from scripts.z13_build_vault_preserving_edits import snapshot_local_edits
    initial = read(RUN/'resume1_vault_finish.json')['protected_existing_edits']
    foreign = {(ROOT/p).resolve() for p in initial}
    own_graph = [p.relative_to(ROOT).as_posix() for p in snapshot_local_edits(ROOT) if p not in foreign]
    paths = ['experiments/project_history.json','dissertation/ЕДИНЫЙ_ПЛАН.md',
             'scripts/z13_finalize_center_hota_result_20261022.py','scripts/build_flyvis_architecture.py',
             'scripts/flyvis_architecture_layout.py','scripts/obsidian_fly_problems.py','scripts/obsidian_confirmed_results.py']
    paths += own_graph
    paths += [p.relative_to(ROOT).as_posix() for p in (ROOT/'dissertation/architecture').iterdir()
              if p.name in ['FlyVis_architecture.drawio','FlyVis_architecture.svg','FlyVis_architecture.png',
                            'FlyVis_architecture_preview.png','FlyVis_architecture_detail.png','FlyVis_architecture_sources.json',
                            'FlyVis_architecture_e3b0d1b.drawio','validation.json']]
    names = ['resume1_controller_result.json','resume1_queue_status.json','resume1_night_finished.json',
             'resume1_finish_recorded.json','resume1_finalize_audit.json','resume1_finalize_result.json',
             'resume1_finalize_launch.json','resume1_owned_refresh.json','resume1_vault_finish.json',
             'result_review_20261022.json','mechanism_review_20261022.json','review_vault.json','review_vault_final.json','review_vault_mechanism.json',
             'review_owned_refresh.json','review_owned_refresh_final.json','review_owned_refresh_mechanism.json','eval/report.json']
    paths += [(RUN/n).relative_to(ROOT).as_posix() for n in names if (RUN/n).is_file()]
    paths += [p.relative_to(ROOT).as_posix() for p in (RUN/'completed_jobs').glob('*.json')]
    paths += [p.relative_to(ROOT).as_posix() for arm in ('C','D','R1') for p in (RUN/arm/'signatures').glob('*/signatures.json')]
    paths += [p.relative_to(ROOT).as_posix() for p in (RUN/'masa_details').glob('*.json')]
    paths = sorted(set(paths))
    for rel in paths:
        p = (ROOT/rel).resolve(); p.relative_to(ROOT.resolve())
        assert p not in foreign and rel != '.gitignore'
        assert p.suffix not in ('.npz','.pt','.log','.joblib')
    record = dict(created=now().isoformat(), paths=paths,
                  sha256={n:sha(ROOT/n) for n in paths if (ROOT/n).is_file()},
                  deleted=[n for n in paths if not (ROOT/n).exists()],
                  excluded_foreign_graph_count=len(foreign),gitignore_deferred=True)
    write(RUN/'publication_scope_20261022.json',record)
    print('Exact own publication scope:',len(paths),'paths;',len(foreign),'foreign notes excluded')


def clarify_mechanism():
    marker = RUN/'mechanism_review_20261022.json'
    if marker.exists():
        raise RuntimeError('Mechanism review already recorded')
    report = read(RUN/'eval/report.json')
    audit = read(RUN/'result_review_20261022.json')
    p = ROOT/'experiments/project_history.json'
    catalog = read(p)
    card = next(c for c in catalog['experiments'] if c['id']=='z13_center_hota_v1')
    head = subprocess.check_output(['git','--no-optional-locks','rev-parse','HEAD'], cwd=ROOT, text=True).strip()
    caveat = ('Реальные вмешательства различались: C1171,D1162,R1 1447 по result_review_20261022.json/events; '
              'одинаковый процент отбора на train не гарантирует одинаковую частоту на dev. '
              'Сравнение полных политик не изолирует представление при равном числе вмешательств. '
              'Причина отдельных ухудшений требует разбора первого расходящегося сопоставления, она не установлена агрегатной HOTA.')
    revise(card,'cannot_claim',card['cannot_claim']+' '+caveat,head,'Уточнение границы сравнения по фактической частоте вмешательств.')
    for r in card['confirmed_results']:
        r['limitations'] += ' '+caveat
    record = dict(at=now().isoformat(),report_sha256=sha(RUN/'eval/report.json'),
                  audit_sha256=sha(RUN/'result_review_20261022.json'),events=audit['events'],
                  pooled_R1_minus_stand={k:report['pooled']['R1_masa|all'][k]-report['pooled']['A_masa|all'][k] for k in ('HOTA','AssA','DetA','IDSw')},
                  code_sha256={n:sha(ROOT/n) for n in ['tracking/z13_ridge_localizer.py','scripts/build_fly_signatures.py']},
                  confirmed='Finite values; confidence is normalized map peak, not calibrated utility; correction changes lost-track center before association; R1 association metrics worsen.',
                  unproven=['Wrong confident peak causes a specific mistaken association','Greater runtime intervention frequency accounts for part of the R1 loss','Changed crops create accumulated feedback error'],
                  next_evidence='First diverging association in a worsened video, with center before/after, correct and assigned detections, and subsequent identity. No new training or dev threshold selection.')
    write(marker,record)
    card['evidence'].append(dict(kind='file',ref=marker.relative_to(ROOT).as_posix(),claim='Измеренная частота вмешательств и отделение механизма кода от причинной гипотезы'))
    write(p,catalog)
    journal('поправка оговорки и ограничение сравнения политик',
            '*Исправление собственной записи.* В предыдущей свежей записи появилась ошибочная фраза «Все34контрольных значения не выдумывались». Число34 не имеет источника и не используется ни в одном выводе. Корректные факты:14парных интервалов и5абсолютных HOTA проверены отдельно; у C/D/R1 нулевой счётчик nonfinite_refusals. Старый текст сохранён для прозрачности.\n\n'
            '*Проверено.* '+caveat+' Источники и код закреплены в runs/z13_center_hota_20261021/mechanism_review_20261022.json. '
            'Неверный уверенный пик, повышенная частота вмешательств R1 и накопление ошибки через вырезки — рабочие версии, не установленная причина. В карточке/витрине добавлено это ограничение. Порог и научные результаты неизменны.')
    print('Mechanism limits and measured intervention counts recorded')


if __name__ == '__main__':
    if '--publication-scope' in sys.argv:
        publication_scope()
    elif '--clarify-mechanism' in sys.argv:
        clarify_mechanism()
    else:
        main()
