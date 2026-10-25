"""Record a literature/code audit; never launch a scientific experiment."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'research_flyvis_deep_2026-10-25'
IDENT = 'z13_deep_identity_research_20261025'


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p, data):
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    assert not (OUT / 'provenance.json').exists()
    from scripts.z13_build_vault_preserving_edits import snapshot_local_edits
    protected = snapshot_local_edits(ROOT)
    write(OUT / 'preservation.json', {
        'graph': {p.relative_to(ROOT).as_posix(): h for p, h in protected.items()},
        'gitignore_sha256': sha(ROOT / '.gitignore'),
        'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip(),
    })
    paths = [
        'AGENTS.md',
        'tracking/z13_candidate_identity.py', 'tracking/z13_filter_descriptor.py',
        'tracking/z13_region_assignment.py', 'tracking/z13_region_assignment_context.py',
        'tracking/z13_readout.py',
        'runs/z13_refusal_causes_20261025/diagnosis.json',
        'runs/z13_proposal_geometry_20261025/recovery1/report.json',
        'runs/z13_box_proposal_20261025/report.json',
        'runs/z13_unmatched_box_20261025/report.json',
        'runs/z13_selected_box_value_20261025/report.json',
        'runs/z13_full_box_supervision_20261025/partial_recorded.json',
        'runs/z13_activity_amplitude_20261025/training/report.json',
        'research_flyvis_next_change_2026-10-23/СВОДКА.md',
        'research_flyvis_missing_proposals_2026-10-25/СВОДКА.md',
        'research_flyvis_missing_proposals_2026-10-25/ПРЕДВАРИТЕЛЬНАЯ_ПРОВЕРКА.md',
    ]
    for name in paths:
        assert (ROOT / name).is_file(), name
    g = read(ROOT / paths[7])
    assert g['incorrect_baseline_rows'] == 1001 and g['no_current_D_candidate_rows'] == 895
    assert sum(g['remaining895_detection_partition'].values()) == 895
    key = 'boxes_eye-centers_eye'
    assert key in g['paired'], list(g['paired'])
    assert abs(g['paired'][key]['mean_per_video'] * 100 - 2.5419) < .001
    s = read(ROOT / paths[10]); p = read(ROOT / paths[11])
    assert not s['primary_pass'] and s['paired']['Dvalue-baseline']['ci95'][0] == 0
    assert p['trained_models'] == 0 and p['scientific_verdict'] is None
    sources = [
        dict(id='keeptrack', url='https://arxiv.org/html/2103.16556v2', examined='Sections3-4; appendicesA-C; tables1,2,3,9,10,11', kind='full_text'),
        dict(id='keeptrack_matcher', url='https://raw.githubusercontent.com/visionml/pytracking/master/ltr/models/target_candidate_matching/target_candidate_matching.py', examined='DescriptorExtractor and matching network', kind='official_code_mutable_master'),
        dict(id='keeptrack_candidates', url='https://raw.githubusercontent.com/visionml/pytracking/master/pytracking/tracker/keep_track/candidates.py', examined='Candidate identities, histories, reassignment and reselection', kind='official_code_mutable_master'),
        dict(id='dmtrack', url='https://arxiv.org/html/2104.12041', examined='Sections3.1-3.5,4.2; table1', kind='full_text'),
        dict(id='dmtrack_code', url='https://github.com/hqucv/dmtrack', examined='README reported results and unfinished ReID release TODO', kind='official_repository'),
        dict(id='siammot', url='https://arxiv.org/html/2105.11595', examined='Sections3-5; tables1,2,7; negative/hard definition', kind='full_text'),
        dict(id='kys', url='https://arxiv.org/html/2003.11014', examined='Sections3,4.1; table1; state propagation and reliability', kind='full_text'),
        dict(id='kys_code', url='https://raw.githubusercontent.com/visionml/pytracking/master/ltr/models/kys/response_predictor.py', examined='Propagation, reliability and recurrent state update', kind='official_code_mutable_master'),
        dict(id='kys_training', url='https://raw.githubusercontent.com/visionml/pytracking/master/ltr/train_settings/kys/kys.py', examined='Lines94-130; entropy configuration and frozen backbone', kind='official_code_mutable_master'),
        dict(id='dam4sam', url='https://arxiv.org/html/2509.13864', examined='Sections3-4,5.7; memory construction, dataset provenance, failures', kind='full_text_extended_same_family_as_CVPR2025'),
        dict(id='tschopp2018', url='https://arxiv.org/html/1806.04793v2', examined='Sections3.1-4.1; selected8DAVIS static-camera sequences, within-video split', kind='full_text'),
        dict(id='flygym', url='https://raw.githubusercontent.com/NeLy-EPFL/flygym-gymnasium/main/flygym_gymnasium/examples/vision/README.md', examined='FlyVis extension and preset-path fly-following example', kind='official_code_documentation'),
        dict(id='wang2024', url='https://computing.hit.edu.cn/_upload/article/files/8c/97/619c2a594f77b400abac90272ca3/be57ea38-389e-4b57-a0f0-749d5561dbd4.pdf', examined='Publisher indexed first page and abstract only; full PDF retrieval timed out', kind='limited_primary_abstract', included_in_recommendation=False),
    ]
    stamp = datetime.now(timezone(timedelta(hours=3)))
    audit = dict(created=stamp.isoformat(), research_only=True, new_model_runs=0,
        not_hota=True, sha256={n: sha(ROOT / n) for n in paths}, public_sources=sources,
        verified_extracts={'geometry_partition': g['remaining895_detection_partition'],
            'oracle_box_minus_center': g['paired'][key], 'selected_value': s['paired']['Dvalue-baseline'],
            'partial_supervision': p},
        claim_review=dict(method_vs_object='FlyVis remains object; borrowed methods are internal readout candidates',
            duplicate_text='New report compares mechanisms with latest failures; previous reports preserved',
            measured_vs_planned='Only historical sources supply measured outcomes; all new mechanisms marked proposed',
            unresolved='No exact FlyVis counterpart found; Chinese full text inaccessible; temporal-cache feasibility not established'),
        report_sha256=sha(OUT / 'СВОДКА.md'))
    write(OUT / 'provenance.json', audit)
    catpath = ROOT / 'experiments/project_history.json'; cat = read(catpath)
    assert not any(c['id'] == IDENT for c in cat['experiments'])
    card = dict(id=IDENT, title='З13: глубокий поиск механизмов личности, предложения и отказа после причинного разбора',
        area='flyvis', state='inconclusive', started='2026-10-25', done='2026-10-25',
        tags=['fly-problem-P8'], question='Какой внутренний механизм ещё не воспроизведён и наиболее близок измеренным ошибкам FlyVis?',
        hypothesis='Обучаемое межкадровое соответствие может отличить цель от соседей лучше фиксированного сопоставления; это предложение, не проверенный результат.',
        expected='Проверить первоисточники, официальные реализации, собственные результаты и границы переноса; выбрать один следующий механизм.',
        acceptance=[], summary='Поиск завершён; эффективность предложенной модернизации не измерена. Это исследовательская запись, не новый эксперимент.',
        obtained='research_flyvis_deep_2026-10-25/СВОДКА.md и provenance.json: сопоставлены KeepTrack, DMTrack, SiamMOT, KYS, DAM4SAM, коннектомный предшественник и фактический пример FlyVis; китайская статья включена только как непроверенная по полному тексту опора. Исправлена общая оговорка об одиночных прогонах KeepTrack/KYS.',
        interpretation='Фиксированный перенос идентификаторов и текущая конкуренция областей не эквивалентны обучаемому межкадровому сопоставлению. Отрицательные упрощения не закрывают полный механизм. Полная рамочная супервизия остаётся не проверенной из-за цены.',
        decision='Рекомендован первый внутренний узел: обучаемое соответствие кандидатов с исходом нет пары. Подготовка пилота начинается с аудита последовательных кэшей и отдельного протокола; поиском не запущены новые расчёты и не продлены старые бюджеты.',
        cannot_claim='Нет нового прироста FlyVis/HOTA, процента к стенду, доказанной причины всех ошибок или готового решения один в один. Повторно используемый train26 не независимая проверка. Рекомендация не означает эффективность.',
        metrics=[], commits=[], files=['research_flyvis_deep_2026-10-25/СВОДКА.md'],
        evidence=[dict(kind='file',ref='research_flyvis_deep_2026-10-25/СВОДКА.md',claim='Ранжирование механизмов и проект различающей проверки'),
            dict(kind='json',ref='research_flyvis_deep_2026-10-25/provenance.json',claim='Хеши локальных источников, проверенные извлечения и область чтения первоисточников')])
    cat['experiments'].append(card)
    catpath.write_text(json.dumps(cat, ensure_ascii=False, indent=1) + '\n', encoding='utf8')
    planpath = ROOT / 'dissertation/ЕДИНЫЙ_ПЛАН.md'; plan = planpath.read_text(encoding='utf8')
    marker = '## Журнал изменений\n\n'; assert plan.count(marker) == 1
    entry = f'''**{stamp:%d.%m.%Y, %H:%M} MSK** — глубокий поиск после разбора предложений и отказа FlyVis

*Постановка.* Автор поручил провести поиск самостоятельно. Сверены первоисточники, официальный код и текущие завершённые опыты; запись `{IDENT}` относится к исследованию, не к новому вычислительному результату.

*Получено.* `research_flyvis_deep_2026-10-25/СВОДКА.md`, хеши и источники в `provenance.json`. Ближайший новый механизм — обучаемое межкадровое сопоставление KeepTrack; китайская DMTrack разделяет резерв предложений и узнавание личности; SiamMOT связывает видимость и рамку с назначением; KYS проверяет перенос пространственной памяти. Полного аналога нашей связки с FlyVis в проверенных источниках не найдено. Китайский PDF 2024 доступен только по индексируемой первой странице; готовое решение ему не приписано.

*Поправка.* Прежняя общая оговорка об одиночных прогонах не относится к ключевым сравнениям KeepTrack и KYS: там пять запусков. Наша фиксированная смесь сходства/расстояния не является полной проверкой обучаемого сопоставления KeepTrack. Простая память и её сброс не эквивалентны пространственному переносу KYS. Прежние тексты сохранены; поправки вынесены в раздел3 нового обзора.

*Вывод.* Отказ, соответствие личности и полезность поправки — разные цели. Эффективность новой реализации неизвестна. `runs/z13_full_box_supervision_20261025/partial_recorded.json` остаётся незавершённым по цене, без научного отрицательного вердикта.

*Решение.* В обзоре ранжированы механизмы и предложена проверка одного внутреннего узла с равными D/C/R1/геометрическим контролями. Бюджет и пригодность временных кэшей ещё не установлены; новые прогоны этим исследованием не запускались, старые окна не продлены. Новых HOTA, процента к стенду и измеренной доли потолка нет. Действующий план исполнения не переписан.

'''
    planpath.write_text(plan.replace(marker, marker + entry, 1), encoding='utf8')
    print('Recorded research only:', IDENT)


if __name__ == '__main__':
    main()
