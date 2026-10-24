"""Read-only independent aggregation of frozen assignment-effect rows; no replay."""
from pathlib import Path
import hashlib
import json
import sys
from datetime import datetime, timezone, timedelta
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RUN = ROOT / 'runs/z13_assignment_effect_20261024/recovery1'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def main():
    from scripts.z13_hota_deadline_guard import WindowsProcesses
    p, report = read(RUN/'protocol.json'), read(RUN/'report.json')
    end = read(RUN/'controller_result.json')
    assert end['exit_code'] == 0 and end['worker_ended'] and end['all_owned_ended'] and not end['remaining']
    ps = WindowsProcesses()
    assert all(ps.creation(int(pid)) != birth for pid, birth in end['owned'].items())
    for rel, expected in p['sha256'].items():
        assert digest(ROOT/rel) == expected, rel
    assert len(report['video_sha256']) == len(p['videos']) == 130
    rows = {}
    for video, expected in report['video_sha256'].items():
        path = RUN/'videos'/(video+'.json')
        assert digest(path) == expected, video
        job = read(path)
        assert job['checks']['raw_replay_exact'] and job['checks']['all_predictions_and_identities_bound']
        assert len(job['rows']) == job['checks']['query_rows']
        rows[video] = job['rows']
    aggregate = {}
    for arm in ('D', 'C', 'R1'):
        counts = dict.fromkeys(report['candidate_counts'][arm], 0)
        values = {}
        missed = 0
        known_queries = 0
        helpful_queries = 0
        for video in p['evaluate26']:
            known = [r for r in rows[video] if r['kind'] >= 0]
            helpful = sum(any(x['beneficial'] for x in r['arms'][arm]) for r in known)
            values[video] = helpful / len(known)
            known_queries += len(known)
            helpful_queries += helpful
            counts['known'] += len(known)
            for row in known:
                for x in row['arms'][arm]:
                    counts['candidates'] += 1
                    counts['beneficial'] += bool(x['beneficial'])
                    counts['harmful'] += bool(x['harmful'])
                    if x['old_center_useful']:
                        counts['center_useful_candidates'] += 1
                        suffix = 'beneficial' if x['beneficial'] else 'harmful' if x['harmful'] else 'neutral'
                        counts['center_useful_'+suffix] += 1
                    elif x['beneficial']:
                        missed += 1
        assert counts == report['candidate_counts'][arm], arm
        assert values == report['per_video'][arm], arm
        a = np.array([values[v] for v in sorted(values)])
        rng = np.random.default_rng(1)
        boot = np.mean(a[rng.integers(0, len(a), size=(5000, len(a)))], axis=1)
        result = dict(mean_per_video=float(a.mean()), ci95=[float(np.quantile(boot, .025)), float(np.quantile(boot, .975))], better=int(sum(a > 0)), n=len(a))
        assert result == report['oracle_beneficial_fraction'][arm], arm
        aggregate[arm] = dict(counts=counts, independent_interval=result,
            beneficial_candidates_not_old_center_useful=missed,
            known_queries=known_queries, queries_with_beneficial_candidate=helpful_queries)
    receipt = dict(at=datetime.now(timezone(timedelta(hours=3))).isoformat(),
        report_sha256=digest(RUN/'report.json'), source_hashes_verified=len(p['sha256']),
        video_hashes_verified=len(rows), all_owned_ended=True, independently_recomputed=True,
        aggregate=aggregate, not_hota=True,
        caveat='Candidate counts are correlated alternatives, not deployed interventions. One-step results omit feedback and later association stages; this audit does not change the prespecified gate.')
    path = RUN/'independent_audit_20261025.json'
    assert not path.exists()
    path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(receipt, ensure_ascii=False))


def decision_audit():
    """Describe saved outcomes only; no new scientific threshold or replay."""
    from collections import Counter
    audit = read(RUN/'independent_audit_20261025.json')
    report = read(RUN/'report.json')
    assert digest(RUN/'report.json') == audit['report_sha256']
    p = read(RUN/'protocol.json')
    table = {}
    for arm in ('D', 'C', 'R1'):
        queries = Counter()
        useful = Counter()
        for video in p['evaluate26']:
            path = RUN/'videos'/(video+'.json')
            assert digest(path) == report['video_sha256'][video]
            for r in read(path)['rows']:
                if r['kind'] < 0:
                    continue
                choices = r['arms'][arm]
                before = choices[0]['own_before']
                assert all(c['own_before'] == before for c in choices)
                queries['known'] += 1
                queries['baseline_own_correct' if before else 'baseline_own_incorrect'] += 1
                if any(c['beneficial'] for c in choices):
                    queries['has_beneficial'] += 1
                for c in choices:
                    if not c['old_center_useful']:
                        continue
                    key = ('correct' if c['own_before'] else 'incorrect')+'->'+('correct' if c['own_after'] else 'incorrect')
                    key += '; net_correct_delta='+str(c['correct_count_delta'])
                    useful[key] += 1
        table[arm] = dict(query_counts=dict(queries), old_useful_outcomes=dict(sorted(useful.items())))
    result = dict(at=datetime.now(timezone(timedelta(hours=3))).isoformat(),
        source_report_sha256=audit['report_sha256'], descriptive_only=True, arms=table,
        caveat='Correctness is the saved first-association identity criterion; neutral does not mean identical assignment. No trajectory feedback, no new HOTA and no acceptance-rule change.')
    path = RUN/'decision_audit_20261025.json'
    assert not path.exists()
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))


def record_audits():
    from scripts.z13_assignment_effect_20261024 import journal
    audit = read(RUN/'independent_audit_20261025.json')
    decision = read(RUN/'decision_audit_20261025.json')
    assert audit['report_sha256'] == decision['source_report_sha256'] == digest(RUN/'report.json')
    journal('независимая проверка последствий и разрыв с метрикой центра',
        'runs/z13_assignment_effect_20261024/recovery1/independent_audit_20261025.json: '
        'заново проверены4396хешей источников,130JSONвидео,все26повидео долей и5000бутстрэп повторов дляC/D/R1;'
        'счётчики и интервалы точно совпали. Собственное дерево завершено. '
        'Из846кандидатовD с прежней меткой полезного центра37полезны сопоставлению,17вредны,792нейтральны '
        'по однокадровому критерию правильности назначений;нейтральность не означает идентичность назначений. '
        'Ещё128из165полезных для сопоставления кандидатов не имели прежней метки полезного центра. '
        'Подробные переходы правильности сохранены вdecision_audit_20261025.json того же каталога. '
        'Это описательный разбор прежних строк,не новый эксперимент,не причинное объяснение всейHOTA. '
        'Критерий перехода2%неизменен:1,633329%[1,176393;2,138276],итогinconclusive. '
        'Обучение этой версии не запускается. Исходный полный итог опубликованe2ae53f. '
        'Следующий выбор узла должен учитывать ошибочные назначения,которые не исправляет ни один кандидат, '
        'а не только расстояние до центра;принятой новой реализации пока нет. '
        'ПриростHOTA и доля измеренного потолка для данного разбора не определены.')
    handoff = ROOT/'runs/z13_weekend_20261024/HANDOFF.md'
    text = ('# Актуально — '+datetime.now(timezone(timedelta(hours=3))).strftime('%d.%m.%Y %H:%M')+' MSK\n\n'
        'assignment_effect/recovery1 завершён и опубликованe2ae53f. Независимыеaudit_20261025/decision_audit_20261025 '
        'проверили исходники/130видео/26повидеоДИ и разобрали последствия. Текущих расчётов этой стадии нет. '
        'Критерий перехода не пройден,обучение её версий запрещено по протоколу. '
        'D:846старыхполезныхцентров→37полезных/17вредных/792нейтральных последствий;128из165 '
        'полезных последствий не имели старой положительнойметки. Это диагностическийразрыв,неHOTA. '
        'Новый узел ещё не выбран;при продолжении сначала использоватьdecision_audit_20261025 '
        'для различения отсутствия подходящего кандидата и неверной оценки уже имеющихся. '
        'Нельзя возобновлять ошибочнуюunperturbedпостановку или объявлять текущий процесс работающим. '
        'Общая авторизация до26.10 07:40сохраняется;меньшие бюджеты завершены.\n\nНиже прежние передачи сохранены.\n\n')
    handoff.write_text(text+handoff.read_text(encoding='utf-8'), encoding='utf-8')


if __name__ == '__main__':
    if '--decision-audit' in sys.argv:
        decision_audit()
    elif '--record' in sys.argv:
        record_audits()
    else:
        main()
