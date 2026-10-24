"""Independent bookkeeping of the activation-boundary ceiling; never reruns it."""
from pathlib import Path
from collections import Counter
import json
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.z13_activation_timing_20261025 import RUN, IDENT, W, read, write, sha, now, journal


def main():
    from scripts.z13_hota_deadline_guard import WindowsProcesses
    from scripts.z13_localization_labels_20261023 import verify_sources
    from scripts.z13_continue import paired
    assert not (RUN / 'finish_recorded.json').exists()
    p, e = read(RUN / 'protocol.json'), read(RUN / 'controller_result.json')
    assert e['worker_ended'] and e['all_owned_ended'] and not e['remaining']
    ps = WindowsProcesses()
    assert all(ps.creation(int(pid)) != birth for pid, birth in e['owned'].items())
    verify_sources(p)
    jobs = [read(path) for path in sorted((RUN / 'videos').glob('*.json'))]
    head = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT).decode().strip()
    source = 'runs/z13_activation_timing_20261025/report.json'
    complete = e['exit_code'] == 0 and len(jobs) == 26 and (RUN / 'report.json').exists()
    if complete:
        r = read(RUN / 'report.json')
        assert r['complete'] and set(j['video'] for j in jobs) == set(p['videos'])
        totals, rates = Counter(), {}
        for job in jobs:
            assert job['exact_baseline'] and sha(ROOT / f"runs/fly_signatures/{job['video']}/signatures.npz") == job['baseline_sha256']
            c = Counter(job['counts'])
            independent = Counter()
            for row in job['rows']:
                state = row['incoming_state']
                independent[state + '_known'] += 1
                independent[state + ('_correct' if row['before_correct'] else '_incorrect')] += 1
                independent[state + ('_high_present' if row['high_present'] else '_high_absent')] += 1
                if 'oracle_center' in row:
                    assert not row['before_correct'] and row['high_present']
                    independent[state + '_oracle_evaluated'] += 1
                    independent[state + '_oracle_beneficial'] += row['beneficial']
                    independent[state + '_oracle_harmful'] += row['harmful']
                    independent[state + '_infield_beneficial'] += row['beneficial'] and row['in_field']
                if state == 'tracked' and not row['before_correct']:
                    independent['tracked_first_incorrect_final_' + row['actual_final_state']] += 1
            for key, value in independent.items():
                assert c[key] == value, (job['video'], key)
            for state in ('tracked', 'lost'):
                assert c[state+'_known'] == independent[state+'_known']
                assert c[state+'_known'] + c[state+'_unknown'] == c[state+'_pool']
            totals.update(c)
            n = c['tracked_known'] + c['lost_known']
            rates[job['video']] = dict(known=n, tracked_only=c['tracked_infield_beneficial']/n,
                lost_only=c['lost_infield_beneficial']/n,
                all_active=(c['tracked_infield_beneficial']+c['lost_infield_beneficial'])/n)
        assert dict(totals) == r['counts'] and rates == r['per_video']
        interval = paired({v:x['all_active'] for v,x in rates.items()}, {v:x['lost_only'] for v,x in rates.items()})
        assert interval == r['paired_added_tracked_center_oracle']
        passed = interval['mean_per_video'] >= .02 and interval['ci95'][0] > 0
        assert r['primary_pass'] == passed
        fields = dict(
            summary='Мишень раннего включения проверена; это диагностический потолок, не результат новой FlyVis.',
            obtained=source + ' → counts: ' + json.dumps(r['counts'], ensure_ascii=False) +
                '; paired_added_tracked_center_oracle: ' + json.dumps(interval, ensure_ascii=False) +
                '. Все26 исходных выходов совпали точно; первичные события videos/*.json.',
            interpretation='В отличие от прежних проверок потерянных треков, здесь отдельно измерены события ещё ведомых треков. ' +
                'Аннотированный центр используется только как диагностическое идеальное действие. ' +
                ('Практический критерий запаса пройден; точность реальной мухи и цена раннего чтения ещё не проверены.' if passed else
                 'Практический критерий запаса для этой правки центра не пройден. Это не доказывает бесполезность ранней памяти, иных размеров рамки или сопоставления идентичностей.'),
            decision=('Подготовить отдельную ограниченную пробу стоимости раннего внутреннего состояния и считывания с одинаковыми C/D/R1 и контролем без мухи; без переноса оракульного прироста на HOTA.' if passed else
                      'Не запускать полное извлечение ранних эпизодов для этой правки центра. Сохранить полный разбор причин; нужна иная проверяемая мишень, а не подбор порога после этого итога.'),
            cannot_claim=p['counterargument'] + ' ' + p['limits'])
        state = 'completed_positive' if passed else 'inconclusive'
    else:
        r, interval, passed = None, None, None
        fields = dict(summary='Сохранён частичный служебный результат; полного вывода о раннем включении нет.',
            obtained='runs/z13_activation_timing_20261025/controller_result.json: exit_code=' + str(e['exit_code']) +
                '; сохранено видео ' + str(len(jobs)) + '/26. Источники проверены, своё дерево окончено. ' +
                'price.json и журналы сохраняют причину остановки.',
            interpretation='Неполный набор не используется для вывода о полезности раннего считывания.',
            decision='Сохранить готовые видео; не повторять их и не продлевать исходный бюджет. Исполнение можно исправить только при проверяемом служебном сбое в прежнем окне.',
            cannot_claim=p['limits'])
        state = 'inconclusive'
    cp = ROOT / 'experiments/project_history.json'
    cat = read(cp)
    card = next(c for c in cat['experiments'] if c['id'] == IDENT)
    for key, value in fields.items():
        if card.get(key) and card[key] != value:
            card.setdefault('revisions', []).append(dict(date=now().strftime('%Y-%m-%d'), commit=head,
                field=key, old=card[key], reason='Сохранён проверенный итог диагностики момента включения.'))
        card[key] = value
    card.update(state=state, done=now().strftime('%Y-%m-%d'))
    card['acceptance'][0].update(met=True if complete else None, evidence='runs/z13_activation_timing_20261025/finish_recorded.json')
    card['acceptance'][1].update(met=passed, evidence=source if complete else 'Полного отчёта нет')
    if complete:
        card['evidence'].append(dict(kind='file', ref=source, claim='Полный диагностический потолок, исходные назначения и одинаковый знаменатель'))
    write(cp, cat)
    journal('итог проверки момента включения до потери',
        '*Получено.* ' + fields['obtained'] + '\n\n*Вывод.* ' + fields['interpretation'] +
        '\n\n*Решение.* ' + fields['decision'] +
        '\n\nСлужебный старт: неподдерживаемый параметр приоритета не запустил контроллер; ошибка сохранена в launch_error_preserved.json. ' +
        'Убран только параметр запуска, приоритет задан сразу после старта; научные источники не изменялись. Существующее окно сохранено и проверено по ненулевому дескриптору.' +
        '\n\nHOTA не измерялась. Процент к стенду и доля достижимого реальной мухой потолка не определены. Архитектурной замены ещё нет: схема не пересобирается ради этой диагностики.')
    write(RUN / 'finish_recorded.json', dict(at=now().isoformat(), complete=complete,
        report_sha256=sha(RUN / 'report.json') if complete else None, all_owned_ended=True,
        independent_counts_and_intervals=complete, primary_pass=passed,
        video_sha256={j['video']:sha(RUN / 'videos' / (j['video']+'.json')) for j in jobs}))
    header = '# Актуально — ' + now().strftime('%d.%m.%Y %H:%M') + ' MSK\n\n'
    header += 'Диагностика z13_activation_timing_v1 окончена, своё дерево завершено. ' + fields['obtained'] + ' ' + fields['decision']
    header += ' Публикацию подтверждает activation_timing_final_publication.json. Предыдущие неудачные версии не продвигать. Авторизация до26.10 07:40МСК сохраняется; новых активных расчётов после этой стадии пока нет. 57 чужих графправок и .gitignore сохранять.\n\nНиже прежние передачи сохранены.\n\n'
    path = W / 'HANDOFF.md'
    path.write_text(header + path.read_text(encoding='utf-8'), encoding='utf-8')


if __name__ == '__main__':
    main()
