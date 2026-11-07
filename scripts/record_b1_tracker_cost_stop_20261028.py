"""Записать один полный проход Б1 и ценовой стоп, не объявляя вывод по20."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.recovery_research_service_20261027 import RUN, read, write, sha, matching, now
from scripts.b123_safe_journal_20261027 import journal


def main():
    folder = RUN / 'b1_tracker/recovery2'
    price = read(folder / 'price.json'); audit = read(folder / 'independent_audit.json')
    stopped = read(folder / 'cost_stop.json'); per = read(folder / 'per_video.json')
    assert not price['proceed'] and not stopped['complete'] and audit['passed'] and not audit['full_series']
    assert audit['report_sha256'] == sha(folder / 'cost_stop.json') and len(stopped['completed_videos']) == 1
    for sub in ['b1_tracker/recovery2', 'b1_tracker_audit/recovery2']:
        c = read(RUN / sub / 'controller_result.json')
        assert c['exit_code'] == 0 and c['all_owned_ended'] and not any(matching(k, v) for k, v in c['owned'].items())
    v = stopped['completed_videos'][0]
    source = (folder / 'per_video.json').relative_to(ROOT).as_posix()
    evidence_base = folder.relative_to(ROOT).as_posix()
    result = dict(complete=False, completed_videos=[v], remaining_videos=stopped['remaining_videos'],
        hota_measured_on_one_video=True, full20_hota_measured=False, scientific_hypothesis_verdict=None,
        first_video_metrics={a: values[v] for a, values in per.items()}, price=price,
        paired_ci_available=False, independent_audit_passed=True, updated=now())
    write(folder / 'partial_result.json', result)
    path = ROOT / 'experiments/project_history.json'; before = path.read_bytes(); cat = read(path)
    card = next(c for c in cat['experiments'] if c['id'] == 'b1_full_tracker_pilot_v1')
    assert not card['obtained']
    card.update(state='inconclusive', done='2026-10-28',
        obtained=f'{source}: первое полное видео {v}, HOTA baseline={per["baseline"][v]["HOTA"]}, appearance/zero={per["appearance"][v]["HOTA"]}, C={per["C"][v]["HOTA"]}, D={per["D"][v]["HOTA"]}, R1/B={per["R1"][v]["HOTA"]}. '
            f'{evidence_base}/price.json: first_full_video_seconds={price["first_full_video_seconds"]}, conservative_remaining_seconds={price["conservative_remaining_seconds"]}, available_seconds={price["available_seconds"]}, proceed=false. '
            f'{evidence_base}/independent_audit.json: passed=true, videos=1, events_checked={audit["events_checked"]}, all_raw_and_json_replayed=true, native_and_zero_exact=true; оба контроллера code0/all_owned_ended.',
        interpretation='Живая интеграция Б1 работает: вмешательство меняет настоящие назначения и последующее состояние, выключенная версия точно повторяет исходный трекер, нулевая поправка сохраняет ветвь внешнего вида. На первом видео D выше исходной базы, но ниже C/R1/B. Одного видео недостаточно для вывода о пользе или вреде Б1; парного интервала по видео нет. Полная серия не допущена по заранее записанной стоимости, не по качеству.',
        decision='Остановить эту реализацию в прежнем бюджете, сохранить первое полное видео и оставшиеся19 без повторения. Не объявлять Б1 подтверждённой или опровергнутой. Для полного Б1 нужен более дешёвый сопоставимый вычислительный путь; длиннее учить старую ограниченную поправку без основания нельзя. По порядку автора перейти к подготовке научной проверки Б2, затем Б3; законченные технические пробы не повторять.',
        cannot_claim='Нет полного20 HOTA и парных интервалов; один seed и одно видео не подтверждают преимущество ядра, отсутствие такого преимущества или всю Б1. Нельзя сравнивать baseline52.981 этого видео с57.901 совокупных20 как один опыт.692.813с — все7вариантов с извлечением/оценкой, не время одного ядраFlyVis. Старые отказы и отдельные исполнения сохранены, бюджет исходного270 не продлён.',
        confirmed_results=[])
    card['acceptance'][0].update(met=False, evidence=evidence_base+'/independent_audit.json: первый полный проход проверен, но требуемые20не выполнены из-за цены')
    card['acceptance'][1].update(met=None, evidence=evidence_base+'/cost_stop.json: научный критерий не оценён на полной выборке')
    card['metrics'] = [dict(key='HOTA', value=per[a][v]['HOTA'], baseline=per['baseline'][v]['HOTA'],
        subset=f'ONE VIDEO ONLY {v}, arm {a}; incomplete20, no CI', source=source, locator=[a, v, 'HOTA'])
        for a in ['baseline', 'appearance', 'C', 'D', 'R1', 'B']]
    card['evidence'] += [dict(kind='file', ref=evidence_base+'/'+name, claim=claim) for name, claim in [
        ('partial_result.json', 'Один полный проход и точная неполнота, не научный итог20'),
        ('price.json', 'Заранее зарегистрированный стоп стоимости'),
        ('independent_audit.json', 'Полный CPU-повтор назначений, голов, выходов и метрик первого видео'),
        ('protocol.json', 'Фиксированная версия восстановления в остатке старого окна')]]
    main_card = next(c for c in cat['experiments'] if c['id'] == 'flyvis_temporal_residual_association_v1')
    updates = dict(summary='Б1 подключена к настоящему первому назначению. Один полный видеопроход всех контролей и независимый аудит завершены; остальные19не допущены по стоимости. Полной научной HOTAБ1 нет.',
        done='Сохранённая одна эпоха C/D/R1/B интегрирована в полный XYWHByteTrack без нового обучения/экспорта. '+card['obtained'])
    for field, text in updates.items():
        if main_card.get(field):
            main_card.setdefault('revisions', []).append(dict(date='2026-10-28', commit='', field=field,
                old=main_card[field], reason='Новый фактический этап: реальная интеграция и один полный проход с ценовым стопом; научный статус и основные формулировки Б1 не меняются.'))
        main_card[field] = text
    main_card['evidence'].append(dict(kind='file', ref=evidence_base+'/partial_result.json', claim='Актуальный остатокБ1после первого полного видеопрохода, не научнаяприёмка'))
    assert path.read_bytes() == before; write(path, cat)
    journal(card['id'], 'Б1: один полный видеопроход проверен, серия остановлена по стоимости',
        '*Получено.* '+card['obtained']+'\n\n*Вывод.* '+card['interpretation']+'\n\n*Решение.* '+card['decision']+
        '\n\n*Ограничения.* '+card['cannot_claim']+'\n\n*Проценты к базе.* Только на первом видео D около+0,52% относительно52,980647; это не результат20и не подтверждённая прибавка проекта. Максимумнеизвестен, доляпотолканевычисляется. Нужны обязательныепроверкикаталога и фундаментальныйобзор без изменения11узловгипотез.')
    write(folder / 'recorded.json', dict(card=card['id'], state='inconclusive', scientific_verdict=None, updated=now()))


if __name__ == '__main__': main()
