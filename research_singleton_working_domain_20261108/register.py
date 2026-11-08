import ast
import subprocess
from research_singleton_working_domain_20261108.common import *


def main():
    proposal = read(Path(__file__).parent/'proposal.json')
    assert not (RUN/'protocol.json').exists()
    RUN.mkdir(parents=True, exist_ok=True)
    write(RUN/'authorization.json', dict(approved=True, recorded=datetime.now(MSK).isoformat(),
        user_answer='Да, только этот этап', question='До15минутCPUна20выбранныхобучающихвидео;безGPU,обученияи26/20val.',
        proposal_sha256=sha(Path(__file__).parent/'proposal.json')), new=True)
    sources = {}
    folders = ['research_singleton_working_domain_20261108', 'tracking',
               'research_bbox_confirmation_inventory_20261108', 'research_bbox_singleton_pilot_20261108',
               'research_bbox_cost_pilot_20261108', 'research_native_confirm_policy_20261108',
               'research_identity_assignment_run_20261108', 'research_bbox_singleton_audit_recovery1_20261108']
    for folder in folders:
        for path in (ROOT/folder).glob('*.py'):
            ast.parse(path.read_text('utf-8'), filename=str(path))
            sources[path.relative_to(ROOT).as_posix()] = sha(path)
    for name in ['scripts/association_log.py','scripts/output_filter_oracle.py','datasets/bdd100k_loader.py',
                 'runs/z13_local_peak_20261021/geometry.json',
                 'runs/flyvis_repairs_20261106/unification_20261107/system_memory_guard.py']:
        sources[name] = sha(ROOT/name)
    for path in (ROOT/'.venv/Lib/site-packages/ultralytics/trackers').rglob('*.py'):
        sources[path.relative_to(ROOT).as_posix()] = sha(path)
    for name in ['proposal.json','checksums.json']:
        path = Path(__file__).parent/name
        sources[path.relative_to(ROOT).as_posix()] = sha(path)
    videos = {}
    for video, row in proposal['selection']['videos'].items():
        for name, digest in row['sha256'].items():
            assert sha(ROOT/name) == digest, name
        videos[video] = dict(split='additional_fit20', raw=row['files']['raw'],
                            baseline=row['files']['baseline'], sources=row['sha256'])
    p = dict(id='singleton_working_domain_inventory_v1_20261108',
        registered=datetime.now(MSK).isoformat(), source_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),
        working_sources_uncommitted=True, sources=sources, videos=videos,
        native_config=proposal['fixed_policy']['native_config'], max_age=2,
        minimum_export_iou=.05, inventory_iou=.2, alpha=.025, matching_threshold=.8,
        data_gate=proposal['data_gate'], active_problem=1, deferred=[2,3,4],
        scope=proposal['requested_action'], scientific_runtime_seconds=900,
        first_video_price_cap_seconds=180, no_automatic_followup=True)
    write(RUN/'protocol.json', p, new=True)
    catalogue = read(ROOT/'experiments/project_history.json')
    assert not any(e['id'] == p['id'] for e in catalogue['experiments'])
    catalogue['experiments'].append(dict(id=p['id'],
        title='Одна свободная рамка — собрать правильные и чужие пары в действующем допуске',
        area='flyvis', state='planned', started='2026-11-08', question=proposal['question'],
        hypothesis='Другие20fit-видео могут дать известные чужие пары именно при age2 и alpha0,025; это проверка готовности данных.',
        expected='Контекст без новых детекций, кадров, активности сети, обучения иHOTA.',
        acceptance=[dict(criterion='Полные20видео;8исходныхколонокNativeточносовпадаютсготовымиbaselineдооценкипар.',met=None),
                    dict(criterion='Независимо проверены стоимости,допуски,редкие фактические метки и назначения.',met=None),
                    dict(criterion='Вcost-qualifiedобласти>=3известныхправильныхи>=3чужих,каждыйклассминимумв2видео;это готовность,не мощность или гарантияриска.',met=None)],
        obtained='', interpretation='', decision='ТолькоэтотCPUэтап;послеитогаостановиться.',
        cannot_claim='НетновогосигналаFlyVis,HOTAилирешенияпервойпроблемы. Личность двухфрагментов не подтверждаеткаждыйнеразмеченныйкадр. Другие20изпрежнегоfit84не независимыйитоговыйтест.',
        evidence=[dict(kind='file',ref='runs/singleton_working_domain_20261108/protocol.json',claim='Согласованный состав данных и ограничения до экспорта')]))
    write(ROOT/'experiments/project_history.json', catalogue)
    path=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';text=path.read_text('utf-8');needle='## Журнал изменений';pos=text.index(needle)+len(needle)
    entry='\n\n**08.11.2026, '+datetime.now(MSK).strftime('%H:%M')+' MSK** — согласован только CPU-этап готовности выборки в рабочем допуске.\n\nАвтор ответил «Да, только этот этап» на предложение `research_singleton_working_domain_20261108/proposal.json`. До 15 минут научного CPU-прохода от отдельного старта; 20 выбранных дополнительных fit-видео, age<=2, alpha=0,025, без GPU, детектора, декодирования кадров, обучения и оценки26/20val. Сначала точное совпадение Native с готовым baseline; затем недостающий контекст, метки и независимая сверка. Источники зафиксированы до выполнения. Карточка `'+p['id']+'`; протокол `runs/singleton_working_domain_20261108/protocol.json`. Автоматическое продолжение к картам или порогам не разрешено.\n'
    path.write_text(text[:pos]+entry+text[pos:],encoding='utf-8')


if __name__ == '__main__':
    main()
