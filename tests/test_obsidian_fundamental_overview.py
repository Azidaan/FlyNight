from scripts.build_obsidian_vault import Linker


def test_source_matrices_do_not_create_obsidian_nodes():
    link=Linker({'trial':'Опорные задачи/Полезная задача'},{},{})
    text=link('C = [[0.9999822202126417, 0.000565], [-0.000119, 1.000168]]; [[0, 1], [1, -1]]. trial','01')
    assert '[[' not in text
    assert 'obsidian://open?' in text and '0.9999822202126417' in text


def test_real_generator_is_sparse_isolated_and_idempotent(tmp_path):
    from scripts.obsidian_fundamental_overview import build
    from scripts.audit_fundamental_obsidian import audit
    vault=tmp_path/'vault';vault.mkdir()
    assert build(vault,authorize_tracking=True)==0
    first=audit(vault)
    assert first['note_count']==55
    assert first['problem_node_count']==10 and first['problem_root_rays']==9
    assert len(first['problem_edges'])==15 and not first['external_problem_edges']
    visible=first['hashes']
    import json
    from pathlib import Path
    history=json.loads((Path(__file__).resolve().parents[1]/'experiments/project_history.json').read_text(encoding='utf-8'))
    full=any(c['id']=='b1_full_tracker_pilot_v1' and c['state'] in ('inconclusive','completed_positive','completed_negative') and any(e.get('ref','').endswith('continuation_20261028/b1_tracker/independent_audit.json') for e in c.get('evidence',[])) for c in history['experiments'])
    history_done=any(c['id']=='history_factorial_tracker_two_epoch_v1' and any('history_tracker_' in e.get('ref','') and e.get('ref','').endswith('/independent_audit.json') for e in c.get('evidence',[])) for c in history['experiments'])
    history_stopped=not history_done and any(c['id']=='history_factorial_tracker_two_epoch_v1' and c['state'] in ('inconclusive','running') and c.get('obtained') for c in history['experiments'])
    first_plan='Разобрать результаты трёх представлений истории' if history_done else ('Завершить полную проверку трёх представлений истории' if history_stopped else ('Подготовить реальные входы истории движения, внешности и наблюдений' if full else 'Довести Б1 до проверки полного сопровождения'))
    assert 'Обзор проекта/Запланировано/'+first_plan in visible
    second_plan='После матрицы истории перейти к Б2 и Б3' if history_done else ('Подготовить научную проверку Б2 и Б3' if history_stopped else 'После Б1 — сравнить три представления истории отдельно и совместно')
    assert 'Обзор проекта/Запланировано/'+second_plan in visible
    assert 'Обзор проекта/Запланировано/Проверить признаки MASA в реальных назначениях трекера' not in visible
    assert 'Обзор проекта/Запланировано/Сравнить новый извлекатель с готовыми ветвями сопоставления' not in visible
    assert 'Обзор проекта/Запланировано/Проверить совместную пользу назначений и калибровки рамок' not in visible
    assert 'Обзор проекта/Запланировано/Сопоставить новую ветвь с сильной MASA на общих видео' not in visible
    assert 'Обзор проекта/Запланировано/Проверить внешний вид в реальных назначениях трекера' not in visible
    assert 'Обзор проекта/Запланировано/Проверить малую поправку рамок вместе с сильным сопоставлением' not in visible
    assert 'Обзор проекта/Запланировано/Обучить отбор полезных продолжений потерянных объектов' not in visible
    assert first['hypothesis_node_count']==25 and first['hypothesis_edge_count']==24
    assert first['root_rays']==5 and not first['external_hypothesis_edges']
    from scripts.obsidian_core_dynamics_hypothesis import paths as core_paths
    assert core_paths().issubset(visible)
    core_note=(vault/(next(iter(core_paths()))+'.md')).read_text(encoding='utf-8')
    assert 'проверка-гипотезы/не-завершена' in core_note
    assert 'временные константы' in core_note and 'Рабочая гипотеза' in core_note
    from scripts.obsidian_core_dynamics_hypothesis import load as core_load
    assert core_load()['authorization_scope'] in core_note
    assert 'Изменение динамики ядра не реализовано' in core_note
    from scripts.obsidian_tracking_hypotheses import paths
    assert paths().issubset(visible)
    assert 'Гипотезы — подходы/Б1.3 — Сшивать обрывки по внешнему виду — признаки MASA' in visible
    assert first['journal_notes']==0 and first['canvas_files']==0
    hashes=first['hashes']
    assert build(vault)==0
    assert audit(vault)['hashes']==hashes


def test_hidden_files_survive_rebuild_and_hypotheses_are_preserved(tmp_path):
    from scripts.obsidian_fundamental_overview import build
    from scripts.audit_fundamental_obsidian import audit
    import hashlib,json
    vault=tmp_path/'vault';vault.mkdir();build(vault,authorize_tracking=True)
    hidden=vault/'06 Журнал/Старая подробная запись.md'
    hidden.parent.mkdir(exist_ok=True);hidden.write_text('Сохранить буквально. [[Несуществующая заметка]]',encoding='utf-8')
    canvas=vault/'Старая доска.canvas';canvas.write_text('{"nodes":[]}',encoding='utf-8')
    manifest=json.loads((vault/'.generated.json').read_text(encoding='utf-8'))
    (vault/'.generated.json').write_text(json.dumps([*manifest,hidden.relative_to(vault).as_posix(),canvas.name]),encoding='utf-8')
    protected={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in vault.rglob('*.md') if p.name.startswith(('Б1','Б2','Б3','Гипотезы'))}
    legacy={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in (vault/'10 Проблемы мухи').glob('*.md')}
    assert len(legacy)>=11
    build(vault)
    assert hidden.read_text(encoding='utf-8').startswith('Сохранить буквально.') and canvas.exists()
    assert all(hashlib.sha256(p.read_bytes()).hexdigest()==h for p,h in protected.items())
    assert all(hashlib.sha256(p.read_bytes()).hexdigest()==h for p,h in legacy.items())
    report=audit(vault)
    assert report['hidden_note_count']>=2 and not report['missing_targets']
