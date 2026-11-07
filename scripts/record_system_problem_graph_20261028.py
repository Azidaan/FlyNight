"""Сохранить проверенный итог оформления карты; научные расчёты не запускает."""
from pathlib import Path
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    from scripts.continuation_service_20261028 import RUN,service
    from scripts.obsidian_system_problems import load
    from scripts.b123_safe_journal_20261027 import journal
    data=load()
    log=RUN/'system_problem_graph_build.log'
    graph_dir=Path(log.read_text(encoding='utf-8').strip().splitlines()[-1])
    graph=service.read(graph_dir/'audit.json')
    assert graph['passed'] and graph['verified_copy_before_live']
    assert graph['protected_hypotheses_byte_identical'] and graph['legacy_problem_files_byte_identical']
    assert not graph['external_problem_edges'] and graph['problem_node_count']==9
    tests=(RUN/'system_problem_graph_tests.log').read_text(encoding='utf-8')
    assert '3 passed' in tests and 'failed' not in tests
    processes={}
    for name,path in [('queue',RUN/'queue.launch.json'),('guard',RUN/'guard.launch.json'),
                      ('visual_controller',RUN/'history_visual_inputs/launch.json')]:
        launch=service.read(path)
        processes[name]=dict(launch=launch,matching=bool(service.matching(launch['pid'],launch['created'])))
    assert all(p['matching'] for p in processes.values())
    progress=service.read(RUN/'history_visual_inputs/status.json')
    record=dict(complete=True,created=service.now(),scope='Graph and rules only; no new scientific result',
        registry_sha256=service.sha(ROOT/'experiments/system_problems.json'),
        graph_audit=str((graph_dir/'audit.json').relative_to(ROOT)),
        problems=len(data['problems']),patterns=len(data['patterns']),
        visible=graph['note_count'],stored=graph['stored_note_count'],
        problem_nodes=graph['problem_node_count'],problem_edges=len(graph['problem_edges']),
        external_problem_edges=graph['external_problem_edges'],
        hypotheses_byte_identical=True,legacy_problem_files_byte_identical=True,
        deleted=graph['deleted_files'],moved=graph['moved_files'],tests=tests.strip(),
        process_identity=processes,visual_progress=progress)
    service.write(RUN/'system_problem_graph_result.json',record)
    journal('system_problems_graph_20261028','Отдельная карта проблем проекта по поручению автора',
        'По прямому поручению автора создана отдельная связная карта «Проблемы» всего проекта. '
        'Реестр experiments/system_problems.json: шесть содержательных проблем и два наблюдаемых повторения; '
        'факт, рабочее объяснение, следующая проверка и критерий решения разделены. '
        'Это оформление существующих фактов, не новый опыт и не установление общих причин. '
        'Генератор scripts/obsidian_system_problems.py, правила AGENTS.md5.1 и инструкция хранилища. '
        'Проверенная копия и итог: '+record['graph_audit']+'. '
        'Сорок видимых заметок, 1344 сохранённых; у группы проблем девять узлов, четырнадцать внутренних рёбер, внешних нет. '
        'Одиннадцать гипотез и прежние P1–P11 побайтно сохранены, удалений и перемещений нет. '
        'Три проверки графа пройдены: runs/continuation_20261028/system_problem_graph_tests.log. '
        'Правило дальнейшего ведения: новые случаи добавлять в историю существующей проблемы, '
        'наблюдаемые повторения отличать от доказанных причин; существенные результаты сопоставлять с открытыми вопросами. '
        'Текущий060 продолжается без изменения источников или весов; проверены PID и время создания очереди, защиты срока и контроллера. '
        'Источник итоговой проверки: runs/continuation_20261028/system_problem_graph_result.json.')
    with (RUN/'HANDOFF.md').open('a',encoding='utf-8') as out:
        out.write('\n\nКарта проблем по прямому поручению автора28.10 готова. '
            'Источник experiments/system_problems.json, генератор scripts/obsidian_system_problems.py, '
            'центр Обзор проекта/Проблемы.md, правила AGENTS5.1. Шесть проблем/два повторения; '
            'факты/рабочие причины/различающая проверка/критерий решения отдельно. '
            'Случаи дописывать в существующие проблемы с источниками, не затирать историю. '
            'Повторение не доказывает общей причины. У карты9узлов/14внутреннихрёбер/0внешних; '
            'гипотезы11 и старыеP1–P11 побайтны;40видимых/1344сохранённых/0удалений/0перемещений. '
            'Итог system_problem_graph_result.json, проверенная копия '+record['graph_audit']+',3graphtests passed. '
            '060 продолжает реальную подготовку входов; его контроллер, очередь и guard проверены PID+birth, '
            'свежий снимок прогресса сохранён в итоговом JSON.065готова. Нового обучения/HOTA1/3/4 пока нет. '
            'До обучения уточнить смысл метки личности после неверного назначения; это рабочий вопрос, не доказанная ошибка данных.\n')
    print(json.dumps(record,ensure_ascii=False))


if __name__=='__main__':
    main()
