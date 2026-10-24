"""Scoped publication of a fixed readout prototype and its architecture branch."""
from pathlib import Path
import inspect
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_observed_contrast_20261025 import RUN,read,write,sha,now
W=ROOT/'runs/z13_weekend_20261024'


def main():
    from scripts import z13_publish_assignment_effect_20261024 as P
    from scripts.z13_publish_native20_final_20261024 import foreign_unchanged
    foreign_unchanged()
    P.RUN=RUN
    extra=[Path(__file__).relative_to(ROOT).as_posix(),'scripts/z13_observed_contrast_20261025.py',
        'scripts/z13_record_observed_contrast_20261025.py','scripts/z13_observed_contrast_console.ps1',
        'tracking/z13_observed_contrast.py','tests/test_z13_observed_contrast.py','scripts/build_flyvis_architecture.py']
    for name in ('protocol.json','report.json','controller.json','controller_result.json','launch.json','finish_recorded.json','D_scores.json','C_scores.json','R1_scores.json'):
        extra.append((RUN/name).relative_to(ROOT).as_posix())
    architecture=ROOT/'dissertation/architecture'
    extra += [(architecture/('FlyVis_architecture'+suffix)).relative_to(ROOT).as_posix() for suffix in ('.drawio','.png','.svg','_preview.png','_detail.png','_sources.json')]
    extra.append('dissertation/architecture/validation.json')
    validation=read(architecture/'validation.json')
    assert validation['diagram_sheets']==1 and not validation['box_overlaps'] and not validation['edges_through_unrelated_nodes']
    for name,digest in validation['outputs'].items():assert sha(architecture/name)==digest
    header=('# Актуально — '+now().strftime('%d.%m.%Y %H:%M')+' MSK\n\n'
        'Проверка z13_observed_contrast_v1 полностью завершена, exit0, своё дерево остановлено. '
        'Первичный report.json в runs/z13_observed_contrast_20261025: D — один полезный выбор из84 при21 неизвестном, '
        'все три основные нижние границы ДИ равны0, primary_pass=false. Версию не применять к HOTA и не подбирать веса. '
        'Однолистовая архитектура обновлена: жёлтая ветвь конкретной реализации считывания с наблюдавшимися фоновыми пиками; '
        'зелёного улучшения нет. Проверка исходов и интервалов независимая, источники неизменны. '
        'Публикацию и зелёные проверки подтверждает observed_contrast_final_publication.json рядом с этим HANDOFF; '
        'если маркера нет, проверить процесс публикации и служебные логи, не повторять научный расчёт. '
        'Готового нового расчётного узла сейчас нет. Продолжение требует иной обоснованной реализации и собственной дешёвой проверки; '
        'скалярная нормировка, абсолютный максимум, эта фиксированная косинусная версия и прежние перечисленные варианты не продвигаются. '
        'Разрешение автора до26.10 07:40МСК остаётся.57 чужих графправок и .gitignore сохранять.\n\nНиже прежние передачи сохранены.\n\n')
    path=W/'HANDOFF.md';path.write_text(header+path.read_text(encoding='utf-8'),encoding='utf-8')
    source=inspect.getsource(P.publish)
    source=source.replace("names=['experiments/project_history.json'","names=EXTRA+['experiments/project_history.json'")
    source=source.replace("'tests/test_z13_assignment_effect.py','-q'","'tests/test_z13_assignment_effect.py','tests/test_z13_refusal_causes.py','tests/test_z13_refusal_trace.py','tests/test_z13_observed_contrast.py','-q'")
    source=source.replace("prefix='assignment_effect_'+mode","prefix='observed_contrast_final'")
    source=source.replace("title='Correct invalid crop-sampling premise and start exact assignment-effect supervision' if mode=='start' else 'Record complete assignment-effect supervision with exact baseline replay'","title='Test observed-background contrast readout and record unpassed gate'")
    ns={**vars(P),'EXTRA':extra};exec(compile(source,str(Path(__file__)), 'exec'),ns)
    ns['publish']('observed_contrast_final')


if __name__=='__main__':main()
