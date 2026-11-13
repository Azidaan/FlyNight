from common import *
journal((OUT/'assignment_audit/card_corrections_journal.txt').read_text(encoding='utf8').strip())
journal('По прямому поручению автора зарегистрирована гипотеза `deferred_light_instance_masks_20261114`: отдельная лёгкая сеть масок экземпляров, сначала YOLO26n-seg, для выделения доступной части каждого объекта. Цвет служит показом маски, а не самостоятельной идентичностью. Источники: experiments/deferred_identity_hypotheses.json; '+OUT.relative_to(ROOT).as_posix()+'/mask_review/hypothesis_integration_v2.json. Предыдущие гипотезы сохранены; новая модель не скачивалась. Проверка качества и вычислительной цены запланирована, положительный исход не заявляется.')
inputs={n:sha(ROOT/n) for n in ['experiments/project_history.json','experiments/system_problems.json','experiments/hypothesis_calendar.json','experiments/deferred_identity_hypotheses.json','scripts/obsidian_level_graph.py','scripts/build_obsidian_vault.py','scripts/obsidian_deferred_identity_hypotheses.py']}
inputs[(OUT/'checks_v2.py').relative_to(ROOT).as_posix()]=sha(OUT/'checks_v2.py')
save('checks_v2_queue.json',dict(runner_sha256=sha(OUT/'run_queue.py'),deadline=read(OUT/'protocol.json')['deadline'],inputs=inputs,jobs=[dict(entry=(OUT/'checks_v2.py').relative_to(ROOT).as_posix(),timeout=600)]))
print('CHECKS_V2_PREPARED')
