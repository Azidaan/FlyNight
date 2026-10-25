"""Save the reviewed context/product/pooling architecture and publish the full result."""
from pathlib import Path
import sys
import inspect
import subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import z13_region_pooling_20261025 as stage
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
OUT=ROOT/'runs/z13_region_pooling_20261025/architecture_milestone'


def main():
    assert not OUT.exists()
    v=read(ROOT/'dissertation/architecture/validation.json')
    assert v['diagram_sheets']==1 and v['nodes']==103 and not v['box_overlaps'] and not v['edges_through_unrelated_nodes'] and v['all_text_fits']
    for name,h in v['outputs'].items():assert sha(ROOT/'dissertation/architecture'/name)==h
    reports=['runs/z13_region_context_20261025/training/report.json','runs/z13_region_product_20261025/training/report.json','runs/z13_region_pooling_20261025/report.json']
    assert all(read(ROOT/p)['complete'] and not read(ROOT/p)['primary_pass'] for p in reports)
    assert all(c['ci95'][0]<=0 for p in reports for c in read(ROOT/p)['paired'].values())
    write(OUT/'review.json',dict(at=now().isoformat(),one_sheet=True,nodes=103,edges=93,
        overview_and_readable_three_branch_crop_viewed=True,all_three_new_branches_yellow=True,
        source_reports={p:sha(ROOT/p) for p in reports},confirmed_results_review='All new paired confidence intervals include zero; no new confirmed improvement. Existing positive results retained; P8 remains partial.',
        publication_service='Initial scoped add returned128, exact stderr was not retained; retry of the named files succeeded. Later final publisher rejected an own newly staged planned card left over from initial publication. Only that unpublished index entry was removed; no working file deleted, no scientific source changed.'))
    stage.journal('архитектурная веха контекста, исходных произведений и пространственного максимума',
        '*Получено.* Три завершённые замены добавлены жёлтыми ветвями на единственный лист: контекст назначения; прямые68поканальных произведений; среднее→максимум внутри той же рамки. '
        'Первичные источники: runs/z13_region_context_20261025/training/report.json; runs/z13_region_product_20261025/training/report.json; runs/z13_region_pooling_20261025/report.json. '
        'Последний Dmax−Dmean +0,006п.п.[−0,116;+0,115], Dmax−исходные назначения −0,096п.п.[−0,323;+0,118]. '
        '*Проверено.*103блока/93стрелки,1лист; общий вид и читаемый фрагмент просмотрены; нет перекрытий/прохождения стрелок через чужие блоки/обрезки текста. '
        'Источники dissertation/architecture/validation.json и runs/z13_region_pooling_20261025/architecture_milestone/review.json. '
        '*Вывод.* Ни одно новое парное сравнение не имеет положительной нижней границы; витрина лучших результатов не расширяется. P8 остаётся частично решённой. '
        '*Решение.* Не продвигать эти реализации кHOTA и не перебирать их пороги. Простая замена среднего на максимум не получила поддержки; общая причина ошибок не установлена. '
        '*Служебная публикация.* Начальный git add вернул128, исходный stderr не сохранён; точная причина не установлена, повтор именованных файлов прошёл. '
        'Финальная защита затем обнаружила собственную ещё не опубликованную запись planned в индексе. Убрана только она из индекса, рабочие файлы не удалялись; научные результаты и старые ошибки сохранены.')
    code=inspect.getsource(stage.publish);old='    names=sorted(set(names));'
    assert code.count(old)==1;code=code.replace(old,'    names += EXTRA_OWN_FILES\n'+old)
    names=[Path(__file__).relative_to(ROOT).as_posix(),'scripts/z13_publish_region_pooling_start_20261025.py','scripts/build_flyvis_architecture.py']
    names += ['dissertation/architecture/'+n for n in ['FlyVis_architecture.drawio','FlyVis_architecture.svg','FlyVis_architecture.png','FlyVis_architecture_preview.png','FlyVis_architecture_detail.png','FlyVis_architecture_sources.json','validation.json','README.md']]
    ns=dict(vars(stage));ns['EXTRA_OWN_FILES']=names
    exec(compile(code,str(Path(__file__)),'exec'),ns);ns['publish']('final')
    write(OUT/'publication.json',dict(at=now().isoformat(),commit=read(stage.RUN/'publication.json')['commit'],push_completed=True))


if __name__=='__main__':main()
