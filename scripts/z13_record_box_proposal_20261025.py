"""Type-exact finish repair and reviewed single-sheet architecture milestone."""
from pathlib import Path
import inspect
import json
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import z13_box_proposal_20261025 as base
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
RUN=base.RUN
ARCH=ROOT/'dissertation/architecture'


def finish():
    assert not (RUN/'finish_recorded.json').exists()
    p=read(RUN/'protocol.json');r=read(RUN/'report.json');choice=read(RUN/'choice.json')['indices']
    values,counts=base.aggregate(p['evaluate26'],choice)
    selected={a:x[choice[a] if a!='Dold' else 0] for a,x in values.items()}
    assert values!=r['all_alpha_evaluate'] and json.loads(json.dumps(values))==r['all_alpha_evaluate']
    assert counts==r['counts'] and selected==r['per_video'] and base.contrasts(selected)==r['paired']
    write(RUN/'finish_type_diagnostic.json',dict(at=now().isoformat(),original_finish_error='AssertionError at all_alpha_evaluate dictionary equality; in-memory integer keys0..3 become JSON string keys',
        original_source_sha256=sha(ROOT/'scripts/z13_box_proposal_20261025.py'),service_source_sha256=sha(Path(__file__)),
        report_sha256=sha(RUN/'report.json'),changed='Normalize only dictionary key types through exact JSON round-trip, then require exact equality of every saved value. Numeric tolerance, outcomes, intervals and scientific criteria unchanged. No replay or training repeated.',all_counts_selected_values_and_paired_exact=True))
    source=inspect.getsource(base.finish)
    old="assert values==r['all_alpha_evaluate']"
    assert source.count(old)==1
    source=source.replace(old,"assert __import__('json').loads(__import__('json').dumps(values))==r['all_alpha_evaluate']")
    ns=dict(vars(base));exec(compile(source,__file__,'exec'),ns);ns['finish']()
    base.journal('служебная сверка ключей сохранённого прототипа','*Ошибка.* При финальной независимой сверке int-ключи alpha0..3 сравнивались со строковыми ключами JSON. Все численные значения, выбор, исходы и интервалы совпадают точно.\n\n*Исправление.* Отдельный scripts/z13_record_box_proposal_20261025.py приводит только типы ключей точным JSON-преобразованием. Старый научный источник и report не изменены; ни одного расчёта повторно не было. Доказательство runs/z13_box_proposal_20261025/finish_type_diagnostic.json.')


def crops():
    from PIL import Image
    Image.MAX_IMAGE_PIXELS=None
    source=read(ARCH/'FlyVis_architecture_sources.json')
    with Image.open(ARCH/'FlyVis_architecture.png') as im:
        for ident in ('z13_anchor_state_v1','z13_spatial_derivatives_v1','z13_box_proposal_v1'):
            n=next(n for n in source['nodes'] if n['id']=='v_'+ident)
            path=RUN/'architecture_milestone'/('v_'+ident+'.png');path.parent.mkdir(parents=True,exist_ok=True)
            im.crop((n['x']-30,n['y']-30,n['x']+n['w']+30,n['y']+n['h']+30)).save(path)


def record():
    out=RUN/'architecture_milestone';assert not (out/'review.json').exists()
    v=read(ARCH/'validation.json')
    assert v['diagram_sheets']==1 and not v['box_overlaps'] and not v['edges_through_unrelated_nodes'] and v['all_text_fits']
    for name,h in v['outputs'].items():assert sha(ARCH/name)==h
    reports=['runs/z13_anchor_state_20261025/replay/report.json',
             'runs/z13_spatial_derivatives_20261025/replay/report.json','runs/z13_box_proposal_20261025/report.json']
    assert all(read(ROOT/n)['complete'] and not read(ROOT/n)['primary_pass'] for n in reports)
    write(out/'review.json',dict(at=now().isoformat(),one_sheet=True,nodes=v['nodes'],edges=v['edges'],overview_and_three_readable_branch_crops_viewed=True,
        all_three_branches_yellow=True,source_reports={n:sha(ROOT/n) for n in reports},
        confirmed_results_review='All three main criteria failed. No new positive strictly-above-zero useful stand contrast. D box effect versus C lower is -2.67e-19, numerically zero rather than positive. No geometry oracle is presented as FlyVis improvement or HOTA.',
        scope='Three internal-path milestones: removal of intermediate search history; local directional descriptor; complete XYWH proposal output. Alternative branches do not imply sequential execution. The diagnostic cause partition is evidence, not a new neural architecture node.'))
    base.journal('архитектурная веха состояния, локального описания и полной рамки','*Получено.* На одном листе добавлены три жёлтые альтернативы: поиск заново от наблюдённого образца; два направления изменения активности соседних колонок; выбранная моделью цели реальная полная рамка вместо только её центра. Все основные критерии не пройдены.\n\n*Проверено.* Источники и прежние результаты сохранены; обзор и три читаемых фрагмента просмотрены. '+str(v['nodes'])+' блоков, '+str(v['edges'])+' стрелок; один лист, перекрытий и проходов через посторонние блоки нет, текст помещается. Проверка runs/z13_box_proposal_20261025/architecture_milestone/review.json. Новых лучших подтверждённых результатов нет; проблемыP8/P9 остаются.\n\n*Решение.* Геометрический запас оправдал рабочий прототип, но его выбор области не дал полезного результата стенду. Порог, вес применения и способ объединения карт на26не перебирать. Раздельно хранить достигнутое понимание причины и отрицательный итог этой реализации; HOTA не измерялась.')


if __name__=='__main__':
    if '--finish' in sys.argv:finish()
    elif '--crops' in sys.argv:crops()
    elif '--record' in sys.argv:record()
    else:raise RuntimeError('Explicit mode required')
