"""Record the authorized saved-row audit before execution and its verified result afterwards."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/z13_readout_overlap_20261018'
CAT=ROOT/'experiments/project_history.json'
PLAN=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md'
ID='z13_readout_overlap_v1'


def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--finish',action='store_true');a=ap.parse_args()
    cat=json.loads(CAT.read_text(encoding='utf-8'))
    if not a.finish:
        if any(c['id']==ID for c in cat['experiments']): raise RuntimeError('Already recorded')
        RUN.mkdir(exist_ok=False)
        sources=['runs/z13_continue_20261017/baseline/rows.json','runs/z13_target_model_20261017/D/rows.json']
        write(RUN/'protocol.json',dict(id=ID,date='2026-10-18',action='Choose one internal target-readout correction',
              sources_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},
              scope='Saved rows only, CPU; no fitting on train26, no model/GPU/video execution',
              decision='If new-only useful rows exist, report complementarity; never call oracle a deployable gain. Select one code-grounded correction, not an oracle selector.',
              diagnostics_for_action=1))
        cat['experiments'].append(dict(id=ID,title='З13: расхождения прежнего считывателя D и модели цели',area='flyvis',state='planned',started='2026-10-18',
              question='Добавляет ли модель цели полезные центры, теряя другие ответы прежнего считывателя?',
              hypothesis='Рабочая версия: ответы дополняют друг друга; причина потерь ещё не установлена.',
              expected='Точное сопоставление сохранённых строк; четыре группы положительных примеров, величины ошибок, разбор по видео. Диагностика 1 для правки узла считывания, согласованной автором 18.10. Без подбора переключателя по train26.',
              acceptance=[dict(criterion='Совпадают ключи строк, видео, метки и опорная геометрия',met=None),dict(criterion='Четыре группы полностью разделяют положительные строки; неизвестные исключены',met=None)],
              tags=['Z13','fly-problem-P8'],runs=['runs/z13_readout_overlap_20261018'],files=['scripts/z13_readout_overlap.py'],
              evidence=[dict(kind='json',ref='runs/z13_readout_overlap_20261018/protocol.json',claim='Постановка до расчёта')]))
        entry='*Постановка.* По согласованию автора — одна диагностика сохранённых ответов D: прежний считыватель против модели цели. Четыре группы полезности и величины ошибок на одинаковых строках; обучение, GPU, HOTA и подбор решения по train26 не выполняются. Протокол: runs/z13_readout_overlap_20261018/protocol.json. Цель — выбрать одну правку внутри считывателя; оракульный выбор только описывает доступные ответы, не является реализованным улучшением.'
    else:
        if (RUN/'recorded.json').exists(): raise RuntimeError('Already finished')
        rep=json.loads((RUN/'report.json').read_text(encoding='utf-8'))
        c=next(c for c in cat['experiments'] if c['id']==ID)
        counts={k:v['n'] for k,v in rep['cells'].items()}
        assert sum(counts.values())==rep['positives']
        result=f"runs/z13_readout_overlap_20261018/report.json: {rep['rows']} строк, {rep['positives']} положительных; оба полезны {counts['both']}, только прежний {counts['old_only']}, только новый {counts['new_only']}, оба не полезны {counts['neither']}."
        c.update(state='inconclusive',done='2026-10-18',obtained=result,
                 interpretation='Ответы дополняют друг друга. Это не доказывает доступность надёжного переключателя или причину потерь.',
                 decision='Перейти к одной рабочей правке узла по коду и литературе; не обучать переключатель на train26.',
                 cannot_claim='Нет нового исполнимого прироста, HOTA, причины ошибки или независимого подтверждения. Оракульный выбор использует истинную разметку.')
        for x in c['acceptance']: x.update(met=True,evidence='runs/z13_readout_overlap_20261018/report.json; проверки соответствия выполнены скриптом')
        c['evidence'].append(dict(kind='json',ref='runs/z13_readout_overlap_20261018/report.json',claim='Разделение сохранённых ответов'))
        entry='*Получено (`'+ID+'`).* '+result+'\n\n*Вывод.* Ответы дополняют друг друга, но причина потерь не установлена. Оракульное объединение не является моделью и не измеряет прирост HOTA.\n\n*Решение.* Одна правка внутри считывания; переключатель по разметке train26 не подбирать. HOTA не измерялась.'
    write(CAT,cat)
    plan=PLAN.read_text(encoding='utf-8');marker='## Журнал изменений\n'
    assert marker in plan
    PLAN.write_text(plan.replace(marker,marker+'\n**18.10.2026** — '+ID+('\n\n' if not a.finish else ', итог\n\n')+entry+'\n',1),encoding='utf-8')
    if a.finish: write(RUN/'recorded.json',{'recorded':True})


if __name__=='__main__': main()
