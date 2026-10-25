"""Audit and publish the completed followup; no scientific execution or cache upload."""
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
from scripts.z13_publish_native20_final_20261024 import foreign_unchanged,graph_rows
from scripts.z13_hota_deadline_guard import WindowsProcesses
from scripts.z13_localization_labels_20261023 import verify_sources
RUN=ROOT/'runs/z13_box_supervision_followup_20261025'
IDENT='z13_box_supervision_followup_v1'
DOC=ROOT/'research_flyvis_deep_2026-10-25/ПРИОРИТЕТ_И_МАТЕМАТИКА.md'


def audit():
    assert read(RUN/'finish_recorded.json')['complete']
    e=read(RUN/'controller_result.json');assert e['exit_code']==0 and e['all_owned_ended'] and not e['remaining']
    ps=WindowsProcesses();assert all(ps.creation(int(pid))!=b for pid,b in e['owned'].items())
    p=read(RUN/'protocol.json');verify_sources(p)
    assert read(RUN/'prediction_reproduction.json')['models']==24
    assert read(RUN/'prediction_reproduction.json')['all_saved_predictions_exact']
    assert read(RUN/'independent_audit.json')['all24_models_rows_action_labels_and_intervals_verified']
    assert read(RUN/'strict_action_gate.json')['passed']
    foreign_unchanged()
    r=read(RUN/'training/report.json')
    return r


def record():
    assert not (RUN/'final_summary.json').exists()
    r=audit();summary={}
    for variant in r['individual']:
        summary[variant]={s:dict(tau=q['tau'],counts=q['counts']) for s,q in r['individual'][variant].items()}
    write(RUN/'final_summary.json',dict(at=now().isoformat(),primary_pass=r['primary_pass'],paired=r['paired'],individual=summary,
        complete=True,not_hota=True,all_owned_ended=True,report_sha256=sha(RUN/'training/report.json')))
    text=DOC.read_text(encoding='utf8');assert '# 4. Измеренный исход настоящего прогона' not in text
    fmt=lambda z:f'{z*100:+.4f}'.replace('.',',')
    body='\n# 4. Измеренный исход настоящего прогона\n\n'+now().strftime('%d.%m.%Y %H:%M')+' МСК. Все24модели закончены, предсказания воспроизведены из сохранённых весов, контроллеры завершили свои деревья.\n\n'
    body+='Источник: `runs/z13_box_supervision_followup_20261025/training/report.json`, `final_summary.json`, `prediction_reproduction.json`.\n\n| Сравнение | Чистая польза назначения, п.п. [95% ДИ] |\n|---|---|\n'
    for key,v in r['paired'].items():body+=f"| {key} | {fmt(v['mean_per_video'])} [{fmt(v['ci95'][0])}; {fmt(v['ci95'][1])}] |\n"
    body+='\nОсновной критерий '+('пройден' if r['primary_pass'] else 'не пройден')+'. Идеальный резерв не являлся прогнозом обучаемой модели. Подробности всех сидов, выбранных на20видео порогов, полезных/вредных/неизвестных действий сохранены; удачный сид не выбирался.\n\n'
    body+='Проверена конкретная прежняя архитектура с согласованным выходом и метками полной рамки. Это не полный SiamMOT, не обучение через дифференцируемое назначение DeepMOT, не оценка всей линии FlyVis. Новые HOTA, процент к стенду и реализованная доля потолка не установлены.\n'
    DOC.write_text(text+body,encoding='utf8')
    from scripts import z13_candidate_reserve_20261021 as m
    old=m.IDENT
    try:
        m.IDENT=IDENT;m.journal('проверенный итог и архитектурная веха полного рамочного действия',
            '*Получено.* runs/z13_box_supervision_followup_20261025/final_summary.json: '+str(r['paired'])+'\n\n*Проверка.* 24модели и все предсказания из весов точно повторены; корневой контроллер exit0, своё дерево завершено.\n\n*Архитектурная веха.* В генератор добавлена ветвь согласованного полного выхода/меток; прежние отрицательные ветви и остановленная по цене попытка сохранены. Причина — впервые измерен исход этой реализации. До публикации проверить один лист, геометрию и читаемость.\n\n*Вывод.* Основной критерий '+str(r['primary_pass'])+'. Математический резерв не предсказывал качество модели. Это конкретное считывание, не вся FlyVis/SiamMOT/DeepMOT; HOTA и процент к стенду не измерены.')
    finally:m.IDENT=old


def publish():
    audit();assert not (RUN/'publication.json').exists()
    assert read(RUN/'final_checks.json')['passed']
    assert read(RUN/'architecture_review.json')['viewed_overview_and_new_branch']
    foreign=foreign_unchanged()
    assert read(RUN/'vault_final.json')['official_generator_completed']
    assert not (ROOT/'.git/index.lock').exists()
    git=lambda *a:subprocess.check_output(['git',*a],cwd=ROOT).decode('utf8').strip()
    assert git('branch','--show-current')=='main' and git('remote','get-url','origin')=='https://gitlab.com/aziteam-group/drbdiss.git'
    assert not git('diff','--cached','--name-only')
    names=[Path(__file__).relative_to(ROOT).as_posix(),'experiments/project_history.json','experiments/metrics_glossary.json','dissertation/ЕДИНЫЙ_ПЛАН.md',
        'runs/z13_weekend_20261024/HANDOFF.md',DOC.relative_to(ROOT).as_posix(),
        'scripts/build_flyvis_architecture.py','scripts/flyvis_architecture_layout.py','runs/z13_math_priority_20261025/publication.json']
    # Primary scientific rows and reports are published; feature/label caches and weights stay local.
    names += [p.relative_to(ROOT).as_posix() for p in RUN.glob('*.json') if p.name!='publication.json']
    for phase in ('cache','training'):
        names += [p.relative_to(ROOT).as_posix() for p in (RUN/phase).glob('*.json')]
    names += [p.relative_to(ROOT).as_posix() for p in (RUN/'training').glob('*/seed*/*.json')]
    names += ['dissertation/architecture/'+n for n in ('FlyVis_architecture.drawio','FlyVis_architecture.svg','FlyVis_architecture.png','FlyVis_architecture_preview.png','FlyVis_architecture_detail.png','FlyVis_architecture_sources.json','validation.json')]
    own_start=set(read(ROOT/'runs/z13_math_priority_20261025/publication.json')['paths'])
    names += [n for state,n in graph_rows() if n not in foreign and ((ROOT/n).is_file() or (state.strip()=='D' and n in own_start))]
    names=sorted(set(names));assert not set(names)&foreign and '.gitignore' not in names
    for n in names:assert (ROOT/n).is_file() or (n in own_start and not (ROOT/n).exists()),n
    for i in range(0,len(names),20):subprocess.run(['git','-c','core.autocrlf=input','add','-f','--',*names[i:i+20]],cwd=ROOT,check=True,capture_output=True)
    staged=set(git('diff','--cached','--name-only','-z').split('\0'))-{''};assert staged and staged<=set(names)
    subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True)
    subprocess.run(['git','commit','-m','Evaluate matched full-box supervision after the action reserve gate','-m','Co-authored-by: Codex <noreply@openai.com>'],cwd=ROOT,check=True)
    commit=git('rev-parse','HEAD');subprocess.run(['git','push','origin','main'],cwd=ROOT,check=True)
    foreign_unchanged();write(RUN/'publication.json',dict(at=now().isoformat(),commit=commit,push_completed=True,paths=names));print(commit)


def readiness_note():
    d=read(RUN/'temporal_readiness.json')
    t=DOC.read_text(encoding='utf8');assert '# 5. Готовность временных данных' not in t
    DOC.write_text(t+'''
# 5. Готовность временных данных после исхода

Дополнительный описательный аудит после результата, без изменения критериев прогона: `runs/z13_box_supervision_followup_20261025/temporal_readiness.json`. Среди 37296 запросов у 28930 есть предыдущий запрос того же наблюдаемого идентификатора трекера; у 14252 поддерживаются реальные области в обоих запросах. Дубликатов пары кадр/идентификатор трекера нет. Группировка не использует истинную личность и будущие кадры. Одинаковые пары признаков всех контролей ещё не проверены.

При вмешательстве только на этих поддержанных парах идеальный резерв на26видео равен +2,069п.п.[1,562;2,556]. Для +2п.п. при нулевом вреде надо использовать около96,7%резерва, а при1%вреда среди остальных — около144,0%, что недостижимо в этом условном сценарии. Это основание сначала расширить доступную историю, а не отрицательный результат полного KeepTrack.

Следующая подготовка должна получить предыдущие наблюдения и реальные области также там, где предыдущего запроса к мухе не было. Затем проверяются охват, точное воспроизведение текущих ответов и равенство данных D/C/R1. В текущем цикле временной считыватель не обучался; готового исполнителя этой новой выборки пока нет.
''',encoding='utf8')
    from scripts import z13_box_supervision_followup_20261025 as m
    m.journal('описательный аудит готовности временного узла после результата',
        'runs/z13_box_supervision_followup_20261025/temporal_readiness.json: поддержаны14252из37296пар; только наблюдаемый trackerID и более ранний кадр. Ограниченный охватом оракул +2,0688п.п.[1,5622;2,5559], не прогноз модели. Для +2п.п. нужно96,7%резерва без вреда или144,0%при1%вреда. Это новый описательный разбор после исхода, критерии опыта не меняются. Следующая подготовка — полные предыдущие наблюдения/области и равные пары, не обучение на усечённых старых парах.')
    m.handoff('z13_box_supervision_followup_v1 завершён и независимо проверен:24модели;D-baseline -0,0625п.п.[-0,2374;+0,0353],все6ДИ включают0;непродвигатькHOTA. Все деревья завершены root exit0. Метки130видео и24модели сохранены,не повторять. Следующий узел — обучаемые межкадровые соответствия реальных областей. Readinessаудит:14252/37296поддержанныхпар;ихoracle2,069п.п.,почти весь запас нужен безвреда. Сначала получить предыдущие наблюдения/области для отсутствующих пар с равными D/C/R1 и точным текущим прогнозом. Готового исполнителя новой выборки нет;не выдавать план за активный прогон. Источники и приоритеты в research_flyvis_deep_2026-10-25/ПРИОРИТЕТ_И_МАТЕМАТИКА.md. Публикация через z13_publish_box_followup_result_20261025.py. Общее окно26.10до07:40МСК,57чужихправок/.gitignoreсохранить; всёскрыто.')
    write(RUN/'architecture_review.json',dict(at=now().isoformat(),viewed_overview_and_new_branch=True,one_sheet=True,new_branch_text_readable=True,no_overlaps=True))
    assert all(c['ci95'][0]<=0<=c['ci95'][1] for c in read(RUN/'training/report.json')['paired'].values())
    write(RUN/'confirmed_review.json',dict(at=now().isoformat(),eligible_new_confirmed_results=0,reason='All six preregistered paired intervals include zero.'))


if __name__=='__main__':
    if '--record' in sys.argv:record()
    elif '--readiness-note' in sys.argv:readiness_note()
    elif '--publish' in sys.argv:publish()
