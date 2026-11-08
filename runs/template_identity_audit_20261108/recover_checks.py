"""Preserve the failed receipt; verify the authorized Moscow-week rollover.

Scientific scores and the completed exact-frame audit are reused unchanged.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import contextlib,hashlib,json,runpy,shutil,subprocess,sys,tempfile,time
from collections import Counter
from datetime import date,datetime,timedelta,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
from system_memory_guard import guard_entrypoint
guard_entrypoint(timeout=900)
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
assert read(OUT/'report_queue_status.json')['state']=='failed'
assert not (OUT/'recovery_checks.json').exists()
failed_log=ROOT/'runs/flyvis_repairs_20261106/unification_20261107/uncapped_checks/90cd054c9ada417d942eeb9d0a55d45c/memory_guard.worker.log'
text=failed_log.read_text(encoding='utf8');assert '62 passed' in text and 'AssertionError' in text
for path,digest in read(OUT/'protocol.json')['sources'].items():assert sha(ROOT/path)==digest,path
for path,digest in read(OUT/'execution_receipt.json')['checksums'].items():assert sha(OUT/path)==digest,path
r=read(OUT/'results.json');direct=[x for x in r['records'] if x['direct_kind']!='unknown']
paired=[]
for row in direct:
    name='_'.join(map(str,[row['video'],row['frame'],row['old'],row['new']]))+'.json'
    ready={};sources={}
    for prior in ['background_patch_gate_20261108','decoder34_identity_gate_20261108']:
        path=ROOT/'runs'/prior/'events'/name;value=read(path);sources[path.relative_to(ROOT).as_posix()]=sha(path)
        ready[prior]=dict(cells=value['cells'],supported=value['supported'],scores={n:{f:v.get(f) for f in ['correlation_max','filter_max']} for n,v in value['variants'].items()})
    paired.append(dict(video=row['video'],frame=row['frame'],old=row['old'],new=row['new'],direct_kind=row['direct_kind'],reference_identity=row['reference_history'][-1]['identity'],candidate_identity=row['current']['identity'],native_current_status=row['native_current']['status'],conditional_label_preserved=row['conditional_label'],ready_scores=ready,sources=sources))
assert len(paired)==5
save(OUT/'direct_pair_scores.json',dict(passed=True,records=paired,scope='Read already completed scores for all five directly matched pairs. No new descriptors, score calculation, threshold or AUROC-based choice.'))
now=datetime.now(timezone(timedelta(hours=3)));actual=read(OUT/'protocol.json')['registered'][:10]
assert actual=='2026-11-09'
path=ROOT/'experiments/project_history.json';cat=read(path);card=next(x for x in cat['experiments'] if x['id']==OUT.name)
changes=[]
for field in ['started','done']:
    if card[field]!=actual:
        changes.append(dict(field=field,old=card[field],new=actual));card.setdefault('revisions',[]).append(dict(date=actual,commit='',field=field,old=card[field],reason='Код подготовлен08.11, но зарегистрированная сверка и её фиксация выполнены после полуночи09.11МСК; источник protocol.json:registered.'))
        card[field]=actual
card['evidence'].append(dict(kind='file',ref=f'runs/{OUT.name}/direct_pair_scores.json',claim='Готовые оценки пяти прямо размеченных пар без пересчёта'))
save(path,cat)
path=ROOT/'experiments/system_problems.json';problems=read(path);problem=next(x for x in problems['problems'] if x['id']=='С2')
case=next(x for x in problem['cases'] if x['component']=='Принадлежность последнего шаблона памяти')
case['date_revision']=dict(old=case['date'],reason='Фактическая регистрация и выполнение09.11МСК, protocol.json:registered');case['date']=actual;save(path,problems)
path=ROOT/'experiments/useful_literature.json';literature=read(path);source=next(x for x in literature['sources'] if x['id']=='yan2021_stark_template_memory')
source['doi']='не проверен';source['sections']+=' DOI-resolver и Crossref API недоступны; номер DOI не принимается из вторичного источника.';save(path,literature)
eligible=[x for x in r['records'] if x['reference_history'][-1]['identity'] is not None and x['conditional_label']['target'] is not None]
assert len(eligible)==29 and all(x['reference_history'][-1]['identity']==x['conditional_label']['target'] for x in eligible)
clarification=('У всех29прямо сопоставленных прошлых рамок есть прежняя метка цели, и все29совпадают с ней. Это проверенное покрытие гипотезы загрязнения; остальные171прошлая рамка не признаны чистыми или чужими. '
    'Новая прямо подтверждённая чужая пара448→453, видео009215e2-62d79d70, кадр920, ранее оставалась неизвестной: сырая детекция имеет однозначное соответствие разметке, а текущее состояние нового трека не проходит прежний рамочный допуск. Старая метка не переписана. '
    'Готовые оценки пяти прямо размеченных пар прочитаны без пересчёта в direct_pair_scores.json. Это обучающие диагностические случаи, а не независимая проверка качества.')
(OUT/'УТОЧНЕНИЕ.md').write_text('# Границы вывода и календарная поправка\n\n'+clarification+'\n\nДата выполнения —09.11.2026МСК, protocol.json:registered; имя папки отражает день подготовки кода08.11. Старые значения дат сохранены в revisions карточки.\n\nПервичная проверка гипотез остановилась на точном сравнении всех байтов после перехода к новой неделе. По прямому правилу автора текущая неделя должна смениться; проверяются отдельно научные листья и разрешённая календарная навигация. Исходный снимок до первого перехода не сохранён, поэтому задним числом его побайтная сохранность не утверждается.\n',encoding='utf8')
path=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';text=path.read_text(encoding='utf8')
entry=f'**{now:%d.%m.%Y, %H:%M} MSK** — Уточнены29прямых опор и календарная проверка гипотез после полуночи.\n\n*Получено (`{OUT.name}`).* {clarification}\n\n*Поправка.* Даты started/done карточки и случая исправлены на09.11МСК по protocol.json:registered, прежние значения сохранены.62первичныхтеста прошли; исходная служебная проверка остановилась на побайтном сравнении календаря при переходе к новой неделе. Журнал сбоя сохранён.\n\n*Решение.* Научные оценки не повторять. Повторить только необходимые проверки изменённых записей; проверить сохранность научных листьев при календарном переходе отдельно от разрешённой смены недели. Источники `runs/{OUT.name}/УТОЧНЕНИЕ.md`, `runs/{OUT.name}/direct_pair_scores.json`.\n\n'
path.write_text(text.replace('## Журнал изменений\n','## Журнал изменений\n\n'+entry,1),encoding='utf8')
save(OUT/'metadata_correction.json',dict(date=now.isoformat(),changes=changes,failed_log_sha256=sha(failed_log),old62_tests_passed=True,reference_targets_compared=29,matching_targets=29))
print('Metadata corrected; checking catalog',flush=True)
with (OUT/'recovery_catalog_validation.log').open('x',encoding='utf8') as stream:
    with contextlib.redirect_stdout(stream):
        try:runpy.run_path(str(ROOT/'scripts/validate_project_history.py'),run_name='__main__')
        except SystemExit as e:
            if e.code:raise
import pytest
assert pytest.main(['tests/test_project_history.py','-q'])==0
from scripts.obsidian_level_graph import Tree,hypotheses
from scripts.build_obsidian_vault import main
from scripts.audit_obsidian_levels import audit
vault=ROOT/'ObsidianFlyVis'
before={p.name:p.read_bytes() for p in (vault/'Карта проекта/Гипотезы').glob('*.md')}
def build(v):
    sys.argv=['scripts/build_obsidian_vault.py','--vault',str(v)];assert main()==0
with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as temp:
    temp=Path(temp);snapshots=[];calendars=[]
    for day in [date(2026,11,8),date(2026,11,9)]:
        copy=temp/str(day);shutil.copytree(vault,copy)
        t=Tree(copy);t.add('Гипотезы','Гипотезы');hypotheses(t,copy,as_of=day);t.save()
        snapshots.append({k:(copy/n['path']).read_bytes() for k,n in t.nodes.items() if k.startswith('hyp:')})
        calendars.append({k:n for k,n in t.nodes.items() if k.startswith('hyp-month:')})
    assert snapshots[0]==snapshots[1] and len(snapshots[0])==len(read(ROOT/'experiments/hypothesis_calendar.json')['entries'])
    assert calendars[1]['hyp-month:2026-11:week:3']['current_week']
    assert not calendars[1]['hyp-month:2026-11:week:2']['current_week']
    preview=temp/'preview';shutil.copytree(vault,preview);build(preview);g=audit(preview)
    build(vault);assert audit(vault)==g
assert before=={p.name:p.read_bytes() for p in (vault/'Карта проекта/Гипотезы').glob('*.md')}
g.update(hypotheses_unchanged=True,calendar_rollover_scientific_leaves_equal=True,prior_snapshot_limitation='Actual pre-first-rollover bytes were not persisted; this proof compares reconstructed08/09 scientific leaves and exact before/after bytes of the recovery build.')
save(OUT/'recovery_graph_audit.json',g)
for path,digest in read(OUT/'protocol.json')['sources'].items():assert sha(ROOT/path)==digest,path
for path,digest in read(OUT/'execution_receipt.json')['checksums'].items():assert sha(OUT/path)==digest,path
save(OUT/'recovery_checks.json',dict(passed=True,source_hashes_stable=True,scientific_scores_reused=True,catalog_tests=58,original_tests=62,graph=g,checksums={n:sha(OUT/n) for n in ['recover_checks.py','results.json','summary.json','direct_pair_scores.json','metadata_correction.json','recovery_catalog_validation.log','recovery_graph_audit.json','УТОЧНЕНИЕ.md']}))
print('Recovery checks passed',flush=True)
