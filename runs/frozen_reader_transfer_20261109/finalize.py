"""Record zero-support coverage, refresh graph only, retain closed execution logs."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import contextlib,hashlib,json,shutil,sys,tempfile
from pathlib import Path
from datetime import datetime,timedelta,timezone
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
from system_memory_guard import guard_entrypoint
guard_entrypoint(timeout=600)
import psutil
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,v):
    with (OUT/n).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')
assert read(OUT/'final_checks.json')['passed'];excluded=read(OUT/'excluded_geometry.json')
assert [(r['clipped'],r['old_cells'],r['joint_cells']) for r in excluded['records']]==[(False,0,0),(True,0,0)]
for path,h in excluded['sources'].items():assert sha(ROOT/path)==h,path
for path,h in read(OUT/'scientific_receipt.json')['checksums'].items():assert sha(OUT/path)==h,path
before={p:sha(ROOT/p) for p in ['experiments/project_history.json','experiments/obsidian_experiment_topics.json','experiments/system_problems.json','experiments/metrics_glossary.json']}
plan=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';s=plan.read_text(encoding='utf8');now=datetime.now(timezone(timedelta(hours=3)));entry=f'**{now:%d.%m.%Y, %H:%M} MSK** — Обрезанный случай с нулевой поддержкой также не получил оценки.\n\n*Получено (`{OUT.name}`).* Дополнительная проверка геометрии двух первоначально исключённыхпар:961→1216безобрезки0→0общихячеек;40→188видеоb1e3e9f5-92377424кадр205обрезансправа, новаяпривязка изменила перенос, но поддержка0→0. По неизменному правилу≥3ячеек обе формулы не дают оценки. Это не отказ решателя назначения и не положительный тест сигнала. Текущая детекция последней пары прямо размечена, оба точных кадра вместе не подтверждены. Источник `runs/{OUT.name}/excluded_geometry.json`.\n\n*Проверки.* Каталог согласован,58тестовкаталога пройдены,13математических/APIтестов переиспользованы поSHA. Исходныйграф сначала проверен на копии, затем обновлён; гипотезы сохранены побайтно. Сейчас после новой записи обновляется только граф; каталог и научные оценки не меняются.\n\n';assert 'Обрезанный случай с нулевой поддержкой также не получил оценки' not in s;plan.write_text(s.replace('## Журнал изменений\n','## Журнал изменений\n\n'+entry,1),encoding='utf8')
from scripts.build_obsidian_vault import main
from scripts.audit_obsidian_levels import audit
vault=ROOT/'ObsidianFlyVis';hyp={p.name:p.read_bytes() for p in (vault/'Карта проекта/Гипотезы').glob('*.md')}
def build(v):sys.argv=['scripts/build_obsidian_vault.py','--vault',str(v)];assert main()==0
with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as temp:
    copy=Path(temp)/'ObsidianFlyVis';shutil.copytree(vault,copy);build(copy);graph=audit(copy);build(vault);assert audit(vault)==graph
assert hyp=={p.name:p.read_bytes() for p in (vault/'Карта проекта/Гипотезы').glob('*.md')};assert before=={p:sha(ROOT/p) for p in before}
logs=[];failures=[]
for path in sorted(OUT.glob('*.log')):
    for line in path.read_text(encoding='utf8',errors='replace').splitlines():
        if not line.startswith('SYSTEM_MEMORY_GUARD '):continue
        receipt=json.loads(line[len('SYSTEM_MEMORY_GUARD '):]);command=receipt['command'];token=command[-1].split('=',1)[-1];source=ROOT/f'runs/flyvis_repairs_20261106/unification_20261107/uncapped_checks/{token}';dest=OUT/'execution_logs'/token;dest.mkdir(parents=True,exist_ok=True)
        for file in source.iterdir():
            if file.is_file():shutil.copyfile(file,dest/file.name)
        for pid,birth in receipt['owned'].items():
            try:assert psutil.Process(int(pid)).create_time()!=birth,(pid,'Still running')
            except psutil.NoSuchProcess:pass
        logs.append(dict(token=token,source_log=path.name,returncode=receipt['returncode'],reason=receipt['reason'],owned_ended=True))
        if receipt['returncode'] or receipt['reason']:failures.append(dict(token=token,returncode=receipt['returncode'],reason=receipt['reason']))
write('execution_receipt.json',dict(passed=True,stages=logs,failed_preparation_or_assertion_stages_preserved=failures,all_previous_owned_workers_ended=True,uncontrolled_initial_syntax_failure_preserved='register.err',pre_guard_manifest_path_failure_preserved='compare.err',interrupted_journal_write='annotate_label_grade.py exited1 after label_grade/card save with transient OSError22 opening unchanged plan; finish_label_journal.py finished only that write.',no_ready_science_repeated=True))
write('final_graph_audit.json',dict(passed=True,graph=graph,hypotheses_unchanged=True,catalog_hashes_after_58_tests=before,catalog_unchanged_after_tests=True,original_final_checks_preserved=True,checksums={p.relative_to(OUT).as_posix():sha(p) for p in [OUT/'ИТОГ.md',OUT/'excluded_geometry.json',OUT/'label_grade.json',OUT/'final_checks.json',OUT/'execution_receipt.json',Path(__file__)]}))
print('Final graph refreshed; catalogue tests reused unchanged, all earlier owned workers ended',flush=True)
