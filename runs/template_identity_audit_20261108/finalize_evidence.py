"""Record a post-hoc counterexample from ready scores, without rescoring."""
import hashlib,json,sys
from pathlib import Path
from datetime import datetime,timedelta,timezone
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
from system_memory_guard import guard_entrypoint
guard_entrypoint(timeout=600)
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
    with p.open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')
assert read(OUT/'recovery_checks.json')['passed']
assert not (OUT/'counterexample.json').exists()
rows=read(OUT/'direct_pair_scores.json')['records']
own=next(x for x in rows if x['video']=='00982f76-83a093ef' and x['old']==88 and x['new']==144)
other=next(x for x in rows if x['video']=='00d7268f-fd4487be' and x['old']==73 and x['new']==106)
assert own['direct_kind']=='own' and other['direct_kind']=='foreign'
table={}
for prior in own['ready_scores']:
    for name,scores in own['ready_scores'][prior]['scores'].items():
        table[name]={field:dict(own=value,foreign=other['ready_scores'][prior]['scores'][name][field],own_above_foreign=value>other['ready_scores'][prior]['scores'][name][field]) for field,value in scores.items()}
assert len(own['ready_scores']['background_patch_gate_20261108']['cells'])==7
assert len(other['ready_scores']['background_patch_gate_20261108']['cells'])==6
assert all(not table[name][field]['own_above_foreign'] for name in ['FlyVis','C34'] for field in ['correlation_max','filter_max'])
for row in [own,other]:
    for path,digest in row['sources'].items():assert sha(ROOT/path)==digest,path
save(OUT/'counterexample.json',dict(passed=True,own_pair=[own['video'],88,144],foreign_pair=[other['video'],73,106],own_cells=7,foreign_cells=6,ready_scores=table,
    mathematical_result='If a>=tau accepts an own example with score a and foreign score b>a, then b>=tau also accepts the foreign example. Thus no monotone high-score threshold correctly accepts this own and rejects this foreign pair.',
    limitations='Post-hoc descriptive counterexample, not an independent test or estimated general error rate. Two videos, different scenes and geometry; near-equal support is not a matched physical-scene control. Direct box identity does not imply uncontaminated neural receptive fields. No threshold selected and no claim that all information is absent from the maps.',
    sources={name:sha(OUT/name) for name in ['results.json','direct_pair_scores.json']}))
fact='При прямом подтверждении обеих рамок своя88→144(00982f76-83a093ef,7ячеек) имеет готовое сходство8T4/T5 0,151713, чужая73→106(00d7268f-fd4487be,6ячеек) —0,341247; для34каналов0,419629и0,481309. Фоновый фильтр сохраняет неверный порядок. Поэтому монотонный порог по этим оценкам не может одновременно принять свою и отклонить чужую. Это выбранный после просмотра контрпример, не новая независимая оценка качества; разные сцены и геометрия сохраняются как ограничение. Источник `runs/'+OUT.name+'/counterexample.json`.'
path=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';text=path.read_text(encoding='utf8');now=datetime.now(timezone(timedelta(hours=3)))
entry=f'**{now:%d.%m.%Y, %H:%M} MSK** — Неверный порядок сходства сохраняется при прямых опорах личности.\n\n*Получено (`{OUT.name}`).* {fact}\n\n*Вывод.* Загрязнение прошлой рамки не объясняет этот контрпример; принадлежность отдельных нейронных сигналов объекту этим не установлена. Математическое ограничение относится к конкретному скалярному сходству, а не ко всем способам чтения карт FlyVis.\n\n*Решение.* Не лечить этот считыватель подбором порога или объявлять ядро неисправным. При следующей замене сохранять отдельные пространственные свидетельства прошлой и текущей цели и проверять принадлежность, затем судьбуID в полном трекере. Готовые оценки не пересчитаны.\n\n'
path.write_text(text.replace('## Журнал изменений\n','## Журнал изменений\n\n'+entry,1),encoding='utf8')
import shutil,tempfile
from scripts.build_obsidian_vault import main
from scripts.audit_obsidian_levels import audit
vault=ROOT/'ObsidianFlyVis';hyp={p.name:p.read_bytes() for p in (vault/'Карта проекта/Гипотезы').glob('*.md')}
def build(v):sys.argv=['scripts/build_obsidian_vault.py','--vault',str(v)];assert main()==0
with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as temp:
    preview=Path(temp)/'ObsidianFlyVis';shutil.copytree(vault,preview);build(preview);g=audit(preview);build(vault);assert audit(vault)==g
assert hyp=={p.name:p.read_bytes() for p in (vault/'Карта проекта/Гипотезы').glob('*.md')}
g['hypotheses_unchanged']=True;save(OUT/'final_graph_audit.json',g)
for path,digest in read(OUT/'protocol.json')['sources'].items():assert sha(ROOT/path)==digest,path
for path,digest in read(OUT/'execution_receipt.json')['checksums'].items():assert sha(OUT/path)==digest,path
save(OUT/'final_checks.json',dict(passed=True,scientific_scores_reused=True,metadata_recovery_passed=True,graph_passed=True,checksums={name:sha(OUT/name) for name in ['finalize_evidence.py','counterexample.json','final_graph_audit.json','recovery_checks.json','results.json','summary.json','УТОЧНЕНИЕ.md']}))
print('Final evidence and graph verified')
