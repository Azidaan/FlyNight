"""Записать только полный аудированный результат в карточку, С7 и граф."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib,json,psutil,shutil,sys,time
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/flyvis_offset_recovery1_20261104';sys.path.insert(0,str(ROOT))
from scripts.obsidian_system_problems import load,problem_path
from scripts.obsidian_fundamental_overview import build
from scripts.audit_fundamental_obsidian import audit
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def ended(pid,birth):
 try:return abs(psutil.Process(int(pid)).create_time()-float(birth))>=.01
 except psutil.NoSuchProcess:return True
def main():
 chain=json.loads((OUT/'chain_result.json').read_text(encoding='utf-8'));r=json.loads((OUT/'report.json').read_text(encoding='utf-8'));a=json.loads((OUT/'independent_audit.json').read_text(encoding='utf-8'))
 assert chain['complete'] and chain['all_owned_ended'] and all(x['exit_code']==0 and x['all_owned_ended'] for x in chain['stages'])
 assert all(ended(pid,birth) for pid,birth in chain['all_owned'].items()) and a['passed'] and a['report_sha256']==sha(OUT/'report.json')
 for f in ['controller_launch.json','guard_launch.json']:
  x=json.loads((OUT/f).read_text());assert ended(x['pid'],x['created']),(f,x)
 assert len(r['rows'])==128 and a['cases']==128 and r['protocol_sha256']==sha(OUT/'protocol.json')
 def n(v):return 'не определена для всех случаев' if v is None else f'{v:.4f}'.replace('.',',')
 paras=[]
 for start,label in [('still','Неподвижное начало'),('moving','Начало с движением')]:
  for contrast,cname in [(1.,'обычный контраст'),(.25,'слабый контраст')]:
   g=r['groups'][f'evaluation_{start}_{contrast}'];f=g['fast_subtract_calibrated'];z=g['fast_raw_calibrated'];s=g['stock_subtract_calibrated'];null=g['zero']
   paras.append(f'{label}, {cname}: ускоренная версия с вычитанием и калибровкой — ошибка {n(f["epe"])}, без вычитания — {n(z["epe"])}, штатная с вычитанием и калибровкой — {n(s["epe"])}, нулевой выход — {n(null["epe"])}. Знаковая задержка: {n(f["latency_mean"])} и {n(s["latency_mean"])} кадра; определена соответственно в {f["latency_identified"]}/16 и {s["latency_identified"]}/16 случаев.')
 conclusion=('Предварительный составной критерий выполнен.' if r['exploratory_gate'] else 'Предварительный составной критерий не выполнен.')
 limits=('Это синтетический глобальный сдвиг на четырёх калибровочных и четырёх отложенных изображениях из исходного TRAIN BDD; '
  'одна модель FlyVis, не HOTA и не оценка объектов. Отложенные сцены не участвовали в подгонке, но ранее могли использоваться '
  'в других проектах; это не независимый итоговый тест. Смещение по кадрам11–20 считается вслепую, даже если движение уже началось. '
  'Знаковая задержка — первый из трёх последовательных кадров правильного направления, а не время работы процессора. '
  'Польза адаптивной динамики не проверена; проблема С7 остаётся открытой.')
 narrative=' '.join(paras)+' '+conclusion+' '+limits
 doc='# Проверка фонового смещения FlyVis\n\n'+'\n\n'.join(paras)+'\n\n'+conclusion+'\n\n'+limits+'\n\nПротокол `protocol.json`, результаты `report.json`, независимый пересчёт `independent_audit.json`, исходный отказ `../flyvis_offset_20261104/chain_result.json`.\n'
 (OUT/'РЕЗУЛЬТАТ.md').write_text(doc,encoding='utf-8')
 catalog=ROOT/'experiments/project_history.json';shutil.copy2(catalog,OUT/'project_history_before_result.json');d=json.loads(catalog.read_text(encoding='utf-8'));card=next(x for x in d['experiments'] if x['id']=='flyvis_offset_recovery1_v1');assert card['state']=='planned'
 missing=any(k.endswith('latency_complete') and not v for k,v in r['checks'].items());state='completed_positive' if r['exploratory_gate'] else ('inconclusive' if missing else 'completed_negative')
 card.update(state=state,done=datetime.now(ZoneInfo('Europe/Moscow')).date().isoformat(),obtained=narrative+' Источник: runs/flyvis_offset_recovery1_20261104/report.json.',interpretation=conclusion+' Прямой перенос на реальные объекты и полное сопровождение не оценивался.',decision='Проблема С7 остаётся открытой; не считать эту проверку подтверждением пользы трекеру.',cannot_claim=limits)
 card['acceptance'][0].update(met=r['exploratory_gate'],evidence='runs/flyvis_offset_recovery1_20261104/report.json:checks');card['evidence'] += [dict(kind='json',ref='runs/flyvis_offset_recovery1_20261104/report.json',claim='Сравнение всех условий'),dict(kind='json',ref='runs/flyvis_offset_recovery1_20261104/independent_audit.json',claim='Независимый пересчёт')];catalog.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 problem=ROOT/'experiments/system_problems.json';shutil.copy2(problem,OUT/'system_problems_before_result.json');d=json.loads(problem.read_text(encoding='utf-8'));item=next(x for x in d['problems'] if x['id']=='С7')
 paths=['runs/flyvis_offset_recovery1_20261104/'+x for x in ('protocol.json','report.json','independent_audit.json')];item['cases'].append(dict(date=datetime.now(ZoneInfo('Europe/Moscow')).date().isoformat(),fact=narrative,evidence=paths));item['cards'].append('flyvis_offset_recovery1_v1')
 for source in paths:item['evidence'].append(dict(source=source,sha256=sha(ROOT/source),claim='Зарегистрированное сравнение фонового смещения и независимый пересчёт'))
 problem.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 plan=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';raw=plan.read_bytes();mark='## Журнал изменений'.encode();nl=b'\r\n' if mark+b'\r\n' in raw else b'\n';now=datetime.now(ZoneInfo('Europe/Moscow'));entry=f'\n**{now:%d.%m.%Y, %H:%M} MSK** — Проверка фонового смещения FlyVis\n\n{narrative} Источники `{paths[1]}` и `{paths[2]}`.\n\n';assert raw.count(mark+nl)==1;plan.write_bytes(raw.replace(mark+nl,mark+nl+entry.encode(),1))
 vault=ROOT/'ObsidianFlyVis';stage=OUT/'vault_checked';backup=OUT/'vault_before';assert not stage.exists() and not backup.exists();shutil.copytree(vault,stage);shutil.copytree(vault,backup)
 before={f.relative_to(vault).as_posix():sha(f) for f in vault.rglob('*') if f.is_file()};name=problem_path(next(x for x in load()['problems'] if x['id']=='С7'))+'.md';preserve=[x for x in before if x.endswith('.md') and x!=name];assert build(stage,preserve)==0
 for f in ['.obsidian/graph.json','.graph-visible.json','.generated.json']:shutil.copy2(backup/f,stage/f)
 checked=audit(stage);assert checked['note_count']==55 and checked['hypothesis_node_count']==25 and checked['problem_node_count']==10 and not checked['external_hypothesis_edges']
 assert set(before)=={x.relative_to(stage).as_posix() for x in stage.rglob('*') if x.is_file()};changed=[x for x,h in before.items() if sha(stage/x)!=h];assert changed==[name] and all(sha(vault/x)==h for x,h in before.items());shutil.copy2(stage/name,vault/name);done=audit(vault);assert done['hashes']==checked['hashes']
 done.update(changed_files=changed,added_files=[],deleted_files=[],moved_files=[],protected_hypotheses_byte_exact=True);(OUT/'graph_audit.json').write_text(json.dumps(done,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print('recorded',state)
if __name__=='__main__':main()
