"""Сохранить предварительный отрицательный контроль и обновить существующую С7."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib,json,shutil,sys
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/flyvis_tau_gain_loso_20261104';sys.path.insert(0,str(ROOT))
from scripts.obsidian_system_problems import load,problem_path
from scripts.obsidian_fundamental_overview import build
from scripts.audit_fundamental_obsidian import audit
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 r=json.loads((OUT/'report.json').read_text(encoding='utf-8'));a=json.loads((OUT/'independent_audit.json').read_text(encoding='utf-8'))
 assert r['complete'] and not r['preliminary_gate'] and a['passed'] and a['report_sha256']==sha(OUT/'report.json')
 g=r['groups']['reverse_1.0'];w=r['groups']['reverse_0.25']
 assert g['fast_calibrated']['epe']>g['stock_calibrated']['epe'] and w['fast_calibrated']['epe']>w['stock_calibrated']['epe']
 fact=('Перекрёстная по четырём ранее изученным исходным изображениям калибровка одним коэффициентом для каждой версии '
  'не выполнила предварительный критерий: на обычном контрасте средняя номинальная ошибка ускоренной версии '
  'после калибровки 1,9841 против 1,9182 у калиброванной штатной; при слабом контрасте 2,4621 против 2,3942. '
  'При обычном контрасте средняя задержка сохранилась 1,0625 против 2,0625 кадра. '
  'При слабом контрасте временной шум неподвижного выхода ускоренной версии после калибровки составил '
  '0,000487 против 0,000298 у калиброванной штатной; средний модуль неподвижного выхода ускоренной версии '
  'вырос после калибровки с 0,2838 до 0,5257. Это описательная проба, не независимый финальный тест.')
 doc=('# Проверка компенсации масштаба FlyVis\n\n'+fact+'\n\n'
  'Для каждого из четырёх изображений коэффициент каждой версии рассчитывался по другим трём методом '
  'наименьших квадратов по обоим контрастам, скоростям, направлениям и событиям. Метки заданы '
  'известным искусственным сдвигом и использовались только при подгонке и оценке; сеть не обучалась '
  'и повторно не запускалась. Ошибка оценивалась усреднённым расстоянием между векторами; '
  'целевая функция подгонки была квадратичной, поэтому её снижение не гарантирует уменьшения этой ошибки.\n\n'
  'Результат не исключает другие способы калибровки или адаптивную динамику и не показывает пользу '
  'для трекера. Все изображения уже использовались при предыдущем анализе; перенос на новые сцены не проверен.\n\n'
  'Источники: `protocol.json`, `report.json`, `independent_audit.json`.\n')
 (OUT/'РЕЗУЛЬТАТ.md').write_text(doc,encoding='utf-8')
 catalog=ROOT/'experiments/project_history.json';data=json.loads(catalog.read_text(encoding='utf-8'));c=next(x for x in data['experiments'] if x['id']=='flyvis_tau_gain_loso_v1');assert c['state']=='planned'
 c.update(state='completed_negative',done='2026-11-04',obtained=fact+' Источник: runs/flyvis_tau_gain_loso_20261104/report.json.',
  interpretation='Данная односкалярная калибровка не прошла заранее зарегистрированный предварительный критерий. Результат не показывает, что любая калибровка невозможна.',
  decision='Сохранить исследовательский результат; не считать С7 решённой и не внедрять эту калибровку как готовое решение.',
  cannot_claim='Ранее изученные четыре изображения не являются независимым итоговым тестом; не измерены реальные траектории, HOTA и генерализация. Заданы синтетические цели.')
 c['acceptance'][0].update(met=False,evidence='runs/flyvis_tau_gain_loso_20261104/report.json:checks')
 c['evidence'] += [dict(kind='json',ref='runs/flyvis_tau_gain_loso_20261104/report.json',claim='Полные результаты по четырём разбиениям'),dict(kind='json',ref='runs/flyvis_tau_gain_loso_20261104/independent_audit.json',claim='Независимый пересчёт коэффициентов и метрик')]
 catalog.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 problems=ROOT/'experiments/system_problems.json';shutil.copy2(problems,OUT/'system_problems_before.json');data=json.loads(problems.read_text(encoding='utf-8'));p=next(x for x in data['problems'] if x['id']=='С7')
 sources=['runs/flyvis_tau_gain_loso_20261104/'+x for x in ['protocol.json','report.json','independent_audit.json']]
 p['cases'].append(dict(date='2026-11-04',fact=fact,evidence=sources));p['cards'].append('flyvis_tau_gain_loso_v1')
 for source in sources:p['evidence'].append(dict(source=source,sha256=sha(ROOT/source),claim='Протокол, предварительный результат или его независимый пересчёт'))
 problems.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 plan=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';raw=plan.read_bytes();mark='## Журнал изменений'.encode();nl=b'\r\n' if mark+b'\r\n' in raw else b'\n';now=datetime.now(ZoneInfo('Europe/Moscow'))
 entry=f'\n**{now:%d.%m.%Y, %H:%M} MSK** — Предварительная перекрёстная калибровка FlyVis\n\n{fact} Источник `{sources[1]}`, независимый пересчёт `{sources[2]}`, дорасчётный протокол `{sources[0]}`. Не доказывает невозможность других способов; проблема С7 остаётся открытой.\n\n';assert raw.count(mark+nl)==1;plan.write_bytes(raw.replace(mark+nl,mark+nl+entry.encode(),1))
 vault=ROOT/'ObsidianFlyVis';stage=OUT/'vault_checked';backup=OUT/'vault_before';assert not stage.exists() and not backup.exists();shutil.copytree(vault,stage);shutil.copytree(vault,backup)
 before={f.relative_to(vault).as_posix():sha(f) for f in vault.rglob('*') if f.is_file()};target=problem_path(next(x for x in load()['problems'] if x['id']=='С7'))+'.md';preserve=[x for x in before if x.endswith('.md') and x!=target]
 assert build(stage,preserve)==0
 for f in ['.obsidian/graph.json','.graph-visible.json','.generated.json']:shutil.copy2(backup/f,stage/f)
 checked=audit(stage);assert checked['note_count']==55 and checked['hypothesis_node_count']==25 and checked['problem_node_count']==10 and not checked['external_hypothesis_edges']
 assert set(before)=={f.relative_to(stage).as_posix() for f in stage.rglob('*') if f.is_file()};changed=[f for f,h in before.items() if sha(stage/f)!=h];assert changed==[target],changed
 assert all(sha(vault/f)==h for f,h in before.items());shutil.copy2(stage/target,vault/target);done=audit(vault);assert done['hashes']==checked['hashes']
 done.update(changed_files=changed,added_files=[],deleted_files=[],moved_files=[],protected_hypotheses_byte_exact=True)
 (OUT/'graph_audit.json').write_text(json.dumps(done,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print('recorded and graph checked')
if __name__=='__main__':main()
