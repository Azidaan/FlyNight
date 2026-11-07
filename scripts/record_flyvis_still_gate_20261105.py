"""Сохранить ограниченный пилот и использованные первоисточники в С7."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib,json,shutil,sys
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/flyvis_still_gate_20261105';sys.path.insert(0,str(ROOT))
from scripts.obsidian_system_problems import load,problem_path
from scripts.obsidian_fundamental_overview import build
from scripts.audit_fundamental_obsidian import audit
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'));r=json.loads((OUT/'report.json').read_text(encoding='utf-8'));a=json.loads((OUT/'independent_audit.json').read_text(encoding='utf-8'));o=json.loads((OUT/'observability_check.json').read_text(encoding='utf-8'))
 assert r['complete'] and not r['exploratory_gate'] and a['passed'] and a['report_sha256']==sha(OUT/'report.json') and o['passed']
 assert all(r['groups'][f'evaluation_still_{ct}']['gate_true']==16 and r['groups'][f'evaluation_moving_{ct}']['gate_true']==0 for ct in [1.,.25])
 assert r['gate_stress_only']['stationary_cases']==32 and r['gate_stress_only']['single_pixel_perturbation_updates']==0
 text=('На ранее изученных синтетических последовательностях точное совпадение 10 переходов кадров разрешило вычитание '
  'во всех 32 условиях с неподвижным началом и не разрешило его ни в одном из 32 условий с движением на четырёх оценочных изображениях. '
  'При обычном контрасте средняя ошибка ускоренной версии с условным правилом составила 0,9194 при неподвижном начале '
  'и 2,0887 при начале с движением; она совпала соответственно с безусловным вычитанием и отсутствием вычитания. '
  'При слабом контрасте и движущемся начале условный отказ дал 2,4512 против 2,2960 у безусловного вычитания, поэтому '
  'предварительный составной критерий не выполнен. Один изменённый пиксель в одном из начальных кадров привёл '
  'к отказу от коррекции во всех 32 отдельных пробах только признака неподвижности. На однотонной синтетической сцене '
  'одинаковые входные кадры соответствовали как покою, так и заданному движению; оба раза признак разрешил вычитание. '
  'Частота таких ошибок на реальных видео и устойчивость полной сети к шуму не измерены.')
 lit=('# Использованные источники для проверки смещения FlyVis\n\n'
  '[Toyama и соавторы, Wallflower: Principles and Practice of Background Maintenance, 1999](https://www.microsoft.com/en-us/research/publication/wallflower-principles-and-practice-of-background-maintenance/), DOI 10.1109/ICCV.1999.791228. Авторы разбирают инициализацию и условное обновление модели фона; это помогло поставить вопрос о ложном принятии движения за фон. Наше условие полного совпадения кадров — отдельный упрощённый контроль, не реализация Wallflower. В статье условие 1,1 относится к предыдущей ошибке прогноза, а не к установленному в присланном тексте глобальному минимуму.\n\n'
  '[Weiss, Simoncelli, Adelson, Motion illusions as optimal percepts, 2002](https://www.nature.com/articles/nn858), DOI 10.1038/nn0602-858. Работа рассматривает неопределённость оценки движения и априорное предпочтение малой скорости; она помогла включить различение неподвижности и невидимого движения как ограничение. Наш однотонный пример — собственная синтетическая иллюстрация, не эксперимент этой статьи.\n\n'
  'Обе работы использованы для постановки и ограничения вывода; ни одна не доказывает, что проблема С7 решена.\n')
 (OUT/'ИСТОЧНИКИ.md').write_text(lit,encoding='utf-8')
 doc=('# Условное вычитание смещения FlyVis\n\n'+text+'\n\n'
  'Для сравнения в тех же условиях причинная фазовая корреляция двух изображений без FlyVis дала среднюю ошибку 0,0119 '
  'при неподвижном начале и обычном контрасте, и 0,0135 при движущемся начале и обычном контрасте. '
  'Это благоприятная для метода задача глобального целочисленного сдвига, не сравнение реальных трекеров.\n\n'
  'Все четыре оценочные сцены уже анализировались раньше; до регистрации дополнительно были просмотрены пять оценочных '
  'переходов фазовой корреляции для проверки знака. Это разведочный повтор, не независимый итоговый тест. '
  'Штатный прогрев состояния FlyVis на первом кадре уже существует, и отдельно проверено вычитание его выхода; '
  'проверенное условное правило не меняет нейронное ядро. С7 остаётся открытой; HOTA и реальные объекты не оценивались.\n\n'
  'Источники: `protocol.json`, `report.json`, `independent_audit.json`, `observability_check.json`, `ИСТОЧНИКИ.md`.\n')
 (OUT/'РЕЗУЛЬТАТ.md').write_text(doc,encoding='utf-8')
 cat=ROOT/'experiments/project_history.json';d=json.loads(cat.read_text(encoding='utf-8'));c=next(x for x in d['experiments'] if x['id']=='flyvis_still_gate_v1');assert c['state']=='planned'
 c.update(state='completed_negative',done=datetime.now(ZoneInfo('Europe/Moscow')).date().isoformat(),obtained=text+' Источник: runs/flyvis_still_gate_20261105/report.json.',
  interpretation='Чистые наблюдаемые сдвиги различаются, но правило не проходит заранее заданный составной критерий и не устойчиво даже к одному изменённому пикселю; однотонная сцена не позволяет определить физическое движение по кадрам.',
  decision='Не считать правило решением С7; не переносить вывод на реальные видео. Использовать найденные ограничения при постановке будущего более устойчивого признака.',
  cannot_claim='Нет независимого финального теста, сенсорного шума во всей FlyVis, движения реальных объектов, HOTA или доказательства пользы ядра; phaseCorrelate работает в благоприятной синтетической постановке.')
 c['acceptance'][0].update(met=False,evidence='runs/flyvis_still_gate_20261105/report.json:checks');c['evidence'] += [dict(kind='json',ref='runs/flyvis_still_gate_20261105/report.json',claim='Условия и количественные результаты'),dict(kind='json',ref='runs/flyvis_still_gate_20261105/independent_audit.json',claim='Независимый пересчёт'),dict(kind='json',ref='runs/flyvis_still_gate_20261105/observability_check.json',claim='Неразличимость однотонных входов')]
 cat.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 prob=ROOT/'experiments/system_problems.json';shutil.copy2(prob,OUT/'system_problems_before.json');d=json.loads(prob.read_text(encoding='utf-8'));s=next(x for x in d['problems'] if x['id']=='С7')
 refs=['runs/flyvis_still_gate_20261105/'+f for f in ('protocol.json','report.json','independent_audit.json','observability_check.json','ИСТОЧНИКИ.md')]
 source_text=' Источник постановки: [Toyama и соавторы, Wallflower, 1999](https://www.microsoft.com/en-us/research/publication/wallflower-principles-and-practice-of-background-maintenance/) — условное обновление фона; наше правило не является Wallflower. Ограничение наблюдаемости согласуется с [Weiss и соавторами, 2002](https://www.nature.com/articles/nn858); статьи не доказывают решения С7.'
 s['cases'].append(dict(date=datetime.now(ZoneInfo('Europe/Moscow')).date().isoformat(),fact=text+source_text,evidence=refs));s['cards'].append('flyvis_still_gate_v1')
 for path in refs:s['evidence'].append(dict(source=path,sha256=sha(ROOT/path),claim='Условное вычитание: протокол, пилот, аудит, ограничение или литературное основание'))
 prob.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 plan=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';raw=plan.read_bytes();marker='## Журнал изменений'.encode();nl=b'\r\n' if marker+b'\r\n' in raw else b'\n';now=datetime.now(ZoneInfo('Europe/Moscow'));entry=f'\n**{now:%d.%m.%Y, %H:%M} MSK** — Исследовательская проверка допуска вычитания смещения FlyVis\n\n{text} Источники `{refs[1]}`, `{refs[2]}`, `{refs[3]}`, литературный разбор `{refs[4]}`. Проверено по сохранённым выходам; сцены и пять оценочных переходов ранее изучены, HOTA нет; С7 открыта.\n\n';assert raw.count(marker+nl)==1;plan.write_bytes(raw.replace(marker+nl,marker+nl+entry.encode(),1))
 road=ROOT/'FLYVIS_TRACKER_RESEARCH_ROADMAP.md'
 with road.open('a',encoding='utf-8') as f:f.write('\n\n## 05.11.2026 — условное вычитание смещения FlyVis\n\n'+text+' Источник: `runs/flyvis_still_gate_20261105/report.json`. С7 открыта.\n')
 vault=ROOT/'ObsidianFlyVis';stage=OUT/'vault_checked';backup=OUT/'vault_before';assert not stage.exists() and not backup.exists();shutil.copytree(vault,stage);shutil.copytree(vault,backup)
 before={f.relative_to(vault).as_posix():sha(f) for f in vault.rglob('*') if f.is_file()};name=problem_path(next(x for x in load()['problems'] if x['id']=='С7'))+'.md';preserve=[x for x in before if x.endswith('.md') and x!=name];assert build(stage,preserve)==0
 for f in ['.obsidian/graph.json','.graph-visible.json','.generated.json']:shutil.copy2(backup/f,stage/f)
 checked=audit(stage);assert checked['note_count']==55 and checked['hypothesis_node_count']==25 and checked['problem_node_count']==10 and not checked['external_hypothesis_edges']
 assert set(before)=={f.relative_to(stage).as_posix() for f in stage.rglob('*') if f.is_file()};changed=[f for f,h in before.items() if sha(stage/f)!=h];assert changed==[name] and all(sha(vault/f)==h for f,h in before.items());shutil.copy2(stage/name,vault/name);done=audit(vault);assert done['hashes']==checked['hashes'];done.update(changed_files=changed,added_files=[],deleted_files=[],protected_hypotheses_byte_exact=True)
 (OUT/'graph_audit.json').write_text(json.dumps(done,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print('result and sources saved')
if __name__=='__main__':main()
