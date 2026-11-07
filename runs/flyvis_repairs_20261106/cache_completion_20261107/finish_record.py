"""Record completed cache checks, preserving earlier audit and user records."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib
import importlib.util
import json
import os

ROOT=Path(__file__).resolve().parents[3]
RUN=ROOT/'runs/flyvis_repairs_20261106'
OUT=Path(__file__).resolve().parent
PLAN='dissertation/ЕДИНЫЙ_ПЛАН.md'
CAT='experiments/project_history.json'
GLOSS='experiments/metrics_glossary.json'
sha=lambda raw:hashlib.sha256(raw).hexdigest()

def main():
    reports={name:json.loads((OUT/name/'result.json').read_bytes()) for name in ('heavy_test','remaining_tests')}
    assert all(r['completed'] and r['returncode']==0 for r in reports.values())
    assert '1 passed' in (OUT/'heavy_test/output.log').read_text()
    assert '59 passed, 1 deselected' in (OUT/'remaining_tests/output.log').read_text()
    pins={}
    for source,rows in [
        ('loading_constraint_followup/frozen_sources.json',None),
        ('final_boundary_review/manifest.json','sources')]:
        data=json.loads((RUN/source).read_bytes())
        pairs=data.items() if rows is None else [(r['path'],r['sha256']) for r in data[rows]]
        for name,digest in pairs:
            assert sha((ROOT/name).read_bytes())==digest,name
            pins[name]=digest
    temporal=json.loads((RUN/'temporal_numpy_addendum_v3/manifest.json').read_bytes())
    for row in temporal['changes']:
        assert sha((ROOT/row['path']).read_bytes())==row['after_sha256']
        pins[row['path']]=row['after_sha256']
    for name in ('evaluation/project_history/index.py','tests/test_project_history_cache_transient.py'):
        assert (ROOT/name).stat().st_mtime < (OUT/'heavy_test/started.json').stat().st_mtime
        pins[name]=sha((ROOT/name).read_bytes())
    report=dict(passed=True,tests_total=60,execution='One test then 59 tests in sequential isolated processes; no tests omitted.',
        runs=reports,source_sha256=pins,
        scope='Cache serialization, unchanged live view, real project index, fingerprint invalidation and existing catalogue tests.',
        limits='No claim about the cause of the earlier reboot or the unassigned Python error dialog; old on-disk caches are not rewritten.')
    assert not (OUT/'acceptance.json').exists()
    (OUT/'acceptance.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    now=datetime.now(timezone(timedelta(hours=3)))
    entry=(f'\n\n**{now:%d.%m.%Y, %H:%M} MSK** — Завершение проверки кэша истории с ограничением памяти\n\n'
        '*Получено.* Исправлено сохранение временного списка неотслеживаемых файлов внутри кэша истории: '
        'запись uncommitted исключается до преобразования в словарь, статистика сохранённой истории согласована; '
        'обычное отображение и повторное получение текущего состояния Git сохранены. '
        'Все 58 прежних тестов истории и два новых теста пройдены последовательно: один наиболее затратный отдельно и остальные 59 вместе '
        '(runs/flyvis_repairs_20261106/cache_completion_20261107/acceptance.json). '
        f'Пики памяти групп процессов: {reports["heavy_test"]["peak_job_bytes"]} и {reports["remaining_tests"]["peak_job_bytes"]} байтов; '
        'запрошенное ограничение 512 МиБ на группу. Новые проверки не запускались параллельно.\n\n'
        '*Граница вывода.* Прежние одновременные проверки документации были прерваны после перезагрузки; '
        'последующие ошибки MemoryError и тайм-аут сохранены в memory_incident_20261107. '
        'Причина прежнего исчерпания памяти и присланного окна аварии Python не установлена. '
        'Текущий отдельный тест завершился с кодом 0; другие процессы не объявляются виновниками. '
        'Исправление уменьшает новые кэши; чтение старого большого кэша и полный захват вывода Git остаются отдельными ограничениями.\n\n'
        '*Решение.* По прямому поручению автора завершить проверку и публиковать седьмой пакет только после успешных проверок '
        'его точного состава. Научные выводы, штатные веса и старые записи этим служебным исправлением не меняются.\n').encode('utf-8')
    (OUT/'journal.txt').write_bytes(entry)
    spec=importlib.util.spec_from_file_location('recorder',RUN/'record_boundary_addendum.py')
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    old=RUN/'boundary_ledger_overlay_v1'
    new=RUN/'boundary_ledger_overlay_v2';new.mkdir(exist_ok=False)
    manifest=json.loads((old/'manifest.json').read_bytes())
    for row in manifest['files']:
        name=row['path'];raw=(ROOT/row['staged_candidate']).read_bytes()
        assert sha(raw)==row['after_sha256']
        changed=helper.append_journal(raw,entry) if name==PLAN else raw
        target=new/'after'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(changed)
        row.update(after_sha256=sha(changed),staged_candidate=target.relative_to(ROOT).as_posix())
    for name in (PLAN,CAT,GLOSS):
        target=new/'baseline'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((old/'baseline'/name).read_bytes())
    manifest['cache_completion_journal_sha256']=sha(entry)
    (new/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    path=ROOT/PLAN
    with helper.open_exclusive(path) as stream:
        previous=stream.read();changed=helper.append_journal(previous,entry)
        backup=RUN/'private_ledger_backups/cache_completion_20261107';backup.mkdir(exist_ok=False)
        (backup/'plan_before.md').write_bytes(previous)
        stream.seek(0);stream.write(changed);stream.truncate();stream.flush();os.fsync(stream.fileno())
        stream.seek(0);assert stream.read()==changed
    (OUT/'record_receipt.json').write_text(json.dumps(dict(before_sha256=sha(previous),after_sha256=sha(changed),
        previous_plan_bytes_preserved=True,catalogue_not_modified=True),indent=2),encoding='utf-8')
    print(json.dumps(dict(tests=60,pins=len(pins),overlay='boundary_ledger_overlay_v2')))

if __name__=='__main__':main()
