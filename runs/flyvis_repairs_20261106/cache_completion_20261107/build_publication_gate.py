"""Require completed sequential checks and the exact final ledger/source bytes."""
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parents[3]
RUN=ROOT/'runs/flyvis_repairs_20261106'
OUT=Path(__file__).resolve().parent
sha=lambda raw:hashlib.sha256(raw).hexdigest()

def main():
    acceptance=json.loads((OUT/'acceptance.json').read_bytes())
    assert acceptance['passed'] and acceptance['tests_total']==60
    for name,digest in acceptance['source_sha256'].items():
        assert sha((ROOT/name).read_bytes())==digest,name
    overlay=RUN/'boundary_ledger_overlay_v2'
    ledger=json.loads((overlay/'manifest.json').read_bytes())
    hashes={r['path']:r['after_sha256'] for r in ledger['files']}
    outcomes={}
    cases=[('publication_validator','validator','main',None),
           ('publication_tests_main_stable','tests','main','59 passed, 1 deselected'),
           ('publication_tests_cache','tests','cache','1 passed, 59 deselected')]
    for directory,mode,group,phrase in cases:
        data=json.loads((OUT/directory/'result.json').read_bytes())
        assert data['completed'] and data['returncode']==0 and not data.get('timed_out')
        assert data['hard_job_memory_bytes']==512*1024*1024
        text=(OUT/directory/'output.log').read_text(encoding='utf-8')
        assert (phrase in text) if phrase else ('OK: каталог согласован с источниками' in text)
        suffix='_stable' if mode=='tests' else ''
        checked=json.loads((overlay/f'validation_after_{mode}_{group}{suffix}.json').read_bytes())
        assert checked['exit_code']==0
        assert (checked['publication_candidates_unchanged'] if mode=='tests' else checked['production_ledger_and_glossary_unchanged'])
        actual={k.replace('\\','/'):v for k,v in checked['input_sha256'].items()}
        assert all(actual[name]==digest for name,digest in hashes.items())
        outcomes[directory]=data
    for row in ledger['files']:
        assert sha((ROOT/row['staged_candidate']).read_bytes())==row['after_sha256']
    report=dict(passed=True,working_tests=60,publication_tests=60,
        execution='Sequential isolated processes: 59 tests and the single memory-intensive test; complete collected test set covered.',
        source_sha256=acceptance['source_sha256'],ledger_sha256=hashes,outcomes=outcomes,
        old_failures_preserved=True,scientific_weights_changed=False)
    assert not (OUT/'publication_gate.json').exists()
    (OUT/'publication_gate.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    peak=max(r['peak_job_bytes'] for r in [*acceptance['runs'].values(),*outcomes.values()])
    final=('Проверка кэша истории и допуск седьмого пакета, 07.11.2026\n\n'
        'Исправлено сохранение временных путей рабочего дерева в кэше истории. Постоянные события сохраняются, '
        'статистика кэша соответствует сохранённым событиям, обычное отображение и восстановление актуального состояния Git проверены. '
        'Адресное воспроизведение до исправления: один отказ и одна успешная проверка; после исправления обе проверки прошли '
        '(memory_incident_20261107/cache_before_v2 и cache_after).\n\n'
        'Все 58 прежних тестов истории и два новых теста пройдены для рабочего проекта и для точного представления документов публикации. '
        'Наиболее затратный тест выполнялся отдельным процессом, остальные 59 вместе; пропусков нет, выбранные группы не пересекаются. '
        'Валидатор публикации завершился успешно; два прежних предупреждения о шаблонных путях старой карточки сохранены. '
        'Источник: acceptance.json и publication_gate.json с адресами отдельных журналов.\n\n'
        f'Все новые проверки запускались последовательно с запросом ограничения Windows Job 512 МиБ. '
        f'Максимальный измеренный пик группы среди этих успешных проверок — {peak} байтов. '
        'Предыдущие MemoryError, тайм-ауты и прерванные запуски сохранены отдельно; они не переименованы в успешные.\n\n'
        'Присланное окно аварии Python не удалось сопоставить с PID. Отдельный тест кэша завершился штатно, код возврата 0; '
        'причина окна и прежней перезагрузки остаётся неустановленной. Это исправление не гарантирует ограничение памяти '
        'при первом чтении прежних больших кэшей; полный захват вывода Git также остаётся отдельным риском.\n\n'
        'Исходники остальных исправлений седьмого пакета совпали с их проверенными контрольными суммами. '
        'Сохранены результаты совместного прогона 822 passed, 2 skipped до изменения кэша истории; новый кэш проверен отдельно '
        'описанными тестами. Обучение, HOTA и изменение штатных весов не выполнялись.\n')
    (OUT/'FINAL_REPORT.txt').write_text(final,encoding='utf-8')
    print(json.dumps(dict(passed=True,tests=60,max_peak_job_bytes=peak)))

if __name__=='__main__':main()
