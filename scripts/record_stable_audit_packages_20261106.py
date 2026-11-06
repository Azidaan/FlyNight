"""Append verified publication and explicitly withdraw automatic geometry promotion."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib
import json
import re

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/flyvis_repairs_20261106'


def main():
    first=json.loads((OUT/'first_stable_remote_verification.json').read_text(encoding='utf-8'))
    second=json.loads((OUT/'second_stable_remote_verification.json').read_text(encoding='utf-8'))
    assert second['verified'] and second['files_verified']==241
    assert first.get('verified')
    validation=json.loads((OUT/'second_stable_package_final/validation.json').read_text(encoding='utf-8'))
    assert validation['passed']
    assert '403 passed' in (OUT/'second_stable_package_final/shared_tests.log').read_text()
    assert '398 passed, 5 skipped' in (OUT/'second_stable_package_final/isolated_tests.log').read_text()
    roi=json.loads((OUT/'coordinate_only_roi_review/report.json').read_text(encoding='utf-8'))
    now=datetime.now(timezone(timedelta(hours=3)))
    card=dict(id='flyvis_system_integrity_stable_packages_20261106',
        title='Авторская система — проверенные защиты исполнения и отказ от неподтверждённой геометрии',
        area='infra',state='infrastructure',started='2026-11-06',done='2026-11-06',
        question='Какие исправления подтверждённых технических дефектов можно сохранить без автоматического изменения откалиброванного поведения?',
        hypothesis='Техническая проверка сохранности исполнения; гипотезы о качестве трекинга и новых весах не проверяются.',
        expected='Раздельно проверить воспроизведение сбоя, обычный путь и зависимых потребителей; сохранить только проверенные файлы и отрицательные свидетельства.',
        acceptance=[dict(criterion='Первый и второй пакеты прошли совместные проверки и проверки в отдельной копии репозитория с точным списком файлов. Критерий записан после технической проверки, не является предварительной регистрацией научного опыта.',met=True,evidence='runs/flyvis_repairs_20261106/first_stable_package/isolated_validation.json; runs/flyvis_repairs_20261106/second_stable_package_final/validation.json'),
                    dict(criterion='Автоматическое включение геометрических кандидатов отклонено после провала заранее записанной проверки координат и пересчёта калибровки.',met=True,evidence='runs/flyvis_repairs_20261106/coordinate_only_roi/protocol_before_run.json; runs/flyvis_repairs_20261106/coordinate_only_roi_review/report.json'),
                    dict(criterion='Опубликованные GitLab-файлы совпадают с соответствующими локальными коммитами по Git blob ID.',met=True,evidence='runs/flyvis_repairs_20261106/first_stable_remote_verification.json; runs/flyvis_repairs_20261106/second_stable_remote_verification.json')],
        obtained='Первый пакет: 99 проверок в рабочем проекте; в отдельной копии 98 пройдено, одна пропущена из-за отсутствующего локального FlyVis runtime (runs/flyvis_repairs_20261106/stable_loading_cache_tests.log; first_stable_package/isolated_validation.json в том же каталоге). Второй пакет: 403 проверки в рабочем проекте, 398 пройдено и пять пропущено в отдельной копии из-за отсутствия локальных кэшей (runs/flyvis_repairs_20261106/second_stable_package_final/shared_tests.log и isolated_tests.log). Независимый пересчёт 3043 интервалов одного сохранённого видео подтвердил изменение средней ошибки конечной точки с 48,20591598764831 до 48,27365642772315 пикселя при изменении только координат (runs/flyvis_repairs_20261106/coordinate_only_roi_review/report.json). Точные прежние координаты и векторы обычного режима подтверждены на настоящей модели CPU (runs/flyvis_system_fixes_20261106/calibration_guard/final_default_report.json).',
        interpretation='Подтверждены защитные исправления загрузки, происхождения кэшей, восстановления контрольной точки, порядка отсева пар при назначении, сохранности Retina и классического движения, сброса производных состояний и общей блокировки GPU. Прибавка качества трекинга не измерялась. Правильность формулы геометрического преобразования не доказала безопасность её автоматического применения к обученному и откалиброванному модулю FlyVis.',
        decision='В обычных конструкторах и сценариях исполнения оставлен legacy_v1 с прежними точками и векторами. Два новых геометрических варианта доступны только явно и не объявлены улучшением. Локальные коммиты e8f674cec565632dbebfa187c8610a3eb38ef729 и 7636309c73729a8f6cd3ade7f6c18e39037a2146 сохранены; их точное содержимое опубликовано отдельными API-коммитами e060d8539accdb934939b92a1e29d2246795858b и c88863e76780f340fc1dc3dca87e53d14ff5a4dc в ветке codex/system-integrity-20261106-e8f674ce. Главная удалённая ветка осталась на 9626126d90390740e052c331cc07337a9e85620e. Проверка последующих исправлений состояния FlyVis, записи HDF5 и временного пути продолжается отдельно.',
        cannot_claim='Нет новой оценки HOTA, обучения научной модели, новой калибровки или доказательства отсутствия всех возможных ошибок. Небольшое ухудшение координатного кандидата на одном видео не доказывает его вредность во всех условиях. Объединённые имена GPU-блокировки действуют для процессов с новым кодом; ранее запущенные процессы сохраняют прежние объекты. Идентификаторы локальных и API-коммитов различаются, совпадение относится к проверенным файлам.',
        metrics=[],evidence=[dict(kind='json',ref=path,claim=claim) for path,claim in [
            ('runs/flyvis_repairs_20261106/first_stable_remote_verification.json','Точное содержимое первого опубликованного пакета'),
            ('runs/flyvis_repairs_20261106/second_stable_remote_verification.json','Точное содержимое второго опубликованного пакета'),
            ('runs/flyvis_repairs_20261106/second_stable_package_final/validation.json','Объединённая и изолированная проверки'),
            ('runs/flyvis_repairs_20261106/coordinate_only_roi_review/report.json','Независимое контрдоказательство автоматическому включению координат'),
            ('runs/flyvis_system_fixes_20261106/calibration_guard/final_default_report.json','Совпадение обычного пути с исходным преобразователем')]])
    cat=ROOT/'experiments/project_history.json'; before=cat.read_bytes(); text=before.decode('utf-8'); data=json.loads(text)
    assert not any(c['id']==card['id'] for c in data['experiments'])
    start=re.search(r'"experiments"\s*:\s*\[',text).end()-1
    _,end=json.JSONDecoder().raw_decode(text,start)
    offset=len(text[:end-1].rstrip())
    insertion=',\n'+'\n'.join('    '+line for line in json.dumps(card,ensure_ascii=False,indent=2).splitlines())+'\n  '
    updated=text[:offset]+insertion+text[offset:]
    assert json.loads(updated)==dict(data,experiments=[*data['experiments'],card])
    entry=f'''\n\n**{now:%d.%m.%Y, %H:%M} MSK** — Два проверенных пакета и окончательное решение о геометрии

*Постановка.* Сохранить воспроизводимые исправления без присвоения техническим тестам научного результата. Проверить собственное предложение о геометрии до автоматического включения; сохранить прежние пользовательские изменения отдельно.

*Получено (`{card['id']}`).* {card['obtained']}

*Вывод.* {card['interpretation']} {card['cannot_claim']}

*Решение.* {card['decision']} HTTPS git push вернул 403; публикация выполнена через уже подключённый GitLab. Слияния, переписывания истории и изменения прежних пользовательских файлов в коммитах не было. Старый пустой index.lock от 26.10.2026 после проверки отсутствия Git-процесса сохранён как `runs/flyvis_repairs_20261106/git_index_lock_stale_20261026.bak`; файл не удалялся. Проверки научных гипотез и граф Obsidian не изменены.
'''.encode('utf-8')
    plan=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md'; old=plan.read_bytes(); marker='## Журнал изменений'.encode()
    assert old.count(marker)==1
    new=old.replace(marker,marker+entry,1); assert new.replace(marker+entry,marker,1)==old
    (OUT/'stable_packages_card.json').write_text(json.dumps(card,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'stable_packages_journal.txt').write_bytes(entry)
    cat.write_bytes(updated.encode('utf-8')); plan.write_bytes(new)
    sha=lambda value:hashlib.sha256(value).hexdigest()
    receipt=dict(at=now.isoformat(),other_cards_unchanged=True,previous_plan_bytes_preserved=True,
        before=dict(catalogue=sha(before),plan=sha(old)),after=dict(catalogue=sha(cat.read_bytes()),plan=sha(new)))
    (OUT/'stable_packages_record_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(receipt))


if __name__=='__main__':main()
