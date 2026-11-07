"""Записать завершённую уборку в отдельный журнал, не создавая узел журнала."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
RUN=ROOT/'runs/b123_20261027'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists();p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
    audit=read(RUN/'graph_fundamental_final/audit.json');assert audit['passed']
    archives=[]
    for folder in ('graph_fundamental_rebuild','graph_fundamental_rebuild_recovery1','graph_fundamental_final'):
        base=RUN/folder/'archived-notes'
        for path in base.rglob('*'):
            if path.is_file():archives.append(dict(old_path=path.relative_to(base).as_posix(),archive=path.relative_to(ROOT).as_posix(),sha256=sha(path)))
    write(RUN/'graph_fundamental_final/archive_inventory.json',dict(files=archives,count=len(archives),
        full_pre_migration_snapshot='runs/b123_20261027/graph_fundamental_rebuild/vault-before',
        scientific_runs_deleted=False))
    protocol=dict(id='busca_checkpoint_inventory',created=datetime.now(timezone.utc).isoformat(),budget_seconds=120,
        author_stop='2026-10-27T07:20:00+03:00',device='cpu',
        checkpoint='weights/busca_official_20261027/model_busca.pth',
        checkpoint_sha256='ed294bfec54e87d84a0d1dbdaf9630d218c59311c1e3f1fd4782e32dc3d478f1',
        purpose='Проверить наличие весов извлекателя внутри полного checkpoint, прежде чем повторять загрузку. Только безопасное чтение тензоров на CPU; без модели, обучения и HOTA.')
    assert sha(ROOT/protocol['checkpoint'])==protocol['checkpoint_sha256']
    folder=RUN/'busca_checkpoint_inventory';write(folder/'protocol.json',protocol)
    source='scripts/b123_busca_checkpoint_inventory_20261027.py'
    write(RUN/'ready_jobs/80_busca_checkpoint_inventory.json',dict(title='BUSCA — проверить полноту полученных авторских весов без GPU',
        folder=folder.relative_to(ROOT).as_posix(),script=source,script_sha256=sha(ROOT/source),protocol_sha256=sha(folder/'protocol.json')))
    from scripts import z13_localization_labels_20261023 as J
    J.IDENT='obsidian_fundamental_overhaul'
    J.journal('Obsidian перестроен по задачам; журнал и технические кружки исключены',
        '*Решение автора.* В Obsidian нужны только фундаментальные записи: задача → содержательная попытка → итог → следующее решение. '
        'Подробная хронология остаётся в этом Markdown-файле, без копии журнала в хранилище. '
        'Разрешено дерево «Гипотезы → Б1–Б3 → подходы», с невидимыми переходами в другие группы. '
        'Правило записано в AGENTS.md и краткую инструкцию Obsidian.\n\n'
        '*Сделано и проверено.* `runs/b123_20261027/graph_fundamental_final/audit.json`:23заметки,10узлов/9рёбер в дереве гипотез,3луча центра,0внешних рёбер,0потерянных целей,0записей журнала,0canvas. '
        'Пять опорных задач агрегируют несколько попыток; Б1–Б3 сохраняют исходный научный текст, добавлены только дочерние подходы BUSCA/LTR. '
        'Матрицы с двойными скобками раньше ошибочно создавали ссылки; числовые пустые заметки перемещены в архив. '
        'Старая «Карта мухи.canvas» содержала209элементов и старую легенду цветов; сохранена в архиве и не пересоздаётся. '
        'Архив и хеши:graph_fundamental_final/archive_inventory.json; первичные runs и научные состояния не изменены.\n\n'
        '*Собственная ошибка миграции.* Первая сборка была применена к открытому хранилищу до полного испытания и остановилась на сохранности CR/LF. '
        'Сохранён её вывод и снимок до изменений; исходные тексты гипотез восстановлены и проверены, запись теперь побайтная. '
        'Повторная сборка на копии и регрессионные проверки:6passed. В дальнейшем сначала испытать миграцию на копии. '
        'Коммит/push блокирует прежний.git/index.lock; он не удалён.\n\n'
        '*Продолжение BUSCA.* Публичная папка авторов открылась; в README у идентификатора model_busca.pth пропущен последний символx. '
        'Получен полный авторский checkpoint143290361байт; хеш в busca_weights_download/recovery1/downloaded_files.json. '
        'Вторая загрузка осталась частичной по исходному сроку02:23:05,старый бюджет не продлён. '
        'Следующее чтение состава checkpoint на CPU зарегистрировано до запуска:busca_checkpoint_inventory/protocol.json; модель и HOTA ещё не запускались.')
    with (RUN/'HANDOFF.md').open('a',encoding='utf-8') as f:
        f.write('\n\n## '+datetime.now(timezone(timedelta(hours=3))).strftime('%H:%M')+' МСК — приоритетные новые правила графа и продолжение BUSCA\n\n'
            'Автор потребовал убрать все пчихи: Obsidian только фундаментальные задачи с замыслом/попыткой/итогом/следующим решением. '
            'Никакого журнала,копии плана,технических карточек или старогоcanvas. Генератор build_obsidian_vault.py теперь вызывает obsidian_fundamental_overview.py. '
            '23заметки; audit_fundamental_obsidian.py подтвердил дерево10узлов/9рёбер/0внешних/все цели. '
            'Автор разрешил самостоятельно добавлять дочерние подходы в дерево Гипотезы→Б1–Б3; внешние переходы только стелс. '
            'Сборка через b123_sync_graph_preserving_hypotheses_20261027.py --update-hypothesis-tree --archive-obsolete-generated; не возвращать legacy_main и старый каталог в граф. '
            'Сначала проверять копию перед миграцией. Все старые файлы в аудируемом архиве,ручная заметка также сохранена в исходном виде. '
            '6тестов графа passed; каталог после научных итогов ранее OK/58passed,он с тех пор не менялся. '
            'BUSCA:model_busca.pth полностью получен,правильный ID заканчивается Nfx;model_feats.partial~29МБ сохранён,срок старой загрузки истёк и не продлевался. '
            'Состав полного checkpoint может включать весь извлекатель: задание80 проверяет это на CPU. '
            'Если веса полные,следующий шаг—совместимость/реальное подключение,не новая загрузка того же содержимого. '
            'Не выдавать подготовку за запуск BUSCA/HOTA. Продолжать BUSCA→LTR в прежнем окне до07:30,один GPU,guard актуален.\n')
    print(json.dumps(dict(graph_verified=True,archive_files=len(archives),next_job='80_busca_checkpoint_inventory.json')))
if __name__=='__main__':main()
