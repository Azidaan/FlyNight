"""Record an explicitly requested UI restart; never restart computation."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,now
from scripts.z13_hota_deadline_guard import WindowsProcesses
from scripts.z13_localization_labels_20261023 import verify_sources
from scripts.z13_val20_common_20261024 import journal


def main():
    run=ROOT/'runs/z13_val20_matched_20261024/common_authorized_recovery1'
    receipt=max(run.glob('console_restart_*.json'),key=lambda p:p.name)
    new=read(receipt);current=run/'console_recovery.json';old=read(current)
    assert old['console_pid']==new['previous_console_pid']
    backup=run/(receipt.stem+'.previous_console.json')
    assert not backup.exists();backup.write_bytes(current.read_bytes())
    write(current,new)
    ps=WindowsProcesses();controller=read(run/'controller.json');monitor=read(run/'completion.json')
    assert ps.creation(controller['pid'])==controller['birth']
    assert ps.creation(monitor['pid'])==monitor['birth']
    verify_sources(read(run/'protocol.json'))
    assert new['console_window_handle']!=0 and new['computation_not_restarted']
    journal('по просьбе автора заново открыта актуальная консоль прогресса',
        'Открыто отдельное окноPowerShell7,PID'+str(new['console_pid'])+
        ';при запуске ненулевойWindowHandle. Источник '+receipt.relative_to(ROOT).as_posix()+
        '. ПрежнийPID'+str(old['console_pid'])+' уже отсутствовал;переиспользованныйPID '
        'не завершался. Канонический console_recovery.json обновлён с сохранением '
        'прежней записи. Контроллер и наблюдатель расчёта сохранилиPID/время создания;'
        'научные файлы/прогон не менялись и не перезапускались. '
        'На открытии показаны143/231задание,centerC на восьмом видео. '
        'Пересборка графа и архитектуры для перезапуска окна не требуется.')
    handoff=ROOT/'runs/z13_weekend_20261024/HANDOFF.md'
    header=('# Актуально — '+now().strftime('%d.%m.%Y %H:%M')+' MSK\n\n'
        'common_authorized_recovery1 уже работает после автоматического запуска16:12МСК. '
        'Контроллер'+str(controller['pid'])+'/birth'+str(controller['birth'])+
        ',монитор'+str(monitor['pid'])+'/birth'+str(monitor['birth'])+'. '
        'Очередь завершилась с queue_result state=launched;не запускать её повторно. '
        'Готовы7из20полныхвидео,считаетсявосьмое;прогресс смотреть вstatus.json. '
        'Прежняя зависавшаяMASA четвёртого ролика завершена1213кадров/73,893с,'
        'проверка совпадения признаков пройдена. Исходный предел25.10 06:00:27МСК '
        'и научные настройки прежние. Новых научных итоговHOTA пока нет.\n\n'
        'По просьбе автора актуальная консоль открыта заново:PowerShell7 PID'+
        str(new['console_pid'])+',console_recovery.json и '+receipt.name+'. '
        'СтарыйPID'+str(old['console_pid'])+' уже отсутствовал. Прогон и наблюдатель '
        'не перезапускались. Каноническая запись консоли обновлена,предыдущая сохранена '
        'в '+backup.name+'. Не создавать дубль этой консоли. Остальной протокол '
        'common_authorized_recovery1 и правила ниже действуют.\n\n'
        'Ниже прежние передачи сохранены.\n\n')
    handoff.write_text(header+handoff.read_text(encoding='utf-8'),encoding='utf-8')
    print('Console receipt/journal/handoff updated;original computation identities and sources verified.')


if __name__=='__main__':main()
