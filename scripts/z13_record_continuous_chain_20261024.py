"""Record explicit author instruction and verify the existing continuation chain."""
from pathlib import Path
from datetime import datetime
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
from scripts.z13_localization_labels_20261023 import journal
from scripts.z13_hota_deadline_guard import WindowsProcesses
W=ROOT/'runs/z13_weekend_20261024';B=ROOT/'runs/z13_matched_trackers_20261024'


def main():
    out=W/'continuous_chain_authorization.json';assert not out.exists()
    ps=WindowsProcesses();controller=read(B/'gpu_recovery3/controller.json');queue=read(B/'stock_controls/queue4.json')
    assert ps.creation(controller['pid'])==controller['birth'] and ps.creation(queue['pid'])==queue['birth']
    assert not (B/'stock_controls/queue4_failure.json').exists()
    status=read(B/'gpu_recovery3/status.json');assert status['state']=='running' and (now()-datetime.fromisoformat(status['updated'])).total_seconds()<600
    paths=['scripts/z13_matched_pipeline_recovery3_20261024.py','scripts/z13_matched_stock_queue4_20261024.py','scripts/z13_matched_stock_execution4_20261024.py','scripts/z13_weekend_driver_20261024.py']
    write(out,dict(at=now().isoformat(),author_request='Давай дальше! Можешь после каждого своего сообщения, где не требуется выбор запускать дальнейшую цепочку прогонов дальше? Для беспрерывной работы',permission_window_source='runs/z13_weekend_20261024/authorization.json',complete_by=read(W/'authorization.json')['complete_by'],agents_rule='AGENTS.md#9.3',agents_sha256=sha(ROOT/'AGENTS.md'),controller=controller,stock_queue=queue,progress=status,continuation_source_sha256={n:sha(ROOT/n) for n in paths},automation_id='claude-flyvis',automation_status='ACTIVE',automation_interval_minutes=120,new_automation_created=False,next='Complete main six-system benchmark and primary verification; existing queue4 then starts registered stock C/B/R1 controls automatically. Failed descriptor implementation is not promoted. New scientific alternative chosen from evidence before a new protocol and cheap pilot.'))
    journal('ответ не останавливает разрешённую цепочку работы','Прямое решение автора24.10: после каждого сообщения,не требующего выбора,продолжать дальнейшую цепочку без нового «давай». Правило добавлено вAGENTS.md9.3. Перед завершением ответа проверены идентичность контроллера recovery3 и очередиqueue4,свежий прогресс,отсутствие новойошибки. Текущее сравнение→проверка→штатнаяC/B/R1 уже обеспечены локальной очередью,без обращений к модели для ожидания. Существующая claude-flyvis обновлена,ACTIVE,интервал120минут сохранён;дубль и двухминутныеопросы не создавались. Новые решения агента выполняются в прежнем разрешении до26.10 07:40МСК;действующие меньшие бюджеты не продлены,проваленная версия не продвигается. Источник: runs/z13_weekend_20261024/continuous_chain_authorization.json.')
    handoff=W/'HANDOFF.md';text=handoff.read_text(encoding='utf-8');prefix=f'''# Непрерывность работы — решение автора {now():%d.%m.%Y %H:%M} МСК

Автор просит после каждого сообщения без необходимого выбора продолжать разрешённую цепочку без нового «давай». ПравилоAGENTS9.3,источник continuous_chain_authorization.json. Перед завершением ответа проверять реальные прогресс и идентичность очереди;не ограничиваться обещанием или расписанием. Готовые переходы выполняет локальная очередь,новый обоснованный узел по плану выбирает агент. Проваленный критерий сохранять и не продвигать неудачную версию. Существующая автоматизация claude-flyvis обновлена,ACTIVE,интервал120минут сохранён;дубли/частыеопросы не создавать. Общее разрешение до26.10 07:40МСК и меньшие бюджеты неизменны. Последние научные результаты и текущая recovery3/queue4 ниже.

''';handoff.write_text(prefix+text,encoding='utf-8')
    print(json.dumps(dict(continuation_verified=True,progress=status,automation_interval_minutes=120),ensure_ascii=False))


if __name__=='__main__':main()
