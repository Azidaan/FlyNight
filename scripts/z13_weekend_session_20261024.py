"""Record the author's weekend authorization without rewriting active protocols."""
from pathlib import Path
from datetime import datetime,timedelta,timezone
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,now
RUN=ROOT/'runs/z13_weekend_20261024'


def journal(title,text):
    path=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';s=path.read_text(encoding='utf-8');anchor='## Журнал изменений\n';assert s.count(anchor)==1
    path.write_text(s.replace(anchor,anchor+f'\n**{now():%d.%m.%Y, %H:%M} MSK** — {title}\n\n'+text+'\n',1),encoding='utf-8')


def main():
    assert not (RUN/'authorization.json').exists()
    msk=timezone(timedelta(hours=3));deadline=datetime(2026,10,26,7,40,tzinfo=msk)
    write(RUN/'authorization.json',dict(at=now().isoformat(),author_request='Давай, работаем нонстопом, время до понедельника, всё это время беспрерывно работаем!',timezone='Europe/Moscow',save_at=(deadline-timedelta(minutes=5)).isoformat(),hard_stop=(deadline-timedelta(seconds=30)).isoformat(),complete_by=deadline.isoformat(),scope='Continue necessary evidence-grounded FlyVis work, repairs, matched pilots, verification, exact-file commits/push. Existing science/inputs/criteria preserved. New runs get individual protocols, bounded budgets and independent owned-tree controllers. One GPU process, visible separate progress consoles. No downloads or heldout75 without distinct authorization. Active benchmark protocols and their current smaller budgets are not silently extended.',order=['Finish same7val main6 and C/B/R1 controls','CPU-only audit of descriptor replay failure while GPU busy','Recover extraction with unchanged parity tolerances if cause verified','Matched36CPU-model descriptor pilot only after complete verified extraction','Only a passing matched pilot advances to frozen-policy closed-loop HOTA; failed version leads to next documented node'],heartbeat='No frequent token-consuming polling automation; local queues perform waiting without model calls.'))
    journal('автор разрешил непрерывную работу на выходных до понедельника','Прямая команда автора: «Давай, работаем нонстопом, время до понедельника, всё это время беспрерывно работаем!». Разрешение сохранено в runs/z13_weekend_20261024/authorization.json. Ближайший обязательный конец —26.10.2026 07:40МСК,сохранение07:35и дерево завершить до07:39:30. Действующие меньшие бюджеты не продлены;каждый следующий расчёт получает отдельный бюджет и контроллер. Текущие6систем/7val и очередьC/B/R1 продолжаются. Параллельно допускается короткийCPUразбор ошибки воспроизведения коэффициентов,без конкурирующегоCUDA. Научные пороги/данные не меняются,загрузки и75отложенных не разрешены. Коммиты/push после зелёныхпроверок;частаяheartbeatавтоматизация не создаётся.')
    print(str(RUN/'authorization.json'))


if __name__=='__main__':main()
