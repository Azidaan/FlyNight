from pathlib import Path
from datetime import datetime,timezone,timedelta
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261029 import RUN,service

def main():
    first=service.read(RUN/'masa_query_pilot/price.json')
    disk=service.read(RUN/'masa_query_pilot/disk_gate.json')
    launch=service.read(RUN/'masa_query_pilot/launch.json')
    assert disk['passed'] and service.matching(launch['pid'],launch['created'])
    for name in ['masa_query_preparation','masa_disk_guard_registration']:
        c=service.read(RUN/name/'controller_result.json')
        assert c['exit_code']==0 and c['all_owned_ended'] and not any(service.matching(k,v) for k,v in c['owned'].items())
    data=dict(first_complete_video=first,disk_gate=disk,gpu_launch=launch,
              queue=service.read(RUN/'queue.launch.json'),guard=service.read(RUN/'guard.launch.json'),
              current_status=service.read(RUN/'masa_query_pilot/status.json'),
              full_result_eta_msk='29.10.2026 20:00–21:00, forecast conditional on complete audit',scientific_verdict=None,updated=service.now())
    service.write(RUN/'launch_verified.json',data)
    stamp=datetime.now(timezone(timedelta(hours=3))).strftime('%d.%m.%Y, %H:%M MSK')
    text=f'''**{stamp}** — MASA-Detic: полный пилот запущен, первое видео сохранено

Получено. runs/continuation_20261029/launch_verified.json: первое полное видео/пятьвариантов71,719с,консервативнооставшиеся19+резерв3025,322с. Дисковаяпроверкапройдена:первыйвыход5407336байт,требуется5574187888байт,свободно88922263552байт. Очередь/guard/GPUпровереныPID+birth,подготовка002/дисковый003code0/allended;каталогOK/59passed121,16с. Нулевойконтрольпервоговидеосовпалсnativeпоraw/JSON/состоянию;первыйGPU/HOTAпроходсохранёнсSHAи не повторяется. Идётвтороевидео. Данные/веса/настройкизамороженыдоval;новогообучениянет.

Вывод. Наодномвидеонаучнаяприёмканеоценивается. Прогнозполногоитогаснезависимымаудитом—29.10 20:00–21:00МСК,условный,необещаниеположительногорезультата. ПослеGPUseriesследует020CPUаудит/025оформлениебезвызовамодели. Еслиаудитоткажет—сохранитьошибку/исходники,необъявлятьнаучныйуспехинеослаблятьдопуск. СтарыйсыройFP32gateосталсянепройденным;новыйвыходнойинтерфейспрошёлотдельнуюзаранеезаписаннуюпроверку1e-5наtrainкадрах. Системаquery-сопоставления не официальныйMASAтрекеринеядроFlyVis.

Решение. Продолжатьдо30.10 07:40МСК,save07:30/всесвоидеревья07:39:30. Редкаяавтоматизацияmasa-detic-30-07-40служитпроверкойиагентскимрешением,непостояннымприсутствиеммодели. Во время обязательныхкаталоговыхтестов025неправитьплан/каталог/словарь/контролируемыеисточники. После025агентчитаетполныеисточники/частныеэффекты/Проблемы/ограничения;готовоеБ1/матрицу/обучениенеповторять.

'''
    with (RUN/'HANDOFF.md').open('a',encoding='utf-8') as f:f.write('\n\n'+text)
    plan=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';body=plan.read_text(encoding='utf-8');marker='## Журнал изменений';assert body.count(marker)==1
    plan.write_text(body.replace(marker,marker+'\n\n'+text,1),encoding='utf-8')
    print(stamp)

if __name__=='__main__':main()
