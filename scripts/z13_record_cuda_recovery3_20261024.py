"""Preserve verified recovery state and architecture milestone after CPU result."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
from scripts.z13_hota_deadline_guard import WindowsProcesses
from scripts.z13_localization_labels_20261023 import verify_sources,journal
W=ROOT/'runs/z13_weekend_20261024';B=ROOT/'runs/z13_matched_trackers_20261024';T=ROOT/'runs/z13_filter_descriptor_fixed_20261024/train'


def main():
    ps=WindowsProcesses();c=read(B/'gpu_recovery3/controller.json');q=read(B/'stock_controls/queue4.json')
    assert ps.creation(c['pid'])==c['birth'] and ps.creation(q['pid'])==q['birth']
    for folder in (B/'gpu_recovery2',B/'gpu_recovery3',B/'stock_controls',T):verify_sources(read(folder/'protocol.json'))
    old=read(B/'gpu_recovery2/controller_result.json');assert old['exit_code']!=0 and old['all_owned_ended'] and not old['remaining']
    assert all(ps.creation(int(pid))!=birth for pid,birth in old['owned'].items())
    from scripts.z13_matched_stock_execution4_20261024 import verify
    verify()
    done=read(T/'finish_recorded.json');report=read(T/'report.json');assert done['report_sha256']==sha(T/'report.json') and not report['primary_pass']
    for name,h in done['models'].items():assert sha(ROOT/name/'model.pt')==h
    from scripts.z13_continue import paired,per_video_useful
    for key,value in report['paired'].items():
        a,b=key.split('-');assert paired(report['per_video'][a],report['per_video'][b])==value
    for arm in ('D','C','R1'):
        for access in ('ridge','mean'):
            for mode in ('activity','geometry'):
                key=f'{arm}_{access}_{mode}';assert per_video_useful(read(T/arm/access/mode/'ensemble_rows.json'))==report['per_video'][key]
    assert read(W/'fixed_verified.json')['checks_green']
    h=W/'HANDOFF.md';text=h.read_text(encoding='utf-8')
    prefix=f'''# Актуально — {now():%d.%m.%Y %H:%M} МСК: recovery3 и полный отрицательный CPU-итог

Текущий GPU: gpu_recovery3, scripts/z13_matched_pipeline_recovery3_20261024.py; контроллер{c['pid']}/birth{c['birth']}. Recovery2 застрял на centerD последнего ролика:25/1205 с03:03:41,повторные py-spy стеки подтвердили ожидание CUDA cuStreamSynchronize. Стек сохранён. Только собственный проверенный PID8944 остановлен;43готовыхзадания проверены,старые ошибки сохранены. Новый процесс прошёл прежнее место и продвигается. Один повтор;служебный контроль600с без обновления завершает своё дерево,повторное такое зависание разбирать отдельно. Исходные конец05:53:48МСК,научные настройки/источники/пороги неизменны.

Stock ждёт через scripts/z13_matched_stock_queue4_20261024.py,realPID{q['pid']}/birth{q['birth']};queue3_failure сохранён. После recovery3 используется scripts/z13_record_matched_recovery3_20261024.py,затем execution4 для stock. Исходные протоколы/источники stock неизменны;execution_v4.json фиксирует служебные хеши. Старый локальный диспетчер продолжает финализацию stock;не запускать дубль.

CPUcache3720/3720 завершёнexit0,все9наборов прошли воспроизведение. CPUtrain36/36 завершёнexit0,карточка z13_filter_descriptor_fixed_v2 записана completed_negative,проверки и граф зелёные:fixed_verified.json. Primary_pass=false. Dкоэффициенты-Dсреднее +0,671906п.п.,95%ДИ[-0,989288;2,389616],12/26;не прошёл заранее записанный критерий>=2п.п.и положительная нижняя граница. Это неопределённость пользы,не доказанное отсутствие любого эффекта. Отдельно Dкоэффициенты-R1коэффициенты +10,795263п.п.[7,331636;14,620246],C-R1+12,769151п.п.[8,306709;17,591296]. Это полезные центры при известной присутствующей цели на повторно использованных26train,ансамбль3сидов;неHOTA,не готовый отказ. Dне лучше геометрического контроля: -4,062409п.п.[-8,994661;-0,129330]. Действующий считыватель не заменять,новую версию кHOTAнепродвигать. Полныйисточник train/report.json;проверенные первичныеensemble_rows и36моделей вfinish_recorded.json. Граф обновлён,архитектурная ветвь добавлена жёлтой,одинлист,перекрытий нет.

Ниже прежние передачи состояния сохранены как история.

'''
    if not text.startswith('# Актуально — '):h.write_text(prefix+text,encoding='utf-8')
    marker=W/'cuda_recovery3_verification.json'
    if not marker.exists():journal('архитектурная веха: проверено описание цели коэффициентами','Полный CPU-итог z13_filter_descriptor_fixed_v2 уже записан: runs/z13_filter_descriptor_fixed_20261024/train/finish_recorded.json. Проверены36хешеймоделей,первичные ensemble_rows и все парные26видео. Критерии не пройдены;действующий считыватель сохранён. В однолистовую архитектуру добавлена жёлтая ветвь замены среднего образца68коэффициентами наблюдаемого фильтра,с исходом+0,67п.п.[−0,99;+2,39],без утверждения о HOTA. Генератор иvalidation:1лист,нетперекрытий и пересечений постороннихблоков;общийвид и читаемыйфрагмент проверены. Новая очередь queue4 следует recovery3,ошибки очередей2/3 сохранены;старые источники неизменны. Полныйисточник: train/report.json;графпроверка runs/z13_weekend_20261024/fixed_vault.json,58тестов пройдены.')
    # Readable inspection of the new block from the same single sheet.
    from PIL import Image
    sources=read(ROOT/'dissertation/architecture/FlyVis_architecture_sources.json');node=next(n for n in sources['nodes'] if n['card']=='z13_filter_descriptor_fixed_v2')
    im=Image.open(ROOT/'dissertation/architecture/FlyVis_architecture.png');x,y,w,h0=(node[n] for n in ('x','y','w','h'));im.crop((x-20,y-20,x+w+20,y+h0+20)).save(W/'descriptor_architecture_detail.png')
    write(marker,dict(at=now().isoformat(),recovery_controller=c,stock_queue=q,completed43_jobs_preserved=True,report_sha256=sha(T/'report.json'),models36_verified=True,all_paired_recomputed=True,primary_rows_verified=True,deadline_unchanged=True,architecture_single_sheet=True))


if __name__=='__main__':main()
