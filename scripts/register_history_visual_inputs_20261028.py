"""Новый общий кэш изображений и фиксированных признаков; не новый опыт MASA."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261028 import RUN,service
read,write,sha,now=service.read,service.write,service.sha,service.now
from scripts.b123_safe_journal_20261027 import journal

def main():
    folder=RUN/'history_visual_inputs';assert not (folder/'protocol.json').exists()
    p=dict(parent_experiment='history_representations_factorial_v1',created=now(),budget_seconds=10800,
        image_size=64,context=3.,minimum_free_bytes=10*1024**3,
        motion=dict(width=320,height=180,horizon_frames=60,threshold=.1,min_response=.1),
        representations='По реальным последовательно декодированным изображениям: текущий серый вид/давность изменений/маска поддержки. Перенос камеры phase-correlation, горизонты60кадров; нет компенсации вращения/перспективы. Ненадёжная оценка илисдвиг>четвертькадра→сбросследа/маска/новаяэпохакамеры. ROIфиксированное3xокружениеТЕКУЩЕЙреальнойрамки,не зависитотистории/GT,одинаковоконтролям.',
        appearance='Тот же уже имеющийся замороженный MASA-R50 FP32/1024,256L2признаков. Это только общийфиксированныйвход программы1/3/4,не новое сравнение/ветвьMASA;веса/BatchNormне меняются. Исторические области кодируются в их прошлом кадре,не будущимконтекстом.',
        causal_binding='Точные пары(video,frame,detection_index)текущихипрошлыхреальныхобластей. Ни GT,ниtargetsне читаются. Признаки одинаковы всем будущим моделям.',
        price_rule='После первого ПОЛНОГО видео2*time*103+120<остаток10800с;приfalseсохранитьпервый/остаток,не продолжать103/непродлевать.',
        training_ready=False,weights_trained=False,hota_measured=False,
        sha256={p:sha(ROOT/p) for p in ['scripts/cache_history_visual_inputs_20261028.py','scripts/audit_history_visual_inputs_20261028.py',
            'tracking/causal_motion_history.py','tracking/frame_masa_appearance.py','tracking/masa_r50.py','weights/masa/masa_r50.pth',
            'runs/continuation_20261028/history_pair_dataset/report.json','runs/continuation_20261028/history_observed_export/report.json']})
    write(folder/'protocol.json',p)
    history=read(ROOT/'experiments/project_history.json');ident='history_visual_inputs_v1';assert not any(c['id']==ident for c in history['experiments'])
    history['experiments'].append(dict(id=ident,title='Изображение истории движения и общий банк признаков объектов',area='flyvis',state='running',started='2026-10-28',
        question='Можно ли подготовить реальные причинные визуальные входы одинаковым вариантам1/3/4 в разрешённом окне?',
        hypothesis='Гипотеза готовности входов:последовательных видео,реальных рамок и существующего извлекателя достаточно;польза представлений поHOTAпока не проверяется.',
        expected='На всех104train/dev получить текущийвид,следдавностиизмененийсмаскойпереносакамеры и общие256фиксированныхпризнаковреальныхобластей. Никакойразметки/обучения/сменывесов;одинаковыеисточникивсемвариантам.',
        acceptance=[dict(criterion=x,met=None) for x in ['Полные104видео с хешами/конечными256L2признаками/неизменнымивесами;первыйполныйпроходпрошёлпредзаписаннуюцену',
            'Всеуказателитекущего/прошлого точносовпадаютсподготовленнымипарами,прошлоестрогопрошлое;GTиtargetsнеиспользуются',
            'Всеиндексы/нормы/хешипроверены,пикселитрёхфиксированныхвидео повторены последовательнымдекодированием безпроизвольныхпереходов']],
        evidence=[dict(kind='file',ref=folder.relative_to(ROOT).as_posix()+'/protocol.json',claim='До GPU: фиксированные входы/перенос камеры/источники/стоимость'),
            dict(kind='file',ref='dissertation/Проверка трёх представлений истории — отдельно и совместно.md',claim='Литературная опора и роль входов;не подтверждениеHOTA')],metrics=[],confirmed_results=[]))
    write(ROOT/'experiments/project_history.json',history)
    for name,sub,script,budget in [('058_history_visual_checks','history_visual_checks','scripts/history_visual_catalog_checks_20261028.py',600),
        ('060_history_visual_inputs','history_visual_inputs','scripts/cache_history_visual_inputs_20261028.py',10800),
        ('065_history_visual_audit','history_visual_audit','scripts/audit_history_visual_inputs_20261028.py',900)]:
        target=RUN/sub
        if sub!='history_visual_inputs':write(target/'protocol.json',dict(parent_experiment=ident,budget_seconds=budget,sha256={script:sha(ROOT/script)}))
        write(RUN/'ready_jobs'/(name+'.json'),dict(name=name,folder=target.relative_to(ROOT).as_posix(),script=script,script_sha256=sha(ROOT/script),protocol_sha256=sha(target/'protocol.json')))
    journal(ident,'Подготовка реальных изображений истории и общего фиксированного внешнего вида',
        'Источник runs/continuation_20261028/history_visual_inputs/protocol.json:доGPUзафиксированыпоследовательноедекодирование,переноскамерыбезGT,след60кадров,маска/сбросприненадёжности,фиксированное3xROI64x64иобщие256FP32признаковсуществующегоизвлекателя. '
        '058каталог/68тестов→060одинGPU10800ссценой2*time*103+120послепервогоПОЛНОГОвидео→065хеши/причинность/нормы/пиксельныйповтортрёхвидео. Это программа1/3/4,не новаяпроверкаMASA;обучение/HOTAне запускаютсяэтимшагом.')
    with (RUN/'HANDOFF.md').open('a',encoding='utf-8') as f:
        f.write('\n\nСледующаяреальнаяреализацияГОТОВА:058history_visual_checks600сCPU→060history_visual_inputs10800сGPU→065audit900сCPU. '
            '060нужноpassed055иоконченноеегоCPUдерево/проверки058. Реальныетекущие/прошлыеобласти,кодируютсяВСВОЁМкадре,GT/targetsне читаются. '
            'ОбщийфиксированныйMASA-R50FP32/1024/256входтолько дляпорученнойматрицы,НЕноваяMASAветвь/сравнение;новыхвесовнет. '
            'MHIтекущийсерый/следдавности/маскапереносакамеры,3xROIоттекущейрамкиНЕотистории,64x64. Камератолькопереносphasecorr320x180,threshold.1/horizon60frames,minresponse.1,maxshiftquarter;ненадёжностьсбрасываетслед/эпоху. Все настройкидоGPUвprotocol. '
            'СуществующийGPUlockтотже. После первогоПОЛНОГОвидео2*time*103+120<остаток10800с,иначестопссохранениембезпродления. Готовоенеповторять. '
            '065проверяетвсеиндексы/прошлое/хеши/нормы,последовательноповторяетпикселивидео0/52/103;неGPUпереинференс. '
            'После065агентДОЛЖЕНреализоватьобщуюобучаемуюголову/последний/EMAконтроли/8сочетаний,зафиксироватьархитектуру/потери/3сида/разбиения/бюджеты/цену,обучитьипроверитьполныйтрекер. '
            'Главнаязацепкаданных:193126пар,131062известны,121061положительны/10001отрицательны,62064unknown. ПриобученииunknownНЕnegative;сильныйперекосположительныхнужноявноучестьодинаковымвсемметодамloss/семплингом,а не заявлятьуспехпонаивнойточности. НикакогоGTввыбористории. '
            'УбазыужеестьКалман/последнее/простаяпамять;сравниватьдобавочнуюпользу. НовыхHOTA1/3/4ещёнет.\n')
    print('058/060/065 prepared')

if __name__=='__main__':main()
