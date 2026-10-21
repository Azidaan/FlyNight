"""Single-sheet diagonal tree layout; imported by build_flyvis_architecture.py.

This file changes presentation only. Experimental content remains in the entry
point's curated definitions and in the project catalogue.
"""
from __future__ import annotations


EDGE_STYLES = {
    'flow': dict(level='I', colour='#BC252D', width=8, arrow=30, dashed=False),
    'recurrence': dict(level='I', colour='#7342A4', width=6, arrow=27, dashed=False),
    'variant': dict(level='II', colour='#B97D00', width=5, arrow=23, dashed=False),
    'confirmed': dict(level='III', colour='#117A46', width=8, arrow=30, dashed=False),
    'future': dict(level='IV', colour='#416FBA', width=5, arrow=24, dashed=True),
}


def build_layout(node, edge, source, cards, lanes, base, confirmed):
    width, height = 9000, 9400
    positions = [(100 + (i % 4)*2200, (600 if i < 4 else 3900) + (i % 4)*110)
                 for i in range(8)]
    originals, hubs, leaves, module_bottom = [], [], {}, {}

    node('title',100,40,8780,150,'FlyVis — от исходной архитектуры к проверенным изменениям',
         'Один лист. Сначала красный путь I по номерам 01–08; затем локальные жёлтые ветви II. Зелёный III — достигнутое улучшение; синий пунктир IV — план.',
         'note',size=28)
    for i,(kind,title,body) in enumerate([
        ('base','I  ИСХОДНАЯ ВЕРСИЯ','Красные блоки и самые толстые красные стрелки — исходный путь. Фиолетовая петля — возврат состояния.'),
        ('tried','II  ИСПЫТАННЫЕ ЗАМЕНЫ','Золотые стрелки ведут к жёлтым вариантам конкретного узла. Подпись различает отрицательный и неопределённый исход.'),
        ('good','III  ПОДТВЕРЖДЁННОЕ УЛУЧШЕНИЕ','Толстая зелёная стрелка ведёт к проверенной замене. Внутри указаны контроль, выборка и доверительный интервал.'),
        ('future','IV  ДАЛЬНЕЙШИЕ ВАРИАНТЫ','Синий пунктир — ещё не проверенное изменение. Он не обозначает достигнутый результат.'),
    ]):
        node('legend'+str(i),100+i*2200,230,2080,200,title,body,kind,size=24)

    def geometry(nid):
        # node() returns IDs, while these placements are known explicitly below.
        return leaves[nid]

    for i,(number,title,variants) in enumerate(lanes):
        x,y=positions[i]
        original='original_'+str(i)
        content='\n\n'.join(t+'\n'+b for t,b,ref in base[i])
        if i==4:
            content='РАЗВЕТВЛЕНИЕ: активность идёт отдельно в декодер движения и отдельно в объектное считывание 06.\n\n'+content
        if i==5:
            content='ПЕРВАЯ ОБЪЕКТНАЯ ВЕРСИЯ ПРОЕКТА З13:\n\n'+base[i][0][0]+'\n'+base[i][0][1]+'\n\nВыход: 721 колонка + отдельный вариант отказа. Более поздние пространственный считыватель и индивидуальная модель цели показаны жёлтыми ответвлениями II.2 и II.5; они не являются последовательными слоями.'
        if i==6:
            content='ИСХОДНЫЙ ВЫБОР ЦЕНТРА:\n'+base[i][0][1]+'\n\nИСХОДНАЯ ОЦЕНКА ПРИСУТСТВИЯ З13:\n1 − вероятность отдельного варианта отказа. Этот балл сам по себе не равен вероятности полезной поправки. Более поздние замены балла и модели оценки показаны в ветвях II.'
        for _,_,ref in base[i]:source(ref)
        node(original,x+80,y,2000,650,number+'  '+title+' — ИСХОДНАЯ ОСНОВА',content,
             'external' if i==7 else 'base',base[i][0][2],size=23)
        originals.append(original)
        hy=y+750
        hub='hub_'+str(i)
        node(hub,x+680,hy,800,130,'II  ВАРИАНТЫ ЗАМЕНЫ УЗЛА','Каждая ветвь ниже — отдельный проверенный вариант.','tried',size=20)
        hubs.append(hub)
        edge(original,hub,'variant',start='bottom',end='top',label='II',
             label_at=(x+1080,y+700))
        top=hy+220
        bottom=top
        for j,(cid,title,body,result) in enumerate(variants):
            state=cards[cid]['state']
            status={'inconclusive':'НЕОПРЕДЕЛЁННО','completed_negative':'КРИТЕРИЙ НЕ ПРОЙДЕН',
                    'closed':'ЭТА ВЕРСИЯ ОСТАНОВЛЕНА'}.get(state,state)
            right=j%2
            nx=x+(1210 if right else 80)
            ny=top+(j//2)*420+(80 if right else 0)
            leaf='v_'+cid
            text=body+'\n'+status+': '+result
            h,fs=360,22
            if cid=='fly_spatial_quality_z13_v2':
                text+='\nУстройство: вложение 32; колонка и среднее соседей; до 64 позиций образца, координаты и внимание; 721 балл + отказ.'
                h,fs=400,20
            elif cid=='z13_target_model_probe_v1':
                text+='\n34 канала + среднее соседей = 68 признаков; отдельная гребневая регрессия по последнему наблюдаемому образцу.'
                h,fs=400,20
            elif cid=='z13_candidate_ranker_v1':
                h,fs=400,20
            elif cid=='z13_pairwise_map_v1':
                h,fs=400,20
            node(leaf,nx,ny,870,h,f'II.{j+1}  '+title,text,'tried',cid,cid,size=fs)
            leaves[leaf]=(nx,ny,870,h)
            # A central bus in the empty gap serves short left/right branches.
            edge(hub,leaf,'variant',points=[(x+1080,ny+h/2)],
                 start='bottom',end='left' if right else 'right',
                 label='II',label_at=(x+(1145 if right else 1015),ny+h/2))
            bottom=max(bottom,ny+h)
        module_bottom[i]=bottom

    # Level I: broad diagonal arrows connect only the module roots.
    for i in (0,1,2,4,5,6):
        x,y=positions[i]; xx,yy=positions[i+1]
        edge(originals[i],originals[i+1],'flow',start='right',end='left',
             label='I.'+str(i+1),label_at=((x+2080+xx+80)/2,(y+yy)/2+325))
    # A dedicated empty horizontal corridor makes the line wrap unambiguous.
    x,y=positions[3];xx,yy=positions[4]
    wrap_y=3820  # leave visible air below the expanded temporal-memory lane
    edge(originals[3],originals[4],'flow',start='right',end='left',
         points=[(8940,y+325),(8940,wrap_y),(140,wrap_y),(140,yy+325)],
         label='I.4  СОСТОЯНИЕ → АКТИВНОСТЬ: ПРОДОЛЖЕНИЕ ПУТИ',label_at=(4500,wrap_y))
    # Recurrent connection has its own colour and empty corridor above the roots.
    x,y=positions[3];xx,yy=positions[2]
    edge(originals[3],originals[2],'recurrence',start='top',end='top',
         points=[(x+1080,500),(xx+1080,500)],label='I  ↺  СОХРАНЁННОЕ СОСТОЯНИЕ',
         label_at=(6680,500))

    # Evidence remains checked against the frozen primary JSON, unchanged by layout.
    cdelta,report=confirmed('z13_target_center_dev_v1','C_center_vs_binary_dev35')
    ddelta,_=confirmed('z13_target_center_dev_v1','D_center_vs_binary_dev35')
    rep,_=confirmed('z13_target_center_dev_v1','D_center_vs_R1_dev35')
    repc,_=confirmed('z13_target_center_dev_v1','C_center_vs_R1_dev35')
    old,_=confirmed('z13_target_center_dev_v1','D_center_vs_prior_dev35')
    x,y=positions[5]
    gy=module_bottom[5]+180
    node('g_center',x+80,gy,2000,640,'III  К ЧЕМУ ПРИШЛИ: ЦЕНТРИРОВАННОЕ СЧИТЫВАНИЕ',
         'ЗАМЕНА: вместо плоской маски объекта — гауссов пик центра последней наблюдаемой рамки. Считыватель — индивидуальная гребневая модель цели. Ядро в этих сравнениях фиксировано.\n\n'
         'ПРОТИВ БИНАРНОЙ ЦЕЛИ: C '+cdelta+'; D '+ddelta+'.\n\n'
         'ПРОТИВ ВХОДА ГЛАЗА R1: C '+repc+'; D '+rep+'.\n\n'
         'ПРОТИВ ПРЕЖНЕГО СЧИТЫВАТЕЛЯ НА ТОЙ ЖЕ D: '+old+'.\n\n'
         'dev35 использовался раньше. Подтверждено улучшение полезных центров; надёжность отбора, преимущество D над C и улучшение HOTA этим не доказаны.',
         'good',report,'z13_target_center_dev_v1',size=24)
    leaf='v_z13_target_model_probe_v1'
    lx,ly,lw,lh=geometry(leaf)
    edge(leaf,'g_center','confirmed',start='right',end='top',
         points=[(x+1020,ly+lh/2),(x+1020,gy-70),(x+1080,gy-70)],
         label='III  МАСКА → ПИК ЦЕНТРА',label_at=(x+1080,gy-80))
    for j,(cid,title,body) in enumerate([
        ('fly_future_stmtrack_memory_v1','IV  STMTrack: память наблюдений','Вместо одного фиксированного образца — отдельные прошлые наблюдения цели. Добавление не должно использовать будущую разметку.'),
        ('fly_future_keeptrack_candidates_v1','IV  KeepTrack: личности кандидатов','Вместо одного максимума — отдельные пики, локальная активность и соответствия во времени. Допускается отсутствие соответствия.'),
    ]):
        nx=x+(1210 if j else 80);ny=gy+800
        node('f_'+cid,nx,ny,870,360,title,body+'\nПЛАН полной версии; упрощённые проверки показаны жёлтыми ветвями.','future',cid,cid,size=23)
        edge('g_center','f_'+cid,'future',start='bottom',end='top',
             points=[(x+1080,gy+720),(nx+435,gy+720)],label='IV',label_at=(nx+435,gy+750))

    x,y=positions[3]
    fy=module_bottom[3]+190
    cid='fly_future_kys_state_v1'
    node('f_'+cid,x+80,fy,2000,350,'IV  KYS: согласованная пространственная память',
         'Возможная замена: переносить память считывателя между соответствующими областями кадров. Нужна проверка рассогласования координат. Статья не обосновывает прямой перенос потенциалов нейронов FlyVis. ПЛАН: результата нет.',
         'future',cid,cid,size=24)
    edge(originals[3],'f_'+cid,'future',start='right',end='right',
         points=[(x+2140,y+325),(x+2140,fy+175)],label='IV',label_at=(x+2140,fy+175))

    # Context notes are kept outside the data tree: they do not acquire flow arrows.
    x,y=positions[0]
    node('input_partial',x+80,module_bottom[0]+180,2000,340,'ПОДТВЕРЖДЁННЫЕ РАННИЕ ЭФФЕКТЫ — ОТДЕЛЬНО ОТ З13',
         'Ламина и локальная нормализация дали функциональные эффекты в отдельных проверках. Они не объединены автоматически с З13 в одну доказанную модель. Проверенный вариант нормализованного потока не улучшил HOTA.',
         'note','tracking/retina.py',size=24)
    x,y=positions[2]
    node('control',x+80,module_bottom[2]+170,2000,350,'КАКИЕ ЯДРА И КОНТРОЛИ СРАВНИВАЮТСЯ',
         'C0 — инициализация до Sintel; C — штатная обученная FlyVis; D — дообученная муха. B — HexGRU (рекуррентная сеть на решётке), состояние 16, выход 34. R0/R1 — временные признаки входа глаза без FlyVis. B и D не равны по ёмкости.',
         'note','scripts/z13_target_model.py',size=24)
    x,y=positions[6]
    node('metric',x+80,module_bottom[6]+170,2000,360,'ЧТО ИЗМЕРЯЕТСЯ И ЧТО ОСТАЁТСЯ НЕИЗВЕСТНЫМ',
         'Полезный центр: ошибка меньше ошибки штатного прогноза и не больше четверти размера известной цели. Средняя доля по видео не равна точности среди выбранных. Неизвестный исход не означает ошибку или успех. Глобальный верхний 1% — диагностика, а не готовое покадровое правило.',
         'note','scripts/z13_train.py',size=24)
    cm,mp=confirmed('z13_map_only104_v1','C_map104_vs_top1')
    dm,_=confirmed('z13_map_only104_v1','D_map104_vs_top1')
    cv,dvp=confirmed('z13_map_transfer104_v1','C_map104_vs_top1_dev35')
    dv,_=confirmed('z13_map_transfer104_v1','D_map104_vs_top1_dev35')
    my=module_bottom[6]+720
    node('g_map104',x+80,my,2000,640,'III  ЧАСТНЫЙ ЭФФЕКТ: ЧТЕНИЕ ПИКОВ КАРТЫ',
         'ЗАМЕНА: максимум карты → обучаемый выбор одного из3 раздельных пиков. Вход — положения и высоты пиков; карта FlyVis остаётся источником, дополнительные сырые векторы не нужны этому варианту.\n\n'
         'ПРОТИВ МАКСИМУМА НА train26: C '+cm+'; D '+dm+'.\n\n'
         'ФИКСИРОВАННЫЙ ПЕРЕНОС НА dev35: C '+cv+'; D '+dv+'.\n\n'
         'Подтверждены эти частные сравнения по полезным центрам. Общий критерий переноса не пройден: превосходство над выбором по прогнозу не доказано. Прежний центрированный считыватель остаётся рабочей версией. Вклад общего модуля отделяется R1; HOTA не измерялась.',
         'good',dvp,'z13_map_transfer104_v1',size=22)
    edge(hubs[6],'g_map104','confirmed',start='right',end='right',
         points=[(x+2140,y+815),(x+2140,my+320)],label='III',label_at=(x+2140,my+320))
    x,y=positions[7]
    node('g_scope',x+80,module_bottom[7]+180,2000,420,'ГРАНИЦА ДОСТИГНУТОГО РЕЗУЛЬТАТА',
         'Зелёные узлы III — подтверждённые сравнения считывания. У карты104 показан частный эффект при непройденном общем критерии. Превосходство D над C и польза новых узлов по HOTA пока не установлены. Архитектурные ветви нельзя складывать в одну модель без отдельной проверки.',
         'note',report,size=25)
    hy=module_bottom[7]+800
    node('f_center_hota',x+80,hy,2000,430,'IV  ТЕКУЩАЯ ПРОВЕРКА СТЕНДА: ЦЕНТРИРОВАННАЯ РЕГРЕССИЯ',
         'Применяется ранее подтверждённый зелёный узел центрированной регрессии. Кандидаты104, парная потеря и окно в него не входят.\n\n'
         'C/D/R1: те же4 наблюдения и состояние; одна политика верхнего1% исходных баллов. Меняется только центр потерянного трека; YOLO26, ByteTrack и MASA прежние.\n\n'
         'Сейчас техническое воспроизведение и замер. HOTA ещё не измерена; по утреннему сроку сохраняются законченные видео и остаток очереди.',
         'future','runs/z13_center_hota_20261021/protocol.json','z13_center_hota_v1',size=23)
    edge(originals[7],'f_center_hota','future',start='right',end='right',
         points=[(8930,y+325),(8930,hy+215)],label='IV',label_at=(8930,hy+215))
    node('footer',100,9200,8780,160,'ПОРЯДОК ЧТЕНИЯ: I — КРАСНЫЙ ПУТЬ 01–08 → II — ЛОКАЛЬНЫЕ ЗАМЕНЫ → III — ДОСТИГНУТЫЙ УЗЕЛ → IV — ПЛАНЫ',
         'Цвет блока — статус; римская метка стрелки — её уровень и смысл. Восемь частей пути пронумерованы арабскими числами. Альтернативы — отдельные ветви; петля состояния показана отдельно.',
         'note',size=25)
    detail=(positions[5][0]+20,positions[5][1]-50,positions[5][0]+2150,gy+1180)
    return width,height,detail
