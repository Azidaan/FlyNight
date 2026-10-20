"""Build the single-sheet, editable FlyVis architecture atlas. No scientific execution.

Curated module replacements are kept here; outcomes and evidence remain in the
experiment catalogue. Colour applies to the explicitly stated comparison only.
"""
from __future__ import annotations

import hashlib
import html
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'dissertation/architecture'
STEM = 'FlyVis_architecture'
DATE = '21.10.2026'
FONT = 'C:/Windows/Fonts/segoeui.ttf'
BOLD = 'C:/Windows/Fonts/segoeuib.ttf'
PALETTE = {'base': ('#FAD8DA','#BC252D'), 'tried': ('#FFF1BA','#BF8A18'),
           'good': ('#DDF3E3','#117A46'), 'future': ('#E8F0FF','#416FBA'),
           'external': ('#E8EDF3','#68788C'), 'note': ('#FFFFFF','#CCD5DF')}

# Each entry explicitly states what replaces what, not merely a paper name.
LANES = [
 ('01', 'ВХОД И АДАПТАЦИЯ ЯРКОСТИ', [
  ('flyvis_retina_frontend_v1','Адаптивный фоторецептор','Вместо gray/255: сжатие яркости I/(I + адаптация), локальное среднее и временная адаптация.','Один входной модуль; результат неоднозначен, это не доказанная польза HOTA.'),
  ('fly_eye_frontend_v1','Глаз перед детектором','Вместо исходного изображения YOLO: кадр после фоторецептора и накопления.','Внешнее применение входного узла. Проверенная версия не помогла детектору.'),
  ('fly_dynamic_norm_v1','Динамическая нормализация','Вместо фиксированной локальной нормализации lcn24: изменяющаяся во времени нормализация сигнала.','Эта реализация не стала подтверждённой заменой lcn24.'),
  ('fly_input_norm_hota_v1','Нормализованный поток в стенде','Вместо прежнего источника движения: поток FlyVis после нормализации контраста.','Улучшение измерения движения не превратилось в подтверждённое улучшение HOTA.'),
 ]),
 ('02', 'ГЕОМЕТРИЯ ГЛАЗА', [
  ('flyvis_stress','Полный кадр / плитки','Вместо одной области: фиксированная сетка перекрывающихся фрагментов полного кадра.','Стресс-проверка не подтвердила пригодность этой версии для трекинга.'),
  ('flyvis_foveation_readout_v1','Отдельный глаз на трек','Вместо общих плиток: отдельная вырезка и состояние каждого трека; прежнее считывание активности.','В проверке сопоставления прирост не получен. Сам механизм вырезки используется в последующих версиях.'),
  ('fly_scale_probe_v1','Масштаб и шаг кадров','Вместо прежнего размера фрагментов и шага: меньшие плитки и прореживание кадров.','Проверенная конфигурация не принята как улучшение; не означает бесполезности любого масштаба.'),
  ('fly_efference_eye_v1','Учёт перемещения глаза','Вместо прежней вырезки: стабилизация входа и калиброванное считывание движения.','Геометрические признаки улучшились, но все технические критерии не пройдены.'),
  ('fly_efference_eye_v2','Перенос начального отклика','Вместо прежней ссылки: перенос начального отклика и калибровка накопленного движения.','Технический критерий не выполнен; улучшение переноса ссылки не подтверждено.'),
 ]),
 ('03', 'СВЯЗИ И ПАРАМЕТРЫ ЯДРА', [
  ('fly_finetune_k1_pilot_v1','Обучение входа и декодера','Вместо штатного решения Sintel: подстройка входа и декодера под видео BDD100K.','Проверенный пилот не дал требуемого улучшения движения.'),
  ('fly_finetune_k1_core_v1','Дообучение ядра для движения','Вместо фиксированного ядра: обучение 734 свободных параметров вместе с декодером на нормализованном входе.','Эта постановка не прошла критерий; не закрывает обучение FlyVis вообще.'),
  ('fly_connectome_null_v1','Перемешанные связи','Контроль вместо исходной схемы: перемешанные связи при сохранении степеней и знаков.','Неопределённый результат. Это проверка вклада связей, не принятая модернизация.'),
  ('fly_flywire_swap_v1','Профили связей FlyWire','Вместо исходных профилей: профили FlyWire и перемешанный контроль при согласованной дозе.','Проверенный перенос не подтвердил нужное преимущество.'),
  ('z13_core_continue_v1','Продолжение обучения D','Вместо сохранённой D: продолжить обучение ядра; контроль — продолжать только считыватель.','Продолжение в этой постановке не улучшило результат.'),
  ('z13_aligned_core_v1','Градиент через модель цели','Вместо замороженной D: обучать ядро через центрированную модель конкретной цели.','На проверенной ступени улучшение не подтверждено; сработала остановка.'),
 ]),
 ('04', 'ВРЕМЯ И СОСТОЯНИЕ', [
  ('fly_persistent_eye_v1','Непрерывный глаз при потере','Вместо закрытия/перезапуска глаза: сохранять состояние и поправлять прогноз остаточным потоком.','Польза HOTA и преимущество непрерывности в этой реализации не подтверждены.'),
  ('z13_candidate_adaptive_v1','Адаптивное накопление','Вместо фиксированного смешивания 0,5: обучаемая доля текущей активности и перенесённой предыдущей.','Основной критерий локализации не выполнен. Это смешивание признаков, не банк отдельных образцов.'),
  ('fly_signature_reid_v1','Сигнатура состояния','Вместо одного измерения движения: использовать отпечаток активности для повторного узнавания.','Результат неоднозначен; не доказана надёжная сшивка траекторий.'),
 ]),
 ('05', 'СЧИТЫВАНИЕ ДВИЖЕНИЯ', [
  ('flyvis_internal_readout_v1','Активность вместо потока','Вместо декодированного движения: T4/T5 либо все 34 выходных типа внутри рамки и кольца.','В проверке сопоставления замена не дала требуемого преимущества.'),
  ('flyvis_readout_nonlinear_diag','Нелинейное считывание','Вместо прежней модели на сохранённых признаках: нелинейная модель для плиток и фовеального глаза.','Проверенная нелинейность не исправила результат.'),
  ('fly_decoder_refit_v1','Новый декодер движения','Вместо штатного декодера Sintel: новый декодер на сохранённой активности.','Результат неоднозначен; вклад представления и декодера полностью не отделён.'),
  ('fly_wide_field_v1','Широкое поле T4/T5','Вместо поколоночного декодера: суммирование направлений в рамке и широком поле, по принципу HS/VS.','Одна испытанная версия не улучшила результат.'),
  ('flyvis_fovea_motion_v1','Остаточный поток','Вместо чистого прогноза трека: поправка по остаточному потоку фовеального глаза.','Проверенное применение не дало нужного выигрыша.'),
  ('fly_gain_tracker_recheck_v1','Исправленный масштаб потока','Вместо прежнего усиления: исправленное усиление потока при применении в трекере.','Исправление масштаба не дало подтверждённой пользы стенду.'),
 ]),
 ('06', 'ОБЪЕКТНОЕ СЧИТЫВАНИЕ', [
  ('fly_object_readout_z13_v1','З13: один средний образец','Вместо движения: сравнение каждой колонки с усреднённым образцом цели, 721 позиция + отказ.','Исходная объектная реализация не прошла критерии применения.'),
  ('fly_spatial_quality_z13_v2','Пространственный образец','Вместо одного вектора: до 64 колонок образца, координаты и сопоставление с каждой колонкой поиска.','Проверенная версия не стала подтверждённым решением для стенда.'),
  ('z13_candidate_foreground_v1','Цель относительно окружения','Вместо одного отклика на цель: вычитать обучаемую долю отклика на окружение вне расширенной рамки.','Прибавка к согласованной D без этого узла не подтверждена.'),
  ('z13_candidate_size_v1','Предсказание размеров','Вместо последнего наблюдаемого размера: две обучаемые поправки log(w), log(h) при тех же центрах.','IoU рамок ухудшился. Исходный размер здесь — наблюдённая рамка, не прогноз Калмана.'),
  ('z13_target_model_probe_v1','Линейная модель цели','Вместо общего обученного сравнения: отдельная гребневая регрессия; цель 1 внутри наблюдаемой рамки, 0 вне.','Бинарная цель не превзошла прежнее считывание. Признаки мухи против входа — отдельный положительный эффект.'),
  ('z13_directional_context_v1','Направленные соседи','Вместо среднего соседей: дополнительно 170 направленных контрастов; всего 238 признаков.','Общий результат неоднозначен; частный выигрыш не означает устойчивый перенос.'),
  ('z13_context_transfer_v1','Перенос направленного контекста','Проверка выбранной версии C и контроля R1 на ранее использованном dev35.','Критерии локализации и выбора поправок не выполнены.'),
 ]),
 ('07', 'НАДЁЖНОСТЬ И ОТКАЗ', [
  ('z13_threshold_diagnostic_v1','Точный порог без округления','Вместо округлённого порога: повтор выбора точного значения и разбор исходов.','Надёжность исходной реализации этим не исправлена.'),
  ('z13_tree_quality_probe_v1','Деревья вместо нейросети','Вместо нейросетевой оценки качества: деревья на наблюдаемых признаках положения.','Проверенная замена не прошла критерий.'),
  ('z13_score_swap_v1','Старый балл + новый центр','Вместо нового балла при новом центре: балл прежнего считывателя D.','Улучшение отбора не подтверждено.'),
  ('z13_next_utility_v1','Оценка полезности ответа','Вместо одной высоты отклика: обучаемая оценка по геометрии и наблюдаемым признакам.','Основной результат неоднозначен.'),
  ('z13_map_utility_v1','14 признаков карты и геометрии','Вместо ограниченной оценки: 6 геометрических + 8 сводных признаков карты ответа.','Критерии основного сравнения не выполнены; исходные центры фиксированы.'),
  ('z13_map_transfer_v1','Перенос оценки полезности','Вместо оценки только на train26: проверка сохранённой модели C/D/R1 на dev35.','Требуемая переносимость выбора не подтверждена.'),
  ('z13_pairwise_utility_v1','Парное обучение ранжированию','Вместо поточечной потери: добавить парную потерю на тех же 14 признаках и фиксированных центрах.','Все четыре основных интервала включают ноль. Новая память цели не добавлена.'),
 ]),
 ('08', 'ПРИМЕНЕНИЕ В СТЕНДЕ', [
  ('fly_motion_trust_cheap_v1','Простые правила доверия','Вместо безусловной поправки потока: меньший вес и отказ при недостатке колонок в рамке.','Проверенные правила не дали нужной пользы.'),
  ('fly_motion_trust_learned_v1','Обучаемое доверие','Вместо простого правила: оценка доверия с признаками мухи и контроль без них.','Первый вариант не принят; затем проверен исправленный протокол.'),
  ('fly_motion_trust_learned_v2','Исправленный протокол доверия','Вместо прежней процедуры: порог на видео подбора, данные в условиях применения.','Исправленная версия также не дала нужного выигрыша.'),
  ('fly_motion_photometric_v1','Фотометрическая проверка','Вместо прежней оценки доверия: проверять согласованность яркости после переноса изображения.','Испытанная проверка не дала нужного выигрыша.'),
  ('fly_spatial_hota_policy_z13_v1','Применение З13 с отказом','Вместо стенда без поправки: центры прежней D при одной политике применения, затем та же MASA.','Полезный прирост HOTA не подтверждён. Это не испытание нового центрированного считывателя.'),
 ])
]

BASE = [
 [('КАДР ВИДЕО','BDD100K MOT → изображение текущего кадра. YOLO26 относится к окружению; разметка будущего кадра не подаётся в FlyVis.','tracking/flyvis_motion.py'),
  ('ИСХОДНЫЙ ВХОД','Оттенки серого → gray/255. Штатная модель обучалась на Sintel для движения. Ретина и lcn24 — отдельные проверенные входные варианты, не обязательные части З13.','tracking/retina.py'),
  ('ГРАНИЦА МОДЕРНИЗАЦИИ','Можно менять подготовку входа глаза. Успех входного узла проверяется отдельно от применения его перед YOLO.','AGENTS.md')],
 [('ОБЛАСТЬ НАБЛЮДЕНИЯ','Полный кадр / фиксированные плитки. В объектной ветви: квадрат вокруг прогноза, сторона max(2·max(w,h), 64 px), сдвиг внутрь кадра.','tracking/foveated_eyes.py'),
  ('BOXEYE — ГЕКСАГОНАЛЬНЫЙ ГЛАЗ','Усреднение и выборка на шестиугольной решётке. extent = 15; 721 колонка; kernel_size = 13 в конфигурации модели 000. Вход эпизода: B × T × 721.','runs/flyvis/results/flow/0000/000/_meta.yaml'),
  ('КООРДИНАТЫ','Кадр ↔ квадрат глаза ↔ ближайшая колонка. Каждая колонка имеет до 6 соседей; отсутствующие краевые соседи заменяются самой колонкой.','tracking/z13_readout.py')],
 [('КОННЕКТОМ — СХЕМА СВЯЗЕЙ','fib25-fib19_v2.2: пространственно повторённые типы клеток и связи. Модель рекуррентная; это не последовательность полносвязных слоёв перцептрона.','runs/flyvis/results/flow/0000/000/_meta.yaml'),
  ('ОБНОВЛЕНИЕ НЕЙРОНА','Вход + смещение + сумма взвешенных ответов соседей − текущая активность. ReLU (обнуление отрицательного ответа); изменение делится на постоянную времени.','tracking/foveated_eyes.py'),
  ('ЧТО ОБУЧАЕТСЯ В D','734 свободных параметра: смещения нейронов, постоянные времени, силы синапсов. C — штатное обученное ядро; D — дообученное. Знаки и число синапсов не входят в эти три группы.','tracking/z13_readout.py')],
 [('ИНИЦИАЛИЗАЦИЯ','Плавное предъявление первого изображения, fade-in (выход на начальное состояние). Затем обновление по последовательности; состояние сохраняется между шагами.','tracking/z13_readout.py'),
  ('ВРЕМЯ В ОБЪЕКТНОЙ ВЕТВИ','4 последних наблюдения создают образец; затем идут кадры поиска. Между кадрами — интерполяция входа и шаги интегрирования. При 30 Гц и max dt=0,02: два шага на кадр.','tracking/flyvis_motion.py'),
  ('ДВА РАЗНЫХ ВИДА ПАМЯТИ','Состояние ядра продолжает обновляться. Образец / коэффициенты модели цели в текущем эпизоде фиксированы. Четыре кадра образца не равны всей временной истории.','scripts/z13_target_model.py')],
 [('АКТИВНОСТЬ ПО КОЛОНКАМ','В З13: последний шаг интегрирования → 34 выходных типа × 721 колонка. T1/T2/T2a/T3; T4a–d, T5a–d; семейства Tm и TmY.','tracking/flyvis_activity.py'),
  ('ИСХОДНЫЙ ДЕКОДЕР ДВИЖЕНИЯ','DecoderGAVP, конфигурация [8,2], ядро 5 → две компоненты потока по колонкам. Декодер движения и объектный считыватель — альтернативные выходные ветви.','runs/flyvis/results/flow/0000/000/_meta.yaml'),
  ('ПЕРЕВОД В ПИКСЕЛИ','Агрегация внутренних шагов; исходный отклик; масштаб вырезки и усиление; изменение знака вертикальной компоненты. Выход: поле движения, а не личность цели.','tracking/flyvis_motion.py')],
 [('З13: ПЕРВЫЙ ОБЪЕКТНЫЙ ВЫХОД','Стандартизация 34 каналов → линейное вложение в 16 чисел → среднее по наблюдаемой рамке. Косинусное сходство образца и каждой колонки поиска.','tracking/z13_readout.py'),
  ('ПРОСТРАНСТВЕННАЯ ВЕРСИЯ','Вложение 32; колонка + среднее соседей; до 64 позиций образца; позиционные признаки и внимание (взвешенное сопоставление). 721 балл + балл отказа.','tracking/z13_spatial.py'),
  ('ИНДИВИДУАЛЬНАЯ МОДЕЛЬ ЦЕЛИ','34 канала + среднее соседей = 68 признаков. Гребневая регрессия с регуляризацией, отдельно по последнему наблюдаемому образцу. Выход — карта поиска.','tracking/z13_target_center.py')],
 [('ЦЕНТР ИЗ КАРТЫ','Максимальная колонка + её 6 соседей → локальное взвешенное среднее → координаты кадра. Далёкие максимумы не усредняются в один центр.','scripts/z13_target_model.py'),
  ('ОЦЕНКА УВЕРЕННОСТИ','В исходном З13: 1 − вероятность отказа. В модели цели: нормированная высота пика. Ни один из этих баллов сам по себе не равен вероятности полезной поправки.','tracking/z13_readout.py'),
  ('ОЦЕНКА ПО ГОТОВОМУ ОТВЕТУ','6 признаков геометрии + 8 сводок карты: пики, разность, энтропия, массы и разброс. Это внешний выбор применения ответа; он не изменяет ядро и полученный центр.','tracking/z13_response_utility.py')],
 [('ОКРУЖЕНИЕ: YOLO26s','Детекции объектов. Сохранённый детектор обучен при 640; в стенде запуск 960, conf 0,01. Вход FlyVis не заменяет детектор.','AGENTS.md'),
  ('XYWHByteTrack → MASA','Прогноз положения и сопоставление детекций; после получения траекторий — процедура сшивки с признаками MASA. Поправка FlyVis применяется отдельной политикой.','AGENTS.md'),
  ('ПРОВЕРКА ПОЛЬЗЫ','TrackEval → HOTA, парный разброс по видео. Контроли: без правки; та же правка без мухи; правка с мухой. Частный выигрыш центра не заменяет это сравнение.','AGENTS.md')]
]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    previous_validation=OUT/'validation.json'
    existing_diagram=OUT/(STEM+'.drawio')
    if existing_diagram.exists():
        if not previous_validation.exists():
            raise RuntimeError('Existing draw.io has no generation receipt; preserve and review it before regeneration.')
        receipt=json.loads(previous_validation.read_text(encoding='utf-8'))
        if sha(existing_diagram)!=receipt.get('outputs',{}).get(existing_diagram.name):
            raise RuntimeError('Manual draw.io edits detected. Preserve and reconcile them; refusing to overwrite.')
    catalogue = json.loads((ROOT/'experiments/project_history.json').read_text(encoding='utf-8'))
    cards = {x['id']: x for x in catalogue['experiments']}
    nodes, edges, sources, linked = [], [], {}, set()
    fonts = {}
    def font(size, bold=False):
        key=(size,bold)
        if key not in fonts: fonts[key]=ImageFont.truetype(BOLD if bold else FONT,size)
        return fonts[key]
    def wrap(text, width, size, bold=False):
        result=[]; f=font(size,bold)
        for para in text.split('\n'):
            line=''
            for word in para.split(' '):
                candidate=(line+' '+word).strip()
                if f.getlength(candidate)>width and line:
                    result.append(line);line=word
                else: line=candidate
                while f.getlength(line)>width:
                    n=max(1,int(len(line)*width/f.getlength(line))-1)
                    result.append(line[:n]);line=line[n:]
            result.append(line)
        return result
    def source(path):
        p=ROOT/path
        if not p.is_file(): raise ValueError(f'Missing source: {path}')
        sources[path]=sha(p)
    def node(id,x,y,w,h,title,body='',kind='base',ref='',card='',size=22):
        title_lines=wrap(title,w-40,25,True)
        body_lines=wrap(body,w-40,size)
        ref_lines=wrap(ref,w-40,15) if ref else []
        used=24+len(title_lines)*32+12+len(body_lines)*(size+7)+len(ref_lines)*20+20
        if used>h: raise ValueError(f'Text overflow {id}: {used}>{h}')
        if ref and (ROOT/ref).is_file(): source(ref)
        if card: linked.add(card)
        d=dict(id=id,x=x,y=y,w=w,h=h,title=title_lines,body=body_lines,footer=ref_lines,
               kind=kind,ref=ref,card=card,size=size)
        nodes.append(d);return id
    def edge(a,b,kind='flow',points=None,start='right',end='left',label='',label_at=None):
        edges.append(dict(a=a,b=b,kind=kind,points=points,start=start,end=end,label=label,label_at=label_at))
    from scripts.flyvis_architecture_layout import build_layout, EDGE_STYLES
    def confirmed(cid, rid):
        c=cards[cid]; r=next(x for x in c['confirmed_results'] if x['id']==rid)
        ev=r['evidence'];path=ROOT/ev['source']
        if sha(path)!=ev['sha256']:raise ValueError('Evidence hash changed: '+rid)
        obj=json.loads(path.read_text(encoding='utf-8'))
        for k in ev['locator']:obj=obj[k]
        lo,hi=obj['ci95']
        if lo<=0:raise ValueError('Nonpositive confirmed interval')
        fmt=lambda x:f'{100*x:+.2f}'.replace('.',',')
        source(ev['source']);linked.add(cid)
        return f"{fmt(obj['mean_per_video'])} п.п.; 95% ДИ [{fmt(lo)}; {fmt(hi)}]",ev['source']
    linked.update(('flyvis_retina_stageB_v1','fly_input_norm_v1'))
    source('scripts/z13_score_swap.py')
    source('scripts/build_flyvis_architecture.py')
    source('scripts/flyvis_architecture_layout.py')
    W,H,DETAIL=build_layout(node,edge,source,cards,LANES,BASE,confirmed)
    byid={n['id']:n for n in nodes}
    # XML is intentionally uncompressed: every block and edge stays editable.
    mx=ET.Element('mxfile',host='app.diagrams.net',agent='FlyVis architecture generator',version='24.7.17')
    diagram=ET.SubElement(mx,'diagram',id='flyvis-single-sheet',name='FlyVis — архитектура')
    model=ET.SubElement(diagram,'mxGraphModel',dx='1800',dy='1000',grid='1',gridSize='10',guides='1',tooltips='1',connect='1',arrows='1',fold='1',page='1',pageScale='1',pageWidth=str(W),pageHeight=str(H),math='0',shadow='0')
    root=ET.SubElement(model,'root');ET.SubElement(root,'mxCell',id='0');ET.SubElement(root,'mxCell',id='1',parent='0')
    # Nodes appear above edges in the visual rendering, with a common explicit coordinate model.
    def port(n,side):
        return {'left':(n['x'],n['y']+n['h']/2),'right':(n['x']+n['w'],n['y']+n['h']/2),
                'top':(n['x']+n['w']/2,n['y']),'bottom':(n['x']+n['w']/2,n['y']+n['h'])}[side]
    def edgepoints(e):
        a,b=byid[e['a']],byid[e['b']]
        pts=[port(a,e['start']),*(e['points'] or []),port(b,e['end'])]
        return [p for i,p in enumerate(pts) if i==0 or p!=pts[i-1]]
    badges=[]
    for i,e in enumerate(edges):
        style=EDGE_STYLES[e['kind']]
        colour=style['colour'];dashed=style['dashed'];sw=style['width']
        anchors={'left':(0,.5),'right':(1,.5),'top':(.5,0),'bottom':(.5,1)}
        ex,ey=anchors[e['start']];ix,iy=anchors[e['end']]
        # Explicit endpoints preserve the reviewed routing in draw.io as in the preview.
        cell=ET.SubElement(root,'mxCell',id='edge'+str(i),parent='1',edge='1',
            style=f'edgeStyle=none;rounded=1;arcSize=20;html=0;endArrow=block;endFill=1;endSize=18;strokeWidth={sw};strokeColor={colour};dashed={int(dashed)};dashPattern=12 8;exitX={ex};exitY={ey};exitDx=0;exitDy=0;entryX={ix};entryY={iy};entryDx=0;entryDy=0;')
        geo=ET.SubElement(cell,'mxGeometry',relative='1',**{'as':'geometry'})
        pts=edgepoints(e)
        ET.SubElement(geo,'mxPoint',x=str(pts[0][0]),y=str(pts[0][1]),**{'as':'sourcePoint'})
        ET.SubElement(geo,'mxPoint',x=str(pts[-1][0]),y=str(pts[-1][1]),**{'as':'targetPoint'})
        arr=ET.SubElement(geo,'Array',**{'as':'points'})
        for x,y in pts[1:-1]:ET.SubElement(arr,'mxPoint',x=str(x),y=str(y))
        cell.set('source',e['a']);cell.set('target',e['b'])
        e['render_points']=pts;e['colour']=colour;e['dashed']=dashed;e['width']=sw;e['level']=style['level'];e['arrow']=style['arrow']
        if e['label']:
            cx,cy=e['label_at'];bw=font(22,True).getlength(e['label'])+28
            badges.append(dict(id='badge'+str(i),x=cx-bw/2,y=cy-21,w=bw,h=42,text=e['label'],colour=colour))
    for n in nodes:
        fill,stroke=PALETTE[n['kind']]
        obj=ET.SubElement(root,'object',id=n['id'],label='',module_status=n['kind'],source=n['ref'])
        if n['ref'] and (ROOT/n['ref']).is_file():obj.set('link',(ROOT/n['ref']).as_uri())
        if n['card']:
            c=cards[n['card']];obj.set('experiment',n['card']);obj.set('catalogue_state',c['state'])
            candidates=sorted((ROOT/'ObsidianFlyVis').rglob(n['card']+'.md'))
            if candidates:obj.set('link',candidates[0].as_uri())
            for field in ('question','hypothesis','interpretation','decision','cannot_claim','evidence','acceptance'):
                obj.set(field,json.dumps(c.get(field,''),ensure_ascii=False))
        value='<b>'+'<br>'.join(map(html.escape,n['title']))+'</b><br><br>'+'<br>'.join(map(html.escape,n['body']))
        if n['footer']:value+='<br><br><font style="font-size:15px;color:#52647A">'+'<br>'.join(map(html.escape,n['footer']))+'</font>'
        obj.set('label',value)
        cell=ET.SubElement(obj,'mxCell',parent='1',vertex='1',style=f'rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};strokeWidth=2;align=left;verticalAlign=top;spacing=20;fontFamily=Segoe UI;fontSize={n["size"]};fontColor=#182B40;dashed={int(n["kind"]=="future")};')
        ET.SubElement(cell,'mxGeometry',x=str(n['x']),y=str(n['y']),width=str(n['w']),height=str(n['h']),**{'as':'geometry'})
    for b in badges:
        cell=ET.SubElement(root,'mxCell',id=b['id'],parent='1',vertex='1',value=b['text'],
            style=f'rounded=1;arcSize=30;whiteSpace=wrap;html=0;fillColor=#FFFFFF;strokeColor={b["colour"]};strokeWidth=2;fontColor={b["colour"]};fontSize=22;fontStyle=1;fontFamily=Segoe UI;align=center;verticalAlign=middle;')
        ET.SubElement(cell,'mxGeometry',x=str(b['x']),y=str(b['y']),width=str(b['w']),height=str(b['h']),**{'as':'geometry'})
    xmlpath=OUT/(STEM+'.drawio');ET.indent(mx);ET.ElementTree(mx).write(xmlpath,encoding='utf-8',xml_declaration=True)
    # Same-layout SVG and PNG are navigation previews, not extra diagram sheets.
    svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
         '<rect width="100%" height="100%" fill="#FFFFFF"/>']
    im=Image.new('RGB',(W,H),'white');draw=ImageDraw.Draw(im)
    for e in edges:
        pts=e['render_points'];col=e['colour']
        svg.append('<polyline points="'+' '.join(f'{x},{y}' for x,y in pts)+f'" fill="none" stroke="{col}" stroke-width="{e["width"]}" stroke-linejoin="round"'+(' stroke-dasharray="12 8"' if e['dashed'] else '')+'/>')
        for p,q in zip(pts,pts[1:]):
            if e['dashed']:
                dist=math.dist(p,q)
                for off in range(0,int(dist),20):
                    v=[(p[0]+(q[0]-p[0])*t/max(dist,1),p[1]+(q[1]-p[1])*t/max(dist,1)) for t in (off,min(off+12,dist))]
                    draw.line(v,fill=col,width=e['width'])
            else:draw.line([p,q],fill=col,width=e['width'])
        p,q=pts[-2:];angle=math.atan2(q[1]-p[1],q[0]-p[0]);length=e['arrow'];arr=[q,(q[0]-length*math.cos(angle-.5),q[1]-length*math.sin(angle-.5)),(q[0]-length*math.cos(angle+.5),q[1]-length*math.sin(angle+.5))]
        draw.polygon(arr,fill=col);svg.append('<polygon points="'+' '.join(f'{x},{y}' for x,y in arr)+f'" fill="{col}"/>')
    for n in nodes:
        x,y,w,h=n['x'],n['y'],n['w'],n['h'];fill,stroke=PALETTE[n['kind']]
        draw.rounded_rectangle((x,y,x+w,y+h),radius=14,fill=fill,outline=stroke,width=2)
        svg.append(f'<g id="{n["id"]}"><title>{html.escape(n["ref"])}</title><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
        ty=y+22
        for lines,size,bold,col in [(n['title'],25,True,'#182B40'),(n['body'],n['size'],False,'#182B40'),(n['footer'],15,False,'#52647A')]:
            for line in lines:
                draw.text((x+20,ty),line,font=font(size,bold),fill=col)
                svg.append(f'<text x="{x+20}" y="{ty+size}" font-family="Segoe UI,Arial,sans-serif" font-size="{size}" font-weight="{700 if bold else 400}" fill="{col}">{html.escape(line)}</text>')
                ty+=size+7
            ty+=12
        svg.append('</g>')
    for b in badges:
        x,y,w,h=b['x'],b['y'],b['w'],b['h'];col=b['colour']
        draw.rounded_rectangle((x,y,x+w,y+h),radius=10,fill='white',outline=col,width=2)
        draw.text((x+14,y+4),b['text'],font=font(22,True),fill=col)
        svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="white" stroke="{col}" stroke-width="2"/><text x="{x+14}" y="{y+27}" font-family="Segoe UI,Arial,sans-serif" font-size="22" font-weight="700" fill="{col}">{html.escape(b["text"])}</text>')
    svg.append('</svg>');(OUT/(STEM+'.svg')).write_text('\n'.join(svg),encoding='utf-8')
    im.save(OUT/(STEM+'.png'),optimize=True)
    ratio=min(2400/W,2400/H)
    im.resize((round(W*ratio),round(H*ratio)),Image.Resampling.LANCZOS).save(OUT/(STEM+'_preview.png'))
    # One detail view of the SAME sheet, useful for visual QA and user preview.
    detail=im.crop(DETAIL);detail.thumbnail((1600,2200));detail.save(OUT/(STEM+'_detail.png'))
    source('experiments/project_history.json')
    dossier=dict(date=DATE,scope='One sheet; architectural replacements, not all tracker experiments',source_sha256=sources,
        cards={cid:dict(source='experiments/project_history.json',locator=f'experiments[id={cid}]',state=cards[cid]['state'],evidence=cards[cid].get('evidence',[])) for cid in sorted(linked)},nodes=nodes,edges=edges)
    (OUT/(STEM+'_sources.json')).write_text(json.dumps(dossier,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    # Structural acceptance: one diagram, unique IDs, live sources and catalogue entries, no box overlap.
    overlaps=[]
    for i,a in enumerate(nodes):
        for b in nodes[i+1:]:
            if min(a['x']+a['w'],b['x']+b['w'])>max(a['x'],b['x']) and min(a['y']+a['h'],b['y']+b['h'])>max(a['y'],b['y']):overlaps.append([a['id'],b['id']])
    assert not overlaps,overlaps
    # Check routed lines, not just box placement. A line may touch its own
    # source/target but must not run through another node's interior.
    def segment_hits_rect(p,q,n):
        lo,hi=0.0,1.0
        bounds=((n['x']+3,n['x']+n['w']-3),(n['y']+3,n['y']+n['h']-3))
        for axis,(lower,upper) in enumerate(bounds):
            delta=q[axis]-p[axis]
            if abs(delta)<1e-12:
                if p[axis]<=lower or p[axis]>=upper:return False
            else:
                a,b=sorted(((lower-p[axis])/delta,(upper-p[axis])/delta))
                lo=max(lo,a);hi=min(hi,b)
                if lo>=hi:return False
        return lo<hi
    blocked=[]
    for e in edges:
        for n in nodes:
            if n['id'] in (e['a'],e['b']):continue
            if any(segment_hits_rect(p,q,n) for p,q in zip(e['render_points'],e['render_points'][1:])):
                blocked.append([e['a'],e['b'],n['id']])
    assert not blocked,blocked
    ids=[x.get('id') for x in mx.iter() if x.get('id')]
    assert len(ids)==len(set(ids))
    assert len(mx.findall('diagram'))==1
    audit=dict(diagram_sheets=1,nodes=len(nodes),edges=len(edges),linked_experiments=len(linked),source_files=len(sources),box_overlaps=overlaps,
        all_text_fits=True,confirmed_source_hashes_verified=True,canvas=[W,H],arrow_levels=EDGE_STYLES,edges_through_unrelated_nodes=blocked,
        original_node_colour=PALETTE['base'][0],outputs={p.name:sha(p) for p in OUT.glob(STEM+'*') if p.suffix in ('.drawio','.svg','.png')})
    (OUT/'validation.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(audit,ensure_ascii=False))

if __name__=='__main__':main()
