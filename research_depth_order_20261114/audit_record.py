from common import *
import numpy as np,cv2
from scipy.stats import mannwhitneyu
from PIL import Image,ImageDraw,ImageFont
base=read(OUT/'result.json');high=read(OUT/'resolution_result.json');z=np.load(ROOT/'research_visible_part_2179_20261114/mask_review/masks_reviewed.npz');verified=[]
for protocol in ['protocol_v2.json','resolution_protocol.json']:
    for n,h in read(OUT/protocol)['inputs'].items():assert sha(ROOT/n)==h,n
for name,result in [('depth_maps.npz',base),('resolution_maps.npz',high)]:
    maps=np.load(OUT/name)
    for row in result['records']:
        f=row['frame'];d=maps[str(f)];a=d[z[f'{f}_target'].astype(bool)];b=d[z[f'{f}_neighbor'].astype(bool)];u=float(mannwhitneyu(b,a,method='asymptotic').statistic)/(len(a)*len(b));k='pair_rank_probability' if name=='depth_maps.npz' else 'rank_probability';assert abs(u-row[k])<1e-12
        medians=[float(np.quantile(a,.5)),float(np.quantile(b,.5))];given=[row['regions']['target']['median'],row['regions']['neighbor']['median']] if name=='depth_maps.npz' else [row['target_median'],row['neighbor_median']];assert medians==given;assert row['gate']==bool(medians[0]<medians[1] and u>.75)
        verified.append(dict(file=name,frame=f,rank_probability=u,target_median=medians[0],neighbor_median=medians[1],gate=row['gate']))
assert base['scientific_passed']==all(r['gate'] for r in base['records']);assert high['passed']==all(r['gate'] for r in high['records'])
save('audit.json',dict(passed=True,comparisons=verified,source_unchanged=True,hashes={n:sha(OUT/n) for n in ['result.json','resolution_result.json','depth_maps.npz','resolution_maps.npz']},scope='Independent rank-statistic and median recomputation; does not supply metric 3D truth.'))
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',20);small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17)
f=1165;im=cv2.imread(str(ROOT/f'research_visible_part_2179_20261114/frames/{f}.png'));s=Image.new('RGB',(1890,490),'#20232a');draw=ImageDraw.Draw(s);l,t,r,b=860,170,1280,450
for j,title in enumerate(['Исходный сложный кадр','Глубина: вход 518, порядок ошибочен','Глубина: вход 1036, порядок верен']):
    if j==0:pic=im[t:b,l:r]
    else:
        d=np.load(OUT/('depth_maps.npz' if j==1 else 'resolution_maps.npz'))[str(f)][t:b,l:r];lo,hi=np.quantile(d,[.02,.98]);pic=cv2.applyColorMap(np.uint8(255*np.clip((d-lo)/(hi-lo),0,1)),cv2.COLORMAP_TURBO)
        for k,col in [('target',(0,255,0)),('neighbor',(255,180,20))]:
            mask=z[f'{f}_{k}'][t:b,l:r].astype('uint8');contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(pic,contours,-1,col,1)
    s.paste(Image.fromarray(cv2.cvtColor(cv2.resize(pic,(630,420)),cv2.COLOR_BGR2RGB)),(630*j,40));draw.text((630*j+5,9),title,font=font,fill='white')
draw.text((7,465),'Тёплое — ближе; холодное — дальше. Шкала отдельно для каждой карты. Контуры ручные: зелёный — дальняя цель, голубой — сосед.',font=small,fill='white');s.save(OUT/'1165_сравнение_разрешения.png')
report='''Глубина как дополнительный признак различения машин

Готовая Depth Anything V2 Small проверена без обучения на трёх старых кадрах эпизода 2179. На вход подавался полный кадр; оценки извлекались только из прежних ручных видимых участков дальней цели и ближнего соседа. Модель выдаёт относительную обратную глубину: большее значение соответствует более близкой поверхности. Это не расстояния в метрах. Источники: protocol_v2.json, официальный репозиторий https://github.com/DepthAnything/Depth-Anything-V2 и пояснение автора https://github.com/DepthAnything/Depth-Anything-V2/issues/174.

При размере входа 518 правильный порядок медиан получен только на кадре 1159. На кадрах 1164 и 1165 он обратный. Вероятности правильного ранжирования случайной пары пикселей из ручных видимых участков соответственно 0,8684; 0,3268; 0,2761. Это доля правильно упорядоченных пар с половинным весом равенств, а не вероятность правильного ID. Источник: result.json. Основная заранее зарегистрированная проверка отрицательная.

После этого отдельно зарегистрировано увеличение размера входа до 1036 с теми же исходными кадрами, моделью и весами. Порядок стал правильным на всех трёх кадрах; вероятности ранжирования 0,9130; 0,8750; 0,8608. Источники: resolution_protocol.json и resolution_result.json. Это положительная последующая диагностика знакомого эпизода. Она не заменяет отрицательный исход первой проверки и не является независимой проверкой переноса.

Подтверждённый механизм на уровне настройки вычисления: изменение разрешения входа при остальных сохранённых условиях устранило два инвертированных порядка глубины. Какой внутренний процесс сети вызвал это, не установлено. Утверждать, что причиной был только размер машины в пикселях, пока нельзя: размер входа также меняет дискретизацию признаков и работу сети. Причина всех ошибок проекта этим не установлена.

Время на CPU с двумя потоками: медиана стандартного варианта около 2,4 секунды на кадр после одного прогрева; увеличенный вариант 33,2–40,5 секунды, первый из трёх кадров без отдельного прогрева нового размера. Это не измерение скорости на видеокарте и не точный сравнительный тест скорости. Оперативную память контролировал общий защитный запуск; записи находятся в queue_v2_execution.json и queue_resolution_execution.json. Точный пик памяти отдельной модели не измерен.

Практический смысл — проверять глубину как мягкое дополнительное свидетельство по видимым участкам. Усреднение всей рамки смешивает машину с человеком, баком и соседом. Две машины на одинаковой глубине этот признак не различит. Для дальнейшего опыта нужны новые видео, заранее закреплённое разрешение и сравнение того же сопоставления с глубиной и без неё. Полный трекер и ядро FlyVis в этом опыте не запускались, рабочие веса и пороги не менялись.

Исходники, веса, все входы и карты сохранены на диске E. Независимый пересчёт шести сравнений подтвердил медианы и ранжирование: audit.json. Ручные области — прежняя интерпретация видимых поверхностей, а не измеренная трёхмерная разметка.
'''
(OUT/'РЕЗУЛЬТАТ.md').write_text(report,encoding='utf8')
cat=read(ROOT/'experiments/project_history.json');card=next(c for c in cat['experiments'] if c['id']=='depth_order_2179_20261114');card.update(state='completed_negative',done='2026-11-14',obtained='При входе 518: правильный порядок 1/3, вероятности ранжирования 0,8684; 0,3268; 0,2761. Источник: research_depth_order_20261114/result.json. Последующая диагностика 1036: правильный порядок 3/3, вероятности 0,9130; 0,8750; 0,8608; resolution_result.json.',interpretation='Исходный вариант не прошёл допуск. При увеличении разрешения два обратных порядка исправлены на знакомом эпизоде. Вывод относится к настройке входа, не к доказанному внутреннему механизму сети.',decision='Сохранить глубину как кандидата дополнительного сигнала; для применения требуется отдельная проверка фиксированного варианта на новых видео.',cannot_claim='Метрическая глубина, надёжность на новых видео, улучшение личности или HOTA, реальная скорость на GPU. Последующая диагностика не заменяет исходную проверку.');
for a,v in zip(card['acceptance'],[False,False,True]):a.update(met=v,evidence='research_depth_order_20261114/audit.json')
card['evidence'] += [dict(kind='file',ref=rel(OUT/n)) for n in ['РЕЗУЛЬТАТ.md','result.json','resolution_protocol.json','resolution_result.json','audit.json']]
(ROOT/'experiments/project_history.json').write_text(json.dumps(cat,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
topics=read(ROOT/'experiments/obsidian_experiment_topics.json');topics['topics'][card['id']]=['Сохранение личности','Содержание области наблюдения','Порядок машин по относительной глубине'];(ROOT/'experiments/obsidian_experiment_topics.json').write_text(json.dumps(topics,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
lit=read(ROOT/'experiments/useful_literature.json');lit['sources'].append(dict(id='yang2024_depth_anything_v2',title='Оценка относительной глубины — Depth Anything V2',problem_group='Сохранение личности',method_group='Разделение перекрывающихся объектов',publication='Lihe Yang et al. Depth Anything V2. NeurIPS 2024.',url='https://arxiv.org/abs/2406.09414',sections='Официальная реализация, готовая малая модель и подготовка входного изображения.',established='Предобученная модель относительной глубины; Small содержит около 24,8 млн параметров. Это не доказательство качества сопоставления объектов.',problem='С1 — смешанные области наблюдения похожих машин.',application='Использованы официальные замороженные веса и исходники для трёх кадров 2179; стандартное и увеличенное разрешение. Первичный опыт не прошёл условие, последующая диагностика дала правильный порядок на знакомом эпизоде.',evidence=['research_depth_order_20261114/РЕЗУЛЬТАТ.md'],limitations='Нет измеренной глубины сцены и независимого переноса на новые видео.'))
(ROOT/'experiments/useful_literature.json').write_text(json.dumps(lit,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
journal('Получено depth_order_2179_20261114: готовая модель глубины при входе 518 дала правильный порядок машин на одном из трёх старых кадров; при отдельно зарегистрированном увеличении до 1036 — на всех трёх. Вероятности ранжирования соответственно 0,8684/0,3268/0,2761 и 0,9130/0,8750/0,8608; research_depth_order_20261114/result.json, resolution_result.json, audit.json. Вывод: разрешение входа причинно влияет на два наблюдаемых отказа; внутренний механизм не установлен. Решение: исходный научный статус отрицательный, положительную последующую диагностику сохранить отдельно; новые видео и сопоставление ещё не проверены.')
print('DEPTH_AUDITED_RECORDED')
