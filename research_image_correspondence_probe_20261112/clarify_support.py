from common import *
a=read(OUT/'detector_box_audit.json');assert a['passed']
text=('Уточнение состава сравнения 574: 22 из 120 точек находятся вне рамки детектора и вне аннотированной рамки кандидата; ещё 28 находятся внутри рамки детектора, но вне аннотированной рамки кандидата. '
      'Поэтому все 50 точек нельзя приписывать одному снятию ограничения по рамке детектора. Здесь разделяются расширенная поддержка шаблона и расхождение рамки детектора с разметкой. '
      'В исходном сравнении C34 сохранённый сдвиг составляет −4,255754475703325 пикселя по x и +2,1278772378516626 по y; relative_template_plan сначала переносит рамку, затем прибавляет сдвиг, а clip_to_candidate=False оставляет только проверку поля глаза. '
      'Причинный вклад каждой части в ложное разрешение ещё не проверен. Источники research_image_correspondence_probe_20261112/detector_box_audit.json, protocol.json, tracking/full_template_correspondence.py.')
path=OUT/'РЕЗУЛЬТАТ.md';s=path.read_text(encoding='utf8');assert '6. Уточнение границ' not in s
path.write_text(s+'\n6. Уточнение границ\n\n'+text+'\n',encoding='utf8')
journal(text)
src=(OUT/'plot_support.py').read_text(encoding='utf8')
src=src.replace("for ax in axes:","det=np.array(next(c['case']['current_box'] for c in read(OUT/'protocol.json')['cases'] if c['case']['old_id']==574))\nfor ax in axes:")
src=src.replace("(m['current_candidate_GT_box'],'orange','Другой объект 00115011')", "(m['current_candidate_GT_box'],'orange','Разметка кандидата 00115011'),(det,'red','Рамка детектора')")
src=src.replace('70 точек внутри рамки кандидата','70 точек внутри разметки кандидата').replace('50 точек вне рамки кандидата','50 точек вне разметки кандидата').replace('Две аннотированные рамки','Разметка двух объектов и детекция').replace('области, выходящей за рамку кандидата','области вне разметки кандидата').replace("support_574.png","support_574_v2.png").replace("plot_manifest.json","plot_manifest_v2.json").replace("'plot_support.py'","'plot_support_v2.py'")
assert not (OUT/'plot_support_v2.py').exists();(OUT/'plot_support_v2.py').write_text(src,encoding='utf8')
print('CLARIFIED')
