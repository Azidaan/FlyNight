"""Save the useful source and the precise limit of the shift-search idea."""
from evaluation_common import *
import subprocess
proof=read(OUT/'search_objective_math.json');assert proof['passed']
path=OUT/'МАТЕМАТИКА_ПОИСКА.md';assert not path.exists()
path.write_text('''# Выбор сдвига и оценка принадлежности

1. Подтверждено кодом

tracking/channel_identity_head.py выбирает сдвиг по среднему корреляций, затем вычисляет обученную оценку на каналах выбранного сдвига. Неравные веса допускают другое положение максимума. Для фактических весов новой C34 построен ограниченный по корреляциям контрпример: search_objective_math.json. Это алгебраический пример, а не измеренный отказ на видео.

2. Что исправление одного максимума не может сделать

При той же геометрии, допустимых сдвигах, коэффициентах и пороге максимум обученного балла по всем сдвигам не меньше его значения на прежнем выбранном сдвиге. Поэтому ранее разрешённая чужая связь останется разрешённой. Простая замена выбора сдвига может вернуть свою связь, но сама по себе не исправляет ложное разрешение чужой. Общий решатель всё ещё может изменить назначения; теорема относится к локальному разрешению.

3. Литературное основание и границы переноса

Bhat, Danelljan, Van Gool, Timofte. Learning Discriminative Model Prediction for Tracking, ICCV 2019. Проверены раздел 3.1, формулы 1–2: https://arxiv.org/pdf/1904.07220. В статье ошибка обучения определяется по пространственным положениям целевой и фоновой областей. Это основание различать обучаемую пространственную модель и замену операции максимума при прежних весах. Нашу теорему монотонности вывели отдельно; статья её не проверяет и пользу FlyVis не доказывает. Прямая загрузка CVF PDF вернула 403; доступная первичная версия arXiv прочитана в нужном разделе.

4. Что уже было у нас

Модель конкретной цели, односторонняя ошибка фона и коэффициенты фильтра уже имеют прежние проверки. Карточки z13_target_model_probe_v1, z13_hinge_target_v1, z13_filter_descriptor_fixed_v2 и их первичные отчёты сохраняются. Называть предложение «обучить пространственный классификатор по DiMP» новым методом нельзя. Возможную новую постановку нужно сопоставить с прежними данными, отрицательными примерами, правилом выбора сдвига и реальными ID. Новое обучение или смена правила поиска здесь не запускались.
''',encoding='utf8')
def atomic(path,data,before):
    temp=path.with_suffix(path.suffix+'.search_review.pending');assert not temp.exists();temp.write_bytes(data)
    for _ in range(30):
        assert path.read_bytes()==before
        try:temp.replace(path);return
        except PermissionError:time.sleep(.1)
    raise PermissionError(str(path))
path=ROOT/'experiments/useful_literature.json';before=path.read_bytes();lit=json.loads(before);identity='bhat2019_spatial_training_vs_posthoc_max'
assert all(s['id']!=identity for s in lit['sources'])
lit['sources'].append(dict(id=identity,title='Пространственное обучение и выбор максимума — DiMP',problem_group='Сохранение личности',method_group='Поиск и обучение пространственного считывателя',
    publication='Goutam Bhat, Martin Danelljan, Luc Van Gool, Radu Timofte. Learning Discriminative Model Prediction for Tracking. ICCV,2019,6182–6191; просмотрена первичная arXiv-версия2.',
    url='https://arxiv.org/pdf/1904.07220',doi='не подтверждён в просмотренном первоисточнике; используется arXiv:1904.07220',sections='Раздел3.1,формулы1–2; прямойCVFPDFвернул403, доступныйarXivпрочитан.',
    established='Модель цели обучается по пространственным положениям, с разным обращением с целевой и фоновой областями.',
    problem='Поиск сдвига по среднему каналов и оценка принадлежности по обученным весам используют разные критерии.',
    application='Источник помог отделить обучение пространственного различения от простой смены максимума. Для последней отдельно доказана монотонность: она не может отвергнуть ранее разрешённую чужую связь при прежней голове.',
    limitation='Теорема и её проверка наши; причинная роль расхождения критериев в реальных отказах не установлена. DiMP-подобные модели в проекте уже проверялись; это не новый готовый метод или доказанная польза FlyVis.',
    evidence=['runs/domain_matched_native_20261109/search_objective_math.json','runs/domain_matched_native_20261109/МАТЕМАТИКА_ПОИСКА.md','runs/z13_target_model_20261017/report.json','runs/z13_hinge_target_20261023/report.json','runs/z13_filter_descriptor_fixed_20261024/train/report.json'],related=['sys:С2']))
atomic(path,(json.dumps(lit,ensure_ascii=False,indent=2)+'\n').encode(),before)
path=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';before=path.read_bytes();nl=b'\r\n' if b'\r\n' in before else b'\n';marker='## Журнал изменений'.encode()+nl
entry='**'+datetime.now(MSK).strftime('%d.%m.%Y, %H:%M MSK')+'** — Проверена математика замены критерия поиска сдвига\n\n*Сделано.* До выбора следующей правки проверена идея заменить максимум среднего каналов максимумом обученного балла. Неравные веса допускают расхождение выбранных сдвигов; ограниченный алгебраический контрпример построен на фактических весах новой головы. При неизменных весах и пороге новый максимум не меньше прежней оценки, поэтому уже разрешённая чужая связь не может стать отвергнутой на локальном этапе. Изменение общего назначения этим не исключается. Источники runs/domain_matched_native_20261109/search_objective_math.json, МАТЕМАТИКА_ПОИСКА.md. Карты, обучение и трекер не повторялись.\n\n*Литература и прежние проверки.* Прочитан раздел3.1,формулы1–2 первичногоDiMP: https://arxiv.org/pdf/1904.07220; прямойCVFPDFвернул403. Полезный источник записан в experiments/useful_literature.json как различие пространственного обучения и простой смены максимума, без приписывания статье нашей теоремы. Прежние модели конкретной цели и ошибки фона найдены в каталоге; идея не объявляется новой. Рабочий пилот не изменён.\n\n'
atomic(path,before.replace(marker,marker+nl+entry.replace('\n',nl.decode()).encode(),1),before)
write('search_math_review_recording.json',dict(passed=True,source_sha256=sha(OUT/'search_objective_math.json'),literature_id=identity,goal_achieved=False))
print('SEARCH_MATH_AND_USEFUL_SOURCE_SAVED',flush=True)
