"""Freeze corrected decoding and broaden input checks before any rollout."""
from common import *
start('cpu')
import gzip
from sparse_views import SparseViews
p=read(OUT/'protocol.json');verify(p)
assert read(OUT/'seek_diagnostic.json')['any_raw_seek_mismatch']
witnesses=[];sources={}
for video in p['videos']:
    events_path=LEGACY/'rollouts/G_R1'/(video+'.events.json.gz');sources[events_path.relative_to(ROOT).as_posix()]=sha(events_path)
    with gzip.open(events_path,'rt',encoding='utf8') as f:events=json.load(f)
    found={}
    for event in events:
        for row in event['rows']:
            for frame,crop in [[h['frame'],h['crop']] for h in row['history']]+row['search']:
                if frame<100:continue
                path=LEGACY/'retinal'/video/(SparseViews.key(frame,crop)+'.npy')
                if path.exists():found[(int(frame),tuple(crop))]=path
    frame,crop=min(found);path=found[(frame,crop)];sources[path.relative_to(ROOT).as_posix()]=sha(path)
    witnesses.append(dict(video=video,frame=frame,crop=list(crop),cached=path.relative_to(ROOT).as_posix()))
for path in [Path(__file__),OUT/'sequential_views.py',OUT/'witness_recovery.py',OUT/'protocol.json',OUT/'seek_diagnostic.json',OUT/'sparse_views.py',OUT/'common.py']:
    sources[path.relative_to(ROOT).as_posix()]=sha(path)
write('input_recovery_protocol.json',dict(registered=datetime.now(MSK).isoformat(),sources=sources,witnesses=witnesses,
    cause='Observed raw decoded frames at nominal100 and117 differ between VideoCapture.set and native sequential reading. This proves a decode-path discrepancy, not a neuronal defect or an established timestamp offset.',
    correction='Keep checkpoint state at next_frameN, open video at0,decode/discard raw0..N-1 WITHOUT reprocessing retina,then process requested suffix. No frame offset guessing or tolerance change.',
    checks='Original two witnesses tested both via checkpoint and from0; add one deterministic cached view at/after100 per ALL20 videos,checkpoint route. Require exact eye-array equality and per-frame dtype. No failed video skipped.',
    amendment='Дописано до прогонов: после выявленного различия seek-пути расширена проверка восстановления на все20 видео. Порогов,моделей,условий сопоставления и качественных исходов нет. Это исправление отсутствующих данных, не новая сигнальная диагностика.',
    unchanged='Original scientific protocol remains immutable. Old failed witness and its outputs are retained; no previous success is overwritten.'))
journal('Исправляется путь восстановления отсутствующих входов', 'Первый входной контроль завершился ошибкой: на кадре117 восстановление отcheckpoint дало максимум0,568359, а обработка отначала совпала с готовым видом точно. Независимо установлено различие сырых кадров при seek и последовательном чтении на номинальных кадрах100/117, `runs/full_template_native_20261109/seek_diagnostic.json`. Кадровый сдвиг не угадывается. Новый путь сохраняет состояние обработки, но декодирует сырой префикс последовательно, не обрабатывая его повторно; исходный протокол не меняется. До любых реальных назначений входной контроль расширен на все20 видео и зарегистрирован в `runs/full_template_native_20261109/input_recovery_protocol.json`.')
print('Corrected decode input gate registered for all20 videos',flush=True)
