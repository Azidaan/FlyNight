"""Retain the first note and correct an inverted enumeration in its prose."""
from common import *
path=OUT/'ПРОМЕЖУТОЧНЫЙ_ИТОГ.md';old=path.read_bytes()
archive=OUT/'ПРОМЕЖУТОЧНЫЙ_ИТОГ_v1.md'
with archive.open('xb') as f:f.write(old)
before='Именно5/6наблюдались в обучении,89/2случая соответственно; более длинные пропуски не проверены этой головой.'
after='В обучении пропуск6кадров встретился89раз,5кадров —2раза; более длинные пропуски не проверены этой головой.'
assert old.decode('utf8').count(before)==1
path.write_bytes(old.decode('utf8').replace(before,after).encode('utf8'))
journal('Уточнение порядка перечисления пропусков в промежуточной записке',
    'В `runs/full_template_native_20261109/ПРОМЕЖУТОЧНЫЙ_ИТОГ.md` исправлено слово «соответственно»:89случаев относятся к6кадрам,2случая — к5кадрам, как уже правильно зафиксировано в первичныхfeatures.json иexecution_protocol.json. Первоначальная записка сохранена вПРОМЕЖУТОЧНЫЙ_ИТОГ_v1.md. Данные, область исполнителя, модели и выводы не менялись.')
print('Enumeration corrected; prior note retained',flush=True)
