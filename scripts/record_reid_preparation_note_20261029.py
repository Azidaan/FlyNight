from pathlib import Path
from datetime import datetime,timezone,timedelta
ROOT=Path(__file__).resolve().parents[1]
stamp=datetime.now(timezone(timedelta(hours=3))).strftime('%d.%m.%Y, %H:%M MSK')
text=f'''**{stamp}** — Прямо поручены MASA-Detic, MGOR и ScaleTrack; подготовка и красные Т10–Т12

Постановка. Автор поручил опробовать три метода, добавить их в граф и подготовить всё для более быстрого запуска. Новое узкое разрешение на официальные исходники/веса этих кандидатов; основной стенд и окружение не менять, общий07:40 сохраняется.

Получено. В experiments/tracking_hypotheses.json добавлены Т10–Т12, в project_history.json — planned reid_masa_detic_v1/reid_mgor_v1/tracking_scaletrack_v1. Протокол и технические препятствия: runs/continuation_20261028/reid_candidates_preparation/protocol.json и ПОДГОТОВКА.md. availability.json содержит14сохранённых первичных текстов/метаданных с SHA, снимки официальных деревьев и проверку локальных зависимостей; OpenReviewAPIвернул403. MASA-Detic требует отсутствующие mmdet/mmcv/mmengine либо отдельный проверенный перенос. MGORконфиг ссылается на локальный обученный файл, опубликованные веса не найдены; ScaleTrackисполнимый код/веса не подтверждены. Это не отрицательные результаты методов. Загрузка MASA-Detic ведётся отдельно, итог только по download_status.json.

Вывод. Ни новой HOTA, ни завершённой проверки трёх методов пока нет. Граф проверен на копии: tracking_reid_added_20261029_032755/audit.json — 53видимых/1359сохранённых/24узла гипотез/23внутреннихребра/4луча/0внешних/0удалений. Б1–Б3, старыеТ1–Т9 и P1–P11 побайтно сохранены. Новые узлы красные; зелёный только после завершённой проверки и сохранённого исхода.

Решение. Продвигать подготовку MASA-Detic без повторения готовых расчётов и без изменения текущей матрицы; MGOR/ScaleTrack довести до проверки доступности необходимых артефактов, не имитировать отсутствующие обученные модели. При фактической готовности зарегистрировать отдельный GPU-протокол и запуск через существующую очередь. Полный итог всех трёх доутра не обещан.

'''
plan=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md'
body=plan.read_text(encoding='utf-8');marker='## Журнал изменений'
assert body.count(marker)==1
plan.write_text(body.replace(marker,marker+'\n\n'+text,1),encoding='utf-8')
with (ROOT/'runs/continuation_20261028/HANDOFF.md').open('a',encoding='utf-8') as f:f.write('\n\n'+text)
print(stamp)
