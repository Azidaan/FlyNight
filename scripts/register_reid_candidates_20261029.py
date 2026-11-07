"""Record explicit author scope before trials; preserve all previous candidates."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib, json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/continuation_20261028/reid_candidates_preparation'

def write(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    OUT.mkdir(exist_ok=True)
    source=ROOT/'experiments/tracking_hypotheses.json'
    original=source.read_bytes()
    backup=OUT/'tracking_hypotheses_before.json'
    if not backup.exists():backup.write_bytes(original)
    data=json.loads(original)
    assert len(data['methods'])==9, 'Register once; never duplicate'
    request='Давай попробуем MASA-Detic и Re-ID — MGOR, и ScaleTrack, тоже запиши их в графы, Опробуем всё. Как всегда красный не опробовано - зелёный опробовано'
    data['additional_author_requests']=[dict(date='2026-10-29',text=request)]
    data['numbering']='Т1–Т12 здесь номера кандидатов ветки Трекинг; это не исторические этапы Т1–Т5 раздела7.8 Единого плана и не Б1–Б3.'
    common='Полная проверка на одинаковых детекциях YOLO26 и заранее фиксированном наборе видео; HOTA/AssA/IDF1/FP/FN/IDSw и парные интервалы. Подготовка и загрузка весов не завершают научную проверку.'
    methods=[dict(id='T10',title='Т10 — Различать объекты более сильными признаками — MASA-Detic',
        status='planned',role='Замена извлекателя внешности при сохранённом детекторе',
        mechanism='Swin-B и адаптер MASA дают признаки конкретного объекта по заданной рамке; сравнение с имеющейся MASA-R50.',
        question='Уменьшает ли замена признаков ошибки личности при прежних детекциях и правилах памяти?',
        limitation='Авторская таблица использует другие детекции и не доказывает прирост нашего стенда. Официальный путь зависит от отсутствующих mmdet/mmcv/mmengine; нужен проверенный перенос без изменения основного окружения. Старые обученные головы нельзя молча подключить к новому распределению признаков.',
        check=common,paper='https://openaccess.thecvf.com/content/CVPR2024/html/Li_Matching_Anything_by_Segmenting_Anything_CVPR_2024_paper.html',
        code='https://github.com/siyuanliii/masa',code_review='Официальный BDD-конфиг и Swin-B прочитаны, исходники/SHA сохранены в reid_candidates_preparation. Вес detic_masa.pth опубликован; исполнения нового извлекателя пока нет.'),
      dict(id='T11',title='Т11 — Узнавать экземпляры разных классов — MGOR',status='planned',
        role='Кандидат универсального повторного распознавания объектов',
        mechanism='Обучение различению экземпляров с переносом между категориями объектов; работа Object-Generalized Re-Identification.',
        question='Лучше ли MGOR различает похожие объекты разных классов нашей выборки, чем текущие признаки?',
        limitation='Официальный код обучения есть, ссылка на готовые обученные веса пока не найдена. Конфиг ожидает локальный logs/transformer_meta_60.pth. Случайная инициализация или обычный ViT не являются обученным MGOR. Метрика поиска изображений не равна HOTA.',
        check=common,paper='https://openaccess.thecvf.com/content/CVPR2026/papers/Chen_Object-Generalized_Re-Identification_A_Step_Towards_Universal_Instance_Perception_CVPR_2026_paper.pdf',
        code='https://github.com/whucsy/OG-ReID',code_review='Дерево официального репозитория, конфигурация, требования и скрипт обучения сохранены с SHA. Готовые веса и воспроизводимый вывод ещё не подтверждены.'),
      dict(id='T12',title='Т12 — Сохранять личность через последовательность соответствий — ScaleTrack',status='planned',
        role='Кандидат замены модуля сопоставления, а не только извлекателя',
        mechanism='Цепочечное контрастивное обучение и пространственно-временное внимание для согласованной личности между кадрами.',
        question='Улучшает ли обученное временное сопоставление сопровождение сверх независимых признаков внешности?',
        limitation='Имеется текст заявки Scaling Open-World Multiple Object Tracking; публичная исполнимая реализация и обученные веса не подтверждены. Не путать с одноимёнными GUI-агентом и робототехническим проектом. Статус принятия статьи не подтверждён; OpenReview запрос вернул403.',
        check=common+' До реализации подтвердить причинность: никакого доступа к будущим кадрам.',
        paper='https://openreview.net/forum?id=INnZIRw2XE',code=None,
        code_review='В проверенном списке публичных репозиториев автора связанный ScaleTrack не найден; в доступной версии статьи код обещан. Это препятствие воспроизведению, не отрицательный научный результат.')]
    data['methods'].extend(methods)
    write(source,data)
    protocol=dict(version=1,created=datetime.now(timezone(timedelta(hours=3))).isoformat(),
        author_request=request,preparation_request='Можешь подготовить всё для этих reid? хочется их быстрее проверить',
        scope='Три конкретных кандидата, разрешена подготовка необходимых официальных исходников и весов; не новые внешние наборы данных и не изменение основного окружения.',
        hard_stop='2026-10-29T07:40:00+03:00',save_at='2026-10-29T07:30:00+03:00',
        existing_priority='Существующие матрица и Б2 сохраняются; одна очередь/один GPU. Пока доступны только подготовительные CPU/сетевые действия.',
        hypotheses=methods,scientific_verdict=None,training_authorized_by_this_protocol=False,
        required_before_gpu=['Официальный код/веса с SHA и лицензией','Отдельный исполнимый путь без смены основного окружения','Проверка численного соответствия оригиналу','Нулевой контроль при неизменных детекциях','Отдельный зарегистрированный протокол и очередь после готовности'],
        comparison='MASA-R50 против MASA-Detic и MGOR с одинаковым потребителем признаков и памятью. ScaleTrack меняет сопоставление: отдельное сравнение всей ветви, выигрыш не приписывать только Re-ID.',
        split='Только существующие train/dev для настройки. Исторический val для полного заранее фиксированного сравнения, не новый независимый итоговый тест.',
        colors='Красный: нет завершённой проверки с первичным результатом. Зелёный: проверка закончена, исход отдельно положительный/отрицательный/неопределённый. Техническая загрузка не делает научный узел зелёным.')
    write(OUT/'protocol.json',protocol)
    catalogue=ROOT/'experiments/project_history.json'
    (OUT/'project_history_before.json').write_bytes(catalogue.read_bytes())
    history=json.loads(catalogue.read_text(encoding='utf-8'))
    for m,identifier in zip(methods,['reid_masa_detic_v1','reid_mgor_v1','tracking_scaletrack_v1']):
        assert not any(c['id']==identifier for c in history['experiments'])
        history['experiments'].append(dict(id=identifier,title=m['title'],area='reid',state='planned',started='2026-10-29',
            question=m['question'],hypothesis=m['mechanism'],expected='Согласованная проверка качества полного трекера; положительный исход заранее не предполагается.',
            acceptance=[dict(criterion='Завершена полная проверка с источниками и независимым аудитом',met=None),
                        dict(criterion='Нижняя граница парного 95% интервала HOTA к согласованному контролю положительна',met=None)],
            cannot_claim=m['limitation'],decision='Подготовить код, веса и совместимость; GPU запуск только после регистрации отдельного полного протокола.',
            evidence=[dict(kind='file',ref='runs/continuation_20261028/reid_candidates_preparation/protocol.json',claim='Прямое поручение, постановка до расчёта, критерии готовности')],
            metrics=[],confirmed_results=[]))
    write(catalogue,history)
    print('Registered T10/T11/T12 and three planned experiments; no GPU launched')

if __name__=='__main__':main()
