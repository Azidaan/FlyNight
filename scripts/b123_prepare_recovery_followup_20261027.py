"""Зафиксировать прямое поручение автора и готовые проверки после текущей серии."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib, json, sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RUN = ROOT / 'runs/b123_20261027'
MSK = timezone(timedelta(hours=3))


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    assert not path.exists(), str(path)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def main():
    grant = RUN / 'recovery_followup_authorization.json'
    auth = read(RUN / 'authorization.json')
    now = datetime.now(MSK)
    value = dict(created=now.isoformat(), user_request='После прогона текущей серией займись Буска и второй работой, до утра, нонстопом',
        original_authorization_sha256=sha(RUN/'authorization.json'), hard_stop=auth['hard_stop'],
        save_at=auth['save_at'], controller_stop=auth['controller_stop'], single_gpu=True,
        sequence=['Complete current B1 saved-query pilot and prepared B2/B3 implementation checks',
                  'BUSCA: reproducibility, bounded cost probe, matched tracker experiment if ready',
                  'LTR: exact paper reproduction readiness, mask recovery probe if ready'],
        scope='Продолжение прямо поручено автором; срок 27.10 07:30 МСК не продлён. Подготовка исходников допускается во время текущего GPU-прогона; вычисления новых методов только после текущей очереди. Данные BDD100K уже локальные: новые наборы данных не загружать. Необходимые публичные исходники и готовые веса допустимы в отдельной папке проекта после проверки размера, происхождения и оставшегося времени. Не менять основное окружение ради старых зависимостей.',
        scientific_gates='До каждого вычислительного этапа: карточка, фиксированные входы/веса/контроль/бюджет. Нулевая ветвь точно повторяет базу. Короткий замер GPU до полной серии; консервативная стоимость удвоенного замера на оставшиеся видео плюс оценка/сохранение должна укладываться до07:20. Один GPU; после каждого метода проверять конец собственного дерева.',
        benefit='Полный тот же трекер на тех же видео: база, простое продолжение без обученной проверки и авторский модуль. HOTA/AssA/IDF1/FP/FN/IDSw и парный 95% интервал по видео; положительная нижняя граница HOTA нужна для утверждения о пользе. Меньше пропусков само по себе не успех. Никакой настройки на итоговых видео.',
        stop='Нет доступной проверяемой реализации/весов или стоимость не помещается — сохранить точный блокер, перейти к следующему разрешённому методу. Собственную замену назвать отдельной адаптацией, не воспроизведением статьи. После готовности — действие, не бесконечная диагностика. Граф Гипотезы не менять.',
        limitations='LTR из IbPRIA и найденный SGAR-MOT — разные реализации. Не подменять одну другой и не переносить опубликованные числа между ними. Текущие технические Б2/Б3 не являются завершением их научной проверки.')
    write(grant, value)
    cat_path = ROOT/'experiments/project_history.json'
    before = cat_path.read_bytes(); cat = json.loads(before)
    jobs = [('40_busca_readiness.json','busca','Восстановление пропусков по истории объекта — готовность BUSCA'),
            ('50_ltr_readiness.json','ltr','Восстановление пропусков по маскам объекта — готовность второй работы')]
    for filename, method, title in jobs:
        ident = method+'_recovery_readiness_v1'
        assert not any(c['id']==ident for c in cat['experiments'])
        folder = RUN/(method+'_readiness')
        script = ROOT/('scripts/b123_'+method+'_readiness_20261027.py')
        protocol = dict(id=ident, created=now.isoformat(), timezone='Europe/Moscow',
            budget_seconds=120, author_stop=auth['save_at'], device='cpu', method=method,
            action='Проверить зафиксированные публичные исходники, наличие необходимых весов и зависимостей, точный узел подключения к нашему трекеру; GPU, обучение, видео и TrackEval не запускать.',
            acceptance=['Снимок исходников привязан к точному коммиту/URL и хешам; авторский код не исполняется',
                        'Отдельно перечислены готовое, отсутствующее и непроверенное; следующий шаг меняет выход трекера',
                        'Результат не называется проверкой HOTA, преимуществом метода или отрицательным научным итогом'],
            continuation='Агент по отчёту подготавливает ограниченный запуск метода с отдельным протоколом и замером. Отсутствие весов — ограничение готовности, не бесполезность метода.',
            sha256={script.relative_to(ROOT).as_posix():sha(script),
                    'scripts/b123_recovery_readiness_common_20261027.py':sha(ROOT/'scripts/b123_recovery_readiness_common_20261027.py'),
                    grant.relative_to(ROOT).as_posix():sha(grant)})
        write(folder/'protocol.json',protocol)
        write(RUN/'ready_jobs'/filename, dict(title=title,folder=folder.relative_to(ROOT).as_posix(),
            script=script.relative_to(ROOT).as_posix(),script_sha256=sha(script),protocol_sha256=sha(folder/'protocol.json')))
        cat['experiments'].append(dict(id=ident,title=title,area='flyvis',state='ready',started=f'{now:%Y-%m-%d}',
            question='Можно ли воспроизводимо запустить метод восстановления потерянного объекта в оставшемся окне?',
            hypothesis='',expected=protocol['action'],acceptance=[dict(criterion=x,met=None) for x in protocol['acceptance']],
            summary='Подготовительная проверка, разрешена автором после текущей серии; не опыт эффективности.',
            obtained='',interpretation='',decision='',cannot_claim=protocol['continuation']+' HOTA не измерена.',
            metrics=[],commits=[],files=[script.relative_to(ROOT).as_posix()],evidence=[dict(kind='file',ref=(folder/'protocol.json').relative_to(ROOT).as_posix(),claim='Условия до проверки')]))
    assert cat_path.read_bytes()==before
    cat_path.write_text(json.dumps(cat,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    from scripts import z13_localization_labels_20261023 as J
    J.IDENT='busca_ltr_followup_authorization'
    J.journal('поручено ночное продолжение: BUSCA, затем восстановление по маскам',
        '*Решение автора.* После текущей серии перейти к BUSCA и второй работе, до утра без ожидания нового сообщения. '
        'Поручение и границы сохранены в `runs/b123_20261027/recovery_followup_authorization.json`; исходная authorization.json не изменена. '
        'До07:30 МСК, сохранение07:20, один GPU. Новые данные не загружаются. Необходимые публичные исходники и готовые веса — только с проверкой происхождения, размера и стоимости в оставшемся окне.\n\n'
        '*Подготовлено.* Две карточки готовности и задания40/50 после уже подготовленных Б2/Б3. Только чтение исходников/окружения; не HOTA. '
        'Следующее действие — восстановление потерянного объекта с контролем ложных продолжений. Нужны воспроизводимость исходной базы и замер перед полной серией. '
        'Вторая работа LTR не равна более позднему SGAR-MOT. Граф «Гипотезы» не меняется.')
    with (RUN/'HANDOFF.md').open('a',encoding='utf-8') as f:
        f.write('\n\n## '+now.strftime('%H:%M')+' МСК — прямое поручение BUSCA → LTR после текущей очереди\n\n'
                'Читай recovery_followup_authorization.json. Автор разрешил это продолжение до07:30; срок не продлён. '
                'Очередь выполняет подготовительные40/50 после20/30; их report.json не научный результат. '
                'Дальше агент должен готовить рабочее вмешательство и измерение, а не ждать нового «давай». '
                'BUSCA первая, LTR вторая; новые наборы не качать, существующие данные не менять. '
                'Источник/готовые веса публичные, проверять размер/происхождение/совместимость, отдельное окружение при необходимости. '
                'Профиль GPU перед полным запуском; дважды измеренная стоимость должна помещаться до07:20. '
                'Не запускать поверх текущего GPU; отдельный граф заморожен; .git/index.lock не удалять.\n')
    print(json.dumps(dict(registered=True,jobs=[x[0] for x in jobs],hard_stop=auth['hard_stop'])))


if __name__=='__main__': main()
