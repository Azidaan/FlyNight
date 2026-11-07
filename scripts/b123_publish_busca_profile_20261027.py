"""Дополнить не записавшуюся постановку; публиковать готовое задание после проверок."""
from pathlib import Path
import hashlib,json,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
RUN=ROOT/'runs/b123_20261027/busca_runtime_profile'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    from scripts import z13_localization_labels_20261023 as J
    J.IDENT='busca_runtime_profile_v1'
    plan=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md'
    if 'поставлен ограниченный профиль авторской BUSCA'.encode() not in plan.read_bytes():
        for attempt in range(6):
            before=plan.read_bytes()
            try:
                J.journal('поставлен ограниченный профиль авторской BUSCA',
                    '*Постановка.* Проверить полноту авторских весов, точность экономного хранения позиционного кодирования и стоимость прямого прохода. '
                    'Пакеты из 1, 2 и 4 объектов, по три повтора после прогрева. Память из 11 реальных прошлых областей, пять текущих кандидатов; '
                    'выбор по сохранённому штатному трекеру, без чтения разметки. Один GPU, 300 секунд, независимый контроллер. '
                    'Это технический профиль, а не оценка HOTA и не результат полного трекера.\n\n'
                    '*Критерии до расчёта.* Все 379 ключей весов должны строго совпасть с архитектурой. '
                    'Экономное хранение независимых осей должно побайтно воспроизвести все 125 строк малой полной таблицы '
                    'и весь позиционный прямой проход. Выход на реальных областях должен быть конечным, вероятности нормированными. '
                    'Стоимость и память сохраняются. Источник: `runs/b123_20261027/busca_runtime_profile/protocol.json`.\n\n'
                    '*Подготовка.* Зависимость positional-encodings версии 6.0.3 получена из PyPI в tools/busca-deps-20261027; '
                    '7519 байт и SHA256 проверены, основное окружение не менялось. Полный checkpoint уже содержит извлекатель, '
                    'но его полнота принимается только после строгой проверки архитектуры. Исходники авторов не изменяются. '
                    'Первый вызов записи журнала вернул OSError22; протокол и карточка сохранились, журнал оставался прежним. '
                    'Постановка дописана отдельно до публикации задания, готовые прогоны не повторяются.')
                break
            except OSError:
                assert plan.read_bytes()==before
                if attempt==5:raise
                time.sleep(.2)
    if '--publish' in sys.argv:
        p=json.loads((RUN/'protocol.json').read_text())
        for name,h in p['sha256'].items():assert sha(ROOT/name)==h,name
        source='scripts/b123_busca_runtime_profile_20261027.py'
        target=ROOT/'runs/b123_20261027/ready_jobs/90_busca_runtime_profile.json'
        assert not target.exists()
        target.write_text(json.dumps(dict(title='BUSCA — реальный прямой проход, полнота весов и стоимость',
            folder=RUN.relative_to(ROOT).as_posix(),script=source,script_sha256=sha(ROOT/source),
            protocol_sha256=sha(RUN/'protocol.json')),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print('Published prepared stage90')
if __name__=='__main__':main()
