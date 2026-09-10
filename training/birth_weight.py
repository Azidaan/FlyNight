"""Доставка признака «возраст объекта в траектории» в функцию потерь YOLO26.

Признак кодируется дробной частью класса в файле метки (см.
`scripts/make_birth_labels.py`): целая часть — класс, дробная — корзина
возраста, `класс + 0.1 * (корзина + 1)`. Здесь дробная часть снимается обратно.

Точка врезки одна на обе ветви. YOLO26 работает в режиме `end2end`, поэтому
функция потерь — `E2ELoss`, и внутри неё ДВЕ копии `v8DetectionLoss`: ветвь
«один ко многим» и ветвь «один к одному». Обе собирают цели через
`v8DetectionLoss.preprocess`, поэтому патч ставится именно туда и действует
сразу на обе.

Что делает патч:
    1. снимает корзину возраста с дробной части класса;
    2. накапливает распределение по корзинам — это и есть проверка гейта:
       доля объектов возраста 0-1 в обучении должна совпасть с измеренной
       по разметке (8.44% на mini-BDD, 8.66% на полном val);
    3. возвращает классу целое значение ДО того, как метку увидит назначатель
       целей, чтобы назначение не отличалось от штатного ни на бит.

При `alpha = 0` (по умолчанию) поведение обучения не меняется вовсе: класс
восстанавливается флорингом, веса не применяются. Это делает патч проверяемым
контролем — прогон с патчем на закодированных метках обязан совпасть с
прогоном без патча на обычных.

Применение весов — Этап 4, здесь его намеренно нет: гейт отвечает на вопрос
«доходит ли признак», а не «помогает ли он».

Использование:
    from training.birth_weight import install, report, reset
    install()
    ... обучение ...
    report()
"""
from collections import Counter

import torch

STEP = 0.1                     # шаг кодирования корзины, как в make_birth_labels
BUCKETS = {0: 'возраст 0', 1: 'возраст 1', 2: 'возраст 2',
           3: 'возраст 3-4', 4: 'возраст 5+', -1: 'без признака'}

STATS: Counter = Counter()
_orig_preprocess = None
_alpha = 0.0


def reset() -> None:
    """Обнулить накопленную статистику."""
    STATS.clear()


def decode(cls: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(целый класс, номер корзины). Корзина -1 означает «признака нет».

    Обычная целая метка даёт дробную часть 0 и читается как «сведений нет»,
    поэтому корзина 0 кодируется дробной частью 0.1, а не нулём: иначе
    валидационные метки без признака молча считались бы первым появлением.
    """
    base = cls.floor()
    frac = cls - base
    bkt = torch.round(frac / STEP).long() - 1
    return base, bkt.clamp_(min=-1)


def _preprocess(self, targets, batch_size, scale_tensor):
    """Патч `v8DetectionLoss.preprocess`: снять признак и восстановить класс."""
    if targets.shape[0]:
        base, bkt = decode(targets[:, 1])
        for b, n in zip(*torch.unique(bkt, return_counts=True)):
            STATS[int(b)] += int(n)
        targets = targets.clone()
        targets[:, 1] = base
    return _orig_preprocess(self, targets, batch_size, scale_tensor)


def install(alpha: float = 0.0) -> None:
    """Поставить патч. alpha пока не используется — применение весов Этап 4."""
    global _orig_preprocess, _alpha
    from ultralytics.utils.loss import v8DetectionLoss

    _alpha = alpha
    if _orig_preprocess is not None:
        print('[birth_weight] патч уже установлен')
        return
    _orig_preprocess = v8DetectionLoss.preprocess
    v8DetectionLoss.preprocess = _preprocess
    print(f'[birth_weight] патч установлен на v8DetectionLoss.preprocess, '
          f'alpha={alpha} (веса не применяются, это гейт доставки)')


def uninstall() -> None:
    global _orig_preprocess
    if _orig_preprocess is None:
        return
    from ultralytics.utils.loss import v8DetectionLoss
    v8DetectionLoss.preprocess = _orig_preprocess
    _orig_preprocess = None


def report() -> dict:
    """Распределение объектов по корзинам, увиденное функцией потерь."""
    total = sum(STATS.values())
    print('\n=== birth_weight: что дошло до функции потерь ===')
    if not total:
        print('  объектов не было — патч не сработал или обучение не шло')
        return {}
    print('  %-14s %10s %8s' % ('корзина', 'объектов', 'доля'))
    out = {}
    for b in sorted(STATS):
        share = 100.0 * STATS[b] / total
        out[b] = share
        print('  %-14s %10d %7.2f%%' % (BUCKETS.get(b, f'? {b}'), STATS[b], share))
    young = STATS.get(0, 0) + STATS.get(1, 0)
    out['young'] = 100.0 * young / total
    print(f'\n  доля возраста 0-1: {out["young"]:.2f}%')
    print('  ожидание по разметке mini-BDD: 8.44%, по полному val: 8.66%')
    print('  ВНИМАНИЕ: цели считаются ДВАЖДЫ (обе ветви E2ELoss), поэтому')
    print('  абсолютные счётчики удвоены; доли от этого не меняются.')
    return out
