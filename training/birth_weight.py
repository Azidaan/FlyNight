"""Взвешивание объектов по их временной роли в траектории (Этапы 2 и 4).

ЗАМЫСЕЛ

Пропуск объекта в середине живущего трека компенсируется предсказанием
трекера; пропуск при первом появлении не компенсируется ничем, потому что
трека ещё нет. Значит цена ошибки детектора зависит от возраста объекта, а
обучение это игнорирует. Здесь вес объекта в функции потерь делается убывающей
функцией возраста.

Измеренное основание (стенд 960, 200 val-видео): доля найденных растёт по
возрасту монотонно, 38.00 -> 45.36 -> 49.86 -> 55.31 -> 63.02 -> 70.98 ночью,
и две трети разрыва между первым появлением и зрелым треком приходятся на
недооценку уверенности, а не на слепоту. Недооценку поднять обучением можно.

ДОСТАВКА ПРИЗНАКА

Формат меток YOLO — ровно пять чисел в строке, номера трека там нет, шестая
колонка читается как полигон сегментации, а поднять число классов нельзя: это
меняет размер головы детектора. Поэтому корзина возраста кодируется ДРОБНОЙ
частью класса (`scripts/make_birth_labels.py`): `класс + 0.1 * (корзина + 1)`.
Максимум 7.5 < 8, проверка меток проходит, аугментации переносят признак
вместе с объектом.

Гейт Этапа 2 подтвердил, что доставка неискажённая: при alpha = 0 обучение
совпало с обычным ДО ПЯТОГО ЗНАКА по всем метрикам.

ТОЧКИ ВРЕЗКИ (две, обе маленькие)

1. `v8DetectionLoss.preprocess` — снимает корзину с дробной части, возвращает
   классу целое значение до того, как метку увидит назначатель целей, и
   сохраняет корзины в раскладке (пакет, объект), выровненной с целями.

2. `v8DetectionLoss.__init__` — оборачивает `self.assigner`. После назначения
   известно соответствие «якорь -> эталонный объект» (`target_gt_idx`) и маска
   переднего плана (`fg_mask`); по ним вес раскладывается на якоря и кладётся
   в `self.class_weights`. Это штатный крючок Ultralytics: строкой
   `bce_loss *= self.class_weights` он умножает классификационную
   составляющую. По умолчанию он пуст, так что мы ничего не вытесняем; если
   у модели свои веса классов, наши на них домножаются.

Почему именно классификационная составляющая. Измеренный механизм отказа —
недооценка уверенности (ступень ПОРОГ, 18.0% всех появлений против 4.5%
слепоты). Уверенность формирует именно классификационная ветвь.

Почему точка врезки одна на обе ветви. YOLO26 работает в режиме `end2end`,
функция потерь — `E2ELoss` с двумя копиями `v8DetectionLoss`; патч ставится на
класс, поэтому действует на обе.

ВЕСА ПО КОРЗИНАМ

`w = 1 + alpha * PROFILE[корзина]`. Профиль по умолчанию убывающий, а не
ступенчатый: полнота растёт по возрасту монотонно, порога нет, и непрерывное
взвешивание вдвое увеличивает мишень метода (5289 объектов против 2295).
Корзина -1 означает «признака нет» (обычная целая метка) и получает вес 1.

Использование:
    from training import birth_weight
    birth_weight.install(alpha=1.0)
    ... обучение ...
    birth_weight.report()
"""
from collections import Counter

import torch

STEP = 0.1                     # шаг кодирования корзины, как в make_birth_labels
BUCKETS = {0: 'возраст 0', 1: 'возраст 1', 2: 'возраст 2',
           3: 'возраст 3-4', 4: 'возраст 5+', -1: 'без признака'}
# убывающий профиль: чем моложе объект, тем дороже его пропуск
PROFILE_SMOOTH = {0: 1.00, 1: 0.75, 2: 0.50, 3: 0.25, 4: 0.0, -1: 0.0}
# ступенька: только первое появление и следующий кадр
PROFILE_BINARY = {0: 1.00, 1: 1.00, 2: 0.0, 3: 0.0, 4: 0.0, -1: 0.0}

STATS: Counter = Counter()
_orig_preprocess = None
_orig_init = None
_orig_assign = None
_alpha = 0.0
_profile = PROFILE_SMOOTH
_w_lut: torch.Tensor | None = None     # таблица весов, индекс = корзина + 1

# Буфер между двумя точками врезки. Держится в МОДУЛЕ, а не на объекте потерь,
# и патчи ставятся на классы, а не на экземпляры: Ultralytics пикклит модель
# вместе с функцией потерь при сохранении чекпойнта, и замыкание или локальный
# объект в её полях роняет сохранение с «Can't pickle local object».
# Порядок вызовов внутри одной ветви строго последовательный —
# preprocess, затем назначатель, — поэтому буфера на модуль достаточно.
_BKT: torch.Tensor | None = None
_LOSS = None


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
    bkt = torch.round((cls - base) / STEP).long() - 1
    return base, bkt.clamp_(min=-1)


def _lut(device) -> torch.Tensor:
    """Таблица весов по корзинам; индекс сдвинут на 1, чтобы вместить -1."""
    global _w_lut
    if _w_lut is None or _w_lut.device != device:
        vals = [1.0 + _alpha * _profile.get(b, 0.0) for b in range(-1, 5)]
        _w_lut = torch.tensor(vals, dtype=torch.float32, device=device)
    return _w_lut


def _preprocess(self, targets, batch_size, scale_tensor):
    """Снять корзину, восстановить класс, сохранить корзины в раскладке целей.

    Корзина едет ШЕСТОЙ колонкой через штатный `preprocess`: он переводит в
    xyxy только колонки 1:5, а паддинг раскладывает одинаково для всех колонок.
    Так корзины оказываются выровнены с целями без собственной логики паддинга.
    Затем колонка отрезается, и наружу уходит тензор обычной формы.
    """
    global _BKT, _LOSS
    if targets.shape[0] == 0:
        _BKT, _LOSS = None, self
        return _orig_preprocess(self, targets, batch_size, scale_tensor)

    base, bkt = decode(targets[:, 1])
    for b, n in zip(*torch.unique(bkt, return_counts=True)):
        STATS[int(b)] += int(n)
    aug = torch.cat([targets[:, :1], base.unsqueeze(1), targets[:, 2:],
                     bkt.to(targets.dtype).unsqueeze(1)], 1)
    out = _orig_preprocess(self, aug, batch_size, scale_tensor)
    _BKT, _LOSS = out[..., 5].round().long(), self
    return out[..., :5].contiguous()


def _assign(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
    """Патч `TaskAlignedAssigner.forward`: разложить веса объектов на якоря.

    После назначения известно соответствие «якорь -> эталонный объект»
    (`target_gt_idx`) и маска переднего плана (`fg_mask`). Вес кладётся в
    `class_weights` объекта потерь — штатный крючок, которым Ultralytics
    умножает классификационную составляющую.
    """
    res = _orig_assign(self, pd_scores, pd_bboxes, anc_points,
                       gt_labels, gt_bboxes, mask_gt)
    if _LOSS is None:
        return res
    base_cw = getattr(_LOSS, '_birth_base_cw', None)
    if _alpha == 0.0 or _BKT is None or _BKT.numel() == 0:
        _LOSS.class_weights = base_cw
        return res
    fg_mask, target_gt_idx = res[3], res[4]
    bkt = _BKT.to(fg_mask.device)          # назначатель мог уйти на процессор
    w_gt = _lut(bkt.device)[bkt + 1]                     # (пакет, объект)
    bi = torch.arange(bkt.shape[0], device=bkt.device)[..., None]
    w = w_gt[bi, target_gt_idx]                          # (пакет, якорь)
    w = torch.where(fg_mask, w, torch.ones_like(w)).unsqueeze(-1)
    _LOSS.class_weights = w if base_cw is None else base_cw * w
    return res


def _init(self, *a, **k):
    """Патч `v8DetectionLoss.__init__`: запомнить свои веса классов, если были.

    Функция модульная, а не локальная: локальная не пикклится вместе с моделью.
    """
    _orig_init(self, *a, **k)
    self._birth_base_cw = self.class_weights


def install(alpha: float = 0.0, profile: str = 'smooth') -> None:
    """Поставить патч. alpha = 0 оставляет обучение тождественным обычному."""
    global _orig_preprocess, _orig_init, _orig_assign, _alpha, _profile, _w_lut
    import ultralytics
    from ultralytics.utils.loss import v8DetectionLoss

    if ultralytics.__version__ != '8.4.84':
        print(f'[birth_weight] ВНИМАНИЕ: патч писался под ultralytics 8.4.84, '
              f'установлена {ultralytics.__version__}. Проверьте, что '
              f'v8DetectionLoss.preprocess и self.class_weights не изменились.')

    _alpha = float(alpha)
    _profile = PROFILE_BINARY if profile == 'binary' else PROFILE_SMOOTH
    _w_lut = None
    if _orig_preprocess is not None:
        print('[birth_weight] патч уже установлен, обновлены только alpha и профиль')
        return

    from ultralytics.utils.tal import TaskAlignedAssigner

    _orig_preprocess = v8DetectionLoss.preprocess
    v8DetectionLoss.preprocess = _preprocess
    _orig_init = v8DetectionLoss.__init__
    v8DetectionLoss.__init__ = _init
    _orig_assign = TaskAlignedAssigner.forward
    TaskAlignedAssigner.forward = _assign
    weights = {BUCKETS[b]: round(1.0 + _alpha * _profile.get(b, 0.0), 3)
               for b in (0, 1, 2, 3, 4)}
    print(f'[birth_weight] патч установлен, alpha={_alpha}, профиль={profile}')
    print(f'[birth_weight] веса по корзинам: {weights}')


def uninstall() -> None:
    global _orig_preprocess, _orig_init, _orig_assign, _BKT, _LOSS
    from ultralytics.utils.loss import v8DetectionLoss
    from ultralytics.utils.tal import TaskAlignedAssigner
    if _orig_preprocess is not None:
        v8DetectionLoss.preprocess = _orig_preprocess
        _orig_preprocess = None
    if _orig_init is not None:
        v8DetectionLoss.__init__ = _orig_init
        _orig_init = None
    if _orig_assign is not None:
        TaskAlignedAssigner.forward = _orig_assign
        _orig_assign = None
    _BKT, _LOSS = None, None


def report() -> dict:
    """Распределение объектов по корзинам, увиденное функцией потерь."""
    total = sum(STATS.values())
    print('\n=== birth_weight: что дошло до функции потерь ===')
    if not total:
        print('  объектов не было — патч не сработал или обучение не шло')
        return {}
    print('  %-14s %10s %8s %8s' % ('корзина', 'объектов', 'доля', 'вес'))
    out = {}
    for b in sorted(STATS):
        share = 100.0 * STATS[b] / total
        out[b] = share
        w = 1.0 + _alpha * _profile.get(b, 0.0)
        print('  %-14s %10d %7.2f%% %8.3f'
              % (BUCKETS.get(b, f'? {b}'), STATS[b], share, w))
    young = STATS.get(0, 0) + STATS.get(1, 0)
    out['young'] = 100.0 * young / total
    print(f'\n  доля возраста 0-1: {out["young"]:.2f}%')
    print('  ВНИМАНИЕ: цели считаются ДВАЖДЫ (обе ветви E2ELoss), поэтому')
    print('  абсолютные счётчики удвоены; доли от этого не меняются.')
    return out
