"""Интеграция DRB в neck YOLO — wrapper-подход (Этап 4, подэтап 4.4).

Как устроено. Голова Detect в Ultralytics берёт входы из трёх слоёв neck
(индексы head.f, у yolo26s — [16, 19, 22]; читаются динамически). Каждый из
этих слоёв оборачивается в DRBWrappedLayer: слой считает свой выход, DRB
очищает его и отдаёт дальше; карта Q сохраняется в last_q для трекера.
Атрибуты .i/.f/.type копируются — конвейер DetectionModel их использует
для маршрутизации, обёртка для него прозрачна.

Почему через собственный Trainer. model.train() Ultralytics ПЕРЕСОБИРАЕТ
модель: trainer.get_model строит чистую DetectionModel из yaml и грузит веса —
обёртки, вставленные заранее, теряются. DRBTrainer переопределяет get_model
и вставляет DRB после сборки. Запуск: YOLO(w).train(trainer=DRBTrainer, ...).

Ограничения wrapper-подхода (для notes/drb_integration_notes.md):
- resume работает только через ветку "weights уже обёрнуты" в get_model;
- чекпойнты DRB-модели пикль-зависимы от кода проекта (models.drb должен
  быть импортируем при загрузке);
- перед финальной версией планируется чистая YAML-регистрация (Этап 5+).
"""
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics.models.yolo.detect.train import DetectionTrainer  # noqa: E402

from models.drb import DRB  # noqa: E402


class DRBWrappedLayer(nn.Module):
    """Обёртка слоя neck: inner -> DRB. Прозрачна для конвейера Ultralytics."""

    def __init__(self, inner: nn.Module, drb: DRB) -> None:
        super().__init__()
        self.inner = inner
        self.drb = drb
        # служебные атрибуты маршрутизации DetectionModel
        self.i = getattr(inner, 'i', None)
        self.f = getattr(inner, 'f', None)
        self.type = getattr(inner, 'type', type(inner).__name__)
        self.np = getattr(inner, 'np', sum(p.numel() for p in inner.parameters()))
        self.last_q: torch.Tensor | None = None

    def forward(self, x):
        y = self.inner(x)
        y_out, q = self.drb(y)
        self.last_q = q.detach() if q is not None else None
        return y_out


def has_drb(det_model: nn.Module) -> bool:
    return any(isinstance(m, DRBWrappedLayer) for m in det_model.model)


def probe_pyramid_shapes(det_model: nn.Module, layer_indices: list[int],
                         imgsz: int = 640) -> dict[int, tuple[int, int, int]]:
    """(C, H, W) выходов указанных слоёв на входе imgsz — через hooks."""
    shapes: dict[int, tuple[int, int, int]] = {}
    handles = []
    for idx in layer_indices:
        def _hook(mod, inp, out, idx=idx):
            shapes[idx] = tuple(out.shape[1:])
        handles.append(det_model.model[idx].register_forward_hook(_hook))
    was_training = det_model.training
    det_model.eval()
    device = next(det_model.parameters()).device
    with torch.no_grad():
        det_model(torch.zeros(1, 3, imgsz, imgsz, device=device))
    for h in handles:
        h.remove()
    if was_training:
        det_model.train()
    return shapes


def insert_drb(det_model: nn.Module, imgsz: int = 640,
               lqa_kernel: int = 3, branches: str = 'both') -> list[int]:
    """Оборачивает DRB слои-входы головы Detect. Идемпотентно.

    branches: 'both' | 'lqa' | 'fga' — конфигурация ветвей (Этап 7).
    Возвращает индексы обёрнутых слоёв (уровни P3/P4/P5 по порядку head.f).
    """
    if has_drb(det_model):
        return [m.i for m in det_model.model if isinstance(m, DRBWrappedLayer)]

    head = det_model.model[-1]
    indices = list(head.f)                      # напр. [16, 19, 22]
    shapes = probe_pyramid_shapes(det_model, indices, imgsz)
    device = next(det_model.parameters()).device
    for idx in indices:
        c, h, w = shapes[idx]
        drb = DRB(c, h, w, lqa_kernel=lqa_kernel, branches=branches).to(device)
        det_model.model[idx] = DRBWrappedLayer(det_model.model[idx], drb)
    return indices


class DRBInputLayer(nn.Module):
    """DRB перед первым слоем: очистка на уровне RGB-пикселей (Серия II, Этап 7).

    В отличие от DRBWrappedLayer (DRB после выхода слоя), здесь DRB применяется
    ко ВХОДУ слоя 0 — то есть к изображению до backbone. Один экземпляр вместо
    трёх. Честная оговорка для раздела 2.5 диссертации: на входе C=3 против
    сотен каналов в neck — статистики LQA работают в совсем другом режиме.
    """

    def __init__(self, inner: nn.Module, drb: DRB) -> None:
        super().__init__()
        self.inner = inner
        self.drb = drb
        self.i = getattr(inner, 'i', None)
        self.f = getattr(inner, 'f', None)
        self.type = getattr(inner, 'type', type(inner).__name__)
        self.np = getattr(inner, 'np', sum(p.numel() for p in inner.parameters()))
        self.last_q: torch.Tensor | None = None

    def forward(self, x):
        x_clean, q = self.drb(x)
        self.last_q = q.detach() if q is not None else None
        return self.inner(x_clean)


def insert_drb_at_input(det_model: nn.Module, imgsz: int = 640,
                        lqa_kernel: int = 3) -> list[int]:
    """Вставляет один DRB на RGB-вход (до backbone). Идемпотентно."""
    if isinstance(det_model.model[0], DRBInputLayer):
        return [0]
    if has_drb(det_model):
        raise RuntimeError('Модель уже содержит DRB в neck — варианты несовместимы')
    device = next(det_model.parameters()).device
    drb = DRB(3, imgsz, imgsz, lqa_kernel=lqa_kernel).to(device)
    det_model.model[0] = DRBInputLayer(det_model.model[0], drb)
    return [0]


def iter_drb(det_model: nn.Module):
    """Пары (индекс слоя, модуль DRB) — для логирования α и доступа к маскам.

    Работает и с neck-вариантом (DRBWrappedLayer), и с input-вариантом.
    """
    for m in det_model.model:
        if isinstance(m, (DRBWrappedLayer, DRBInputLayer)):
            yield m.i, m.drb


def collect_q_maps(det_model: nn.Module) -> dict[int, torch.Tensor]:
    """Карты Q последнего forward: {индекс слоя: (B,1,H,W)}. Для трекера (Этап 9)."""
    return {m.i: m.last_q for m in det_model.model
            if isinstance(m, (DRBWrappedLayer, DRBInputLayer))
            and m.last_q is not None}


# Параметры блока, которым штраф за величину весов противопоказан: у них
# нейтраль не в нуле. У маски частот нейтраль — единица (спектр проходит без
# изменений), у логита α — выбранный стартовый вклад ветви. Под общим правилом
# оба равномерно гасли, и блок вместо обучения затухал.
DECAY_FREE_KEYS = ('fga.delta', 'w_lqa', 'w_fga', 'fga.pred')
# Разгон достаётся ТОЛЬКО весам ветвей, но не маске. Причина в измерениях.
# Маска стала радиальной, и каждый её параметр теперь собирает градиент
# примерно со 137 отсчётов спектра вместо одного — она и так учится
# на два порядка быстрее прежнего, разгонять её сверху значит взорвать.
# А вес ветви, наоборот, стал вектором по каналам: каждый элемент видит
# только свой канал, и его градиент разбавился во столько раз, сколько
# каналов. Ему разгон нужен.
LR_BOOST_KEYS = ('w_lqa', 'w_fga', 'fga.pred')


def freeze_outside_drb(det_model: nn.Module) -> tuple[int, int]:
    """Замораживает всё, кроме параметров блока. Возвращает (сколько, сколько).

    Первая фаза приёма LP-FT (Kumar et al., ICLR 2022): полное дообучение
    обновляет нижние слои, пока новая часть ещё не подогнана, и разрушает
    предобученные признаки — у них дообучение проигрывает обучению одной
    головы 7% на сдвинутом распределении. Лечение: сперва подогнать новую
    часть на неподвижных признаках, и лишь потом размораживать всё.

    Наш случай ровно такой. Замерено: первые четыре-пять эпох блок портит
    признаки (на четвёртой подъём был 0.130, точность 0.3311 против 0.3527
    у контроля), и двенадцати эпох не хватает, чтобы долг отработать. Плюс
    прямое подтверждение дефекта самого рецепта: дообучение уронило модель
    с 0.4396 до 0.4192 БЕЗ всякого блока.

    Параметры без градиента обычные оптимизаторы пропускают, поэтому из
    групп их выбрасывать не нужно — достаточно снять requires_grad.
    """
    keep = {id(p) for _, drb in iter_drb(det_model) for p in drb.parameters()}
    frozen = kept = 0
    for p in det_model.parameters():
        if id(p) in keep:
            kept += 1
        else:
            p.requires_grad_(False)
            frozen += 1
    return frozen, kept


def make_bn_freezer():
    """Callback on_train_batch_start: нормировка замороженной части — в вывод.

    requires_grad=False останавливает веса, но НЕ бегущие средние
    BatchNorm: они пересчитываются в режиме обучения независимо от
    градиента. Без этого «замороженная» сеть продолжала бы меняться, и
    первая фаза перестала бы быть подгонкой блока к неподвижным признакам.
    """
    cache: dict[int, set[int]] = {}

    def _freeze(trainer) -> None:
        m = trainer.model
        keep = cache.get(id(m))
        if keep is None:
            keep = {id(x) for _, drb in iter_drb(m) for x in drb.modules()}
            cache.clear()
            cache[id(m)] = keep
        for mod in m.modules():
            if isinstance(mod, _BatchNorm) and id(mod) not in keep:
                mod.eval()

    return _freeze


def separate_block_params(optimizer, model, lr_mult: float = 1.0,
                          keys=DECAY_FREE_KEYS) -> int:
    """Выносит параметры блока в свои группы. Возвращает их число.

    Две причины. Первая — штраф за величину: у маски и логитов α нейтраль не
    в нуле, и общее правило их разрушает. Вторая — скорость обучения: блок
    приделан к предобученной сети, его веса случайны, и на общей скорости он
    почти не двигается. Ultralytics так же поступает со своей головой
    детектора, давая ей тройную скорость.

    Группа не создаётся с нуля, а клонируется с теми же моментом и прочими
    настройками — меняется только weight_decay и, если задано, lr. Иначе
    пришлось бы гадать, какие ключи ждёт конкретный оптимизатор: у MuSGD
    их больше, чем у обычного SGD. Клонирование происходит ДО создания
    планировщика, поэтому множитель переживает и разогрев, и спад скорости.
    """
    def hit(nm, ks):
        return any(k in nm for k in ks)

    plain, boost = set(), set()
    for nm, pm in model.named_parameters():
        if not hit(nm, keys):
            continue
        if lr_mult != 1.0 and hit(nm, LR_BOOST_KEYS):
            boost.add(id(pm))
        else:
            plain.add(id(pm))
    if not (plain or boost):
        return 0

    extra = []
    for g in optimizer.param_groups:
        for ids, mult in ((plain, 1.0), (boost, lr_mult)):
            mine = [pm for pm in g['params'] if id(pm) in ids]
            if not mine:
                continue
            g['params'] = [pm for pm in g['params'] if id(pm) not in ids]
            twin = {k: v for k, v in g.items() if k != 'params'}
            twin['params'] = mine
            twin['weight_decay'] = 0.0
            if mult != 1.0 and 'lr' in twin:
                twin['lr'] = twin['lr'] * mult
            extra.append(twin)
    for g in extra:
        optimizer.add_param_group(g)
    return sum(len(g['params']) for g in extra)


class DRBTrainer(DetectionTrainer):
    """DetectionTrainer, вставляющий DRB в пересобранную модель.

    Использование: YOLO(weights).train(trainer=DRBTrainer, data=..., ...)
    Вариант ветвей задаётся класс-атрибутом drb_variant (см. make_drb_trainer):
    'both' | 'lqa' | 'fga' — DRB в neck; 'input' — один DRB на RGB-входе.
    """

    drb_variant: str = 'both'
    drb_alpha_init: float = 0.0     # начальное значение w_LQA и w_FGA
    drb_lr_mult: float = 1.0        # во сколько раз быстрее учить сам блок
    drb_mask_init: str = 'flat'     # стартовая форма характеристики
    drb_mask_k: float = 0.3         # глубина подъёма или подавления
    drb_fusion: str = 'add'         # 'add' или 'gated', см. DRB.forward
    drb_adaptive: bool = False      # предсказывать подъём по кадру
    drb_freeze_base: bool = False   # учить только блок (первая фаза LP-FT)
    drb_mask_fixed: bool = False    # характеристику задать и не обучать

    def get_model(self, cfg=None, weights=None, verbose=True):
        if isinstance(weights, nn.Module) and (
                has_drb(weights) or isinstance(weights.model[0], DRBInputLayer)):
            return weights                      # resume: модель уже с DRB
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        if self.drb_variant == 'input':
            wrapped = insert_drb_at_input(model)
            print(f'[DRB] вставлен на RGB-вход (слой {wrapped})')
        else:
            wrapped = insert_drb(model, branches=self.drb_variant)
            print(f'[DRB] обёрнуты слои {wrapped} '
                  f'(входы головы Detect, ветви: {self.drb_variant})')
        if self.drb_alpha_init != 0.0:
            set_alpha_init(model, self.drb_alpha_init)
        set_mask_init(model, self.drb_mask_init, self.drb_mask_k)
        set_fusion(model, self.drb_fusion)
        set_adaptive(model, self.drb_adaptive)
        if self.drb_mask_fixed:
            n = freeze_mask(model)
            print(f'[DRB] характеристика зафиксирована на {n} уровнях — '
                  f'обучаемых частотных параметров ноль')
        return model

    def build_optimizer(self, model, name='auto', lr=0.001, momentum=0.9,
                        decay=1e-5, iterations=1e5):
        """Как у Ultralytics, но параметры блока — без штрафа за величину.

        Заморозка делается здесь, а не в get_model: Ultralytics строит
        оптимизатор после подготовки модели, и снятый requires_grad успевает
        подействовать на то, какие параметры оптимизатор реально трогает.
        """
        if self.drb_freeze_base:
            frozen, kept = freeze_outside_drb(model)
            print(f'[DRB] заморожено {frozen} тензоров сети, учится только '
                  f'блок — {kept} тензоров')
        opt = super().build_optimizer(model, name, lr, momentum, decay,
                                      iterations)
        moved = separate_block_params(opt, model, self.drb_lr_mult)
        speed = ('обычная скорость' if self.drb_lr_mult == 1.0
                 else f'веса ветвей ×{self.drb_lr_mult:g}, маска ×1')
        print(f'[DRB] в своих группах: {moved} параметров '
              f'(маска частот и веса ветвей), без штрафа за величину, {speed}')
        return opt


def set_alpha_init(det_model: nn.Module, w0: float) -> float:
    """Задаёт стартовые веса ветвей w_LQA = w_FGA = w0. Возвращает α = σ(w0).

    При w0 = 0 вклад ветвей стартует с 50%, и блок на старте усиливает
    признаки примерно в 1.8 раза — предобученная голова получает не тот
    масштаб, к которому привыкла. Отрицательное w0 делает старт близким к
    тождественному: блок «включается» по мере того, как обучение находит
    его полезным (приём нулевой инициализации остаточной ветви).
    """
    alpha = float(torch.sigmoid(torch.tensor(w0)))
    with torch.no_grad():
        for _, drb in iter_drb(det_model):
            for name in ('w_lqa', 'w_fga'):
                if hasattr(drb, name):
                    getattr(drb, name).fill_(w0)
    print(f'[DRB] стартовые веса ветвей w = {w0:g} (вклад каждой ветви {alpha:.1%})')
    return alpha


BAND_LO, BAND_HI = 0.15, 0.60          # границы полосы пропускания


def set_adaptive(det_model: nn.Module, on: bool, gain_max: float = 0.5) -> None:
    """Включает предсказание подъёма высоких частот по самому кадру.

    Статическая маска сходится к нейтрали: усреднённая по дню и ночи, она
    вынуждена выбрать «ничего не делать». Замер 21.08 показал, что кадры
    отличаются сильно — у ночных вдвое меньше высокочастотной энергии, —
    и различает их дескриптор почти безошибочно. См. FGA.enable_adaptive.
    """
    if not on:
        return
    k = 0
    for _, drb in iter_drb(det_model):
        fga = getattr(drb, 'fga', None)
        if fga is not None:
            fga.enable_adaptive(gain_max)
            k += 1
    print(f'[DRB] подъём высоких частот предсказывается по кадру '
          f'(уровней {k}, потолок подъёма {gain_max:g})')


def set_fusion(det_model: nn.Module, mode: str) -> None:
    """Как ветви подмешиваются в признаки: 'add' или 'gated'. См. DRB.forward."""
    if mode not in ('add', 'gated'):
        raise ValueError(f'fusion: add|gated, получено {mode!r}')
    for _, drb in iter_drb(det_model):
        drb.fusion = mode
    if mode == 'gated':
        print('[DRB] ветви вносят поправку через затвор, а не прибавляют копию')


def set_mask_init(det_model: nn.Module, kind: str, k: float = 0.3) -> None:
    """Стартовая форма частотной характеристики.

    'flat' — единицы, блок стартует тождественным.

    'highpass' — характеристика линейно растёт с частотой. Замысел был такой:
    мелкие объекты живут в высоких частотах, поднимем их. Диагностика 21.08
    этот замысел опровергла: ночь почти не мешает НАХОДИТЬ мелкие объекты
    (полнота 0.5138 против 0.5186 днём), поднимать нечего. Оставлено для
    аблации как заведомо неверный вариант.

    'bandpass' — гасятся оба края: снизу крупноструктурная избыточность,
    сверху шум. Именно это и нужно по замерам: ночью на мелких объектах
    падает точность (ложные срабатывания, −28% AP@0.5 при неизменной
    полноте) и локализация (47% -> 38%). Обе беды даёт шум, а он живёт в
    верхних частотах вместе с деталью. Приём из работ по подавлению частот
    для мелких объектов, где показано, что одностороннего среза мало.
    Прогон это опроверг: 0.4073 — худший результат из всех версий блока.
    Собственный замер спектров объясняет почему: у ночных кадров BDD100K
    высокочастотной энергии ВДВОЕ МЕНЬШЕ (0.119 против 0.140) и шума тоже
    меньше (0.0030 против 0.0076) — это видео с регистратора, камера уже
    подавила шум. Верхний срез убивал сигнал, а не помеху.

    'lowcut' — та же нижняя половина полосовой, но БЕЗ верхнего среза:
    гасится только крупноструктурная избыточность, детали проходят
    нетронутыми. Ровно то, что делает высокочастотный модуль HS-FPN в
    своей FPN, получая +2.2 AP и +2.6 у крошечных объектов. Новых констант
    не вводится — отсечка та же BAND_LO, что и у полосовой, чтобы разница
    между прогонами сводилась к одному: убран верхний срез.

    Смысл эта форма имеет только вместе с запретом на обучение маски (см.
    freeze_mask). Обучаемая маска трижды из трёх сходилась к нейтрали.
    """
    if kind == 'flat':
        return
    if kind not in ('highpass', 'bandpass', 'lowcut'):
        raise ValueError(
            f'mask_init: flat|highpass|bandpass|lowcut, получено {kind!r}')
    with torch.no_grad():
        for _, drb in iter_drb(det_model):
            fga = getattr(drb, 'fga', None)
            if fga is None:
                continue
            r = (torch.arange(fga.bands, dtype=torch.float32) + 0.5) / fga.bands
            lo = ((BAND_LO - r) / BAND_LO).clamp(0.0, 1.0)
            if kind == 'highpass':
                w = 1.0 + k * r
            elif kind == 'lowcut':
                w = 1.0 - k * lo
            else:
                hi = ((r - BAND_HI) / (1.0 - BAND_HI)).clamp(0.0, 1.0)
                w = 1.0 - k * (lo + hi)
            fga.delta.copy_((w - 1.0).repeat(fga.in_channels))
    if kind == 'highpass':
        print(f'[DRB] маска стартует подъёмом высоких частот: 1.0 -> {1 + k:.2f}')
    elif kind == 'lowcut':
        print(f'[DRB] маска срезает низ: ниже {BAND_LO:g} придавлено до '
              f'{1 - k:.2f}, выше — без изменений')
    else:
        print(f'[DRB] маска стартует полосовой: края придавлены до '
              f'{1 - k:.2f}, полоса {BAND_LO:g}..{BAND_HI:g} нетронута')


def freeze_mask(det_model: nn.Module) -> int:
    """Запрещает обучать частотную характеристику. Возвращает число тензоров.

    Смысл — исключить наблюдавшийся режим отказа. Обучаемая маска сходилась
    к нейтрали трижды из трёх, тремя разными параметризациями: по отсчётам
    спектра, по радиальным кольцам и предсказанная по кадру. Причина
    правдоподобна: свёрточный стек уже обучен и уже реализует нужную
    частотную избирательность, домножать его выход на дополнительную
    спектральную маску нечем, и оптимизация находит это первой.

    Литература сходится туда же. Число обучаемых частотных параметров и
    выигрыш связаны обратно: FcaNet с фиксированным базисом DCT обходит
    свой же обучаемый вариант, HS-FPN с одним скаляром отсечки даёт +2.2,
    FDA-DETR с четырьмя числами на масштаб +3.9, частотная ветвь NF-DETR
    с обучаемыми весами +0.7, наши 23 308 параметров — минус 1.6.

    Вес ветви при этом обучаемым остаётся: это сила вмешательства, а не
    форма фильтра. Так же устроен FDA-DETR.
    """
    n = 0
    for _, drb in iter_drb(det_model):
        fga = getattr(drb, 'fga', None)
        if fga is not None and hasattr(fga, 'delta'):
            fga.delta.requires_grad_(False)
            n += 1
    return n


def make_drb_trainer(variant: str = 'both', alpha_init: float = 0.0,
                     lr_mult: float = 1.0, mask_init: str = 'flat',
                     fusion: str = 'add',
                     adaptive: bool = False,
                     freeze_base: bool = False,
                     mask_fixed: bool = False) -> type[DRBTrainer]:
    """Фабрика Trainer-класса с зафиксированным вариантом DRB.

    Нужна потому, что model.train(trainer=...) принимает класс, а не
    экземпляр — параметризовать можно только через класс-атрибут.
    """
    if variant not in ('both', 'lqa', 'fga', 'input'):
        raise ValueError(f'variant: both|lqa|fga|input, получено {variant!r}')
    return type(f'DRBTrainer_{variant}', (DRBTrainer,),
                {'drb_variant': variant, 'drb_alpha_init': float(alpha_init),
                 'drb_lr_mult': float(lr_mult), 'drb_mask_init': mask_init,
                 'drb_fusion': fusion, 'drb_adaptive': bool(adaptive),
                 'drb_freeze_base': bool(freeze_base),
                 'drb_mask_fixed': bool(mask_fixed)})
