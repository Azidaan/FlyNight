"""Совместимость инструментов оценки bdd100k с новыми версиями библиотек.

Здесь чинятся две несовместимости, обе — следствие того, что цепочка
bdd100k -> scalabel -> motmetrics писалась под старые версии, а в
окружении стоят новые.

Пакет scalabel (0.3.1) писался под pydantic 1, где поле вида
`x: Optional[T]` без значения по умолчанию считалось необязательным и
молча получало None. В pydantic 2 такое поле стало ОБЯЗАТЕЛЬНЫМ: его
можно передать как None, но передать надо.

Из-за этого штатная оценка трекинга bdd100k падает на собственном конфиге
таксономии и на разборе кадров: не хватает то `subcategories` и `color` у
категорий, то `poseSigmas` у конфига, и так по каждому необязательному
полю. Латать по одному бессмысленно — здесь чинится причина.

Вторая: motmetrics 1.4.0 зовёт np.asfarray, убранную в NumPy 2.0.

Применять до первого разбора данных scalabel.
"""
import typing

from pydantic import BaseModel


def _relax(model: type[BaseModel]) -> int:
    """Необязательным полям без значения по умолчанию проставляет None."""
    fixed = 0
    for field in model.model_fields.values():
        if not field.is_required():
            continue
        if type(None) in typing.get_args(field.annotation):
            field.default = None
            fixed += 1
    if fixed:
        model.model_rebuild(force=True)
    return fixed


def _fix_numpy() -> bool:
    """Возвращает np.asfarray, убранную в NumPy 2.0.

    motmetrics 1.4.0 зовёт её при построении матрицы расстояний по IoU, и
    без неё оценка падает на первом же видео.
    """
    import numpy as np

    if hasattr(np, 'asfarray'):
        return False

    def asfarray(a, dtype=np.float64):
        return np.asarray(a, dtype=dtype)

    np.asfarray = asfarray
    return True


def apply() -> int:
    """Чинит модели scalabel и NumPy. Возвращает число поправленных полей."""
    import scalabel.label.typing as t

    _fix_numpy()
    total = 0
    for name in dir(t):
        obj = getattr(t, name)
        if isinstance(obj, type) and issubclass(obj, BaseModel) and obj is not BaseModel:
            total += _relax(obj)
    return total
