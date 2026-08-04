"""Smoke-тест температурной защиты (utils/thermal.py).

Сторож решает, обрывать ли двухнедельный прогон, поэтому проверяется
и то, что он срабатывает, и то, что он НЕ срабатывает зря: одиночный
выброс не должен стоить суток обучения.

Запуск: python scripts/smoke/test_thermal.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.thermal import (  # noqa: E402
    STOP_TEMP, WARN_TEMP, ThermalGuard, read_temps,
)

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = '') -> None:
    print(f'[{"OK " if cond else "FAIL"}] {name}' + (f' — {detail}' if detail else ''))
    if not cond:
        FAILURES.append(name)


def feed(guard: ThermalGuard, temps: list[float], step: float = 20.0,
         t0: float = 1000.0) -> list[tuple[str, str]]:
    return [guard.update(t0 + i * step, v) for i, v in enumerate(temps)]


def main() -> int:
    print('=== штатная работа ===')
    g = ThermalGuard()
    out = feed(g, [55, 60, 64, 66, 67, 68])
    check('рабочая температура не тревожит',
          all(lvl == 'ok' for lvl, _ in out), str([l for l, _ in out]))

    print('=== порог внимания ===')
    g = ThermalGuard()
    out = feed(g, [70, 74, 76, 78])
    check('выше порога внимания — предупреждение, но не останов',
          out[-1][0] == 'warn' and all(lvl != 'stop' for lvl, _ in out),
          out[-1][1])

    print('=== перегрев ===')
    g = ThermalGuard()
    out = feed(g, [80, 85, 88, 89, 90])
    levels = [lvl for lvl, _ in out]
    check('устойчивый перегрев даёт останов', 'stop' in levels, str(levels))
    check('останов не с первого же замера выше порога',
          levels.index('stop') >= levels.index('warn') + 2, str(levels))

    print('=== одиночный выброс ===')
    g = ThermalGuard()
    out = feed(g, [60, 62, 95, 61, 60, 62])
    check('единичный скачок не останавливает прогон',
          all(lvl != 'stop' for lvl, _ in out), str([l for l, _ in out]))

    print('=== резкий рост ===')
    g = ThermalGuard()
    out = feed(g, [64, 70, 77], step=30.0)
    check('резкий рост замечается', out[-1][0] == 'warn'
          and 'резкий рост' in out[-1][1], out[-1][1])

    print('=== нет данных ===')
    g = ThermalGuard()
    check('отсутствие датчика не роняет сторож',
          g.update(1000.0, None) == ('ok', ''))

    print('=== пороги осмысленны ===')
    check('порог внимания ниже порога останова', WARN_TEMP < STOP_TEMP,
          f'{WARN_TEMP} < {STOP_TEMP}')
    check('порог останова ниже аппаратного выключения (~100 °C)',
          70 < STOP_TEMP < 100, str(STOP_TEMP))

    print('=== чтение с железа ===')
    t = read_temps()
    check('температура видеокарты читается',
          isinstance(t['gpu'], float) and 0 < t['gpu'] < 120, str(t['gpu']))
    print(f'      видеокарта {t["gpu"]} °C, загрузка {t["load"]}%, '
          f'процессор {t["cpu"]} (источник {t["cpu_src"]})')

    print(f'\n--- {"OK: температурная защита работает" if not FAILURES else f"FAIL: {FAILURES}"} ---')
    return 1 if FAILURES else 0


if __name__ == '__main__':
    raise SystemExit(main())
