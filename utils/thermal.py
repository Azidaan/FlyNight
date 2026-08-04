"""Чтение температур и защита железа при длительных обучениях.

Обучение идёт неделями без человека, поэтому перегрев должен останавливать
прогон сам. Модуль даёт две вещи: снятие температур и пороговую проверку.

Что доступно на этой машине (проверено):
  - видеокарта: nvidia-smi, работает всегда;
  - процессор: датчика нет. ACPI-зона (MSAcpi_ThermalZoneTemperature) кидает
    исключение, psutil.sensors_temperatures на Windows отсутствует. Если
    поставить LibreHardwareMonitor (бесплатный, открытый), он публикует
    датчики в WMI — код подхватит их автоматически, менять ничего не нужно.

О порогах. 75 °C для игровой видеокарты — штатная рабочая температура под
нагрузкой, а не авария: RTX 5070 начинает снижать частоты в районе 83-87 °C
и сама выключится задолго до повреждения. Поэтому 75 °C здесь — порог
ПРЕДУПРЕЖДЕНИЯ (видно в панели и в журнале), а останов — на STOP_TEMP.
Если нужен буквальный останов на 75 °C, поменяйте STOP_TEMP ниже.
"""
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

WARN_TEMP = 75.0        # видно в панели, пишется в журнал, прогон продолжается
STOP_TEMP = 87.0        # прогон останавливается (зона снижения частот)
RISE_DELTA = 12.0       # «резкий рост»: столько градусов
RISE_WINDOW_S = 120.0   # за столько секунд
SUSTAIN_READINGS = 3    # столько замеров подряд выше STOP_TEMP — не выброс

_NO_WINDOW = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
_cpu_source: str | None = None      # 'lhm' | 'acpi' | 'none'; определяется один раз


def _run(cmd: list[str], timeout: float = 8.0) -> str:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           encoding='utf-8', errors='replace',
                           creationflags=_NO_WINDOW)
        return p.stdout if p.returncode == 0 else ''
    except (OSError, subprocess.SubprocessError):
        return ''


def gpu_temperature() -> float | None:
    """Температура видеокарты, °C."""
    out = _run(['nvidia-smi', '--query-gpu=temperature.gpu',
                '--format=csv,noheader,nounits'])
    m = re.search(r'\d+', out)
    return float(m.group()) if m else None


def gpu_load() -> int | None:
    """Загрузка видеокарты, %."""
    out = _run(['nvidia-smi', '--query-gpu=utilization.gpu',
                '--format=csv,noheader,nounits'])
    m = re.search(r'\d+', out)
    return int(m.group()) if m else None


def _cpu_from_lhm() -> float | None:
    """CPU Package из LibreHardwareMonitor, если он запущен."""
    out = _run(['powershell', '-NoProfile', '-Command',
                "(Get-CimInstance -Namespace root/LibreHardwareMonitor "
                "-ClassName Sensor -ErrorAction Stop | Where-Object "
                "{ $_.SensorType -eq 'Temperature' -and $_.Name -like '*Package*' } "
                "| Select-Object -First 1 -ExpandProperty Value)"], timeout=12)
    m = re.search(r'[\d.,]+', out)
    return float(m.group().replace(',', '.')) if m else None


def _cpu_from_acpi() -> float | None:
    """ACPI-зона: на большинстве десктопов недоступна, но проверить дёшево."""
    out = _run(['powershell', '-NoProfile', '-Command',
                "(Get-CimInstance -Namespace root/wmi -ClassName "
                "MSAcpi_ThermalZoneTemperature -ErrorAction Stop | "
                "Select-Object -First 1 -ExpandProperty CurrentTemperature)"],
               timeout=12)
    m = re.search(r'\d+', out)
    return round(int(m.group()) / 10 - 273.15, 1) if m else None


def cpu_temperature() -> float | None:
    """Температура процессора, °C, или None — если датчика нет."""
    global _cpu_source
    if _cpu_source is None:                       # один раз выбираем источник
        if _cpu_from_lhm() is not None:
            _cpu_source = 'lhm'
        elif _cpu_from_acpi() is not None:
            _cpu_source = 'acpi'
        else:
            _cpu_source = 'none'
    if _cpu_source == 'lhm':
        return _cpu_from_lhm()
    if _cpu_source == 'acpi':
        return _cpu_from_acpi()
    return None


def read_temps() -> dict:
    """Снимок: {'gpu': °C|None, 'cpu': °C|None, 'load': %|None, 'cpu_src': str}."""
    return {'gpu': gpu_temperature(), 'cpu': cpu_temperature(),
            'load': gpu_load(), 'cpu_src': _cpu_source or 'none'}


class ThermalGuard:
    """Следит за температурой: помечает предупреждение и требует останова.

    Не убивает процессы сам — только выносит вердикт. Кто именно
    останавливает прогон, решает вызывающая сторона.
    """

    def __init__(self, warn: float = WARN_TEMP, stop: float = STOP_TEMP) -> None:
        self.warn, self.stop = warn, stop
        self._history: list[tuple[float, float]] = []   # (время, температура)
        self._hot_streak = 0

    def update(self, now: float, temp: float | None) -> tuple[str, str]:
        """(уровень, пояснение); уровень: 'ok' | 'warn' | 'stop'."""
        if temp is None:
            return 'ok', ''
        self._history.append((now, temp))
        self._history = [(t, v) for t, v in self._history
                         if now - t <= RISE_WINDOW_S]

        if temp >= self.stop:
            self._hot_streak += 1
            if self._hot_streak >= SUSTAIN_READINGS:
                return 'stop', (f'{temp:.0f} °C держится {self._hot_streak} '
                                f'замера подряд при пороге {self.stop:.0f} °C')
            return 'warn', f'{temp:.0f} °C — выше порога останова, проверяю ещё раз'
        self._hot_streak = 0

        if len(self._history) >= 2:
            oldest = min(v for _, v in self._history)
            if temp - oldest >= RISE_DELTA and temp >= self.warn:
                return 'warn', (f'резкий рост: +{temp - oldest:.0f} °C '
                                f'за {RISE_WINDOW_S:.0f} с, сейчас {temp:.0f} °C')
        if temp >= self.warn:
            return 'warn', f'{temp:.0f} °C (порог внимания {self.warn:.0f} °C)'
        return 'ok', ''


def main() -> int:
    """Разовый замер: python utils/thermal.py"""
    t = read_temps()
    print(f'видеокарта: {t["gpu"]} °C, загрузка {t["load"]}%')
    if t['cpu'] is None:
        print('процессор: датчик недоступен '
              '(поставьте LibreHardwareMonitor — подхватится сам)')
    else:
        print(f'процессор: {t["cpu"]} °C (источник {t["cpu_src"]})')
    print(f'пороги: внимание {WARN_TEMP:.0f} °C, останов {STOP_TEMP:.0f} °C')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
