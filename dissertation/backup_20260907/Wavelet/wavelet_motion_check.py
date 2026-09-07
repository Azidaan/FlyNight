"""
Проверка мишени и валидированный вейвлетный предсказатель движения.

Что установлено на синтетике (см. отчёт):
  1. Вейвлет с 2 нулевыми моментами (db2) обнуляет равномерное движение
     в детализирующих коэффициентах: энергия 2.5e-14 против 1.77 у Хаара.
  2. Критичные детали реализации: край mode='symmetric' (НЕ periodization,
     НЕ zero - там ошибка растёт в 10-60 раз), обязательное снятие линейного
     тренда перед разложением.
  3. Выигрыш против ЧЕСТНОЙ базы (МНК-сглаживание) растёт с силой дрожания:
     +3% при нулевом дрожании, +11% при среднем, +24% при сильном.
     То есть механизм именно в удалении дрожания.

ГЛАВНОЕ: перед реализацией в трекере выполнить check_target() на реальных
GT-траекториях BDD100K. Если спектрального разделения нет - идея не имеет
мишени на этих данных, и её надо закрыть, не тратя GPU.
"""
import numpy as np
import pywt

FPS = 30


# ---------------------------------------------------------------- предсказатель
def wavelet_predict(history, gap, basis="db2", level=3, drop=(1, 2), fit_win=16):
    """
    history : 1-D массив координаты (например центр рамки по x) до окклюзии
    gap     : на сколько кадров вперёд предсказать
    Возвращает массив предсказанных координат длины gap.
    """
    x = np.asarray(history, float)
    n = len(x)
    if n < 2 ** level * 2:                      # окно слишком короткое для разложения
        return _lsq_predict(x, gap, min(fit_win, n))

    idx = np.arange(n)
    k, b = np.polyfit(idx, x, 1)                # снятие тренда - ОБЯЗАТЕЛЬНО
    trend = k * idx + b
    xd = x - trend

    coeffs = pywt.wavedec(xd, basis, level=level, mode="symmetric")
    out = []
    for i, c in enumerate(coeffs):
        lvl = level - i + 1 if i > 0 else None
        out.append(np.zeros_like(c) if lvl in drop else c)
    clean = pywt.waverec(out, basis, mode="symmetric")[:n] + trend
    return _lsq_predict(clean, gap, fit_win)


def _lsq_predict(h, gap, win=16):
    win = min(win, len(h))
    y = h[-win:]
    xx = np.arange(win)
    k, b = np.polyfit(xx, y, 1)
    return k * (win - 1 + np.arange(1, gap + 1)) + b


# ------------------------------------------------------------ проверка мишени
def check_target(tracks, gmc_residual=True, fps=FPS):
    """
    tracks: список 1-D массивов координат GT-траекторий (после вычета глобального
            движения камеры, если gmc_residual=True).

    Считает усреднённый спектр мощности остатка и проверяет, есть ли
    разделимые полосы: низкая (движение, <2 Гц) и высокая (дрожание, 3-8 Гц).

    КРИТЕРИЙ: если доля энергии в полосе 3-8 Гц заметно превышает уровень
    белого шума и видна как локальный подъём - мишень есть.
    Если спектр монотонно спадает без подъёма - дрожания нет, идея мертва.
    """
    spectra = []
    for tr in tracks:
        x = np.asarray(tr, float)
        if len(x) < 64:
            continue
        idx = np.arange(len(x))
        k, b = np.polyfit(idx, x, 1)
        resid = x - (k * idx + b)               # убираем равномерное движение
        w = np.hanning(len(resid))
        sp = np.abs(np.fft.rfft(resid * w)) ** 2
        freqs = np.fft.rfftfreq(len(resid), d=1.0 / fps)
        # нормируем на общую энергию, чтобы усреднять треки разной амплитуды
        sp = sp / (sp.sum() + 1e-12)
        spectra.append(np.interp(np.linspace(0, fps / 2, 128), freqs, sp))

    if not spectra:
        return None
    avg = np.mean(spectra, axis=0)
    f = np.linspace(0, fps / 2, 128)

    low = avg[(f >= 0.2) & (f < 2.0)].sum()
    mid = avg[(f >= 3.0) & (f <= 8.0)].sum()
    high = avg[f > 10.0].sum()

    print("Доля энергии остатка траектории по полосам:")
    print(f"  движение/манёвр 0.2-2 Гц : {low:.4f}")
    print(f"  дрожание       3-8 Гц    : {mid:.4f}")
    print(f"  шум           >10 Гц     : {high:.4f}")
    print(f"\nОтношение полосы дрожания к шумовой полке: {mid / (high + 1e-12):.2f}")
    print("Мишень есть, если это отношение заметно больше 1 и в 3-8 Гц виден подъём.")
    return f, avg


if __name__ == "__main__":
    # демонстрация на синтетике с известным дрожанием
    rng = np.random.default_rng(1)
    demo = []
    for _ in range(50):
        n = 256
        t = np.arange(n) / FPS
        m = 200 + rng.uniform(-3, 3) * np.arange(n)
        j = sum(rng.uniform(.6, 1.8) * np.sin(2 * np.pi * fr * t + rng.uniform(0, 6.3))
                for fr in rng.uniform(3, 8, 4))
        demo.append(m + j + rng.normal(0, .8, n))
    print("=== ДЕМО на синтетике, где дрожание ЗАВЕДОМО есть ===")
    check_target(demo)
    print("\nНа реальных данных ожидается похожая картина, если дрожание значимо.")
