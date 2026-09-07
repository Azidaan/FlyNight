import numpy as np
import pywt

np.random.seed(0)
N = 256          # длина окна траектории (кадров)
t = np.arange(N)

# Три эталонных сигнала движения
const_pos  = np.full(N, 100.0)                 # объект стоит
const_vel  = 100.0 + 2.5 * t                   # равномерное движение (полином 1-й степени)
const_acc  = 100.0 + 2.5 * t + 0.01 * t**2     # равноускоренное (полином 2-й степени)

signals = {"стоит (полином 0)": const_pos,
           "равномерное (полином 1)": const_vel,
           "ускорение (полином 2)": const_acc}

bases = ["haar", "db2", "db3", "db4", "coif1", "bior2.2"]

print("Энергия детализирующих коэффициентов уровня 1 (чем ближе к нулю, тем лучше базис")
print("описывает это движение как 'норму' и не засоряет детали)\n")
hdr = f"{'базис':<10}" + "".join(f"{k:>26}" for k in signals)
print(hdr)
print("-" * len(hdr))

for b in bases:
    w = pywt.Wavelet(b)
    row = f"{b:<10}"
    for name, sig in signals.items():
        # одноуровневое разложение; cD = детализирующие (высокочастотные) коэффициенты
        cA, cD = pywt.dwt(sig, w, mode="periodization")
        # берём середину, чтобы исключить краевые эффекты
        core = cD[3:-3]
        energy = np.sqrt(np.mean(core**2))
        row += f"{energy:>26.3e}"
    print(row)

print("\nЧисло нулевых моментов у базисов:")
for b in bases:
    w = pywt.Wavelet(b)
    print(f"  {b:<10} vanishing moments = {w.vanishing_moments_psi}")
