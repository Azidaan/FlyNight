import numpy as np, pywt
FPS = 30

def make_track(N, jit_amp, rng):
    t = np.arange(N)/FPS
    motion = 200.0 + rng.uniform(-3,3)*np.arange(N)
    lo, hi = int(.15*N), int(.85*N)
    for _ in range(rng.integers(1,3)):
        c = rng.integers(lo, max(lo+1,hi)); amp = rng.uniform(-40,40)
        w = rng.uniform(.7,1.5)*FPS
        motion += amp/(1+np.exp(-(np.arange(N)-c)/w))
    jit = np.zeros(N)
    for f in rng.uniform(3,8,4):
        jit += jit_amp*rng.uniform(.6,1.8)*np.sin(2*np.pi*f*t+rng.uniform(0,6.28))
    return motion, motion + jit + rng.normal(0,.8,N)

# ---------- предсказатели ----------
def p_2pt(h, gap):                       # соломенное чучело: скорость по 2 точкам
    v = h[-1]-h[-2]; return h[-1]+v*np.arange(1,gap+1)

def p_lsq(h, gap, win=16):               # честная база: МНК-прямая по последним win кадрам
    y = h[-win:]; x = np.arange(win)
    k, b = np.polyfit(x, y, 1)
    return (k*(win-1+np.arange(1,gap+1)) + b)

def p_kalman(h, gap, q=1e-2, r=0.8**2):  # настоящий фильтр Калмана, пост. скорость
    x = np.array([h[0], 0.0]); P = np.diag([10.0, 10.0])
    F = np.array([[1.0,1.0],[0.0,1.0]]); H = np.array([[1.0,0.0]])
    Q = q*np.array([[0.25,0.5],[0.5,1.0]])
    for z in h[1:]:
        x = F@x; P = F@P@F.T + Q
        y = z - H@x; S = H@P@H.T + r; K = (P@H.T)/S
        x = x + (K.flatten()*y); P = (np.eye(2)-K@H)@P
    out=[]
    for _ in range(gap):
        x = F@x; out.append(x[0])
    return np.array(out)

def p_wave(h, gap, basis="db2", level=3, drop=(1,2), win=16):
    x = np.asarray(h,float); idx=np.arange(len(x))
    k,b = np.polyfit(idx,x,1); tr=k*idx+b; xd=x-tr
    co = pywt.wavedec(xd, basis, level=level, mode="symmetric")
    out=[]
    for i,c in enumerate(co):
        lvl = level-i+1 if i>0 else None
        out.append(np.zeros_like(c) if lvl in drop else c)
    clean = pywt.waverec(out, basis, mode="symmetric")[:len(x)] + tr
    return p_lsq(clean, gap, win)        # одинаковая экстраполяция, честно

OBS, TRIALS, GAP = 64, 600, 20
preds = {
    "2-точечная разность (чучело)": p_2pt,
    "МНК по 16 кадрам": p_lsq,
    "фильтр Калмана (пост.скор.)": p_kalman,
    "вейвлет db2 + МНК": lambda h,g: p_wave(h,g,"db2"),
    "вейвлет haar + МНК": lambda h,g: p_wave(h,g,"haar"),
    "вейвлет db4 + МНК": lambda h,g: p_wave(h,g,"db4"),
}

for amp in [0.0, 1.0, 3.0]:
    rng = np.random.default_rng(2024)
    tks = [make_track(OBS+GAP, amp, rng) for _ in range(TRIALS)]
    print(f"\n=== дрожание x{amp}, окклюзия {GAP} кадров, медианная ошибка (пикс) ===")
    ref = None
    for name, f in preds.items():
        errs=[]
        for tr,o in tks:
            errs.append(np.mean(np.abs(f(o[:OBS],GAP)-tr[OBS:OBS+GAP])))
        m=np.median(errs)
        if name.startswith("фильтр Калмана"): ref=m
        print(f"  {name:<32}{m:8.2f}", end="")
        if ref is not None and not name.startswith("фильтр Калмана"):
            print(f"   ({100*(ref-m)/ref:+6.1f}% против Калмана)")
        else:
            print()
