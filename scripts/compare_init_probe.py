"""Сравнение проб инициализации DRB между собой и с baseline (узел Е панели).

Вопрос, который решает проба: с какого вклада должны стартовать ветви DRB.
При вкладе 50% (как записано в плане) блок на старте усиливает признаки
примерно в 1.8 раза, и предобученная голова детектора получает не тот
масштаб, к которому привыкла — первые эпохи уходят на восстановление.
Отрицательный стартовый вес делает блок почти тождественным, и он
«включается» по мере того, как обучение находит его полезным.

Эталон — контрольный прогон base12: те же 12 эпох, тот же сид и та же
конфигурация данных, но без блока. Стосерийный baseline для этого не
годится: на 12-й эпохе у него втрое выше скорость обучения и ещё включена
мозаичная аугментация, то есть он учится в другом режиме.

Выход: experiments/results/init_probe.csv + таблица в консоль.

Запуск: python scripts/compare_init_probe.py
"""
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

PROBE_ROOT = PROJECT_ROOT / 'runs' / 'init_probe'
BASELINE = PROBE_ROOT / 'base12' / 'seed42' / 'results.csv'   # контроль
OUT_CSV = PROJECT_ROOT / 'experiments' / 'results' / 'init_probe.csv'
VARIANTS = [('w0', 0.0, 'вклад 50% (план)'),
            ('w2', -2.0, 'вклад 12%'),
            ('w4', -4.0, 'вклад 2%'),
            ('boost', -4.0, 'тождеств. + разгон'),
            ('drb12', -2.0, 'DRB, условия контроля'),
            ('ft_base', None, 'дообуч. без блока'),
            ('ft_drb', -2.0, 'дообуч. с блоком')]


def read_curve(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def col(row: dict, needle: str) -> float | None:
    for k, v in row.items():
        if needle in k and v not in (None, ''):
            try:
                return float(v)
            except ValueError:
                return None
    return None


def series(rows: list[dict], needle: str) -> list[float]:
    return [v for v in (col(r, needle) for r in rows) if v is not None]


def main() -> int:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    base = read_curve(BASELINE)
    curves = {}
    for pid, w, label in VARIANTS:
        rows = read_curve(PROBE_ROOT / pid / 'seed42' / 'results.csv')
        if rows:
            curves[pid] = (w, label, rows)

    if not curves:
        print('[WARN] ни одна проба ещё не посчитана — запустите их в узле Е')
        return 1

    n = min(len(r) for _, _, r in curves.values())
    print(f'Сравнение на первых {n} эпохах (сид 42, одинаковые данные '
          f'и гиперпараметры; отличается только стартовый вес ветвей)\n')

    header = f'{"вариант":<22}{"стартовый вес":>15}{"mAP50 на конце":>17}{"лучший mAP50":>15}{"ошибка рамок":>15}'
    print(header)
    print('-' * len(header))

    rows_out = []
    if base:
        b = series(base, 'mAP50(B)')[:n]
        bl = series(base, 'train/box_loss')[:n]
        if b:
            print(f'{"контроль (без DRB)":<22}{"—":>15}{b[-1]:>17.4f}'
                  f'{max(b):>15.4f}{bl[-1]:>15.4f}')
            rows_out.append(['base12', '', f'{b[-1]:.5f}', f'{max(b):.5f}',
                             f'{bl[-1]:.5f}', n])
    else:
        print('[WARN] контрольный прогон без DRB не посчитан — пробы сравнимы '
              'только между собой (запустите «Контроль» в узле Е)')

    best_id, best_val = None, -1.0
    for pid, (w, label, rows) in curves.items():
        m = series(rows, 'mAP50(B)')[:n]
        loss = series(rows, 'train/box_loss')[:n]
        if not m:
            continue
        wtxt = '—' if w is None else f'{w:g}'
        print(f'{label:<22}{wtxt:>15}{m[-1]:>17.4f}{max(m):>15.4f}'
              f'{(loss[-1] if loss else float("nan")):>15.4f}')
        rows_out.append([pid, '' if w is None else w, f'{m[-1]:.5f}', f'{max(m):.5f}',
                         f'{loss[-1]:.5f}' if loss else '', n])
        if m[-1] > best_val:
            best_id, best_val = label, m[-1]

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
        w_ = csv.writer(f)
        w_.writerow(['вариант', 'стартовый_вес', 'mAP50_конец', 'mAP50_лучший',
                     'ошибка_рамок', 'эпох'])
        w_.writerows(rows_out)

    print(f'\nЛучший по mAP50 на {n}-й эпохе: {best_id} ({best_val:.4f})')
    if len(curves) < len(VARIANTS):
        missing = [lbl for pid, _, lbl in VARIANTS if pid not in curves]
        print(f'Ещё не посчитаны: {", ".join(missing)}')
    print(f'[OK ] {OUT_CSV}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
