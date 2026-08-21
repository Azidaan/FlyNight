"""Запуск штатной оценки bdd100k с починкой совместимости pydantic.

Вызывать вместо `python -m bdd100k.eval.run`: аргументы те же, разница
только в том, что перед разбором данных применяется scalabel_compat.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.scalabel_compat import apply  # noqa: E402

if __name__ == '__main__':
    n = apply()
    print(f'[COMPAT] поправлено необязательных полей scalabel: {n}', flush=True)
    from bdd100k.eval.run import run
    run()
