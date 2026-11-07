"""Ограниченно повторить временный отказ открытия журнала, сохраняя исходные байты."""
from pathlib import Path
import sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
def journal(ident,title,text):
    from scripts import z13_localization_labels_20261023 as original
    original.IDENT=ident;path=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md'
    for attempt in range(6):
        before=path.read_bytes()
        try:return original.journal(title,text)
        except OSError:
            assert path.read_bytes()==before,'Не повторять запись после частичного изменения'
            if attempt==5:raise
            time.sleep(.2)
