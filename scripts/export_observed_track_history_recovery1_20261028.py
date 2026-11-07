"""Та же подготовка; исправить вызов установленного конструктора до запуска."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

if __name__=='__main__':
    import scripts.export_observed_track_history_20261028 as original
    code=Path(original.__file__).read_text(encoding='utf-8')
    old="XYWHByteTrack(SimpleNamespace(**p['tracker_config']),frame_rate=30)"
    assert code.count(old)==1
    code=code.replace(old,"XYWHByteTrack(SimpleNamespace(**p['tracker_config']))")
    old="folder=RUN/'history_observed_export';p=read(folder/'protocol.json')"
    assert code.count(old)==1
    code=code.replace(old,"folder=RUN/'history_observed_export_recovery1';p=read(folder/'protocol.json')")
    ns={**vars(original),'__name__':'observed_export_checked_constructor'}
    exec(compile(code,str(Path(original.__file__)),'exec'),ns);ns['main']()
