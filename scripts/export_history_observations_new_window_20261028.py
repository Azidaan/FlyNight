"""Экспорт реально назначенных детекций в новом разрешённом окне."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261028 import RUN,service

if __name__=='__main__':
    import scripts.export_observed_track_history_20261028 as original
    code=Path(original.__file__).read_text(encoding='utf-8')
    old="XYWHByteTrack(SimpleNamespace(**p['tracker_config']),frame_rate=30)";assert code.count(old)==1
    code=code.replace(old,"XYWHByteTrack(SimpleNamespace(**p['tracker_config']))")
    old="previous=RUN/'history_sources_after_partial'";assert code.count(old)==1
    code=code.replace(old,"previous=RUN/'history_sources'")
    ns={**vars(original),'__name__':'history_observations_new_window','RUN':RUN}
    exec(compile(code,str(Path(original.__file__)),'exec'),ns);ns['main']()
