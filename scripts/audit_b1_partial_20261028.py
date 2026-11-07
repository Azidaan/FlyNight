"""Точный CPU-повтор19готовых видео; не подмена полного протокола20."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.recovery_research_service_20261027 import RUN,read,write,sha,matching

def main():
    folder=RUN/'b1_partial_audit';p=read(folder/'protocol.json');saved=read(folder/'partial_source.json')
    assert sha(folder/'partial_source.json')==p['partial_source_sha256']
    for path,digest in {**p['sha256'],**saved['sha256']}.items():assert sha(ROOT/path)==digest,path
    import scripts.audit_b1_tracker_20261028 as original
    code=Path(original.__file__).read_text(encoding='utf-8')
    substitutions={
        "source = RUN / 'b1_tracker'; folder = RUN / 'b1_tracker_audit'":"source = RUN / 'b1_tracker/continuation_recovery1'; folder = RUN / 'b1_partial_audit'",
        "assert controller['exit_code'] == 0 and controller['all_owned_ended'] and not controller['timed_out']":"assert controller['exit_code'] == 1 and controller['all_owned_ended'] and not controller['timed_out']; assert sha(source / 'controller_result.json') == saved['controller_sha256']",
        "r = read(source / ('report.json' if complete else 'cost_stop.json'))":"assert not complete; r = saved",
        "assert len(videos) == (len(p['sources']) if complete else 1)":"assert len(videos) == 19 and len(p['sources']) == 20; assert [s['video'] for s in videos] == saved['completed_videos']",
        "if complete:\n                assert metrics == r['per_video'][name][v]":"assert metrics == read(source / 'per_video.json')[name][v]",
        "report_sha256=sha(source / ('report.json' if complete else 'cost_stop.json'))":"report_sha256=sha(folder / 'partial_source.json'), scientific_verdict=None, requested_videos=20",
        "write(source / 'independent_audit.json', result); write(folder / 'report.json', result)":"write(folder / 'report.json', result)"}
    for old,new in substitutions.items():assert code.count(old)==1,old;code=code.replace(old,new)
    ns={**vars(original),'__name__':'strict_partial_auditor','saved':saved}
    exec(compile(code,str(Path(original.__file__)), 'exec'),ns);ns['run']()

if __name__=='__main__':main()
