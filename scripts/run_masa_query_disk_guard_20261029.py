"""Separate safety wrapper; preserve frozen scientific worker unchanged."""
from pathlib import Path
import sys,shutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261029 import RUN,service

def main():
    source=ROOT/'scripts/run_masa_detic_query_pilot_20261029.py'
    p=service.read(RUN/'masa_query_pilot/protocol.json')
    assert service.sha(source)==p['sha256'][source.relative_to(ROOT).as_posix()]
    code=source.read_text(encoding='utf-8')
    anchor='    def status(**kw):\n'
    assert code.count(anchor)==1
    code=code.replace(anchor,anchor+"        assert shutil.disk_usage(ROOT).free>5*1024**3,'Disk reserve5GB'\n")
    anchor="            if vi==0:\n"
    assert code.count(anchor)==1
    code=code.replace(anchor,anchor+"                saved_bytes=sum(f.stat().st_size for f in out.rglob('*') if f.is_file())\n                free=shutil.disk_usage(ROOT).free\n                disk_need=2*saved_bytes*19+5*1024**3\n                service.write(folder/'disk_gate.json',dict(passed=disk_need<free,first_video_bytes=saved_bytes,required_bytes=disk_need,free_bytes=free,updated=service.now()))\n                assert disk_need<free,'Conservative disk gate failed; keep complete first video'\n")
    ns={'__name__':'masa_query_disk_guard_worker','__file__':str(source),'shutil':shutil}
    exec(compile(code,str(source),'exec'),ns);ns['main']()

if __name__=='__main__':main()
