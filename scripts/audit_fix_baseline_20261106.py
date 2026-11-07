"""Snapshot existing work before bounded lifecycle repairs; never stage others' work."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib, json, subprocess, zipfile
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/flyvis_repairs_20261106'
FILES=['tracking/history_pair_runtime.py','tracking/lazy_appearance_association.py',
       'tracking/query_appearance_comparison.py','tracking/frame_masa_appearance.py',
       'tracking/query_assignment_write.py','tracking/query_memory_write.py',
       'tracking/query_bounded_write.py','tracking/causal_motion_history.py',
       'dissertation/ЕДИНЫЙ_ПЛАН.md','experiments/project_history.json','experiments/system_problems.json']
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    target=OUT/'parent_baseline.zip'
    if target.exists():raise FileExistsError(target)
    records={}
    with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for name in FILES:
            data=(ROOT/name).read_bytes();archive.writestr(name,data)
            records[name]=dict(sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))
    status=subprocess.check_output(['git','status','--porcelain=v1','-z'],cwd=ROOT)
    (OUT/'git_status_before.bin').write_bytes(status)
    report=dict(at=datetime.now(timezone(timedelta(hours=3))).isoformat(),
        deadline='2026-11-07T01:00:00+03:00',files=records,
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        intent='Fix verified lifecycle reset/stale-state defects; exact fresh-vs-reset outputs; no scientific settings, data, weights or old results changed.')
    (OUT/'parent_baseline.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Saved baseline for',len(records),'files')
if __name__=='__main__':main()
