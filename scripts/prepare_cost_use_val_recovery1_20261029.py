"""Repair only the old-report schema check; preserve the failed preparation."""
from pathlib import Path
import sys,inspect
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import prepare_cost_use_val_20261029 as original

def main():
    from scripts.continuation_service_20261029 import RUN,service
    c=service.read(RUN/'cost_use_val_preparation/controller_result.json')
    assert c['exit_code']==1 and c['all_owned_ended'] and not any(service.matching(k,v) for k,v in c['owned'].items())
    audit=service.read(RUN/'masa_audit_fp32_recovery1/report.json')
    assert audit['passed'] and audit['complete'] and len(audit['checked'])==20
    assert not (RUN/'cost_use_val/protocol.json').exists(),'No old new-val output may be overwritten'
    source=inspect.getsource(original.main).replace("checks=RUN/'cost_use_val_preparation'","checks=RUN/'cost_use_val_preparation_recovery1'")
    old="assert r['passed'] and all(service.sha(ROOT/k)==h for k,h in r['sha256'].items())"
    assert source.count(old)==1
    source=source.replace(old,"assert r['video']==s['video'] and r['native_zero_exact'] and set(r['metrics'])=={'native','zero','r50_1024','r50_1333','detic_1333'}\n        assert all(service.sha(ROOT/k)==h for k,h in r['sha256'].items())")
    old="p['sha256']={**old['sha256'],**dev['sha256'],**{f:service.sha(ROOT/f) for f in files}}"
    assert source.count(old)==1
    source=source.replace(old,"files.extend(['scripts/prepare_cost_use_val_recovery1_20261029.py','scripts/cost_use_val_recovery1_20261029.py','scripts/audit_cost_use_val_recovery1_20261029.py','runs/continuation_20261029/cost_use_val_preparation/controller_result.json','runs/continuation_20261029/cost_use_val_preparation/stderr.log'])\n    p['preparation_recovery']='Old per-video MASA report has native_zero_exact and metrics, no passed field. Full old20 audit passed separately. Original330 code1 and source preserved; no new val output existed. Correct schema gate only, no criteria/arithmetic/input change.'\n    "+old)
    source=source.replace("('340','cost_use_val','scripts/cost_use_val_20261029.py')","('340','cost_use_val','scripts/cost_use_val_recovery1_20261029.py')")
    source=source.replace("('350','cost_use_val_audit','scripts/audit_cost_use_val_20261029.py')","('350','cost_use_val_audit','scripts/audit_cost_use_val_recovery1_20261029.py')")
    ns=dict(vars(original));exec(compile(source,__file__,'exec'),ns);ns['main']()

if __name__=='__main__':main()
