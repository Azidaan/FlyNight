"""Validate saved exposure in both precisions; preserve the original failed controller."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.flyvis_two_frame_input_20261107 import OUT,read,write,sha
import numpy as np

def main():
    old=read(OUT/'controller.json');assert old['passed'] is False and old['failed']=='audit'
    original_audit=read(OUT/'independent_audit.json');assert original_audit['passed']
    p=read(OUT/'protocol.json');done=read(OUT/'inference_completed.json')
    first=read(OUT/'first_case.json'); rows=[]
    receipt=dict(source_sha256=sha(Path(__file__)),failed_controller_sha256=sha(OUT/'controller.json'),
        original_protocol_sha256=sha(OUT/'protocol.json'),reused_quality_audit_sha256=sha(OUT/'independent_audit.json'),
        reason='Worker subtracted float32 arrays; extra audit compared with unrounded float64 subtraction. Verify EXACT original float32 statistic plus independent float64 statistic. No tolerance or scientific gate change, no repeated inference or quality evaluation.')
    write(OUT/'exposure_recovery1_protocol.json',receipt)
    for case in p['cases']:
        raw=OUT/'raw'/f"{case['uid']}.npz"
        assert sha(raw)==done['raw_sha256'][raw.name]
        with np.load(raw,allow_pickle=False) as z:
            x=z['maps'].astype('float64');f=z['maps']
            row=dict(uid=case['uid'],t4_max=float(abs(x[3:,:4]-x[0,:4]).max()),t5_max=float(abs(x[3:,4:]-x[0,4:]).max()),identity_max=float(abs(x[1]-x[0]).max()))
            if not rows:
                assert case['uid']==first['uid']==p['first_gate']['uid']
                fp32=dict(t4_max=float(abs(f[3:,:4]-f[0,:4]).max()),t5_max=float(abs(f[3:,4:]-f[0,4:]).max()),identity_max=float(abs(f[1]-f[0]).max()))
                for key,value in fp32.items():
                    assert first[key]==value,key
                assert first['passed'] and all(row[k]>max(row['identity_max'],p['first_gate']['minimum_map_change']) for k in ['t4_max','t5_max'])
            rows.append(row)
    write(OUT/'exposure_audit.json',dict(passed=True,count=len(rows),rows=rows,t4_changed=sum(r['t4_max']>1e-7 for r in rows),t5_changed=sum(r['t5_max']>1e-7 for r in rows),
        first_gate_exact=True,original_fp32_statistic_exact=fp32,independent_first_fp64=rows[0],recovery_protocol_sha256=sha(OUT/'exposure_recovery1_protocol.json')))
    write(OUT/'recovery1_controller.json',dict(passed=True,reused_inference=True,reused_quality_audit=True,exposure_audit_sha256=sha(OUT/'exposure_audit.json'),original_controller_sha256=sha(OUT/'controller.json')))
    print('Exposure verified; original failed controller preserved; zero repeated inference',flush=True)

if __name__=='__main__':main()
