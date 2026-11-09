"""Algebraic readiness only: fixed 19-shift contrast, not a video experiment."""
import json,hashlib,math
from pathlib import Path
from datetime import datetime,timezone,timedelta
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
axes=[(u,v) for u in range(-2,3) for v in range(-2,3) if max(abs(u),abs(v),abs(u+v))<=2];assert len(axes)==19
U=[[j for j,(a,b) in enumerate(axes) if max(abs(u-a),abs(v-b),abs((u-a)+(v-b)))>=2] for u,v in axes]
assert all(U) and all(i not in x for i,x in enumerate(U))
def contrast(r,i):return r[i]-sum(r[j] for j in U[i])/len(U[i])
r=[.2+math.sin(i)/10 for i in range(19)];shift=.03
error=max(abs(contrast(r,i)-contrast([v+shift for v in r],i)) for i in range(19));assert error<1e-15
flat=[.7]*19;assert max(abs(contrast(flat,i)) for i in range(19))<1e-15
# All raw scores negative does not prohibit positive contrast, even without fitting.
i=axes.index((0,0));raw=[-.8]*19;raw[i]=-.4
assert max(raw)<0 and contrast(raw,i)>0
protocol=dict(registered=datetime.now(timezone(timedelta(hours=3))).isoformat(),source_sha256=sha(Path(__file__)),
    primary_source='https://www.cs.colostate.edu/~draper/papers/bolme_cvpr10.pdf',section='3.5 Failure Detection and PSR; author extended version',
    checked_before_data='19fixedaxialoffsets,nonneighborUdistance>=2; subtract per-channel Umean at the ORIGINAL equal-channel winning shift. Do not change the shift selector or jointly optimize a latent max.',
    distinctions='MOSSE PSR=(peak-mean)/std on a trained response map with11x11exclusion. This is an unnormalized per-channel contrast on19fixed shifts, an engineering hypothesis. Oldmaputility used peakz,separatedgap,entropy andspread on721location responses, not this34channelvector at fixed shift.',
    invalid_gemini_stop='All old raw head scores negative is not a valid universal stop condition for a contrast representation or retrained head. The bounded negative-profile counterexample has positive contrast. Max-minus-median is also not identical to nonneighbor-mean subtraction.',
    limits='Constant-background cancellation is algebraic, not evidence that foreign matches are flat or true matches sharp. A distractor may have a sharper peak. No causality,classification or ID benefit proved.',
    no_core=True,no_fit=True,no_tracking=True)
with (OUT/'math_protocol.json').open('x',encoding='utf8') as f:json.dump(protocol,f,ensure_ascii=False,indent=2)
with (OUT/'math_checks.json').open('x',encoding='utf8') as f:json.dump(dict(passed=True,axes=axes,nonneighbor_indices=U,counts=[len(x) for x in U],constant_offset_max_error=error,constant_profile_zero=True,
    negative_raw_counterexample=dict(raw_max=max(raw),contrast=contrast(raw,i)),source_sha256=sha(Path(__file__)),protocol_sha256=sha(OUT/'math_protocol.json'),goal_achieved=False),f,ensure_ascii=False,indent=2)
print('SHIFT_CONTRAST_MATH_PASSED',min(map(len,U)),max(map(len,U)),flush=True)
