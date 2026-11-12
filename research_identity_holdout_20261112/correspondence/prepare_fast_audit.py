"""Create a separately fingerprinted, mathematically identical fast auditor."""
import ast,json,hashlib
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
base=(OUT/'audit.py').read_text(encoding='utf8')
old='offsets=np.array([(dx,dy) for dy in range(-2,3) for dx in range(-2,3) if dx or dy])\n'
new=old+'''def boolean_field(gray):
    # Independent24Boolean comparators, no production packedCensus/POPCOUNT.
    field=np.zeros(gray.shape+(24,),dtype=bool)
    center=gray[2:-2,2:-2]
    for bit,(dx,dy) in enumerate(offsets):
        field[2:-2,2:-2,bit]=gray[2+dy:gray.shape[0]-2+dy,2+dx:gray.shape[1]-2+dx]<center
    return field

def exact_histogram_median_and_sum(hamming):
    # Hamming is an integer in0..24; CDF orderstatistics equal fullsorting.
    hist=np.bincount(hamming,minlength=25)
    assert len(hist)==25
    cumulative=np.cumsum(hist);N=len(hamming)
    indices=np.searchsorted(cumulative,[(N-1)//2+1,N//2+1])
    return float(indices.sum()/2),int(np.dot(hist,np.arange(25,dtype=np.int64)))
'''
assert base.count(old)==1;base=base.replace(old,new)
old="center0=a[pixels[:,1],pixels[:,0]];bits0=a[pixels[:,1,None]+offsets[None,:,1],pixels[:,0,None]+offsets[None,:,0]]<center0[:,None];expected=[];cost=[];prior="
new="field0=boolean_field(a);field1=boolean_field(b);bits0=field0[pixels[:,1],pixels[:,0]];expected=[];cost=[];prior="
assert base.count(old)==1;base=base.replace(old,new)
old="xy=pixels+[dx,dy];bits1=b[xy[:,1,None]+offsets[None,:,1],xy[:,0,None]+offsets[None,:,0]]<b[xy[:,1],xy[:,0],None];hamming=np.count_nonzero(bits0!=bits1,axis=1);ordered=sorted(int(x) for x in hamming);N=len(ordered);median=float(ordered[N//2] if N%2 else (ordered[N//2-1]+ordered[N//2])/2);expected.append((dx,dy));cost.append((median,sum(ordered),sum((v-w)**2 for v,w in zip([dx,dy],prior)),dx,dy))"
new="xy=pixels+[dx,dy];bits1=field1[xy[:,1],xy[:,0]];hamming=np.count_nonzero(bits0!=bits1,axis=1);median,total=exact_histogram_median_and_sum(hamming);expected.append((dx,dy));cost.append((median,total,sum((v-w)**2 for v,w in zip([dx,dy],prior)),dx,dy))"
assert base.count(old)==1;base=base.replace(old,new)
base=base.replace("with (OUT/'audit.json').open('xb')", "with (OUT/'audit_fast_v2.json').open('xb')")
base=base.replace("print('HOLDOUT_CORRESPONDENCE_AUDIT'", "print('HOLDOUT_CORRESPONDENCE_AUDIT_FAST_V2'")
base=base.replace("out=dict(passed=True,records=", "out=dict(passed=True,audit_version='directBoolean_precomputed_histogram_v2',records=")
base=base.replace("Independent domain enumeration and direct Boolean24bit comparisons; SciPy interpolation; scalar channel/head arithmetic.","Independent complete domain enumeration and precomputed24Boolean comparisons; EXACT0..24histogrammedian/integersum checked againstEVERY storedcosttuple, domain/tie/winner unchanged; SciPy interpolation; scalar channel/head arithmetic.")
base=base.replace("p=read(OUT/'protocol.json');res=", "p=read(OUT/'protocol.json');fast_protocol=read(OUT/'audit_fast_protocol_v2.json');assert fast_protocol['scientific_protocol_sha256']==sha(OUT/'protocol.json')\nfor n,h in fast_protocol['inputs'].items():assert sha(ROOT/n)==h,n\nres=")
base=base.replace("scope='No new fits", "fast_audit_protocol_sha256=sha(OUT/'audit_fast_protocol_v2.json'),auditor_sha256=sha(Path(__file__)),scope='No new fits")
base='\n'.join(line.rstrip() for line in base.splitlines()).rstrip()+'\n';ast.parse(base)
path=OUT/'audit_fast_v2.py'
with path.open('xb') as f:f.write(base.encode('utf8'))
inputs={}
for name in ['protocol.json','result.json','samples.npz','core_result.json','audit.py','audit_fast_v2.py','prepare_fast_audit.py']:
    f=OUT/name;inputs[str(f.relative_to(ROOT)).replace('\\','/')]=sha(f)
core=json.loads((OUT/'core_result.json').read_text(encoding='utf8'))
for record in core['records']:inputs[record['path']]=record['sha256'];assert sha(ROOT/record['path'])==record['sha256']
protocol=dict(registered=datetime.now(ZoneInfo('Europe/Moscow')).isoformat(),scientific_protocol_sha256=sha(OUT/'protocol.json'),purpose='Technical EXACT acceleration of independent auditor only, no scientificoutput/selection/map/model/threshold changes.',frozen_original_auditor_sha256=sha(OUT/'audit.py'),auditor_sha256=sha(path),changes=['Precompute the same24directBoolean comparisons once pergray instead of gathering24neighbors at everytranslation. No production Census helper import.','Hamming values0..24 aggregated into25integer bins; median via exact twoCDF orderstatistics and integer sum. No approximate quantile, trimming, candidate reduction or pixel subsampling.','Independently enumerate entire feasible domain; compare EVERY saved costtuple, samelexicographic tie and winningtranslation, then sameallrow interpolation/channel/head checks.'],output='audit_fast_v2.json',failure_policy='Any mismatch remainsfailure; do not weaken1e-9. Preserve interrupted originalaudit as interruptednotpassed; parentmay make explicit binding to actualfast_v2 resultafterpassed.',entrypoint='research_identity_holdout_20261112/correspondence/audit_fast_v2.py',deadline='2026-11-13T07:35:00+03:00',inputs=inputs)
with (OUT/'audit_fast_protocol_v2.json').open('xb') as f:f.write((json.dumps(protocol,ensure_ascii=False,indent=2).rstrip()+'\n').encode('utf8'))
print('FAST_AUDIT_REGISTERED',sha(path),len(inputs))
