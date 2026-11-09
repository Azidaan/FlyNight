"""Fix the final model and actual-failure screen before coefficients or scores."""
import json,gzip,hashlib,subprocess
from pathlib import Path
from datetime import datetime,timezone,timedelta
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent;NATIVE=ROOT/'runs/domain_matched_native_20261109';OLD=ROOT/'runs/full_template_native_20261109';TRAIN=ROOT/'research_domain_matched_reader_20261109'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert read(OUT/'audit.json')['passed'] and read(OUT/'summary.json')['gates']['C34__mask_only'] and not read(OUT/'summary.json')['gates']['C34__aligned']
cases=[{k:r[k] for k in ['video','frame','old_id','raw_id','kind']} for r in read(NATIVE/'id_fate.json')['records'] if r['arm']=='C34' and r['scored'] and r['kind']!='unknown'];assert len(cases)==5
files=[Path(__file__),OUT/'deploy_screen.py',OUT/'protocol.json',OUT/'features.json',OUT/'predictions.json',OUT/'summary.json',OUT/'audit.json',NATIVE/'deployment.json',NATIVE/'id_fate.json',TRAIN/'features.json',ROOT/'research_geometry_consistency_20261109/coordinate_report.json',ROOT/'runs/z13_local_peak_20261021/geometry.json']
files += [ROOT/n for n in ['tracking/channel_identity_head.py','tracking/observed_spatial_reader.py','tracking/geometric_template_correspondence.py','tracking/full_template_correspondence.py','tracking/spatial_identity_assignment.py','tracking/channel_balanced_correspondence.py','tracking/hex_linear_interpolation.py']]
def key(v,s):return hashlib.sha256(json.dumps([v,s],separators=(',',':')).encode()).hexdigest()[:32]
for case in cases:
    v=case['video'];path=NATIVE/'rollouts/C34'/(v+'.events.json.gz');files += [path,OLD/'rollouts/C34'/(v+'.events.json.gz')]
    with gzip.open(path,'rt',encoding='utf8') as f:events=json.load(f)['events']
    e=next(e for e in events if e['frame']==case['frame']);q=next(q for q in e['proposals'] if q['old_id']==case['old_id'] and q['raw_id']==case['raw_id']);s=[[h['frame'],h['crop']] for h in q['history']]
    files += [NATIVE/'features'/v/(key(v,seq)+'.C34.npy') for seq in [s,s+q['search']]]
protocol=dict(registered=datetime.now(timezone(timedelta(hours=3))).isoformat(),head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),sources={p.relative_to(ROOT).as_posix():sha(p) for p in files},cases=cases,
    selection='Mask-only C34 is the only passed component gate. Aligned failed and is not deployed. Same-mask image models serve as controls despite their negative point gates. No choice by these fivevalcases.',
    action='Fit4finalmodels onALL26trainingpairs: C34mask,spatialimage-mask,temporalimage-mask,C34oldfull-scalecontrol. Reuse readyfull26geometrynormalization andG7. Independently verify final loss/gradient and foldscale-control into equivalent raw-correlation coefficients.',
    screen='AllfivepreviouslydirectlylabeledcurrentC34queries,4own/1foreign. Same histories and maps. Retain every previously correct decision and strictly decrease totalwronglabels from3. This is a new pre-native mechanism gate recorded after the CVgate but before finalcoefficients/scores; no thresholds,featurewindows or candidates are adjusted to pass it.',
    geometry='Reuse old saved scales/offsets ONLY when old/currentmean,covariance,referencebox/crop,currentbox/crop are exactlyequal. Otherwise restore only the missing fivecasegeometry once and disclose it. Keep originalbox coordinates andmean-correlation winner overcommon visible support.',
    scope='CPU only,4newfits,no core,retina,video,detector,nativerolloutoroldmodelrefits. Ifscreenfails no newfullnativepilot. Ifpass, a separatelyfrozen fullnativecomparison with geometric,image,scale controls and actualIDfate is still required. This is not independentvalidation or proof of coreidentity.',save_stop='2026-11-09T23:54:00+03:00')
with (OUT/'deployment_screen_protocol.json').open('x',encoding='utf8') as f:json.dump(protocol,f,ensure_ascii=False,indent=2)
print('MASK_FINAL_MODEL_AND_FIVECASE_GATE_REGISTERED',flush=True)
