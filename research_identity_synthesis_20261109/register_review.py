"""Freeze the mathematical check after the complete Gemini response."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent
capture=json.loads((OUT/'gemini_capture_verified.json').read_text(encoding='utf8'));assert capture['passed'] and capture['requestExactAfterWhitespaceNormalization'] and capture['answerEndMarker']
paths=[OUT/'check_math.py',OUT/'gemini_request.txt',OUT/'gemini_answer.txt',OUT/'synthesis.json',OUT/'authorization.json']
value=dict(scope='Mathematical prerequisites and counterexamples only; no new real score, neural map or training.',sources={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},checks=['A nonuniform linear rule can reverse a wrong arithmetic mean, without claiming reliable learning.','Ridge-logistic Hessian positive at finite logits; both classes are needed for a finite unpenalized-intercept optimum.','If argmax denotes list indices, MAD changes under renumbering; use physical2D displacements.','Identical foreign visible patterns have identical peak agreement; consensus does not guarantee identity.'],sources_checked=dict(Peduzzi='DOI resolver403. PubMed original abstract read: EPV2..25 and seven predictors, conventional coefficient estimation, not our34-variable penalized model or a channel-consensus tracking rule.',vanSmeden='Publisher original full text read, DOI10.1186/s12874-016-0267-3: performance depends on more than EPV; no universal cutoff. This does not prove our sample sufficient.'))
with (OUT/'math_protocol.json').open('x',encoding='utf8') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')
print('Mathematical checks registered')
