from common import ROOT, OUT, read, sha, write, now
import ast
source = (OUT / 'classmask_probe.py').read_text(encoding='utf8')
source = '\n'.join(line.rstrip() for line in source.splitlines()).rstrip() + '\n'
ast.parse(source)
(OUT / 'classmask_probe.py').write_bytes(source.encode('utf8'))
previous = read(OUT / 'assignment_loss_protocol_v2.json')
inputs = {path: sha(ROOT / path) for path in [previous['archive'], previous['detections'], previous['native_protocol'],
          'tracking/xywh_bytetrack.py', 'tracking/spatial_identity_assignment.py', '.venv/Lib/site-packages/ultralytics/trackers/byte_tracker.py',
          '.venv/Lib/site-packages/ultralytics/trackers/utils/matching.py', 'datasets/bdd100k_loader.py',
          'research_identity_holdout_20261112/recovery/assignment_loss_result_v2.json', 'research_identity_holdout_20261112/recovery/column109699_metadata.json']}
write('classmask_protocol.json', dict(created=now(), archive=previous['archive'], detections=previous['detections'], native_protocol=previous['native_protocol'], inputs=inputs,
                                     worker_sha256=sha(OUT / 'classmask_probe.py'),
                                     question='Does class compatibility alone alter the1160 owner, and after removing2445 does it prevent class0rival2316 fromtaking109699?',
                                     prior_results='Nativeowner2445(.21115), remove-only2445 givesclass0rival2316(.69752) not2179(.71539). Raw109699class2; exactGT at1160absent. These outcomes alreadyknown; this is an explicitlylatemechanism diagnostic, not independent performance evidence.',
                                     interventions=['Classmaskalone: set savedcorrectedcost=1e6 onlywhere savedrowclass differscurrentdetectorclass; compatibleentries/rows/columns unchanged.',
                                                    'Sameclassmask plusdeleteONLY2445row; allremainingclass-compatiblecosts/candidates unchanged.'],
                                     fixed='Samecostlimit0.8, no new weights/costcalibration, fullsavedfirstmatrix. Originalbaseline+identity proof andindependentScipyaugmentedobjective dummy0.4.',
                                     acceptance=['Baselineactual_first reproduced exactly.', 'Allunmaskedcompatibleentries unchanged exactly; no GT used in classmask.',
                                                 'IndependentLapJV/SciPyobjectiveerror<1e-9; report actual109699owner and2179assignment forBOTHcontrols regardlessoutcome.',
                                                 'No claimofown109699, actualGTerrorat1160, usefuldeployedclassfix orHOTA. No further intervention variants afterthese2.'],
                                     maximum_seconds=120, deadline='2026-11-13T07:35:00+03:00'))
print(sha(OUT / 'classmask_protocol.json'))
