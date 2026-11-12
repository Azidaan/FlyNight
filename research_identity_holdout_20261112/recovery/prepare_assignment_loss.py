from common import ROOT, OUT, read, sha, write, now
import ast
worker = OUT / 'assignment_loss_probe.py'
source = '\n'.join(line.rstrip() for line in worker.read_text(encoding='utf8').splitlines()).rstrip() + '\n'
ast.parse(source)
worker.write_bytes(source.encode('utf8'))
native_path = 'runs/domain_matched_native_20261109/protocol.json'
native = read(ROOT / native_path)
archive = 'runs/domain_matched_native_20261109/rollouts/C34/b1ca2e5d-84cf9134.events.json.gz'
detections = native['videos']['b1ca2e5d-84cf9134']['detections']
inputs = [archive, native_path, detections, 'tracking/xywh_bytetrack.py', 'tracking/spatial_identity_assignment.py',
          '.venv/Lib/site-packages/ultralytics/trackers/byte_tracker.py', '.venv/Lib/site-packages/ultralytics/trackers/utils/matching.py',
          '.venv/Lib/site-packages/ultralytics/utils/metrics.py', 'runs/full_template_native_20261109/native_runtime.py', 'runs/domain_matched_native_20261109/runtime_adapter.py']
write('assignment_loss_protocol.json', dict(created=now(), archive=archive, native_protocol=native_path, detections=detections,
                                           worker_sha256=sha(worker), inputs={path: sha(ROOT / path) for path in inputs},
                                           question='What creates nativecost2179≈0.715 versus2445≈0.211 at saved1160, and does removingONLY2445 from the unchanged first costmatrix give2179 the109699column?',
                                           known_before_probe='Savedtrace shows2179 losesassignmentat1160, isLost1161, and2445 owns109699. ExactGT1159 maps2179→00122657,2445→00122746;1165 maps2445→00122657. NoGT at1160, so counterfactual assignment is not a correctness result.',
                                           method='Independent scalarfloat64 andsource-equivalentfloat32 IoU/confidence formula onALLsavedrows×highcolumns. Formula candidate1−IoU*detconfidence includingfloat32 IoU-distance roundtrip. Appearance/history not directfeatures unless sourceaudit showsotherwise; history alreadyaffects savedKalmanmean.',
                                           intervention='Baseline savedcorrectedfloat32matrix copiedidentitywitness; remove ONLYrowtid2445, allotherrows/columns/entries andcostlimit0.8 unchanged. Fullglobalfirstsolver, no frozenassignments andno downstream lifecycle replay. No safe-deploymentclaim for removinganactive competitor.',
                                           controls='Reproduce actual_first exactly, identitycopyexact, preserve survivingentries exactly; independentSciPyaugmentedassignmentobjective withdummycost0.4 verifies LapJV optimum forbaseline/counterfactual.',
                                           acceptance=['Source/currentfingerprints compared witharchivedprotocol whereavailable; never labelcurrenthash historical ifabsent/mismatched.',
                                                       'Fullsavedbase matrix reproduced fromtypedformula within1e-6; scalar64 discrepancy reported, notsilentlymasked.',
                                                       'Baselineactual_first exact, unalteredremainingcostentries, independentassignmentobjectiveerror<1e-9.',
                                                       'Report actualnewowner of109699 and2179 assignedcolumn, evenifanotherrow wins; noGT1160 interpolation, nofulltracker/modelcalls orweightschanges.'],
                                           safe_cutoff='2026-11-13T07:35:00+03:00', maximum_seconds=120))
print(sha(OUT / 'assignment_loss_protocol.json'))
