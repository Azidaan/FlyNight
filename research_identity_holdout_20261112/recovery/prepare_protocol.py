"""Freeze code and ALL externally selected A events before policy outcomes."""
from common import *
import argparse, ast
parser = argparse.ArgumentParser()
parser.add_argument('--selection', required=True)
parser.add_argument('--bank-plan', required=True)
parser.add_argument('--exclusions', required=True)
parser.add_argument('--bank-file', required=True)
args = parser.parse_args()
sources = ['common.py', 'prepare_runtime.py', 'annotations.py', 'worker.py', 'independent.py', 'audit.py', 'prepare_protocol.py']
for name in sources:
    path = OUT / name
    source = path.read_text(encoding='utf8').replace('\r\n', '\n').replace('\r', '\n')
    source = '\n'.join(line.rstrip() for line in source.splitlines()).rstrip() + '\n'
    ast.parse(source)
    path.write_bytes(source.encode('utf8'))
runtime = read(OUT / 'runtime_cases.json')
preparation = read(OUT / 'runtime_preparation.json')
evaluation = read(OUT / 'evaluation_plan.json')
external = [Path(path).resolve() for path in [args.selection, args.bank_plan, args.exclusions]]
for path in external:
    path.relative_to(ROOT)
paths = {str(path.relative_to(ROOT)).replace('\\', '/'): sha(path) for path in external}
paths.update(preparation['inputs'])
paths.update(evaluation['inputs'])
paths.update({str((OUT / name).relative_to(ROOT)).replace('\\', '/'): sha(OUT / name) for name in ['runtime_cases.json', 'runtime_preparation.json', 'evaluation_plan.json', 'historical_observations.json']})
paths['tracking/appearance_recovery_policy.py'] = sha(ROOT / 'tracking/appearance_recovery_policy.py')
bank_file = Path(args.bank_file).resolve()
bank_file.relative_to(ROOT)
excluded = set(read(external[2])['excluded_videos'])
assert all(case['video'] not in excluded for case in runtime['cases'])
protocol = dict(created=now(), experiment_id='identity_holdout_frozen_recovery_20261112', inputs=paths, sources={name: sha(OUT / name) for name in sources},
                selection=dict(manifest=str(external[0].relative_to(ROOT)).replace('\\', '/'), events=len(runtime['cases']), videos=sorted({case['video'] for case in runtime['cases']}),
                               scope='ALL independently selected A events and all their exposed LOST episode rows; no label/own-candidate/score filtering. Old repair-choice cases,26fitpairs anddevelopment videos excluded by frozen reviewer ledger. Reserved B remains untouched.'),
                fixed_policy=dict(path='tracking/appearance_recovery_policy.py', sha256=paths['tracking/appearance_recovery_policy.py'], thresholds='Existing borrowed0.8appearance/0.5proximity/0.7NMS/0.1excluded-low, strictmutualNN andobserved-native episodeeligibility; no changes allowed.'),
                pending_descriptor_bank=str(bank_file.relative_to(ROOT)).replace('\\', '/'),
                binding='Parent registers descriptor_binding.json AFTER extraction and BEFORE repair outcomes: descriptor_file exactly pending_descriptor_bank, inputs include bankSHA and extractionprotocol/result/audit/execution hashes. NPZkeys video|rawframe, boxslots from frozen sharedplan.',
                runtime_boundary='runtime_cases.json contains only observable metadata and descriptorslots; production gets only tid/state/class/predictedbox,raw_id/box/conf/class, complete vectors,frozenbaselinepairs, observableeligible_tids. GT never enters policy or case selection.',
                data_completeness='All savedcompetitorrows andrawcar candidates retained. Missing legitimate observedreference makes competition_completeFalse anddeferALLrows; missing expected bankkey/invalidvector is a technicalerror, never silentrowdrop.',
                evaluation='Exact persisted rawframe clockanchors withmatching annotationname/frameIndex; scalar IoU>=0.5. Missing clock/source, unmatched orambiguous identities remainunknown. Evaluate ALLnew proposals and allsavedbaselinepairs; preserve unavailable labels andvalidity denominators.',
                secondary_historical_evaluation='Separate expected-ID continuity task: deduplicate stored actualhistorical observations by(frame,box), inspect ALLrecent exactlyanchored observations. Need>=1 and everyanchored observation mustmatch uniquely SAMEGTidentity, otherwiseunknown/historyproblem. Preserve actual_source_anchor_frame plusALLanchored observations and descriptor_last_source_frame; NEVERlabel unannotated lastdescriptor pixels bycarryingoldGT. Missing exactcurrentGT stillunknown. No nearest/interpolation. This evaluation neverchangesruntime/policy.',
                modes={'baseline': 'Frozen original assignedpublicpairs; optionalbranch disabled.', 'fixed_repair': 'Exactlythe frozen productionpolicy, entirepool.', 'descriptor_permutation': 'Cyclicshift currentvectors byone position within frame, metadataunchanged; ALL outcomes reported.', 'missing_competitors': 'competition_completeFalse mustdeferall.', 'no_descriptors': 'VectorsNone mustdeferall.', 'remove_all_own': 'Separate EVAL-only counterfactual forEACHeligibleLOSTrow with exactunambiguous referenceidentity and exactcurrentclock. Remove everyrawcandidate whosehitscontainthatidentity, includingambiguoushits. Recomputegroups withALLrowsstillcompeting. Zeroownmatches stillvalid; unknownreference/currentcontrol explicitlyskipped, not a pass.'},
                secondary_control='remove_all_historical_own uses ONLY separatelyvalid historicalexpectedID +exactcurrentclock and otherwiseidentical policy. Report separately from literal-lastsource control; do not inflate evidence wherecandidatepoolsduplicate. Anytestedrowcontinuation afterallmatchingrawboxesremoved is a failure of that registered counterfactual.',
                decomposition='For EVERYeligibleLOSTrow save final action/reason, allraw own-match availability, originalconfidence, NMSrepresentative/aliases, rawcosine/proximity, representativeappearance/proximity/class/low/occupancy gates and reverse-nearestcompetitors. These are fixed accounting traces, not new interventions or threshold search.',
                acceptance=['Frozen policy andallinput/source/bank hashes unchanged before/after; no previousvideo leakage according to explicit exclusionledger.',
                            'ALLselectedevents retained; full savedrows andrawcars preserved; originalpublicpairs/occupiedaliases neverseized andnewproposalsunique.',
                            'Independent scalar fsum cosine/IoU/NMS/allpolicybranches, exactclock labels andallcounterfactualpool generation agree within1e-9. Technicalaudit passing is not the scientific gate.',
                            'To claim boundedpositive recovery evidence: at leastone correctown NEWproposal on A andzero knownforeign orunknown NEWproposals. Otherwise report failure/uncertainty, do not promote or tune.',
                            'Apply the same newproposal own/foreign/unknown accounting to secondary historicalexpectedID continuity SEPARATELY. Anypositive result describes that task, neverprimary lastsource identity orphysicalpixels. Report annotationcoverage, conflicts andmissinglabels.',
                            'On eachvalid remove-all-own control the testedrow mustnotcontinue; anycontinue is a recorded counterfactualfailure. Invalidcontrols remainexplicitunknown.',
                            'Report every descriptor-permutation proposal andlabel; missing/no-descriptor controls mustdeferall. Do not remove unfavorable events.',
                            'Compare baseline andfixedpolicy byevent/video andsourceobservability. Avoid treating repeated same-video events as independentpopulation evidence.',
                            'No fulltracker/publicIDlifecycle/HOTAclaim, no new neuralcorecall orpolicytuning. Diagnose failures only throughregistered exactgates andsourcefacts; reservedB untouched.'],
                cpu=dict(CUDA_VISIBLE_DEVICES='', threads=2, new_fits=0, policy_neural_calls=0, core_calls=0, tracker_calls=0),
                limits=['Previously unseen for repair choice is not necessarily unseen by all historical base models; reviewer ledger defines precise exclusion.', 'GT IoU is not physical pixelownership.', 'Originalbaselinepublic assignments copied, not a complete lifecycle rerun.', 'Sparse exactclock labels may leave proposalsunknown; never convertunknown tosuccess.'],
                safe_cutoff='2026-11-13T07:35:00+03:00', author_stop='2026-11-13T07:40:00+03:00', maximum_seconds=1800, outer_guard='Parent owns boundedCPU launch/controller; thisworker never launchesGPU or other agents.')
write('proposed_protocol.json', protocol)
print('HOLDOUT_RECOVERY_PROTOCOL_READY', sha(OUT / 'proposed_protocol.json'), len(runtime['cases']), flush=True)
