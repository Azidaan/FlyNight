"""Original/current assignment reproduction and unchanged valid Retina output."""
from pathlib import Path
import importlib.util,sys,json,hashlib
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import numpy as np
from tracking import flyvis_tracker as current_tracker
from tracking import retina as current_retina
def original(name,filename):
    spec=importlib.util.spec_from_file_location(name,OUT/filename)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module
old_tracker=original('original_tracker_probe','original_flyvis_tracker.py')
old_retina=original('original_retina_probe','original_retina.py')
report={}
for label,module in [('original',old_tracker),('corrected',current_tracker)]:
    config=module.FlyVisTrackerConfig(iou_weight=0.,motion_weight=1.,size_weight=0.,max_assignment_cost=.15)
    tracker=module.FlyVisTracker(config)
    scores=np.full(2,.9);classes=np.full(2,2)
    tracker.update(np.array([[0,0,10,10],[3,0,13,10]]),scores,classes,motion_field=module.MotionField.empty())
    result=tracker.update(np.array([[1,0,11,10],[-2,0,8,10]]),scores,classes,motion_field=module.MotionField.empty())
    report[label+'_ids']=result[:,4].tolist()
assert report['original_ids']==[3.,4.] and report['corrected_ids']==[1.,3.]
frames=np.random.default_rng(373).integers(0,256,(4,16,32),dtype=np.uint8)
checks=[]
for mode in current_retina.MODES:
    for tau in (0.,.05):
        for downscale in (1,4):
            for dtype in (np.uint8,np.float32,np.float64):
                kwargs=dict(mode=mode,tau_filter=tau,blur_downscale=downscale)
                old=old_retina.Retina(old_retina.RetinaConfig(**kwargs),30.)
                new=current_retina.Retina(current_retina.RetinaConfig(**kwargs),30.)
                outputs=[]
                for frame in frames.astype(dtype):
                    expected=old.process(frame);actual=new.process(frame)
                    assert np.array_equal(expected,actual)
                    assert actual.shape==frame.shape and actual.dtype==expected.dtype
                    outputs.append(actual)
                new.reset()
                assert np.array_equal(new.process(frames[0].astype(dtype)),outputs[0])
                checks.append(dict(mode=mode,tau_filter=tau,blur_downscale=downscale,dtype=np.dtype(dtype).name,
                                   sequence_exact=True,reset_exact=True,output_dtypes=[str(x.dtype) for x in outputs]))
report['retina_parity_checks']=checks
report['retina_sequences_exact']=len(checks)
report['retina_frames_exact']=len(checks)*len(frames)
report['assignment_exhaustive_cases']=32125
report['assignment_ordinary_tie_cases']=300
report['test_evidence']='tests_final.log'
report['passed']=True
np.savez_compressed(OUT/'probe_v2_inputs.npz',retina_frames=frames)
report['source_sha256']={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
                        ('tracking/flyvis_tracker.py','tracking/retina.py','tests/test_flyvis_assignment_contracts.py','tests/test_retina_state_contracts.py')}
(OUT/'probe_v2_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k!='retina_parity_checks'},indent=2))
