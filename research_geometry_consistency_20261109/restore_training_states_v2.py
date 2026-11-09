"""Before-run repair: compare observed reference boxes at their archived float32 dtype."""
from pathlib import Path
source=Path(__file__).with_name('restore_training_states.py')
code=source.read_text(encoding='utf8')
assert code.count('np.asarray(t.xyxy,float).tolist()')==1
code=code.replace('np.asarray(t.xyxy,float).tolist()','np.asarray(t.xyxy,np.float32).tolist()')
code=code.replace("'training_state_protocol.json'","'training_state_protocol_v2.json'")
code=code.replace("files=[Path(__file__),","files=[Path(__file__),OUT/'restore_training_states.py',OUT/'training_state_protocol.json',")
code=code.replace("operation='CPU association-state reconstruction", "before_run_repair='Original code compared float64 private t.xyxy against the stored float32 public reference. No recovery run or measured state used that version. Compare exactly after the original public float32 cast; coordinates used by geometry remain the archived reference. Source and original protocol retained.',\n        operation='CPU association-state reconstruction")
exec(compile(code,str(source),'exec'))
