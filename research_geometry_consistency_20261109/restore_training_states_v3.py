"""Recover after the pre-hook's raw-detection versus BOTrack API mismatch."""
from pathlib import Path
source=Path(__file__).with_name('restore_training_states.py')
code=source.read_text(encoding='utf8')
assert code.count('np.asarray(t.xyxy,float).tolist()')==1
assert code.count('dists=self.get_dists(pool,high)')==1
code=code.replace('np.asarray(t.xyxy,float).tolist()','np.asarray(t.xyxy,np.float32).tolist()')
code=code.replace('dists=self.get_dists(pool,high)','dists=self.get_dists(pool,self.init_track(high,img))')
code=code.replace("'training_state_protocol.json'","'training_state_protocol_v3.json'")
code=code.replace("files=[Path(__file__),","files=[Path(__file__),OUT/'restore_training_states.py',OUT/'restore_training_states_v2.py',OUT/'training_state_protocol.json',OUT/'training_state_protocol_v2.json',")
code=code.replace("operation='CPU association-state reconstruction", "recovery='v2failed before the first saved state: pre-first hook receives raw DetSlice, whereas get_dists expects BOTrack detections. Reproduce the native init_track conversion before the pure distance calculation. Original worker6aa9a3f5ba0c43138f5ca1e3c055fde2 and all sources/protocols retained. Public reference comparison remains float32, matching the saved source.',\n        operation='CPU association-state reconstruction")
exec(compile(code,str(source),'exec'))
