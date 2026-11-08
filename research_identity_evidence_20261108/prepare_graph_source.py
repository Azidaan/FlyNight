from pathlib import Path
root=Path(__file__).resolve().parents[1];old=root/'research_singleton_admission_20261108/rebuild_graph.py';text=old.read_text(encoding='utf-8')
text=text.replace("RUN=ROOT/'runs/singleton_admission_20261108/graph_update'","RUN=ROOT/'runs/identity_evidence_20261108/graph_update'")
text=text.replace("assert read(ROOT/'runs/native_confirm_policy_20261108/comparison_stage_complete.json')['passed']","assert read(ROOT/'runs/bbox_normalized_signals_20261108/full/queue_result.json')['passed'];assert read(ROOT/'runs/bbox_readout_calibration_20261108/queue_result.json')['passed']")
text=text.replace('research_singleton_admission_20261108/rebuild_graph.py','research_identity_evidence_20261108/rebuild_graph.py')
with (Path(__file__).parent/'rebuild_graph.py').open('x',encoding='utf-8') as f:f.write(text)
