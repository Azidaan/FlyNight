from common import *
receipt=read(ROOT/'runs/domain_matched_native_20261109/execution_protocol.json'); sources={}
for name in ['runs/domain_matched_native_20261109/runtime_adapter.py','runs/full_template_native_20261109/native_runtime.py']:
    h=sha(ROOT/name); assert h==receipt['sources'][name]
    lines=(ROOT/name).read_text(encoding='utf8').splitlines()
    sources[name]=dict(sha256=h,matches_historical_execution_receipt=True,fragments=[dict(line=i+1,text=line) for i,line in enumerate(lines) if ('def after(' in line or 'self.histories.setdefault' in line or 'box=np.asarray(t.xyxy' in line or 'raw_origin_method=method' in line or 'self.histories[tid][-1].update' in line)])
save('source_localization.json',dict(passed=True,sources=sources,fact='Planner.after stores the filtered t.xyxy box; runtime adapter adds raw_id provenance afterward without replacing that box. Thus filtered and raw boxes are genuinely different input routes, not swapped file versions.',limit='Code path and stored input provenance, not proof that this choice alone caused an ID switch.'))
