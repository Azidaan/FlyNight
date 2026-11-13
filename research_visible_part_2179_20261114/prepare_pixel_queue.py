from common import *
inputs=dict(read(OUT/'protocol.json')['inputs'])
for n in ['pixel_probe.py','common.py','run_queue.py','mask_review/masks_reviewed.npz','protocol_mask_refinement.json']:
    inputs[(OUT/n).relative_to(ROOT).as_posix()]=sha(OUT/n)
save('pixel_execution_protocol.json',dict(registered=now(),inputs=inputs,mask_outcomes_not_used_for_selection=True))
save('pixel_queue.json',dict(runner_sha256=sha(OUT/'run_queue.py'),deadline=read(OUT/'protocol.json')['deadline'],inputs=inputs,jobs=[dict(entry=(OUT/'pixel_probe.py').relative_to(ROOT).as_posix(),timeout=480)]))
