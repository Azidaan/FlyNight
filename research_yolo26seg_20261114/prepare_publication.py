from common import *
source=(ROOT/'research_visible_part_2179_20261114/publish.py').read_text(encoding='utf8')
source=source.replace("read(D/'checks_v2.json')['passed']", "read(D/'checks.json')['passed']")
source=source.replace("('visible_components_2179_20261114','выполнено')]", "('visible_components_2179_20261114','выполнено'),('yolo26n_seg_visible_part_20261114','без-положительного-итога'),('flyvis_visible_mask_reader_20261114','без-положительного-итога')]")
start=source.index('# Explicitly authorized new hypothesis and its narrow generator source fix.')
end=source.index('# Explicit task-owned tree;',start)
source=source[:start]+source[end:]
source=source.replace("['audit.json','summary.json','feature_outputs.npz','level_features.npz','mask_review/masks_reviewed.npz']", "['audit.json','summary.json','stage_2179/vectors.npz','stage_heldout/vectors.npz','flyvis_probe/predictions.npz']")
source=source.replace("skip={'source_renderer_repeat'", "skip={'tmp','anchors','continuity','candidate_panels','candidate_frames','sequential_frames','source_renderer_repeat'")
source=source.replace("scope='Previous five-case positive checks remain green. New one-event pixel/level causal diagnostic is green within its stated manual-mask scope; it is not independent tracking improvement. Other historical results retain their states.'", "scope='Earlier local manual-mask causal success remains green. New automatic-mask transfer and frozen-FlyVis linear mask-readout tests are completed negative under their registered gates. Existing protected hypothesis tree preserved bytewise; no general closure of segmentation/FlyVis.'")
with (OUT/'publish.py').open('x',encoding='utf8',newline='\n') as f:f.write(source)
for name in ['verify_staging.py']:
 with (OUT/name).open('x',encoding='utf8',newline='\n') as f:f.write((ROOT/'research_visible_part_2179_20261114'/name).read_text(encoding='utf8'))
print('PUBLISHER_PREPARED')
