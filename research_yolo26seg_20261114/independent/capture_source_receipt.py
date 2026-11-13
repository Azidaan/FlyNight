"""Record installed source bytes without importing Ultralytics or torch."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib,json
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
package=ROOT/'.venv/Lib/site-packages/ultralytics'
names=['__init__.py','models/yolo/segment/predict.py','models/yolo/detect/predict.py','utils/ops.py','utils/nms.py','utils/downloads.py','engine/model.py','engine/predictor.py','engine/results.py','nn/tasks.py','cfg/models/26/yolo26-seg.yaml','cfg/default.yaml']
out=dict(created=datetime.now(timezone(timedelta(hours=3))).isoformat(), installed_version='8.4.84', sources={name:dict(path=(package/name).relative_to(ROOT).as_posix(),sha256=hashlib.sha256((package/name).read_bytes()).hexdigest()) for name in names}, primary_sources=['https://docs.ultralytics.com/tasks/segment','https://docs.ultralytics.com/reference/models/yolo/segment/predict','https://docs.ultralytics.com/datasets/segment','https://arxiv.org/abs/2012.02310'], gpu_used=False, inference_used=False, downloads=False)
with (HERE/'installed_source_receipt.json').open('x',encoding='utf8') as stream:json.dump(out,stream,ensure_ascii=False,indent=2)
print(json.dumps(dict(recorded=len(names),version=out['installed_version'])))
