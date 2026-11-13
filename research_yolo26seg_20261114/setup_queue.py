from common import *
stage=sys.argv[1]
inputs=dict(read(OUT/'protocol.json')['inputs']);inputs[(OUT/'worker.py').relative_to(ROOT).as_posix()]=sha(OUT/'worker.py')
manifest=OUT/('case2179.json' if stage=='2179' else 'holdout/manifest.json');inputs[manifest.relative_to(ROOT).as_posix()]=sha(manifest)
if stage!='2179':
 for event in read(manifest)['events']:
  inputs[event['masks']]=sha(ROOT/event['masks'])
  for r in event['roles']:inputs[r['image']]=sha(ROOT/r['image'])
save(stage+'_queue.json',dict(runner_sha256=sha(OUT/'run_queue.py'),deadline=read(OUT/'protocol.json')['deadline'],inputs=inputs,jobs=[dict(entry=(OUT/'worker.py').relative_to(ROOT).as_posix(),args=[stage],timeout=1800)]))
print('QUEUE_PREPARED',stage)
