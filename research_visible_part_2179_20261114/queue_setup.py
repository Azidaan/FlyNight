from common import *
name=sys.argv[1]
jobs={'science':[('trace.py',180),('extract.py',600),('worker.py',180),('audit.py',180)],'checks':[('checks.py',600)]}[name]
inputs=dict(read(OUT/'protocol.json')['inputs'])
inputs[(OUT/'run_queue.py').relative_to(ROOT).as_posix()]=sha(OUT/'run_queue.py')
save(name+'_queue.json',dict(runner_sha256=sha(OUT/'run_queue.py'),deadline=read(OUT/'protocol.json')['deadline'],inputs=inputs,jobs=[dict(entry=(OUT/file).relative_to(ROOT).as_posix(),timeout=timeout) for file,timeout in jobs]))
