"""Bind one execution to byte-exact sources, inputs, environment and outputs."""
from pathlib import Path
import hashlib,json,sys,subprocess,platform,zipfile,importlib.metadata
from datetime import datetime,timezone,timedelta
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024**2),b''):h.update(chunk)
    return h.hexdigest()
def stamp():return datetime.now(timezone(timedelta(hours=3))).isoformat()
def git(*args):return subprocess.check_output(['git','--no-optional-locks',*args],cwd=ROOT).decode().strip()
def begin(out,video,weights,settings,*,launcher="pipeline_real_clip.py",arguments=("--passport",),
          extra_inputs=(),result_file="pipeline_real_clip.json"):
    files=set()
    for directory in ('tracking','scripts','models','datasets','evaluation','tools/flyvis-runtime/flyvis','tools/flyvis-runtime/datamate'):
        base=ROOT/directory
        if not base.is_dir():
            raise FileNotFoundError(base)
        files.update(p for p in base.rglob('*') if p.is_file() and p.suffix in ('.py','.yaml','.yml','.json') and '__pycache__' not in p.parts)
    files.update(HERE/p for p in (launcher,'run_fingerprint.py','system_memory_guard.py'))
    files.update(ROOT/p for p in ('requirements.txt','requirements.in'))
    sources={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(files)}
    modeldir=ROOT/'runs/flyvis/results/flow/0000/000'
    input_files=[video,weights,*extra_inputs]+[p for p in modeldir.rglob('*') if p.is_file() and '__cache__' not in p.parts]
    input_files += [p for p in (ROOT/'runs/flyvis/connectome').rglob('*') if p.is_file() and '__cache__' not in p.parts]
    inputs={p.relative_to(ROOT).as_posix():sha(p) for p in input_files}
    environment={'python':sys.version,'executable':sys.executable,'platform':platform.platform(),
                 'packages':sorted(({'name':d.metadata['Name'],'version':d.version} for d in importlib.metadata.distributions()),key=lambda d:d['name'].lower()),
                 'flyvis_runtime_packages':sorted(({'name':d.metadata['Name'],'version':d.version} for d in importlib.metadata.distributions(path=[str(ROOT/'tools/flyvis-runtime')])),key=lambda d:d['name'].lower())}
    archive=out/'frozen_sources.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for name,digest in sources.items():
            raw=(ROOT/name).read_bytes()
            if hashlib.sha256(raw).hexdigest()!=digest:
                raise ValueError(f"Source changed while archiving: {name}")
            z.writestr(name,raw)
    protocol=dict(schema='flynight-run-fingerprint-v1',run_id=out.name,started_msk=stamp(),
                  branch=git('branch','--show-current'),commit=git('rev-parse','HEAD'),tree=git('rev-parse','HEAD^{tree}'),
                  tracked_worktree_status=git('status','--porcelain=v1','--untracked-files=no'),
                  source_sha256=sources,input_sha256=inputs,settings=settings,environment=environment,
                  source_archive_sha256=sha(archive),command=[sys.executable,str(HERE/launcher),*arguments],
                  result_file=result_file,cwd=str(ROOT))
    (out/'run_protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
    return protocol
def finish(out,protocol,*,extra_output_files=(),scope=None):
    differences=[]
    for kind in ('source_sha256','input_sha256'):
        for name,expected in protocol[kind].items():
            if sha(ROOT/name)!=expected:differences.append(name)
    if git('rev-parse','HEAD')!=protocol['commit']:
        raise ValueError("HEAD changed during execution")
    if differences:
        raise ValueError(f"Changed sources or inputs: {differences}")
    with zipfile.ZipFile(out/'frozen_sources.zip') as z:
        for name,expected in protocol['source_sha256'].items():
            if hashlib.sha256(z.read(name)).hexdigest()!=expected:
                raise ValueError(f"Archived source differs: {name}")
    # Supervisor logs are still open until this worker exits; hash only finalized outputs.
    outputs={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name not in ('memory_guard.json','memory_guard.worker.log','supervisor.json','run_checksums.json')}
    for path in sorted(extra_output_files):
        outputs[path.resolve().relative_to(out.resolve()).as_posix()] = sha(path)
    result=dict(schema='flynight-run-fingerprint-v1',run_id=protocol['run_id'],finished_msk=stamp(),
                commit=protocol['commit'],branch=protocol['branch'],passed=True,
                sources_unchanged=True,inputs_unchanged=True,source_count=len(protocol['source_sha256']),
                input_count=len(protocol['input_sha256']),outputs_sha256=outputs,
                excluded_live_files=['memory_guard.json','memory_guard.worker.log','supervisor.json'],
                scope=scope or 'Byte identity and completed three-frame CPU execution; not full-video accuracy or proof of bitwise repeatability.')
    (out/'run_checksums.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    digest=sha(out/'run_checksums.json')
    (out/'run_checksums.sha256').write_text(digest+'  run_checksums.json\n',encoding='ascii')
    return result
