import os,sys,json,hashlib
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT=Path('E:/Dissertation');OUT=Path(__file__).resolve().parent
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']=os.environ['OPENBLAS_NUM_THREADS']='2'
for k,n in [('TEMP','tmp'),('TMP','tmp'),('TMPDIR','tmp'),('MPLCONFIGDIR','matplotlib'),('TORCH_HOME','torch'),('CUDA_CACHE_PATH','cuda')]:
 p=ROOT/'.cache'/n;p.mkdir(parents=True,exist_ok=True);os.environ[k]=str(p)
sys.path.insert(0,str(ROOT))
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def now():return datetime.now(ZoneInfo('Europe/Moscow')).isoformat()
def rel(p):return Path(p).resolve().relative_to(ROOT).as_posix()
def save(n,v):
 p=OUT/n;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf8',newline='\n') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def journal(text):
 p=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';raw=p.read_bytes();s=raw.decode('utf8');nl='\r\n' if '\r\n' in s else '\n';mark='## Журнал изменений'+nl
 extra='\n**'+datetime.now(ZoneInfo('Europe/Moscow')).strftime('%d.%m.%Y, %H:%M MSK')+'** — '+text+'\n\n'
 assert s.count(mark)==1 and p.read_bytes()==raw;p.write_bytes(s.replace(mark,mark+extra.replace('\n',nl),1).encode('utf8'))
 with (OUT/'journal_additions.txt').open('ab') as f:f.write(extra.encode('utf8'))
