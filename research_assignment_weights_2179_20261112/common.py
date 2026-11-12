import os,sys,json,hashlib
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
for key,folder in [('TEMP',OUT/'temp'),('TMP',OUT/'temp'),('CUDA_CACHE_PATH',ROOT/'.cache/cuda'),('TORCH_HOME',ROOT/'.cache/torch')]:
    folder.mkdir(exist_ok=True,parents=True);os.environ[key]=str(folder)
sys.path.insert(0,str(ROOT))
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def now():return datetime.now(ZoneInfo('Europe/Moscow')).isoformat()
def save(n,v):
    p=OUT/n;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('xb') as f:f.write((json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode())
def journal(message):
    p=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';raw=p.read_bytes();s=raw.decode();nl='\r\n' if '\r\n' in s else '\n';mark='## Журнал изменений'+nl;assert s.count(mark)==1
    addition='\n**'+datetime.now(ZoneInfo('Europe/Moscow')).strftime('%d.%m.%Y, %H:%M MSK')+'** — '+message+'\n\n'
    assert p.read_bytes()==raw;p.write_bytes(s.replace(mark,mark+addition.replace('\n',nl),1).encode())
    with (OUT/'journal_additions.txt').open('ab') as f:f.write(addition.encode())
