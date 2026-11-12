import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import json,hashlib,sys
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent
ID='saved_shift_cross_probe_20261112';ARMS=['C34','R134_spatial','R134_temporal']
sys.path.insert(0,str(ROOT))
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def cast(v):
    if isinstance(v,dict):return {k:cast(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [cast(x) for x in v]
    return v.tolist() if hasattr(v,'tolist') else v
def save(n,d):
    with (OUT/n).open('x',encoding='utf8') as f:json.dump(cast(d),f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def now():return datetime.now(ZoneInfo('Europe/Moscow')).isoformat()
def verify():
    p=read(OUT/'protocol.json')
    for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
    return p
def journal(message):
    p=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';raw=p.read_bytes();s=raw.decode('utf8');nl='\r\n' if '\r\n' in s else '\n';marker='## Журнал изменений'+nl;assert s.count(marker)==1
    entry='\n**'+datetime.now(ZoneInfo('Europe/Moscow')).strftime('%d.%m.%Y, %H:%M MSK')+'** — '+message+'\n\n'
    assert p.read_bytes()==raw;p.write_bytes(s.replace(marker,marker+entry.replace('\n',nl),1).encode('utf8'))
    with (OUT/'journal_additions.txt').open('a',encoding='utf8') as f:f.write(entry)
