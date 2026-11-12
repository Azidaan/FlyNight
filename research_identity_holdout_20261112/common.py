"""Utilities for the bounded, label-free repair test. Preserve old experiments."""
import hashlib,json,sys
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[1]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def now():return datetime.now(ZoneInfo('Europe/Moscow')).isoformat()
def save(n,obj):
 p=OUT/n
 with p.open('xb') as f:f.write((json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode())
def journal(message):
 p=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';raw=p.read_bytes();s=raw.decode('utf8');nl='\r\n' if '\r\n' in s else '\n';marker='## Журнал изменений'+nl;assert s.count(marker)==1
 entry='\n**'+datetime.now(ZoneInfo('Europe/Moscow')).strftime('%d.%m.%Y, %H:%M MSK')+'** — '+message+'\n\n'
 assert p.read_bytes()==raw;p.write_bytes(s.replace(marker,marker+entry.replace('\n',nl),1).encode())
 with (OUT/'journal_additions.txt').open('ab') as f:f.write(entry.encode())
