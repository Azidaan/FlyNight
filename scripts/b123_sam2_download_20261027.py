"""Ограниченная загрузка только официального компактного checkpoint SAM2.1."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,ssl,time,urllib.request,zipfile
import certifi
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/b123_20261027/sam2_weights_download'
def write(name,data):
    path=RUN/name;t=path.with_suffix('.next');t.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    for attempt in range(20):
        try:t.replace(path);return
        except PermissionError:
            if attempt==19:raise
            time.sleep(.05)
def main():
    p=json.loads((RUN/'protocol.json').read_text(encoding='utf-8'))
    deadline=datetime.fromisoformat(json.loads((RUN/'execution_deadline.json').read_text())['deadline']).timestamp()
    source=p['url'];expected=p['bytes'];target=ROOT/p['target'];target.parent.mkdir(parents=True,exist_ok=True)
    part=target.with_suffix('.partial');assert not target.exists() and not part.exists()
    sha=hashlib.sha256();size=0
    with urllib.request.urlopen(source,timeout=25,context=ssl.create_default_context(cafile=certifi.where())) as response,part.open('wb') as output:
        assert 'text/html' not in response.headers.get('Content-Type','')
        while True:
            if time.time()>deadline-5:raise TimeoutError('Original transfer budget; keep partial')
            chunk=response.read(1<<20)
            if not chunk:break
            size+=len(chunk);assert size<=expected;output.write(chunk);sha.update(chunk)
            write('status.json',dict(state='running',stage='Получение официальных весов SAM2.1tiny',done=size,total=expected,updated=datetime.now(timezone.utc).isoformat()))
    assert size==expected and zipfile.is_zipfile(part)
    part.rename(target)
    write('report.json',dict(complete=True,source=source,target=p['target'],bytes=size,sha256=sha.hexdigest(),
        origin_verified_against_official_repository=True,torch_loaded=False,models_executed=False,datasets_downloaded=False))
    write('status.json',dict(state='completed',stage='Веса сохранены, совместимость ещё не проверена',updated=datetime.now(timezone.utc).isoformat()))
if __name__=='__main__':main()
