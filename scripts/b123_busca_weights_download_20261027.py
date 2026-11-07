"""Получить два файла из проверенной публичной папки авторов, без стороннего кода."""
from pathlib import Path
from datetime import datetime,timezone
from html.parser import HTMLParser
import hashlib,json,ssl,urllib.request,urllib.parse,zipfile
import certifi
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/b123_20261027/busca_weights_download'
class Form(HTMLParser):
    def __init__(self):super().__init__();self.action=None;self.fields={}
    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs)
        if tag=='form':self.action=attrs.get('action')
        if tag=='input' and attrs.get('name'):self.fields[attrs['name']]=attrs.get('value','')
def write(name,value):
    target=RUN/name;temp=target.with_suffix('.next');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');temp.replace(target)
def main():
    p=json.loads((RUN/'protocol.json').read_text(encoding='utf-8'))
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==p['source_sha256']
    context=ssl.create_default_context(cafile=certifi.where());outputs=[]
    for item in p['files']:
        target=ROOT/item['target'];target.resolve().relative_to((ROOT/'weights').resolve())
        target.parent.mkdir(parents=True,exist_ok=True);assert not target.exists()
        url='https://drive.usercontent.google.com/download?'+urllib.parse.urlencode(dict(id=item['id'],export='download',confirm='t'))
        response=urllib.request.urlopen(url,context=context,timeout=25)
        if 'text/html' in response.headers.get('Content-Type',''):
            raw=response.read(1_000_001);response.close();assert len(raw)<=1_000_000
            (RUN/(item['name']+'.confirmation.html')).write_bytes(raw)
            form=Form();form.feed(raw.decode('utf-8'))
            assert form.action and urllib.parse.urlparse(form.action).hostname=='drive.usercontent.google.com'
            assert form.fields.get('id')==item['id']
            url=form.action+'?'+urllib.parse.urlencode(form.fields)
            response=urllib.request.urlopen(url,context=context,timeout=25)
        assert 'text/html' not in response.headers.get('Content-Type','')
        part=target.with_suffix('.partial');assert not part.exists()
        size=0;digest=hashlib.sha256()
        with response,part.open('wb') as f:
            while True:
                chunk=response.read(1<<20)
                if not chunk:break
                size+=len(chunk);assert size<=item['expected_bytes'];f.write(chunk);digest.update(chunk)
                write('status.json',dict(state='running',stage='Получение авторских весов',file=item['name'],done=size,total=item['expected_bytes'],updated=datetime.now(timezone.utc).isoformat()))
        assert size==item['expected_bytes']
        assert zipfile.is_zipfile(part),'Expected torch ZIP container'
        part.rename(target)
        outputs.append(dict(path=item['target'],source_file_id=item['id'],bytes=size,sha256=digest.hexdigest()))
        write('downloaded_files.json',outputs)
    write('report.json',dict(complete=True,outputs=outputs,origin=p['origin'],torch_loaded=False,models_executed=False,datasets_downloaded=False))
    write('status.json',dict(state='completed',stage='Авторские веса получены;совместимость модели ещё не проверена',updated=datetime.now(timezone.utc).isoformat()))
if __name__=='__main__':main()
