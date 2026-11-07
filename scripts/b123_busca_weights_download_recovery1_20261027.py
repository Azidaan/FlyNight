"""Продолжить частичную загрузку; старый журнал и абсолютный срок сохраняются."""
from pathlib import Path
import hashlib,json,sys,time,urllib.request
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
RUN=ROOT/'runs/b123_20261027/busca_weights_download/recovery1'
def main():
    from scripts import b123_busca_weights_download_20261027 as original
    p=json.loads((RUN/'protocol.json').read_text(encoding='utf-8'))
    source=(ROOT/'scripts/b123_busca_weights_download_20261027.py').read_text(encoding='utf-8')
    assert hashlib.sha256(source.encode()).hexdigest()==p['original_source_sha256']
    source=source.replace("assert not part.exists()", "offset=part.stat().st_size if part.exists() else 0")
    source=source.replace("size=0;digest=hashlib.sha256()", "size=offset;digest=hashlib.sha256();digest.update(part.read_bytes() if offset else b'')")
    source=source.replace("part.open('wb')", "part.open('ab' if offset else 'wb')")
    # The source hash assertion binds the original file through __file__.
    namespace={'__file__':str(ROOT/'scripts/b123_busca_weights_download_20261027.py'),'__name__':'service_import'}
    exec(compile(source,str(Path(__file__)), 'exec'),namespace)
    namespace['RUN']=RUN
    opener=urllib.request.urlopen
    def ranged(url,*args,**kwargs):
        raw=url.full_url if isinstance(url,urllib.request.Request) else url
        item=next((x for x in p['files'] if x['id'] in raw),None)
        part=(ROOT/item['target']).with_suffix('.partial') if item else None
        if part and part.exists():
            offset=part.stat().st_size
            request=urllib.request.Request(raw,headers={'Range':f'bytes={offset}-'})
            response=opener(request,*args,**kwargs)
            if 'text/html' not in response.headers.get('Content-Type',''):
                assert response.status==206 and response.headers.get('Content-Range','').startswith(f'bytes {offset}-'), 'Server did not support exact partial resume'
            return response
        return opener(url,*args,**kwargs)
    namespace['urllib'].request.urlopen=ranged
    def write(name,value):
        target=RUN/name;temp=target.with_suffix('.next')
        temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        for attempt in range(40):
            try:temp.replace(target);return
            except PermissionError:
                if attempt==39:raise
                time.sleep(.05)
    namespace['write']=write
    namespace['main']()
if __name__=='__main__':main()
