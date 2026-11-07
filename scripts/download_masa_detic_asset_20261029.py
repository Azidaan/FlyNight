"""Fetch exactly the author-requested MASA-Detic artifact and verify upstream SHA."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os, shutil, time, urllib.request

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/continuation_20261028/reid_candidates_preparation'
END=datetime.fromisoformat('2026-10-29T07:30:00+03:00').timestamp()

def main():
    items=json.loads((OUT/'masa/weights_index.json').read_text(encoding='utf-8'))
    entry=next(x for x in items if x['path']=='detic_masa.pth')
    expected=entry['lfs']['oid'];size=entry['size']
    target=ROOT/'weights/masa_detic_author_20261029/detic_masa.pth'
    target.parent.mkdir(exist_ok=True)
    partial=target.with_suffix('.pth.partial')
    assert shutil.disk_usage(ROOT).free>size+5*1024**3
    def sha(path):
        h=hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
        return h.hexdigest()
    status=OUT/'download_status.json'
    def save(state,**extra):
        status.write_text(json.dumps(dict(state=state,updated=datetime.now(timezone.utc).isoformat(),**extra),indent=2)+'\n',encoding='utf-8')
    if target.exists():
        assert target.stat().st_size==size and sha(target)==expected
        save('complete',reused=True,path=str(target),sha256=expected,bytes=size)
        return
    assert not partial.exists(), 'Preserve incomplete download for explicit range recovery'
    assert time.time()<END
    save('downloading',bytes=0,total=size,deadline=END)
    request=urllib.request.Request('https://huggingface.co/dereksiyuanli/masa/resolve/main/detic_masa.pth',headers={'User-Agent':'Dissertation-author-request/1.0'})
    downloaded=0
    with urllib.request.urlopen(request,timeout=20) as response, partial.open('xb') as output:
        while True:
            if time.time()>=END:raise TimeoutError('Global save deadline; partial preserved')
            block=response.read(4*1024**2)
            if not block:break
            output.write(block);downloaded+=len(block)
            save('downloading',bytes=downloaded,total=size,deadline=END)
    assert downloaded==size
    assert sha(partial)==expected, 'Upstream LFS SHA mismatch; never load this artifact'
    os.rename(partial,target)
    save('complete',reused=False,path=str(target),sha256=expected,bytes=size)
    print('MASA-Detic weights downloaded and upstream SHA256 verified; not executed')

if __name__=='__main__':main()
