"""Save pinned official MGOR source without importing it or installing packages."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib,json,urllib.request

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/continuation_20261028/reid_candidates_preparation'

def main():
    tree=json.loads((OUT/'mgor/tree.json').read_text(encoding='utf-8'))
    commit=tree['sha'];target=ROOT/'tools/reid_candidates/mgor'/commit
    target.mkdir(parents=True,exist_ok=True)
    files=[x for x in tree['tree'] if x['type']=='blob' and (x['path'].endswith(('.py','.yml','.yaml','.md','.sh')) or Path(x['path']).name in ('LICENSE','requirements.txt'))]
    def one(entry):
        relative=entry['path'];dest=(target/relative).resolve();dest.relative_to(target.resolve())
        assert entry.get('size',0)<2*1024**2
        url=f'https://raw.githubusercontent.com/whucsy/OG-ReID/{commit}/{relative}'
        content=urllib.request.urlopen(url,timeout=20).read(2*1024**2)
        expected=entry['sha']
        actual=hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
        assert actual==expected,relative
        if dest.exists():assert dest.read_bytes()==content
        else:dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(content)
        return dict(path=dest.relative_to(ROOT).as_posix(),git_blob_sha1=actual,sha256=hashlib.sha256(content).hexdigest())
    with ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(one,files))
    probes={}
    for name,url in [('github_releases','https://api.github.com/repos/whucsy/OG-ReID/releases'),
                     ('hf_mgor','https://huggingface.co/api/models?search=MGOR&limit=30'),
                     ('hf_ogreid','https://huggingface.co/api/models?search=OG-ReID&limit=30')]:
        try:
            raw=urllib.request.urlopen(url,timeout=20).read(2*1024**2)
            (OUT/'mgor'/f'{name}.json').write_bytes(raw)
            probes[name]=dict(url=url,entries=len(json.loads(raw)),sha256=hashlib.sha256(raw).hexdigest())
        except Exception as error:probes[name]=dict(url=url,error=repr(error))
    report=dict(source_revision=commit,files=results,probes=probes,code_executed=False,
        weights_found=False,weights_absence_not_proven_globally=True,scientific_verdict=None,
        next_required='Verified trained MGOR checkpoint matching this source; no random-weight substitute')
    (OUT/'mgor/source_prepared.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(files=len(results),probes=probes),ensure_ascii=False))

if __name__=='__main__':main()
