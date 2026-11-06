"""Prepare exact Git blobs in bounded transfer parts for the connected GitLab API."""
from pathlib import Path
import argparse
import base64
import json
import subprocess

ROOT=Path(__file__).resolve().parents[2]
PART_SIZE=50000


def git(*args,input=None):
    return subprocess.check_output(['git',*args],cwd=ROOT,input=input)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--commit',required=True);parser.add_argument('--parent',required=True)
    parser.add_argument('--package',required=True);parser.add_argument('--part',type=int)
    args=parser.parse_args();out=(ROOT/args.package).resolve();out.relative_to(ROOT)
    if args.part is not None:
        meta=json.loads((out/'remote_export_metadata.json').read_text(encoding='utf-8'))
        assert meta['commit']==args.commit and meta['parent']==args.parent
        if args.part==-1:print(json.dumps(meta));return
        encoded=(out/'remote_actions_local_only.json').read_text(encoding='ascii')
        assert 0<=args.part<meta['parts']
        print(json.dumps(encoded[args.part*PART_SIZE:(args.part+1)*PART_SIZE]));return
    assert git('rev-parse',args.commit+'^').decode().strip()==args.parent
    names=set((out/'source_paths.txt').read_text(encoding='utf-8').splitlines()+(out/'artifact_paths.txt').read_text(encoding='utf-8').splitlines())
    changes={}
    rows=git('diff-tree','--no-commit-id','--name-status','-r','-z',args.commit).decode().strip('\0').split('\0')
    for status,path in zip(rows[::2],rows[1::2]):
        assert status in ('A','M') and path in names
        changes[path]=status
    assert set(changes)==names
    paths=sorted(names)
    packed=git('cat-file','--batch',input=('\n'.join(args.commit+':'+p for p in paths)+'\n').encode())
    offset=0;actions=[];blobs={}
    for path in paths:
        end=packed.index(b'\n',offset)
        blob_id,kind,size=packed[offset:end].decode().split()
        assert kind=='blob';size=int(size);offset=end+1
        raw=packed[offset:offset+size];offset+=size
        assert packed[offset:offset+1]==b'\n';offset+=1
        actions.append(dict(action='create' if changes[path]=='A' else 'update',file_path=path,
                            encoding='base64',content=base64.b64encode(raw).decode('ascii')))
        blobs[path]=blob_id
    assert offset==len(packed)
    encoded=json.dumps(actions,separators=(',',':'))
    (out/'remote_actions_local_only.json').write_text(encoded,encoding='ascii')
    meta=dict(commit=args.commit,parent=args.parent,files=len(actions),characters=len(encoded),
        parts=(len(encoded)+PART_SIZE-1)//PART_SIZE,blob_ids=blobs)
    (out/'remote_export_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in meta.items() if k!='blob_ids'}))


if __name__=='__main__':main()
