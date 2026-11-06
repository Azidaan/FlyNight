"""Verify a whole staged package using batched Git object queries."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess

ROOT=Path(__file__).resolve().parents[2]


def git(*args,input=None):
    return subprocess.check_output(['git',*args],cwd=ROOT,input=input)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('package');parser.add_argument('--manifest-sha',required=True)
    args=parser.parse_args();out=(ROOT/args.package).resolve();out.relative_to(ROOT)
    raw=(out/'manifest.json').read_bytes();assert hashlib.sha256(raw).hexdigest()==args.manifest_sha
    manifest=json.loads(raw)
    for row in manifest['sources']+manifest['artifacts']:
        assert hashlib.sha256((ROOT/row['path']).read_bytes()).hexdigest()==row['sha256'],row['path']
    names=sorted(set((out/'source_paths.txt').read_text().splitlines()+(out/'artifact_paths.txt').read_text().splitlines()))
    assert all('\n' not in p and '"' not in p for p in names)
    changed=set(git('diff','--cached','--name-only','-z').decode().strip('\0').split('\0'))
    assert changed<=set(names),sorted(changed-set(names))
    hashes=git('hash-object','--stdin-paths',input=('\n'.join(names)+'\n').encode()).decode().splitlines()
    index=git('cat-file','--batch-check',input=('\n'.join(':'+p for p in names)+'\n').encode()).decode().splitlines()
    assert len(hashes)==len(index)==len(names)
    mismatch=[p for p,a,b in zip(names,hashes,index) if a!=b.split()[0]]
    assert not mismatch,mismatch
    report=dict(passed=True,head=git('rev-parse','HEAD').decode().strip(),
        allowlist=names,staged_changed=sorted(changed),staged_blob_ids=dict(zip(names,hashes)))
    (out/'root_staging_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(passed=True,allowlist=len(names),changed=len(changed),head=report['head'])))


if __name__=='__main__':main()
