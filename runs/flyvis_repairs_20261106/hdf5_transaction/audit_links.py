"""Read-only link inventory and two-pass SHA256 of actual FlyVis HDF5 roots."""
from pathlib import Path
import hashlib
import json
import time

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).parent
ROOTS=('runs/flyvis','tools/flyvis-runtime/data','tools/flyvis-runtime/flyvis/data')


def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        while block:=stream.read(1024*1024):value.update(block)
    return value.hexdigest()


def main():
    started=time.monotonic();rows=[];roots=[]
    for name in ROOTS:
        root=ROOT/name
        paths=sorted(root.rglob('*.h5')) if root.is_dir() else []
        roots.append(dict(path=name,exists=root.exists(),files=len(paths)))
        for path in paths:
            before=path.stat()
            rows.append(dict(path=path.relative_to(ROOT).as_posix(),resolved=str(path.resolve()),
                bytes=before.st_size,hard_links=before.st_nlink,symlink=path.is_symlink(),
                inode=before.st_ino,mtime_ns=before.st_mtime_ns,sha256=digest(path)))
    for row in rows:
        path=ROOT/row['path'];after=path.stat()
        row['unchanged_after_second_read']=(digest(path)==row['sha256'] and
            (after.st_size,after.st_nlink,after.st_ino,after.st_mtime_ns)==
            (row['bytes'],row['hard_links'],row['inode'],row['mtime_ns']))
    report=dict(roots=roots,files=len(rows),bytes=sum(r['bytes']for r in rows),
        hardlinked=[r['path']for r in rows if r['hard_links']>1],
        symlinked=[r['path']for r in rows if r['symlink']],
        all_original_bytes_and_identity_unchanged=all(r['unchanged_after_second_read']for r in rows),
        seconds=time.monotonic()-started,rows=rows,
        root_provenance='flyvis_adapter.DEFAULT_DATA_DIR and FLYVIS_ROOT_DIR => runs/flyvis; vendor resolve_root_dir fallback tools/flyvis-runtime/data, plus existing packaged flyvis/data. A source search found no other project datamate.set_root_dir or FLYVIS_ROOT_DIR setter.',
        limits='Read-only inventory of these actual roots, not arbitrary future paths or external directories. No HDF5 reader/writer, Torch, FlyVis initialization or GPU was invoked.')
    (OUT/'actual_hdf5_links.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
