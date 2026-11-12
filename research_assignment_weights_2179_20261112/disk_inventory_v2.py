"""Read-only storage inventory; do not follow reparse points or read contents."""
import os,json,time,heapq,ctypes,shutil
from pathlib import Path
from collections import defaultdict
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent
OUT.mkdir(exist_ok=True)
os.environ['CUDA_VISIBLE_DEVICES']=''
def now():return datetime.now(timezone.utc).isoformat()
def dump(name,value):
    p=OUT/name;t=p.with_suffix('.pending');t.write_bytes((json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode());t.replace(p)
profile=Path(os.environ['USERPROFILE']);local=Path(os.environ['LOCALAPPDATA']);roaming=Path(os.environ['APPDATA'])
requested=[ROOT,profile/'.codex',profile/'.cache',profile/'.torch',local/'Temp',local/'pip'/'Cache',local/'NVIDIA'/'ComputeCache',local/'NVIDIA'/'DXCache']
for base in [local,roaming]:
    with os.scandir(base) as entries:
        requested.extend(Path(e.path) for e in entries if e.is_dir(follow_symlinks=False) and any(s in e.name.lower() for s in ['codex','openai','chatgpt']))
paths=[]; unavailable_roots=[]
dump('disk_status.json',dict(stage='checking_roots',started=now()))
for path in requested:
    try:
        if path.exists() and path not in paths:paths.append(path)
    except OSError as exc:unavailable_roots.append(dict(path=str(path),error=str(exc)))
started=time.monotonic();reports=[];last=0
dump('disk_status.json',dict(stage='starting',started=now(),targets=[str(p) for p in paths]))
for root in paths:
    stack=[root];total=files=directories=0;errors=[];skipped=[];groups=defaultdict(lambda:[0,0]);top=defaultdict(lambda:[0,0]);roles=defaultdict(lambda:[0,0]);largest=[]
    while stack:
        directory=stack.pop();directories+=1
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    try:
                        stat=entry.stat(follow_symlinks=False)
                        if getattr(stat,'st_file_attributes',0)&0x400:
                            if len(skipped)<40:skipped.append(entry.path)
                            continue
                        if entry.is_dir(follow_symlinks=False):stack.append(Path(entry.path));continue
                        if not entry.is_file(follow_symlinks=False):continue
                        size=stat.st_size;files+=1;total+=size;relative=Path(entry.path).relative_to(root);parts=relative.parts
                        top[parts[0]][0]+=size;top[parts[0]][1]+=1
                        key='/'.join(parts[:2]) if len(parts)>1 else '(root files)';groups[key][0]+=size;groups[key][1]+=1
                        role=next((p for p in parts[:-1] if p=='publication_text' or p=='before' or p.startswith('graph_')),None)
                        if role:roles[role][0]+=size;roles[role][1]+=1
                        item=(size,entry.path,stat.st_mtime)
                        if len(largest)<30:heapq.heappush(largest,item)
                        elif size>largest[0][0]:heapq.heapreplace(largest,item)
                    except OSError as exc:
                        if len(errors)<40:errors.append(dict(path=entry.path,error=str(exc)))
        except OSError as exc:
            if len(errors)<40:errors.append(dict(path=str(directory),error=str(exc)))
        if time.monotonic()-last>5:
            dump('disk_status.json',dict(stage='scanning',updated=now(),root=str(root),files=files,logical_bytes=total,directories=directories,completed_roots=len(reports)));last=time.monotonic()
    compact=lambda d:[dict(path=k,logical_bytes=v[0],files=v[1]) for k,v in sorted(d.items(),key=lambda x:x[1][0],reverse=True)[:40]]
    reports.append(dict(root=str(root),logical_bytes=total,files=files,directories=directories,top_level=compact(top),second_level=compact(groups),duplicate_artifact_roles=compact(roles),largest_files=[dict(path=p,logical_bytes=n,modified_unix=m) for n,p,m in sorted(largest,reverse=True)],errors=errors,skipped_reparse=skipped))
    dump('disk_inventory_partial.json',dict(updated=now(),reports=reports))
drives={d:dict(zip(['total','used','free'],shutil.disk_usage(d+':/'))) for d in ['C','E']}
dump('disk_inventory.json',dict(passed=True,completed=now(),elapsed_seconds=time.monotonic()-started,reports=reports,unavailable_roots=unavailable_roots,drives=drives,read_only=True,contents_read=False,logical_sizes_not_allocated_sizes=True,reparse_points_skipped=True,source='os.scandir/stat and shutil.disk_usage; no cleanup or downloads'))
dump('disk_status.json',dict(stage='complete',updated=now(),roots=len(reports)))
print('READ_ONLY_DISK_INVENTORY_COMPLETE',len(reports),flush=True)
