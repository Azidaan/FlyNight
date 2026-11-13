from common import *
source=(PREV/'publish.py').read_text(encoding='utf8')
old="assert read(D/'audit.json')['passed'];assert not git('diff','--cached','--name-only').strip()"
new="assert read(D/'audit.json')['passed'] and read(D/'checks.json')['passed'];assert not git('diff','--cached','--name-only').strip()"
assert source.count(old)==1;source=source.replace(old,new)
old="('association_weights_2179_copy_20261112','без-положительного-итога')]"
new="('association_weights_2179_copy_20261112','без-положительного-итога'),('history_representation_2179_20261113','без-положительного-итога')]"
assert source.count(old)==1;source=source.replace(old,new)
source=source.replace("read(D/'storage_summary.json')['calculation_artifact_bytes']", "sum((D/n).stat().st_size for n in ['result.json','audit.json','summary.json','trace_v2.json','descriptors.npz'])")
source=source.replace("skip={'before','execution'", "skip={'temp','before','execution'")
old="    if path.match('queue3*') or path.match('publication*') :continue"
new="    if path.match('queue3*') or path.match('publication*') or path.name=='trace.json':continue"
assert source.count(old)==1;source=source.replace(old,new)
source=source.replace("['.py','.json','.md','.txt','.log','.csv','.png','.svg','.sha256','.zip']", "['.py','.json','.md','.txt','.log','.csv','.png','.svg','.sha256','.zip','.npz']")
with (OUT/'publish.py').open('x',encoding='utf8',newline='\n') as f:f.write(source)
print('SAFE_PUBLISHER_PREPARED')
