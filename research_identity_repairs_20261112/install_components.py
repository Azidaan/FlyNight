"""Install reusable dormant components; no full-runtime activation."""
from common import *
assert read(OUT/'geometry832/audit.json')['passed']
installed=[]
for source,target in [('geometry832/registration.py','tracking/census_correspondence.py'),('geometry832/selection.py','tracking/correspondence_repair_guard.py')]:
 src=OUT/source;dst=ROOT/target;assert not dst.exists(),target
 dst.write_bytes(src.read_bytes());assert sha(dst)==sha(src)
 installed.append(dict(source=source,target=target,sha256=sha(dst)))
save('installed_components.json',dict(installed=installed,activation='Explicit reusable components only. No edits to live tracker, constructor, weights or thresholds. Universal Census replacement failed and is not enabled.',checked=now()))
print('INSTALLED_DORMANT_COMPONENTS',len(installed))
