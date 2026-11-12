from common import *
import subprocess,zipfile,psutil
p=read(OUT/'publication.json');names=list(p['files']);data=subprocess.check_output(['git','cat-file','--batch'],input=('\n'.join('HEAD:'+n for n in names)+'\n').encode());position=0
for name in names:
 end=data.index(b'\n',position);header=data[position:end].decode();position=end+1
 _,kind,size=header.split();assert kind=='blob';size=int(size);actual=data[position:position+size];position+=size+1
 source=Path(p['files'][name]['source']);assert sha(source)==p['files'][name]['sha256'];expected=source.read_bytes()
 if source.suffix.lower() not in ['.jpg','.png','.npz','.zip']:expected=expected.replace(b'\r\n',b'\n')
 assert actual==expected,name
formatting=read(OUT/'publication_formatting.json');assert sha(OUT/'frozen_source_bytes.zip')==formatting['archive_sha256']
with zipfile.ZipFile(OUT/'frozen_source_bytes.zip') as z:
 assert z.testzip() is None
 for row in formatting['files']:assert hashlib.sha256(z.read(row['file'])).hexdigest()==row['frozen_sha256'],row['file']
for n,h in read(OUT/'audit.json')['scientific_artifacts'].items():assert sha(ROOT/n)==h,n
q=read(OUT/'queue7_execution.json');assert q['complete'] and q['passed']
for job in q['jobs']:
 for pid,birth in job['result']['owned'].items():
  try:alive=psutil.Process(int(pid)).create_time()==birth
  except psutil.NoSuchProcess:alive=False
  assert not alive,pid
save('publication_verification.json',dict(passed=True,head=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),checked_blobs=len(names),archived_frozen_texts=len(formatting['files']),science_unchanged=True,all_owned_publication_workers_ended=True))
print('PUBLISHED_BLOBS_FROZEN_ARCHIVE_AND_SCIENCE_VERIFIED',len(names))
