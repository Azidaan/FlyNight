from common import *
import urllib.request
url='https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-seg.pt'
destination=ROOT/'weights/yolo26seg_official_20261114/yolo26n-seg.pt'
destination.parent.mkdir(exist_ok=True)
assert not destination.exists()
request=urllib.request.Request(url,headers={'User-Agent':'Dissertation-research'})
with urllib.request.urlopen(request,timeout=90) as response,destination.open('xb') as output:
    final_url=response.url
    while True:
        block=response.read(1024*1024)
        if not block:break
        output.write(block)
assert destination.stat().st_size>1_000_000
save('weights_receipt.json',dict(source=url,redirect_host=urllib.parse.urlparse(final_url).hostname,path=destination.relative_to(ROOT).as_posix(),bytes=destination.stat().st_size,sha256=sha(destination),downloaded=now(),source_basis='Official installed Ultralytics downloads.py v8.4.0 release URL, model explicitly requested by user.'))
print('OFFICIAL_WEIGHTS_SAVED',destination.stat().st_size)
