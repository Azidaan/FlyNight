"""E2E tests against a running loopback atlas; keep only numerical QA, never the images."""
import base64,json,sys,urllib.request,urllib.error
from pathlib import Path
import numpy as np


def run(base):
    from urllib.parse import urlparse
    u=urlparse(base)
    assert u.scheme=='http' and u.hostname=='127.0.0.1' and u.port
    base=base.rstrip('/')
    def request(method,path,payload=None,origin=None):
        req=urllib.request.Request(base+path,data=None if payload is None else json.dumps(payload).encode(),method=method,
            headers={'Content-Type':'application/json','Origin':origin or base})
        try:
            with urllib.request.urlopen(req,timeout=240) as response:return response.status,json.loads(response.read())
        except urllib.error.HTTPError as exc:return exc.code,json.loads(exc.read())
    assert request('GET','/api/status')[1]['available']
    assert request('POST','/api/respond',{'frames':[]},'http://somewhere.invalid')[0]==403
    assert request('POST','/api/respond',{'kind':'image','frames':['invalid']})[0]==400
    # Two video frames differ only by a moving bar.
    a=np.full((256,256),32,dtype='uint8');a[:,65:95]=220
    b=np.roll(a,15,axis=1)
    enc=lambda x:base64.b64encode(x.tobytes()).decode()
    def get(kind,sequence):
        status,j=request('POST','/api/respond',{'kind':kind,'frames':[enc(x) for x in sequence]})
        assert status==200,j.get('error')
        eye=np.array(j['eye']);maps=np.array(j['maps'])
        assert len(j['channels'])==len(set(j['channels']))==34 and j['device']=='cpu'
        assert maps.shape==(len(sequence),34,721) and eye.shape==(len(sequence),721)
        assert np.isfinite(maps).all() and np.isfinite(eye).all() and eye.min()>=-1e-6 and eye.max()<=1.00001
        return eye,maps,j['checkpoint_sha256']
    e0,m0,w0=get('image',[a])
    e,m,w=get('video',[a,b])
    assert w==w0 and np.array_equal(e0[0],e[0]) and np.max(abs(m0[0]-m[0]))<1e-6
    de=float(np.max(abs(e[0]-e[1])));dm=float(np.max(abs(m[0]-m[1])));assert de>0 and dm>1e-6
    return dict(passed=True,host='127.0.0.1',transmitted_outside_pc=False,real_flyvis_cpu=True,
                image_shape=list(m0.shape),video_shape=list(m.shape),checkpoint_sha256=w,
                first_frame_identity_max=float(np.max(abs(m0[0]-m[0]))),eye_changed=de,activity_changed=dm,
                rejects_foreign_origin=True,rejects_bad_payload=True)


if __name__=='__main__':
    result=run(sys.argv[1]);print(result)
    if len(sys.argv)>2:
        with Path(sys.argv[2]).open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
