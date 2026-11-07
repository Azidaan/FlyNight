"""Local file inference for the atlas; archive playback also works directly from index.html."""
import argparse
import base64
import json
import logging
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import threading
from urllib.parse import urlsplit
import webbrowser

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DIST = (HERE / 'dist').resolve()
WORKER = ROOT / 'scripts/flyvis_atlas_live_worker_20261107.py'
PY = ROOT / '.venv/Scripts/python.exe'
BUSY = threading.Lock()
LIMIT = 1100000


class Local(SimpleHTTPRequestHandler):
    server_version = 'FlyVisAtlas/1'
    def __init__(self, *args, **kw):
        super().__init__(*args, directory=str(DIST), **kw)

    def log_message(self, fmt, *args):
        pass

    def list_directory(self, path):
        self.send_error(404)
        return None

    def allowed(self):
        return self.headers.get('Host') in (f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}')

    def send_head(self):
        if not self.allowed():
            self.send_error(403);return None
        try:
            Path(self.translate_path(self.path)).resolve().relative_to(DIST)
        except (ValueError,OSError):
            self.send_error(404);return None
        return super().send_head()

    def end_headers(self):
        self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Referrer-Policy','no-referrer')
        super().end_headers()

    def data(self, value, code=200):
        body=json.dumps(value,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode('utf-8')
        self.send_response(code);self.send_header('Content-Type','application/json; charset=utf-8')
        self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(body)));self.end_headers()
        try:self.wfile.write(body)
        except (BrokenPipeError,ConnectionResetError,ConnectionAbortedError):pass

    def do_GET(self):
        if not self.allowed():return self.data({'error':'Запрос не с локального адреса.'},403)
        if urlsplit(self.path).path=='/api/status':
            return self.data(dict(available=PY.is_file() and WORKER.is_file(),busy=BUSY.locked(),maxFrames=12))
        return super().do_GET()

    def do_POST(self):
        if not self.allowed() or self.headers.get('Origin') not in (None,'http://'+self.headers.get('Host','')):
            return self.data({'error':'Допустим только запрос из локального атласа.'},403)
        if self.path!='/api/respond':return self.data({'error':'Нет такого метода.'},404)
        if self.headers.get_content_type()!='application/json':return self.data({'error':'Нужен JSON.'},415)
        try:
            n=int(self.headers.get('Content-Length','0'))
        except ValueError:n=0
        if not 0<n<=LIMIT:return self.data({'error':'Допустимо не более 12 кадров по 256 × 256.'},413)
        try:
            self.connection.settimeout(15)
            req=json.loads(self.rfile.read(n))
            if not isinstance(req,dict) or set(req)!={'kind','frames'} or req['kind'] not in ('image','video'):raise ValueError
            frames=req['frames']
            if not isinstance(frames,list) or not 1<=len(frames)<=12 or req['kind']=='image' and len(frames)!=1:raise ValueError
            if any(not isinstance(s,str) or len(s)!=87384 or len(base64.b64decode(s,validate=True))!=65536 for s in frames):raise ValueError
        except (ValueError,TypeError,OSError):return self.data({'error':'Неверный формат кадров.'},400)
        if not BUSY.acquire(blocking=False):return self.data({'error':'Один расчёт уже выполняется. Дождись его окончания.'},409)
        try:
            folder=ROOT/'tmp/flyvis_atlas_live';folder.mkdir(parents=True,exist_ok=True)
            with tempfile.TemporaryDirectory(dir=folder) as name:
                src=Path(name)/'input.json';dst=Path(name)/'response.json'
                src.write_text(json.dumps(req),encoding='utf-8')
                env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONIOENCODING':'utf-8'}
                # Worker has its own independent supervisor with a wall-time limit.
                proc=subprocess.run([str(PY),str(WORKER),str(src),str(dst)],cwd=ROOT,env=env,stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                if proc.returncode or not dst.is_file():
                    latest=proc.stdout.decode('utf-8',errors='replace')[-1500:]
                    logging.warning('FlyVis worker: %s',latest)
                    return self.data({'error':'Расчёт не выполнен. Возможны занятая проверка или нехватка свободной памяти. Подробность — в окне запуска атласа.'},503)
                return self.data(json.loads(dst.read_bytes()))
        except OSError as exc:
            logging.warning('FlyVis file: %s',exc)
            return self.data({'error':'Не удалось запустить локальную модель.'},503)
        finally: BUSY.release()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=0);parser.add_argument('--no-browser',action='store_true');ns=parser.parse_args()
    with ThreadingHTTPServer(('127.0.0.1',ns.port),Local) as server:
        url=f'http://127.0.0.1:{server.server_port}/'
        print('FlyVis Atlas: '+url+'   |   Ctrl+C to close',flush=True)
        if not ns.no_browser:webbrowser.open(url)
        try:server.serve_forever()
        except KeyboardInterrupt:pass
