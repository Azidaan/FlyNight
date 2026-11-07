"""Проверить публичную папку весов; только HTTP-чтение, без входа в аккаунт."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,re,ssl,sys,urllib.request,urllib.error
import certifi
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/b123_20261027/busca_public_weights_probe'
URLS=[('folder','https://drive.google.com/drive/folders/1SvXexLRS0pbnZkxcSjQEEW3Sb6usiGw-'),
      ('busca_view','https://drive.google.com/file/d/15LB6SPHtDc-4_fLQtRIzU1YWOTF6vNf/view'),
      ('feature_view','https://drive.google.com/file/d/1ZNU0yNkhMTlLRSOC0PR82SwK1ic9OJ8Y/view')]
def write(name,value):
    path=RUN/name;temp=path.with_suffix('.next')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');temp.replace(path)
def main():
    p=json.loads((RUN/'protocol.json').read_text(encoding='utf-8'))
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==p['source_sha256']
    result=[];context=ssl.create_default_context(cafile=certifi.where())
    for name,url in URLS:
        write('status.json',dict(state='running',stage='Публичный адрес '+name,updated=datetime.now(timezone.utc).isoformat()))
        try:
            request=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'})
            with urllib.request.urlopen(request,timeout=20,context=context) as response:
                data=response.read(2_000_001)
                assert len(data)<=2_000_000,'HTML size bound exceeded'
                content_type=response.headers.get('Content-Type','')
                final_url=response.url
                code=response.status
            assert 'text/html' in content_type
            target=RUN/(name+'.html');target.write_bytes(data)
            text=data.decode('utf-8',errors='replace')
            ids=sorted(set(re.findall(r'/file/d/([A-Za-z0-9_-]{20,})',text)))
            snippets=[]
            for match in re.finditer(r'model_busca|model_feats|motsynth|market1501',text,re.IGNORECASE):
                snippets.append(text[max(0,match.start()-180):match.end()+180])
            result.append(dict(name=name,url=url,http_status=code,final_url=final_url,
                content_type=content_type,html_sha256=hashlib.sha256(data).hexdigest(),
                candidate_file_ids=ids,snippets=snippets[:20]))
        except Exception as e:
            result.append(dict(name=name,url=url,error=repr(e),http_status=getattr(e,'code',None)))
    write('report.json',dict(complete=True,public_endpoints_only=True,results=result,
        weights_downloaded=False,datasets_downloaded=False,models_executed=False,
        next_action='Если публичная папка содержит новые официальные ссылки, сверить имя/размер и получить веса. Если доступ не подтверждён, сохранить ограничение воспроизведения и перейти к разрешённой второй работе; не повторять одни404адреса бесконечно.'))
    write('status.json',dict(state='completed',stage='Проверка публичных адресов завершена',updated=datetime.now(timezone.utc).isoformat()))
if __name__=='__main__':main()
