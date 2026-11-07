"""Офлайновый разбор готовности; сторонние модели не импортируются и не исполняются."""
from pathlib import Path
from datetime import datetime, timezone
import ast, hashlib, importlib.util, json, sys

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/b123_20261027'


def run(method):
    folder=RUN/(method+'_readiness')
    def write(name,value):
        p=folder/name;t=p.with_suffix('.next')
        t.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');t.replace(p)
    p=json.loads((folder/'protocol.json').read_text(encoding='utf-8'))
    for name,digest in p['sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    write('status.json',dict(state='running',stage='Проверка исходников и готовности',updated=datetime.now(timezone.utc).isoformat()))
    source=RUN/'recovery_source_snapshot/manifest.json'
    manifest=json.loads(source.read_text(encoding='utf-8')) if source.exists() else None
    sources=[]
    if manifest:
        for entry in manifest['files']:
            target=ROOT/entry['path'];target.resolve().relative_to(ROOT.resolve())
            assert hashlib.sha256(target.read_bytes()).hexdigest()==entry['sha256'],entry['path']
            if entry['repository']==('lorenzovaquero/BUSCA' if method=='busca' else 'ManuelBGomez/SGAR-MOT'):
                sources.append(entry)
    packages=['torch','torchvision','yaml','cv2','positional_encodings'] if method=='busca' else ['torch','segment_anything','sam2','aot']
    installed={n:importlib.util.find_spec(n) is not None for n in packages}
    names=('model_busca.pth','model_feats.pth') if method=='busca' else ('sam_vit_h_4b8939.pth','R50_DeAOTL_PRE_YTB_DAV.pth')
    weights=[]
    for root in (ROOT/'weights',ROOT/'tools/busca',ROOT/'tools/ltr'):
        if root.exists():
            for candidate in root.rglob('*'):
                if candidate.is_file() and candidate.name in names:
                    weights.append(dict(path=candidate.relative_to(ROOT).as_posix(),bytes=candidate.stat().st_size,
                                        verified_compatible=False))
    imports=set()
    for entry in sources:
        target=ROOT/entry['path']
        if target.suffix=='.py':
            try: tree=ast.parse(target.read_text(encoding='utf-8-sig'))
            except SyntaxError:continue
            for node in ast.walk(tree):
                if isinstance(node,ast.Import):imports.update(alias.name.split('.')[0] for alias in node.names)
                elif isinstance(node,ast.ImportFrom) and node.module:imports.add(node.module.split('.')[0])
    if method=='busca':
        mechanism='BUSCA: потерянные после сопоставления траектории, память наблюдённых изображений/рамок, текущая калмановская рамка, соседи и обучаемый отказ. Наш XYWH отличается от авторского XYAH; чужой адаптер целиком не переносить.'
        next_action='Отдельный подключаемый модуль восстановления после последней ассоциации XYWHByteTrack. Сначала точный контроль при выключенной ветви; затем авторские веса и замер на фиксированном train-клипе; потом одинаковые база/простое продолжение/BUSCA на заранее выбранном полном наборе.'
        blockers=['Официальные веса BUSCA и его собственного извлекателя внешнего вида должны быть получены и проверены; наши MASA-признаки не взаимозаменяемы с ними.',
                  'Совместимость старых PyTorch/NumPy/API с текущим окружением ещё не проверена; основное окружение не понижать.',
                  'Дообучение на BDD100K не воспроизведено; запуск общих весов назвать переносом готовой модели, не повтором BDD-опыта статьи.']
    else:
        mechanism='LTR IbPRIA: 27 масок SAM по сетке в последней наблюдённой рамке, обучаемый выбор маски и DeAOT. Найденный SGAR-MOT использует SAM2/MIR и является другой работой.'
        next_action='Проверить точную доступность исходников и весов выбора маски LTR. Если их нет, зафиксировать ограничение воспроизведения; отдельную адаптацию по маскам регистрировать под собственным именем, не выдавая за LTR.'
        blockers=['Полная авторская реализация LTR и веса селектора пока не подтверждены.',
                  'SAM/DeAOT и селектор требуют отдельных весов и измерения памяти/стоимости на12ГБ.',
                  'SGAR-MOT не заменяет LTR автоматически; измерения в README относятся к другому методу.']
    result=dict(complete=True,method=method,preparation_only=True,source_manifest=str(source.relative_to(ROOT)) if manifest else None,
        source_files=len(sources),installed_modules=installed,weights=weights,imports=sorted(imports),
        mechanism=mechanism,next_action=next_action,unresolved=blockers,
        ready_for_full_tracker_run=False,HOTA_measured=False,
        reason='Доступность исходников не заменяет проверку весов, точной базы и стоимости; требуются следующие рабочие ступени.',
        updated=datetime.now(timezone.utc).isoformat())
    write('report.json',result)
    write('status.json',dict(state='completed',stage='Подготовительный отчёт; следующий запуск требует реализации',updated=datetime.now(timezone.utc).isoformat()))
    print(json.dumps(result,ensure_ascii=False))

