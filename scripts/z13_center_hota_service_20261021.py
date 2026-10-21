"""Execution-only byte-hash guards, avoiding whole-corpus rehash per video job.

Scientific source/config/protocol files are unchanged. All bytes are verified
at sealing, probe entry and final evaluation; per-job checks hash code, models
and the current video's inputs, and stat-check the remaining immutable inputs.
"""
from pathlib import Path
import hashlib
import json
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import z13_center_hota_20261021 as science
RUN=science.RUN;CONTRACT=RUN/'service_contract.json';_probe_verified=False


def fingerprint(path):
    s=path.stat();return dict(size=s.st_size,mtime_ns=s.st_mtime_ns)


def verify_files(root,expected,stats,video=None,full=False):
    checked=0
    for rel,digest in expected.items():
        path=(root/rel).resolve();path.relative_to(root.resolve());stamp=fingerprint(path)
        current=video is not None and (video in path.parts or path.stem==video)
        code_model=path.suffix in ('.py','.json','.yaml','.pt','.pth') or path.name in ('best_chkpt','RECORD')
        if full or current or code_model or stamp!=stats[rel]:
            if science.sha(path)!=digest:raise RuntimeError('Frozen execution input changed: '+rel)
            checked+=1
    return checked


def seal():
    if CONTRACT.exists():raise RuntimeError('Existing service contract; do not overwrite')
    p=science.read(RUN/'protocol.json');science.verify(p)
    extra=['scripts/z13_center_hota_service_20261021.py','tests/test_z13_center_hota_service.py','scripts/z13_train.py','scripts/run_bdd_video_rate_experiment.py','scripts/build_masa_embeddings.py','evaluation/hota_eval.py','datasets/subset_filter.py','scripts/buffer_extend_eval.py','tracking/offline_association.py','tracking/flyvis_motion.py','tracking/flyvis_activity.py','tracking/flyvis_adapter.py','tracking/z13_spatial_localizer.py','tools/flyvis-runtime/flyvis/network/network.py']
    extra.extend('runs/z13_data/'+v+'.npz' for v in p['threshold_videos'])
    expected={**p['sha256'],**{n:science.sha(ROOT/n) for n in extra}}
    science.write(CONTRACT,dict(created=science.now().isoformat(),reason='Prepared before execution. Rehashing the whole frozen video/feature corpus for each of176jobs would consume the authorized window; verification scopes do not change any calculation.',scientific_protocol_sha256=science.sha(RUN/'protocol.json'),sha256=expected,fingerprints={n:fingerprint(ROOT/n) for n in expected},contract='All bytes at seal/probe start/final efficacy; all code/config/model bytes every job; current-video bytes every job; remaining files stat-guarded, unexpected stats force byte hash. No report accepted without final whole-corpus verification.'))
    science.journal('служебный контроль неизменности перед запуском стенда','*Сделано.* Полный манифест входов рассчитан в исходном протоколе. Для176заданий подготовлен отдельный `scripts/z13_center_hota_service_20261021.py`: все байты проверены при опечатывании; повторно в начале пробы и перед итоговой HOTA. На каждом задании пересчитываются хеши кода/конфигураций/моделей и текущего видео; остальные входы проверяются по размеру/времени, любое изменение вызывает сверку байтов. Это уменьшает служебное чтение всего корпуса на каждом видео, а не меняет научные настройки. `service_contract.json` отдельно замораживает эту версию, исходный научный протокол и скрипты не переписаны. Добавлены хеши исходных26 эпизодных файлов для технического воспроизведения и кода исполнения. Итог не принимается без полной конечной проверки.')


def guard(p):
    global _probe_verified
    c=science.read(CONTRACT);assert science.sha(RUN/'protocol.json')==c['scientific_protocol_sha256']
    assert all(c['sha256'][n]==v for n,v in p['sha256'].items())
    stage=sys.argv[sys.argv.index('--stage')+1] if '--stage' in sys.argv else 'probe'
    control='--full-control' in sys.argv
    if (stage!='probe' or control) and (RUN/'technical.json').exists():
        assert science.sha(CONTRACT)==science.read(RUN/'technical.json')['execution_service_contract_sha256']
    full=(stage=='probe' and not control and not _probe_verified) or stage=='eval'
    video=sys.argv[sys.argv.index('--video')+1] if '--video' in sys.argv else None
    checked=verify_files(ROOT,c['sha256'],c['fingerprints'],video,full)
    if full:_probe_verified=True
    return checked


def main():
    if '--seal' in sys.argv:seal();return
    if not CONTRACT.exists():raise RuntimeError('Seal exact execution sources before launch')
    science.verify=guard
    if '--full-control' in sys.argv:
        source=(ROOT/'scripts/z13_center_hota_controller_20261021.py').read_text(encoding='utf-8')
        old="str(ROOT/'scripts/z13_center_hota_20261021.py')";new="str(ROOT/'scripts/z13_center_hota_service_20261021.py')"
        assert source.count(old)==1
        scope=dict(__name__='guarded_center_hota_controller',__file__=str(ROOT/'scripts/z13_center_hota_controller_20261021.py'))
        exec(compile(source.replace(old,new),'guarded_center_hota_controller','exec'),scope);scope['main']();return
    science.main()
    stage=sys.argv[sys.argv.index('--stage')+1] if '--stage' in sys.argv else 'probe'
    if stage=='probe':
        report=science.read(RUN/'technical.json');report['execution_service_contract_sha256']=science.sha(CONTRACT);report['initial_whole_corpus_verified']=_probe_verified;science.write(RUN/'technical.json',report)


if __name__=='__main__':main()
