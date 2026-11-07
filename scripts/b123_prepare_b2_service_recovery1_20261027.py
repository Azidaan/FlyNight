from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
RUN=ROOT/'runs/b123_20261027';OLD=RUN/'b2_gradient';NEW=OLD/'recovery1'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists();p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
    old=read(OLD/'controller_result.json');assert old['all_owned_ended'] and old['exit_code']!=0
    import psutil
    for pid,birth in old['owned'].items():
        if psutil.pid_exists(int(pid)):assert abs(psutil.Process(int(pid)).create_time()-birth)>=.01
    from ultralytics.cfg import get_cfg
    cfg=get_cfg(overrides={'task':'detect'})
    assert (cfg.box,cfg.cls,cfg.dfl)==(7.5,.5,1.5)
    assert not hasattr(get_cfg({'task':'detect'}),'box')
    original=read(OLD/'protocol.json');deadline=read(OLD/'execution_deadline.json')['deadline']
    assert datetime.now(timezone.utc)<datetime.fromisoformat(deadline)
    p={**original,'author_stop':deadline,'service_resume':'Missing loss gains are populated by the existing get_cfg defaults through overrides, not treating the partial inference dict as a complete cfg. Original scientific action/arms/loss/steps unchanged. First trial failed before first optimizer step. Original absolute deadline retained.'}
    write(NEW/'protocol.json',p)
    files=['scripts/b123_b2_gradient_recovery1_20261027.py','scripts/b2_gradient_profile_20261027.py',
           'tracking/b1_fast_stimulus.py','tracking/b1_index_copy_stimulus.py',
           '.venv/Lib/site-packages/ultralytics/cfg/default.yaml',
           'runs/b123_20261027/b2_gradient/failure.json','runs/b123_20261027/b2_gradient/controller_result.json']
    write(NEW/'service.json',dict(reason='get_cfg(partial inference args) omits box/cls/dfl;get_cfg(overrides=args) includes installed defaults. No package update or detector-weight change.',
        original_deadline=deadline,completed_steps_repeated=0,
        defaults={'box':cfg.box,'cls':cfg.cls,'dfl':cfg.dfl},cpu_configuration_check_passed=True,
        stimulus_service='Previously checked equivalent distinct-index stimulus operations, original zeros/gradients assertions retained.',sha256={n:sha(ROOT/n) for n in files}))
    name='scripts/b123_b2_gradient_recovery1_20261027.py'
    write(RUN/'ready_jobs/61_b2_gradient_recovery1.json',dict(title='Б2 — служебное восстановление полной конфигурации потерь, прежний срок',
        folder=NEW.relative_to(ROOT).as_posix(),script=name,script_sha256=sha(ROOT/name),protocol_sha256=sha(NEW/'protocol.json')))
    from scripts import z13_localization_labels_20261023 as J
    J.IDENT='b2_predetection_gradient_v1'
    J.journal('техническая ошибка Б2: не заполнены коэффициенты потерь, подготовлено восстановление',
        '*Получено.* `runs/b123_20261027/b2_gradient/failure.json`:AttributeError box до первого optimizer.step;controller_result.json:exit1,all_owned_ended=true. '
        'Причина в get_cfg(partial),а не в отсутствии параметров установленногоdefault.yaml. CPU-проверка воспроизвела пропуск и стандартное заполнение через overrides. '
        'Отдельнаяrecovery1,старые исходники/ошибка сохранены;те же3шага/контроли/веса/функцияпотерь. Прежний абсолютный срок '+deadline+'. '
        'Коэффициенты стандартной установленной конфигурации7,5/0,5/1,5,источник и хеш в recovery1/service.json. Это не результат качества Б2.')
    print(json.dumps(dict(prepared=True,original_deadline=deadline,cpu_configuration_check_passed=True)))
if __name__=='__main__':main()
