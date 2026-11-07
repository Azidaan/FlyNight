"""Тот же научный опыт и старый срок, ограниченные повторы записи Windows."""
from pathlib import Path
from datetime import datetime,timezone
import json,sys,hashlib,time,math
import psutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));OLD=ROOT/'runs/b123_20261027/supported_transfer';RUN=OLD/'recovery1';ORIGINAL=ROOT/'scripts/b123_supported_transfer_20261027.py'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.recovery1.next')
    for attempt in range(10):
        try:t.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');t.replace(p);return True
        except PermissionError:
            if attempt==9:
                if p.name=='status.json':return False
                raise
            time.sleep(.1)
def prepare():
    assert not RUN.exists();end=read(OLD/'controller_result.json');assert end['all_owned_ended'] and end['exit_code']!=0
    for pid,birth in end['owned'].items():
        if psutil.pid_exists(int(pid)):assert abs(psutil.Process(int(pid)).create_time()-birth)>.01
    assert 'PermissionError' in (OLD/'stderr.log').read_text(encoding='utf-8')
    assert not (OLD/'completed_videos.json').exists() and not (OLD/'report.json').exists()
    previous=read(OLD/'protocol.json');deadline=read(OLD/'execution_deadline.json')['deadline'];seconds=datetime.fromisoformat(deadline).timestamp()-time.time();assert seconds>120
    p=dict(previous);p.update(author_stop=deadline,budget_seconds=math.floor(seconds),recovery='Service only: bounded atomic writes; original deadline unchanged',original_worker_sha256=sha(ORIGINAL),original_protocol_sha256=sha(OLD/'protocol.json'))
    p['sha256']=dict(previous['sha256']);p['sha256'][Path(__file__).resolve().relative_to(ROOT).as_posix()]=sha(Path(__file__))
    write(RUN/'protocol.json',p);write(RUN/'service_recovery.json',dict(old_controller=end,old_deadline=deadline,new_author_stop=p['author_stop'],
        old_outputs_preserved=True,completed_video_count=0,scientific_inputs_and_thresholds_unchanged=True,source_sha256=sha(Path(__file__))))
    from scripts.b123_safe_journal_20261027 import journal
    journal(p['id'],'восстановление записи статуса на прежнем сроке','*Ошибка.* Первый новый видеопроход прервался после500кадров YOLO из-за WinError5при атомарной замене status.json; полного видео или кэша не сохранено. Старые исходник/логи/контроллеркод1 сохранены. '
        '*Исправление.* Отдельнаяrecovery1повторяетвременнуюошибкузаписидо10раз,постояннаяошибкаслужебногостатусанеснимаетконтрольсрока;научныефайлыпослеисчерпанияповтороввызываютошибку. '
        'Веса,выборка,пороги,правило и исходныйдедлайн '+deadline+' сохранены. Незавершённое первое видео будет пересчитано;готовые видео отсутствуют. Источник:runs/b123_20261027/supported_transfer/recovery1/service_recovery.json. Это служебное восстановление,не новый научный результат.')
    print('Prepared service recovery within original deadline')
def run():
    p=read(RUN/'protocol.json');assert sha(ORIGINAL)==p['original_worker_sha256'];namespace={'__name__':'frozen_worker_import','__file__':str(ORIGINAL)}
    exec(compile(ORIGINAL.read_text(encoding='utf-8'),str(ORIGINAL),'exec'),namespace);namespace['RUN']=RUN;namespace['write']=write;namespace['run']()
if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    else:
        try:run()
        except Exception as e:write(RUN/'status.json',dict(state='failed',stage=repr(e),updated=datetime.now(timezone.utc).isoformat()));raise
