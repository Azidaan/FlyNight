"""Не запускать без принятого человеком author_override.json; первыйGPUнеповторять."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import os,shutil,sys,time,subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261028 import RUN,service
read,write,sha,now=service.read,service.write,service.sha,service.now


def main():
    folder=RUN/'history_tracker_author_continuation1';auth=read(folder/'author_override.json')
    assert auth['accepted'] is True and auth['human_message'] and auth['scope']=='Continue same frozen32tracker matrix beyond original090cost gate with new explicit stage deadlines'
    assert not (folder/'protocol.json').exists(),'Do not overwrite a registered continuation'
    original=RUN/'history_tracker_pilot';oldc=read(original/'controller_result.json')
    assert oldc['exit_code']==0 and oldc['all_owned_ended'] and not any(service.matching(k,v) for k,v in oldc['owned'].items())
    oldr=read(original/'report.json');assert not oldr['complete'] and len(oldr['completed'])==1
    oldp=read(original/'protocol.json');oldprice=read(original/'price.json');assert not oldprice['proceed']
    gpu_end=datetime.fromisoformat(auth['gpu_end']).timestamp();audit_end=datetime.fromisoformat(auth['audit_end']).timestamp()
    save=datetime.fromisoformat(read(RUN/'authorization.json')['save_at']).timestamp()
    assert time.time()+300<gpu_end<audit_end<save
    entry=oldr['completed'][0];v=entry['video'];copied={}
    for path,digest in entry['sha256'].items():
        src=ROOT/path;assert sha(src)==digest
        dst=folder/src.relative_to(original);dst.parent.mkdir(parents=True,exist_ok=True)
        if dst.exists():assert sha(dst)==digest
        else:
            try:os.link(src,dst)
            except OSError:shutil.copy2(src,dst)
        assert sha(dst)==digest;copied[dst.relative_to(ROOT).as_posix()]=digest
    first=dict(entry);first.update(sha256=copied,reused_from=original.relative_to(ROOT).as_posix(),original_report_sha256=sha(original/'videos'/v/'report.json'),reused_before_decode=True)
    write(folder/'videos'/v/'report.json',first)
    write(folder/'resume_seed.json',dict(video=v,sha256=copied,first_gpu_video_repeated=False,scientific_settings_unchanged=True,source_controller_sha256=sha(original/'controller_result.json'),source_report_sha256=sha(original/'report.json')))
    shutil.copy2(original/'frozen.json',folder/'frozen.json');assert sha(folder/'frozen.json')==sha(original/'frozen.json')
    p=dict(oldp);p.update(created=now(),author_override_sha256=sha(folder/'author_override.json'),author_gpu_absolute_end=auth['gpu_end'],author_audit_absolute_end=auth['audit_end'],
        price_rule='Original090first full video already costed/negative. Explicit new human authorization creates this separate continuation;reuse first bySHA beforedecode. Same weights/alpha/32/40,remaining39only. Old090price/deadline unchanged.')
    for name in ['scripts/run_history_author_continuation_20261028.py','scripts/audit_history_author_continuation_20261028.py','scripts/record_history_author_continuation_20261028.py','scripts/record_history_full_tracker_20261028.py','tracking/history_cost_replay.py','tests/test_history_cost_replay.py']:
        p['sha256'][name]=sha(ROOT/name)
    # Предыдущийисходныйаудитсохраняется;новаяобёрткаиспользуетточную
    # операциютипаNumPy2,провереннуюво20891событии. Допускинеослабляются.
    history=read(ROOT/'experiments/project_history.json');c=next(c for c in history['experiments'] if c['id']=='history_factorial_tracker_two_epoch_v1')
    date=datetime.now(timezone(timedelta(hours=3))).date().isoformat()
    for key in ['state','done','obtained','interpretation','decision','cannot_claim']:
        if key in c:c.setdefault('revisions',[]).append(dict(date=date,commit='',field=key,old=c[key],reason='New explicit human authorization to continue same frozen matrix; old stop preserved'))
    c['state']='running';c.pop('done',None)
    c['decision']='После нового прямого разрешенияавторапродолжить39оставшихсявидеовотдельномокне,неповторяяготовоепервое. Старую090цену/срокнепереписывать;модель/alpha/критерии не меняются.'
    c['evidence'] += [dict(kind='file',ref=(folder/'author_override.json').relative_to(ROOT).as_posix(),claim='Новоеявноерешениеавторапослеценовогостопа'),dict(kind='file',ref=(folder/'resume_seed.json').relative_to(ROOT).as_posix(),claim='ПервыйготовыйGPUвидеопроходнеповторяется')]
    write(ROOT/'experiments/project_history.json',history)
    for i,args in enumerate([['scripts/validate_project_history.py'],['-m','pytest','tests/test_project_history.py','tests/test_history_cost_replay.py','-q']]):
        with (folder/f'precheck_{i}.log').open('w',encoding='utf-8') as out:r=subprocess.run([sys.executable,*args],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,timeout=300)
        assert r.returncode==0
    p['budget_seconds']=int(gpu_end-time.time()-45);assert p['budget_seconds']>300
    write(folder/'protocol.json',p)
    write(folder/'price.json',dict(original_price=oldprice,original_gate_proceed=False,author_override=True,original_first_video_reused=True,new_available_seconds=p['budget_seconds'],conservative_remaining_seconds=oldprice['conservative_remaining_seconds'],proceed=oldprice['conservative_remaining_seconds']<p['budget_seconds']))
    # GPUиCPUдинамическиеостаткификсируютсяпередпубликацией. Аудитполучает
    # свойсрокизновогоявногорешения,непродлеваетстарый095заднимчислом.
    configs=[('120_history_author_gpu',folder,'scripts/run_history_author_continuation_20261028.py',p['budget_seconds']),
        ('125_history_author_audit',RUN/'history_tracker_author_audit1','scripts/audit_history_author_continuation_20261028.py',int(audit_end-gpu_end-45)),
        ('128_history_author_record',RUN/'history_tracker_author_record1','scripts/record_history_author_continuation_20261028.py',540)]
    for name,f,script,budget in configs:
        if f!=folder:write(f/'protocol.json',dict(budget_seconds=budget,parent_experiment=c['id'],absolute_end=auth['audit_end'] if 'audit' in name else '2026-10-29T07:30:00+03:00',sha256={script:sha(ROOT/script)}))
        write(RUN/'ready_jobs'/(name+'.json'),dict(name=name,folder=f.relative_to(ROOT).as_posix(),script=script,script_sha256=sha(ROOT/script),protocol_sha256=sha(f/'protocol.json')))
    from scripts.b123_safe_journal_20261027 import journal
    journal('history_author_continuation1','Новое прямое решение автора: продолжение фиксированной матрицы после ценового стопа',
        'Источник '+(folder/'author_override.json').relative_to(ROOT).as_posix()+':'+auth['human_message']+'. '
        'Отдельнаяпапкаhistory_tracker_author_continuation1;осталось39видео,первое32вариантноевидеоподключенопоSHAдодекодирования. '
        'Старые090/095цены/отказы/срокисохранены. Теже30весов/.2/32/40/критерии12сравнений,никакогоподборапоval. НовыеGPUдо'+auth['gpu_end']+',аудитдо'+auth['audit_end']+'. '
        'ЧисленнаяошибкааудитаNumPy2исправленавотдельнойобёртке;исходнаяоперацияточновоспроизведена,допускинеослаблены.')


if __name__=='__main__':main()
