"""Исправить неоднозначную адресацию одинаковых рамок, не меняя научный расчёт."""
from pathlib import Path
from datetime import datetime
import sys,time,math
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.recovery_research_service_20261027 import RUN,read,write,sha,now,matching

def prepare_sources():
    old=ROOT/'scripts/audit_lazy_appearance_association_20261027.py';new=ROOT/'scripts/audit_lazy_appearance_association_recovery1_20261027.py';assert not new.exists()
    source=old.read_text(encoding='utf-8')
    before="            jj=[next(j for j,d in enumerate(detections) if np.array_equal(np.asarray(d.xyxy),b)) for b in event['candidate_boxes']]"
    after="""            jj=[]
            # Рамка сама по себе не уникальна: разные YOLO-детекции могут иметь
            # одинаковые координаты, но разные баллы и геометрические стоимости.
            for col,b in enumerate(event['candidate_boxes']):
                options=[j for j,d in enumerate(detections) if j not in jj and np.array_equal(np.asarray(d.xyxy),b)
                         and np.allclose(original[ii,j],np.asarray(event['original'])[:,col],atol=1e-7,rtol=0)]
                assert options,(event['frame'],col,'No unique original candidate')
                jj.append(options[0])"""
    assert source.count(before)==1;source=source.replace(before,after);new.write_text(source,encoding='utf-8')
    helper=ROOT/'scripts/prepare_lazy_appearance_validation_20261027.py';fixed=ROOT/'scripts/prepare_lazy_appearance_validation_recovery1_20261027.py';assert not fixed.exists()
    source=helper.read_text(encoding='utf-8').replace('scripts/lazy_appearance_validation_audit_20261027.py','scripts/lazy_appearance_validation_audit_recovery1_20261027.py').replace('scripts/audit_lazy_appearance_association_20261027.py','scripts/audit_lazy_appearance_association_recovery1_20261027.py')
    fixed.write_text(source,encoding='utf-8')
    write(RUN/'lazy_audit_recovery1_prelaunch.json',dict(created=now(),old_source_sha256=sha(old),new_source_sha256=sha(new),
        reason='First real video replay: frame602, two different candidate columns share identical coordinates; old lookup picked column15 twice. Unique lookup also verifies original geometry-cost column; no GT or feature changes.',
        scientific_outputs_changed=False,original_failure_preserved=True))

def after_failure():
    old=RUN/'lazy_appearance_dev_audit';result=read(old/'controller_result.json');assert result['exit_code']!=0 and result['all_owned_ended'] and not any(matching(k,v) for k,v in result['owned'].items())
    end=datetime.fromisoformat(read(old/'execution_deadline.json')['deadline']).timestamp();remaining=math.floor(end-time.time()-20);assert remaining>120
    folder=old/'recovery1';assert not folder.exists();script='scripts/lazy_appearance_dev_audit_recovery1_20261027.py'
    p=dict(parent_experiment='lazy_appearance_association_v1',created=now(),budget_seconds=remaining,parent_absolute_deadline=read(old/'execution_deadline.json')['deadline'],author_stop=read(RUN/'authorization.json')['save_at'],
        sha256={s:sha(ROOT/s) for s in [script,'scripts/audit_lazy_appearance_association_recovery1_20261027.py','scripts/prepare_lazy_appearance_validation_recovery1_20261027.py']},
        correction='Only bijective addressing of duplicate-coordinate candidates with original cost columns. Same saved science, no rerun GPU/TrackEval; previous900s absolute deadline not extended.')
    write(folder/'protocol.json',p);write(RUN/'ready_jobs/137_lazy_appearance_dev_audit_recovery1.json',dict(name='137_lazy_appearance_dev_audit_recovery1',folder=folder.relative_to(ROOT).as_posix(),script=script,script_sha256=sha(ROOT/script),protocol_sha256=sha(folder/'protocol.json')))

if __name__=='__main__':
    if '--sources' in sys.argv:prepare_sources()
    else:after_failure()
