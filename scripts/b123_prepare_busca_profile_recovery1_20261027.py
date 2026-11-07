"""Восстановить только заполнение недостающих кандидатов по исходному BUSCA."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,math,time
ROOT=Path(__file__).resolve().parents[1];OLD=ROOT/'runs/b123_20261027/busca_runtime_profile';NEW=OLD/'recovery1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists();p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
    result=json.loads((OLD/'controller_result.json').read_text());assert result['all_owned_ended'] and result['exit_code']!=0
    deadline=json.loads((OLD/'execution_deadline.json').read_text())['deadline']
    remaining=datetime.fromisoformat(deadline).timestamp()-time.time();assert remaining>30
    source=ROOT/'scripts/b123_busca_runtime_profile_20261027.py';text=source.read_text(encoding='utf-8')
    text=text.replace("RUN=ROOT/'runs/b123_20261027/busca_runtime_profile'","RUN=ROOT/'runs/b123_20261027/busca_runtime_profile/recovery1'")
    old='order=np.argsort(np.sum((centers-ref)**2,axis=1))[:5];assert len(order)==5'
    assert text.count(old)==1;text=text.replace(old,'order=np.argsort(np.sum((centers-ref)**2,axis=1))[:5]')
    old='candidates=[get_bbox_crop(im,box,normalize=False) for box in candidate_boxes]'
    assert text.count(old)==1
    text=text.replace(old,old+'\n    from busca.tracking import missing_candidate_bbox\n    while len(candidates)<5:\n        candidates.append(np.zeros((384,128,3),dtype=np.uint8))\n        candidate_boxes=np.concatenate([candidate_boxes,missing_candidate_bbox(flavour="ltrb")[None]],axis=0)')
    target=ROOT/'scripts/b123_busca_profile_recovery1_20261027.py';assert not target.exists();target.write_text(text,encoding='utf-8')
    p=json.loads((OLD/'protocol.json').read_text());p.update(author_stop=deadline,budget_seconds=math.floor(remaining),
        service_recovery='В кадре оказалось меньше пяти реальных областей. Использовать штатное дополнение нулевыми изображениями и missing_candidate_bbox, как в associate_embeddings авторов. Сеть,веса,порог,цельипервоначальныйсрокнеменяются.',
        original_deadline=deadline,original_failure='Assertion len(order)==5 after successful strict weight coverage and PE equivalence')
    p['sha256'][target.relative_to(ROOT).as_posix()]=sha(target)
    write(NEW/'protocol.json',p)
    write(OLD/'service_recovery1.json',dict(original_script_sha256=sha(source),recovery_script=target.relative_to(ROOT).as_posix(),
        recovery_script_sha256=sha(target),original_deadline=deadline,old_result=result))
    write(ROOT/'runs/b123_20261027/ready_jobs/91_busca_runtime_profile_recovery1.json',dict(title='BUSCA — штатное дополнение отсутствующих кандидатов, прежний срок',
        folder=NEW.relative_to(ROOT).as_posix(),script=target.relative_to(ROOT).as_posix(),script_sha256=sha(target),protocol_sha256=sha(NEW/'protocol.json')))
    print('Published recovery within original deadline',deadline)
if __name__=='__main__':main()
