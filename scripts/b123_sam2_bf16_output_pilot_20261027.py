"""Новый численный вариант полного пилота; замороженный FP32не меняется."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
RUN=ROOT/'runs/b123_20261027/sam2_bf16_output_pilot';IDENT='sam2_bf16_mask_output_pilot_v1'
ORIGINAL=ROOT/'scripts/b123_sam2_output_pilot_20261027.py'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def prepare():
    assert not RUN.exists()
    profile=ROOT/'runs/b123_20261027/sam2_precision_profile';result=read(profile/'report.json');end=read(profile/'controller_result.json')
    assert result['complete'] and result['all_outputs_finite'] and result['all_image_requests_causal'] and result['bf16_faster']
    assert end['exit_code']==0 and end['all_owned_ended']
    old=read(ROOT/'runs/b123_20261027/sam2_output_pilot/protocol.json')
    p=dict(old);p.update(id=IDENT,created=datetime.now(timezone(timedelta(hours=3))).isoformat(),
        precision='torch.autocast(cuda,bfloat16) on each mask step; weights remain FP32',
        original_worker_sha256=sha(ORIGINAL),budget_seconds=7200,
        action=old['action']+' Единственное изменение алгоритма вычислений относительно FP32 — BF16autocast. Пороги, выборка, модель и веса прежние.',
        adaptation=old['adaptation']+' Этот вариант выполняет масочные шаги в BF16; изменение масок измерено отдельно, эквивалентность качества не предполагается.',
        limitation=old['limitation']+' Нельзя объединять этот результат с первым FP32видео в одну серию; это отдельный численный вариант.')
    p['sha256']=dict(old['sha256'])
    for name in ['scripts/b123_sam2_bf16_output_pilot_20261027.py','tracking/sam2_bf16_recovery.py',
                 'runs/b123_20261027/sam2_precision_profile/report.json']:
        p['sha256'][name]=sha(ROOT/name)
    write(RUN/'protocol.json',p)
    path=ROOT/'experiments/project_history.json';before=path.read_bytes();cat=json.loads(before)
    assert not any(c['id']==IDENT for c in cat['experiments'])
    cat['experiments'].append(dict(id=IDENT,title='Восстановление потерянного объекта — масочный перенос в пониженной точности',area='flyvis',state='ready',started='2026-10-27',
        question='Даст ли та же собственная масочная ветвь полезную HOTA в доступном бюджете при BF16?',
        hypothesis='BF16ускорит вычисление, но знак HOTA этого численного варианта остаётся неизвестным.',expected=p['action'],summary=p['adaptation'],
        acceptance=[dict(criterion=x,met=None) for x in p['acceptance']],obtained='',interpretation='',decision='',cannot_claim=p['limitation'],metrics=[],commits=[],
        files=['scripts/b123_sam2_bf16_output_pilot_20261027.py','tracking/sam2_bf16_recovery.py'],
        evidence=[dict(kind='file',ref=(RUN/'protocol.json').relative_to(ROOT).as_posix(),claim='Новый численный вариант,тот же контроль и ценовой допуск до запуска')]))
    assert path.read_bytes()==before;write(path,cat)
    from scripts.b123_safe_journal_20261027 import journal
    journal(IDENT,'поставлен отдельный полный масочный пилот в BF16',
        '*Постановка.* '+p['action']+'\n\n*Условия.* Первое полное видео заново определяет цену2*time*19+120; '
        '7200снаодномGPU,непозже07:20. СтарыйFP32пилот остановлен,его бюджет не продлевается. '
        'Его первое видео не смешивается с новой серией и не выдаётся за повтор той же численной реализации. '
        'Исполняется замороженный прежний код с единственной заменой класса масочного шага, исходный файл не меняется. '
        'Детектор,XYWH,разметка,веса,пороги,геометрия и выборка прежние. '
        'Короткая парная проверка цены и пиксельных масок сохранена: sam2_precision_profile/report.json. '
        'BF16необъявляетсяточноэквивалентнымFP32. HOTAвновьнужноизмерить.\n\n'
        '*Принятие.* '+p['acceptance'][-1]+'. '+p['limitation']+' '
        'Источник:runs/b123_20261027/sam2_bf16_output_pilot/protocol.json.')
    print('Prepared separate BF16 full pilot; validate before publication')
def run():
    p=read(RUN/'protocol.json');assert sha(ORIGINAL)==p['original_worker_sha256']
    source=ORIGINAL.read_text(encoding='utf-8')
    old='from tracking.sam2_mask_recovery import load_predictor,MaskOutputRecovery'
    assert source.count(old)==1
    replacement='from tracking.sam2_mask_recovery import load_predictor\n    from tracking.sam2_bf16_recovery import BFloat16MaskOutputRecovery as MaskOutputRecovery'
    source=source.replace(old,replacement)
    write(RUN/'source_transform.json',dict(original_source_sha256=p['original_worker_sha256'],
        transform='Only replace MaskOutputRecovery import with BFloat16MaskOutputRecovery; override output folder.',
        transformed_source_sha256=hashlib.sha256(source.encode('utf-8')).hexdigest(),scientific_precision_changed=True))
    namespace={'__file__':str(ORIGINAL),'__name__':'frozen_worker_import'}
    exec(compile(source,str(ORIGINAL), 'exec'),namespace)
    namespace['RUN']=RUN
    namespace['run']()
if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    else:
        try:run()
        except Exception as e:write(RUN/'status.json',dict(state='failed',stage=repr(e),updated=datetime.now(timezone.utc).isoformat()));raise
