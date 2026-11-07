"""Record localized numerical cause without calling scientific problems solved."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261029 import RUN,service

def main():
    c=service.read(RUN/'masa_fp32_preparation/controller_result.json')
    assert c['exit_code']==0 and c['all_owned_ended']
    assert not any(service.matching(k,v) for k,v in c['owned'].items())
    old=service.read(RUN/'masa_query_pilot/report.json');assert old['complete'] and len(old['completed'])==20
    markers=list((RUN/'masa_query_audit/completed').glob('*.json'))
    assert len(markers)==20 and all(service.read(x)['passed'] for x in markers)
    service.write(RUN/'masa_query_audit/discrete_component_verified.json',dict(
        full_discrete_replay_passed=True,videos=20,trackers=5,feature_audit_passed=False,
        full_audit_passed=False,scientific_verdict=None,
        failed_original_controller_sha256=service.sha(RUN/'masa_query_audit/controller_result.json'),
        markers={x.relative_to(ROOT).as_posix():service.sha(x) for x in markers},updated=service.now()))
    path=ROOT/'experiments/system_problems.json';problems=service.read(path)
    problem=next(x for x in problems['problems'] if x['id']=='С6')
    evidence=[dict(source='runs/continuation_20261029/masa_feature_numerics/report.json',
        sha256=service.sha(RUN/'masa_feature_numerics/report.json'),claim='Matched TF32on/off intervention on same inputs/weights, all three variants'),
        dict(source='runs/continuation_20261029/masa_query_audit/controller_result.json',
        sha256=service.sha(RUN/'masa_query_audit/controller_result.json'),claim='Original full-audit failure retained')]
    problem['cases'].append(dict(date='2026-10-29',fact='MASA: все20/5GPUготовы,вседискретныесостояния/JSON/raw/метрикиповторены,ноCPUпризнакинеуложилисьв1e-5. НаодномфиксированномкадречестныйGPUповторисходногоTF32точенсохранённымданным;отключениеТОЛЬКОTF32снизилоошибкуCPU/GPUдляR501024/R501333/Detic1333с3.326e-4/2.893e-4/1.545e-4до5.066e-7/4.843e-7/5.537e-7. Это установленный локальный механизм данного численного отказа, не причина неудач памяти/ядра/Б2.',
        mechanism='scripts/run_masa_detic_query_pilot_20261029.py: implicit torch.backends.cudnn.allow_tf32=True; tensors float32 did not guarantee full-mantissa convolution',
        intervention='Separate masa_query_fp32_recovery1: cudnn/matmul TF32False and highest matmul precision; same weights/actions/20videos/tolerance, old execution preserved',
        next_check='Full20/5CPUaudit and fixed three-video encoder checks must pass unchanged1e-5; only then claim the new execution accepted',evidence=evidence))
    problem['evidence'].extend(evidence);problem['updated']='2026-10-29'
    service.write(path,problems)
    text='''29.10 около19:36МСК — ПРИОРИТЕТНОЕ УТОЧНЕНИЕ. Исходная010полностьюготова20/5/code0/allended.020точноповторилавсе20raw/JSON/events/метрикииzero-state,ноотказаланапервомencoderCPUчекеR501024=.000332631>1e-5;025отказназависимости,НЕполныйнаучныйитог. Старые20GPU/контроллеры/маркерысохранены;discrete_component_verified.jsonподтверждаетТОЛЬКОпройденныйкомпонент.030фиксированныйкадр/все3encoder: исходныйGPUточенсохранённымвекторам,TF32offдаётCPUсогласие<5.54e-7,размерпакетанеобъясняетрасхождение. Конкретныйузел—неявныйcudnn.allow_tf32=True. ВC6добавленслучай/SHA/различающаяпроверка;непереноситьпричинныйвыводнапамять/Б2/научнуюпользу.

035подготовилаОТДЕЛЬНУЮстрогуюFP32версиюпослепроверкикаталога:040masa_query_fp32_recovery1GPU→050masa_audit_fp32_recovery1CPU→055masa_record_fp32_recovery1. Допуски/веса/.1/20/5/истории/данные/статистиканеизменены;явноTF32conv=false/matmul=false/highest. Это новоеисправленноеисполнение, меняющееарифметикупризнаковизависимыеназначения,а не перезаписьисходных20. Не повторятьготовыевидеопроходыВНУТРИкаждойверсии;tracking_completedбезметрик восстанавливатьтолькоCPU. Смотретьсвежиеstatus/price/controller. Независимыйаудитвсех20/3encoderс1e-5обязателен;полныйрезультатнеобъявлятьзаранее. Общийsave30.10 07:30/stop07:40неизменён,однаqueue7916/guard17624. Никакогообучения/скачивания/сменыокружения.
'''
    with (RUN/'HANDOFF.md').open('a',encoding='utf-8') as f:f.write('\n\n'+text)
    service.write(RUN/'masa_fp32_preparation/registered_cause.json',dict(text=text,evidence=evidence,updated=service.now()))

if __name__=='__main__':main()
