"""Зарегистрировать одну фиксированную композицию, без нового выбора по val."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.recovery_research_service_20261027 import RUN,read,write,sha,now,matching
from scripts.b123_safe_journal_20261027 import journal

def main():
    assert read(RUN/'lazy_appearance_validation/independent_audit.json')['passed']
    for folder in ['lazy_appearance_validation','lazy_appearance_validation_audit']:
        c=read(RUN/folder/'controller_result.json');assert c['exit_code']==0 and c['all_owned_ended'] and not any(matching(k,v) for k,v in c['owned'].items())
    folder=RUN/'fixed_composition';assert not folder.exists()
    old=read(RUN/'lazy_appearance_validation/protocol.json');sources=old['sources'];bound=['tracking/fixed_box_calibration.py','tracking/pixel_box_refinement.py',
        'scripts/run_fixed_composition_20261027.py','runs/recovery_research_20261027/lazy_appearance_validation/report.json',
        'runs/recovery_research_20261027/lazy_appearance_validation/independent_audit.json','runs/recovery_research_20261027/pixel_box_numeric_control/calibration.npz',
        'runs/recovery_research_20261027/pixel_box_numeric_control/frozen_before_validation.json','runs/recovery_research_20261027/pixel_box_numeric_control/report.json',
        'runs/recovery_research_20261027/pixel_box_numeric_control/independent_audit.json','runs/z13_val20_matched_20261024/native_masa_recovery2/report.json']
    for s in sources:
        v=s['video'];bound += [s['baseline'],s['alignment'],f'runs/recovery_research_20261027/lazy_appearance_validation/appearance/{v}.json',f'runs/recovery_research_20261027/pixel_box_numeric_control/numeric_control/{v}.json']
        if s['group']=='development20':s['masa']=f'runs/z13_val20_matched_20261024/xywh/evaluated_masa/{v}.json';bound.append(s['masa'])
    p=dict(id='fixed_association_geometry_composition_v1',created=now(),budget_seconds=1200,device='cpu',author_stop=read(RUN/'authorization.json')['save_at'],sources=sources,groups=old['groups'],
        action='К уже фиксированным выходам реальных назначений применить прежнюю линейную калибровку рамок с alpha0,5. '
            'Ничего не обучать и не выбирать заново. Пересчитать12числовых признаков по фактическим рамкам и наличиюIDна предыдущем запросе каждой ветви. '
            'Только координаты выхода;состояние/ID/классы/баллы/число исходной ветви сохраняются. На40 сравнить композицию с исходной базой и обеими отдельными поправками. '
            'На общих первых20 отдельно сравнить готовую сильную офлайнMASA, новую онлайнветвь и ту же фиксированную калибровку поверхMASA. Это не новая реализацияMASA.',
        hypothesis='Две зафиксированные малые поправки разных компонентов могут сохранять пользу совместно; эффекты заранее не складываются.',
        counterargument='Обе версии отбирались наdev,но ихval уже просмотрен. Это последующий поисковый контроль композиции,не новый независимый итоговыйтест. '
            'Смена распределения рамок/ID может разрушить калибровку. Сильная офлайнMASA использует иной временной режим,поэтому сравнение не изолирует одну архитектуру.',
        acceptance=['Никакой подгонки/выбора новых весов или силы по val;полные40,нулевая калибровка точно сохраняет каждую родительскую ветвь,прежняя числовая ветвь точно воспроизведена',
            'ПолнаяHOTA/AssA/IDF1/FP/FN/IDSw;положительнаянижняя95%парнаяграницакомпозициикбазеИобоимродителямна40. Сравнения сMASAтолькообщие20иотдельныйвывод.',
            'НезависимаяпроверкавсехJSON/прошлого/исходныххешей/нулевого контроля/интервалов;baselineне меняется'],
        cost_gate='Первыйполныйвидеопроходвсех6выходовсоценкой:2*time*39+180<остаток1200с;иначестопбезпродления.',
        limitation='Последующая поисковая проверка ранее просмотренных40историческихval. Новыхданных/весов/обучения/GPUнет. '
            'MASA—готоваяофлайнсшивкатолько20,не новая онлайнсистема. Пакетныйвыходгеометрии не меняет состояние. ВкладядраFlyVisне измеряется. '
            'Композиция не становится научной проверкойБ1–Б3;основнойbaselineнеизменен.',sha256={f:sha(ROOT/f) for f in set(bound)})
    write(folder/'protocol.json',p)
    path=ROOT/'experiments/project_history.json';before=path.read_bytes();cat=json.loads(before);assert not any(c['id']==p['id'] for c in cat['experiments'])
    cat['experiments'].append(dict(id=p['id'],title='Совместить фиксированные поправки назначений и рамок',area='flyvis',state='ready',started='2026-10-27',
        question='Сохраняется ли малая польза двух отдельных поправок в одной системе и каков результат на общем сильном контроле?',hypothesis=p['hypothesis'],expected=p['action'],summary=p['counterargument'],
        obtained='',interpretation='',decision='',cannot_claim=p['limitation'],acceptance=[dict(criterion=c,met=None) for c in p['acceptance']],metrics=[],commits=[],files=bound[:3],
        evidence=[dict(kind='file',ref=folder.relative_to(ROOT).as_posix()+'/protocol.json',claim='Фиксированные родители,критерииисильныйконтрольдо расчёта композиции')]))
    assert path.read_bytes()==before;write(path,cat)
    audit=RUN/'fixed_composition_audit';write(audit/'protocol.json',dict(budget_seconds=600,parent_experiment=p['id'],author_stop=p['author_stop']))
    jobs=[('150_fixed_composition','fixed_composition','scripts/run_fixed_composition_20261027.py'),('155_fixed_composition_audit','fixed_composition_audit','scripts/audit_fixed_composition_20261027.py')]
    for name,sub,script in jobs:
        write(RUN/(name+'_prepared.json'),dict(name=name,folder=(RUN/sub).relative_to(ROOT).as_posix(),script=script,script_sha256=sha(ROOT/script),protocol_sha256=sha(RUN/sub/'protocol.json')))
    journal(p['id'],'поставлена фиксированная совместная проверка назначений и координат',
        '*Постановка.* '+p['action']+'\n\n*Возражение.* '+p['counterargument']+'\n\n*Критерии.* '+json.dumps(p['acceptance'],ensure_ascii=False)+' '+p['cost_gate']+
        '\n\n*Ограничения.* '+p['limitation']+' Литературноеоснование назначенийDeepSORT1703.07402/DeepOCSORT2302.11813 уже прочитано;калибровкаготоваизtrain/dev,новая теория пользы не объявляется. Публикацияпослезелёныхпровероккаталога.')
    print('Prepared fixed composition without publishing until catalogue checks')

if __name__=='__main__':main()
