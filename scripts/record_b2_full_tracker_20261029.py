"""Карточка только после первичных результатов и независимого полного аудита."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261028 import RUN,service


def main():
    folder=RUN/'b2_tracker_record';source=RUN/'b2_tracker_pilot';audit=RUN/'b2_tracker_audit'
    for f in (source,audit):
        c=service.read(f/'controller_result.json')
        assert c['exit_code']==0 and c['all_owned_ended'] and not any(service.matching(k,v) for k,v in c['owned'].items())
    r=service.read(source/'report.json');a=service.read(audit/'report.json')
    full=bool(r.get('complete') and a.get('passed') and a.get('full_series'))
    h=service.read(ROOT/'experiments/project_history.json');card=next(c for c in h['experiments'] if c['id']=='b2_predetection_tracker_sparse_v1')
    result=dict(obtained='runs/continuation_20261028/b2_tracker_pilot/report.json и independent_audit.json: '
                +'полная40видео/8вариантов/независимыйCPUаудит.' if full else
                'runs/continuation_20261028/b2_tracker_pilot/report.json и b2_tracker_audit/report.json:полная научная серия не подтверждена; scientific_verdict=null.',
                interpretation='Пользу всей системы отделять от добавки обучения ядра D-C; одна эпоха/seed26/10обучающихцелейнавидео/40историческихval ограничены.'
                               if full else 'Неполнота серии или аудита не является отрицательным научным результатом Б2.',
                decision='Сопоставить полный итог с критериями, частными подтверждениями и открытыми проблемами; затем агент выбирает следующее действие, Б3 после Б2.'
                         if full else 'Сохранить готовые GPU/CPUвидеопроходы/SHA/цены/остаток; меньшие бюджеты не продлевать и готовое не повторять.',
                cannot_claim='Не полная Б2 и не новый независимый итоговый тест; не переносить пользу системы на ядро без D-C. '
                             'Фиксированные P3координаты не совмещают объекты; память ровно5queryкадров, состояния заново инициализируются для окна. '
                             'CPUполныйrawhead сравнивается с заранее заданным допуском, не побитно; дискретный трекер использует проверенные сохранённые GPUдетекции.')
    for field,value in result.items():
        if card.get(field) and card[field]!=value:card.setdefault('revisions',[]).append(dict(date='2026-10-29',commit=None,field=field,old=card[field],reason='Фактический итог зарегистрированного пилота'))
        card[field]=value
    card.update(state=('completed_positive' if r['main_acceptance'] else 'inconclusive') if full else 'inconclusive',done='2026-10-29')
    for i,item in enumerate(card['acceptance']):item.update(met=(r['main_acceptance'] if i==len(card['acceptance'])-1 else True) if full else None,evidence='runs/continuation_20261028/b2_tracker_pilot/report.json')
    if full:
        src=source.relative_to(ROOT).as_posix()+'/report.json'
        card['metrics']=[dict(key='HOTA',value=m['HOTA'],baseline=r['pooled']['native']['HOTA'],subset='40историческихval;'+n+';1эпоха/seed26/разреженныйfit',source=src,locator=['pooled',n,'HOTA']) for n,m in r['pooled'].items()]
        for control,x in r['primary_D_vs_controls'].items():
            if x['ci95'][0]<=0:continue
            card.setdefault('confirmed_results',[]).append(dict(id='b2_D_vs_'+control+'_historical40',title='Б2 — частная разность HOTA D к '+control,
                category='tracker',metric='HOTA',unit='points',comparison='D против '+control,subset='40историческихval',
                conditions='Одна эпоха seed26/10обучающихцелейнакаждомtrainвидео,одинаковыйYOLO960/полныйтрекер;полныйнезависимыйаудит.',
                limitations='Частнаяноминальная95%разность;6первичныхсравненийтакжепоказанысBonferroni. '+card['cannot_claim'],
                evidence=dict(source=src,sha256=service.sha(source/'report.json'),locator=['primary_D_vs_controls',control])))
    service.write(ROOT/'experiments/project_history.json',h)
    from scripts.b123_safe_journal_20261027 import journal
    journal(card['id'],'Сохранён итог полного трекерного пилота Б2',card['obtained']+' '+card['interpretation']+' '+card['cannot_claim'])
    for i,args in enumerate([['scripts/validate_project_history.py'],['-m','pytest','tests/test_project_history.py','tests/test_b2_causal_detector.py','tests/test_b2_tracker_runtime.py','tests/test_b2_tracker_summary.py','-q']]):
        with (folder/f'check_{i}.log').open('w',encoding='utf-8') as out:
            code=subprocess.run([sys.executable,*args],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,timeout=420).returncode
        assert code==0
    service.write(folder/'report.json',dict(passed=True,scientific_series_audited=full,state=card['state'],agent_review_and_fundamental_graph_required=True,updated=service.now()))


if __name__=='__main__':main()
