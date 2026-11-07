"""Записать подтверждённую зависимость данной головы от признака,сохранив границы вывода."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.recovery_research_service_20261027 import RUN,read,write,sha,matching,now
from scripts.b123_safe_journal_20261027 import journal

def main():
    source='runs/recovery_research_20261027/masa_feature_contribution/report.json';r=read(ROOT/source);a=read(RUN/'masa_feature_contribution/independent_audit.json')
    assert r['complete'] and a['passed'] and a['report_sha256']==sha(ROOT/source) and r['feature_added_value_supported']
    for sub in ['masa_feature_contribution','masa_feature_contribution_audit/recovery1']:
        c=read(RUN/sub/'controller_result.json');assert c['exit_code']==0 and c['all_owned_ended'] and not any(matching(k,v) for k,v in c['owned'].items())
    path=ROOT/'experiments/project_history.json';before=path.read_bytes();cat=json.loads(before);card=next(c for c in cat['experiments'] if c['id']=='masa_feature_contribution_v1');assert not card['obtained']
    oldsource='runs/stitch_v2/gap150_dist5/report.json';old=read(ROOT/oldsource)
    context={k:old['diff'][k] for k in ['base-none|test|HOTA','base_masa-none|test|HOTA','base_masa-base|test|HOTA','base_masa-base|val7|HOTA']}
    write(RUN/'masa_feature_contribution/historical_separate_training_context.json',dict(source=oldsource,source_sha256=sha(ROOT/oldsource),comparisons=context,
        caveat='Другие35test/7val,раздельнообученные головы,не текущая20выборка. Превосходствообученнойbase_masaнадобученнойbaseнеподтверждено;этонеопровергаетзависимостьнынешнейзамороженнойголовыотпризнака.'))
    card.update(state='completed_positive',done='2026-10-27',obtained=source+':pooled='+json.dumps({k:x['HOTA'] for k,x in r['pooled'].items()},ensure_ascii=False)+';paired='+json.dumps(r['paired'],ensure_ascii=False)+
        ';neutral_masaчислоссылок=0навсех20. runs/recovery_research_20261027/masa_feature_contribution/independent_audit.json:passed=true;256recovery1code0/all_owned_ended,все80контрольныхJSON/оценкиповтореныточно.',
        interpretation='В данной замороженной10-признаковойголове правильный MASA-признак действительно нужен для измеренной прибавки:нейтрализация дала0связей и исходнуюHOTA,перестановки дали меньшуюHOTA,полныйвариант лучше всех4контролей по положительнымпарныминтервалам. '
            'Это условный эффект одного входа фиксированного решения,не доказательство того,что другой обученный сшиватель безMASA не может дать ту же пользу. '
            'В старомраздельнообученномсравнении '+oldsource+':diff.base_masa-base|test|HOTA +0,030[−0,228;0,298],наval7 +0,272[−0,064;0,640]:дополнительная польза MASA над обученной базовой сшивкой там не подтверждена. '
            'Готовая связка исправляет фрагментациюID,а не детекции;источникконкретнойулучшеннойсвязинельзяопределятьлишьпообщемуIDSw.',
        decision='СохранитьодинпонятныйMASAузелподБ1свыводомиздвухразныхтиповконтролей. Необъявлятьвсюприбавкууникальнойзаслугойоднойсети. '
            'Текущийразборзавершённауровнепроверкикомпонентазамороженнойсвязки;новыепокадровыевариантынезапускать. По последнему прямомупорядкуавтора перейти кБ1→Б2→Б3,неповторятьготовоеобучение.')
    for criterion in card['acceptance']:criterion.update(met=True,evidence=source+':paired.full_vs_*;runs/recovery_research_20261027/masa_feature_contribution/independent_audit.json:passed=true')
    card['metrics']=[dict(key='HOTA',value=r['pooled']['full']['HOTA'],baseline=r['pooled']['neutral_masa']['HOTA'],subset='20 historical val, fixed-head feature intervention',source=source,locator=['pooled','full','HOTA'])]
    card['confirmed_results']=[dict(id='correct_masa_feature_vs_neutral_fixed_head20',title='Правильный признак MASA полезен в данной фиксированной голове',category='tracker',metric='HOTA',unit='points',
        comparison='Настоящий MASA-признак против нейтрального входа при неизменных голове/порогах/кандидатах/остальныхпризнаках',subset='20 исторических val',
        conditions='Точный повтор первоначальных20MASAJSON. Нейтральныйвход—среднееобучающегоscaler;контролиперестановок42/43/44такжехуже,ничегонепереучивалось.',
        limitations='Условнаяпроверказамороженнойголовы,неравноёмкоепереобучениебезMASA,невсеобщаянеобходимостьилидостаточность. Историческийvalненовыйнезависимыйтест. Старыйотдельнообученныйконтрольнадругихданныхнеподтвердилпревосходстваbase_masaнадbase.',
        evidence=dict(source=source,sha256=sha(ROOT/source),locator=['paired','full_vs_neutral_masa']))]
    card['evidence'] += [dict(kind='file',ref=source,claim='Все20причинныеконтролейодноговходафункцииготовойголовы'),dict(kind='file',ref='runs/recovery_research_20261027/masa_feature_contribution/independent_audit.json',claim='Все80контрольныхвыходов/полныеоценки/CIпроверены'),
        dict(kind='file',ref='runs/recovery_research_20261027/masa_feature_contribution/historical_separate_training_context.json',claim='ОграничениесильнымранееобученнымконтролембезMASA')]
    assert path.read_bytes()==before;write(path,cat)
    journal(card['id'],'разбор показал вклад правильного MASA-признака в фиксированной связке',
        '*Получено.* '+card['obtained']+'\n\n*Вывод.* '+card['interpretation']+'\n\n*Решение.* '+card['decision']+
        '\n\n*Проценты.* СовокупнаяHOTAfull58,485787/neutral57,900829:около+1,01%кданнойбазе. Максимумнеизвестен. '
        'Старый255отказрежимаBLASсохранён,256точноповторилвсеоценкибезослабленияпровероквостаткепрежнего300сокна. Нужныобязательныекаталог/тесты/обновлениеоднойMASAзаметки.')
    print('Recorded fixed-head feature contribution and contrary historical retrained comparison')

if __name__=='__main__':main()
