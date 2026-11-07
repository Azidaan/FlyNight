"""Зарегистрировать конкретное отделение вклада MASA, не новую гипотезу графа."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.recovery_research_service_20261027 import RUN,read,write,sha,now
from scripts.b123_safe_journal_20261027 import journal

def main():
    folder=RUN/'masa_feature_contribution';assert not folder.exists();old='runs/z13_val20_matched_20261024/native_masa_recovery2/report.json';r=read(ROOT/old)
    files=['scripts/run_masa_feature_contribution_20261027.py','scripts/stitch_eval_v2.py','scripts/eval_fly_reid.py',old,'runs/z13_center_hota_20261021/masa_head.joblib']
    for v in r['videos']:
        files += [f'runs/z13_val20_matched_20261024/xywh/{sub}/{v}.json' for sub in ['evaluated_raw','evaluated_masa']]
        files += [f'runs/z13_val20_matched_20261024/xywh/signatures/{v}/signatures.npz',f'runs/z13_val20_matched_20261024/xywh/masa/{v}.npz',f'runs/z13_val20_matched_20261024/xywh/dino_compat/{v}.npz',f'runs/z13_val20_matched_20261024/cache/{v}/alignment.json',f'data/bdd100k/labels/box_track_20/val/{v}.json']
    p=dict(id='masa_feature_contribution_v1',created=now(),budget_seconds=900,device='cpu',videos=r['videos'],author_stop=read(RUN/'authorization.json')['save_at'],
        action='Сохранить все пары/остальные признаки/замороженную голову/порог.6/жадную сшивку. Сравнить исходный полный MASA-признак с нейтральным (среднее обучающего scaler,нулевой стандартизованный вклад) и тремя перестановками его значений между теми же парами внутри каждого видео(seed42/43/44). Ничего не обучать/не выбирать по20val.',
        hypothesis='Измеренная прибавка всей связки может требовать корректного MASA-признака либо в основном объясняться другими признаками/сшивкой;вопрос проверяется на той же замороженной голове.',
        counterargument='Отключение одного признака у готовой головы не равно заново обученному сшивателю безMASA. Перестановка нарушает зависимости признаков;это проверка зависимости данного решения,не доказательство универсальной необходимости/достаточностиMASA.',
        acceptance=['Все20,полная исходная сшивка точно повторяет прежние20JSON/число связей;остальные входы/порог/голова/пары неизменны,unknownнефильтруютсяизкандидатов',
            'ПолнаяHOTA/AssA/IDF1/FP/FN/IDSw. Добавочный вклад корректногоMASAпризнака у данной головы подтверждён,еслинижняя95%парнаяграницаполнойверсиикнейтральнойивсем3перестановкам>0',
            'Независимое восстановление всех контрольных выходов и95%интервалов;частные положительные контрасты не скрывают общий непройденный критерий'],
        cost_gate='ПервоеПОЛНОЕвидео4контрольныхвыходов:2*time*19+120<остаток900с,иначестопбезпродления.',
        limitation='Последующая диагностическая проверка на20историческихval после просмотра полезной связки,не новый независимый итоговыйтест. Только CPU на готовых признаках;нет нового детектора/весов/GPU/обучения. Разметка используется только для оценки,не для кандидатов/вероятностей/выбора. Это не причинная абляция одинакового переобучения и не новый результатБ1–Б3.',sha256={f:sha(ROOT/f) for f in files})
    write(folder/'protocol.json',p)
    path=ROOT/'experiments/project_history.json';before=path.read_bytes();cat=json.loads(before);assert not any(c['id']==p['id'] for c in cat['experiments'])
    cat['experiments'].append(dict(id=p['id'],title='Отделить вклад признака MASA от остальных частей сшивателя',area='flyvis',state='ready',started='2026-10-27',question='Какая часть измеренной пользы готовой связки зависит от правильного MASA-признака?',
        hypothesis=p['hypothesis'],expected=p['action'],summary=p['counterargument'],obtained='',interpretation='',decision='',cannot_claim=p['limitation'],acceptance=[dict(criterion=c,met=None) for c in p['acceptance']],metrics=[],commits=[],files=files[:3],evidence=[dict(kind='file',ref=folder.relative_to(ROOT).as_posix()+'/protocol.json',claim='Конкретные контроли/критерии до сравнения20')]))
    assert path.read_bytes()==before;write(path,cat)
    write(RUN/'masa_feature_contribution_audit/protocol.json',dict(parent_experiment=p['id'],budget_seconds=300,author_stop=p['author_stop']))
    for name,sub,script in [('250_masa_feature_contribution','masa_feature_contribution',files[0]),('255_masa_feature_contribution_audit','masa_feature_contribution_audit','scripts/audit_masa_feature_contribution_20261027.py')]:
        write(RUN/(name+'_prepared.json'),dict(name=name,folder=(RUN/sub).relative_to(ROOT).as_posix(),script=script,script_sha256=sha(ROOT/script),protocol_sha256=sha(RUN/sub/'protocol.json')))
    journal(p['id'],'по прямому поручению сначала отделяем источник прибавки связки с MASA',
        '*Постановка.* '+p['action']+'\n\n*Возражение.* '+p['counterargument']+'\n\n*Критерии.* '+json.dumps(p['acceptance'],ensure_ascii=False)+' '+p['cost_gate']+
        ' Автор уточнил порядок:сначала разборMASA,затемБ1–Б3. Это одна проверка компонентов готовой связки на CPU,не новая покадроваяMASA и не новаяБ4. '+p['limitation'])
    print('Registered one CPU feature-contribution experiment before evaluation')

if __name__=='__main__':main()
