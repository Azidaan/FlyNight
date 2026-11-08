"""Record only a completed, independently verified full result; preserve revisions."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_identity_assignment_run_20261108.common import *
RUN=ROOT/'runs/identity_evidence_20261108'
STUDY=ROOT/'runs/bbox_normalized_signals_20261108'


def ru(value,places=6):return f'{value:.{places}f}'.replace('.',',')


def main():
    full=STUDY/'full';assert read(full/'queue_result.json')['passed'];r=read(full/'comparison_report.json');audit=read(full/'independent_audit.json');assert audit['comparison_sha256']==sha(full/'comparison_report.json')
    for path in (full/'children').glob('*.ended.json'):
        c=read(path);assert c['returncode']==0 and c['all_owned_ended'] and c['guardian']['reason'] is None
    cat=read(ROOT/'experiments/project_history.json');card=next(e for e in cat['experiments'] if e['id']=='bbox_singleton_normalized_maps_v1_20261108');stamp=datetime.now(MSK);date=stamp.date().isoformat();image=r['comparisons']['IMAGE-B0'];fly=r['comparisons']['FLYVIS-B0'];point=r['pooled'];base=point['B0'];accepted=bool(r['accepted'].get('IMAGE',False));state='completed_positive' if accepted else 'inconclusive' if image['harm_passed'] and image['ci_corrected'][0]<=0<=image['ci_corrected'][1] else 'completed_negative'
    def change(field,value):
        old=card.get(field,'')
        if old==value:return
        if old:card.setdefault('revisions',[]).append(dict(date=date,commit='',field=field,old=old,reason='Полный20valпилот завершён и независимо проверен; промежуточная формулировка сохранена.'))
        card[field]=value
    gain=r['identity_gain']['IMAGE'];comparison='runs/bbox_normalized_signals_20261108/full/comparison_report.json';independent='runs/bbox_normalized_signals_20261108/full/independent_audit.json';image_rates={a:r['comparisons']['IMAGE-'+a] for a in ['RANDOM_IMAGE','PERM_IMAGE']}
    observed=f"Полный20val: HOTA B0{ru(base['HOTA'])},IMAGE{ru(point['IMAGE']['HOTA'])},разность{ru(image['delta_pooled_HOTA'])}п.п.;95%ДИ[{ru(image['ci95'][0])};{ru(image['ci95'][1])}],поправленный20ДИ[{ru(image['ci_corrected'][0])};{ru(image['ci_corrected'][1])}]. IMAGE: сменID{image['harm']['IDSw']:+d},FP{image['harm']['FP']:+d},FN{image['harm']['FN']:+d},известныхчужихвозвратов{image['harm']['wrong_known_returns']:+d};правильныхвозвратов{gain['correct_returns']:+d},устойчивых{gain['sustained_success']:+d}. FlyVis−B0{ru(fly['delta_pooled_HOTA'])}п.п.,95%ДИ[{ru(fly['ci95'][0])};{ru(fly['ci95'][1])}]. ПринятиезаранеезаданнойальтернативыIMAGE={accepted};основнойSUPPORTнепрошёлусловнуюпроверкуи не продвигается ({comparison}; {independent})."
    supplement=read(RUN/'image_geometry_comparison.json');assert supplement['full_report_sha256']==sha(full/'comparison_report.json');diagnostic=read(RUN/'return_count_diagnostic.json');assert diagnostic['passed'] and diagnostic['delta_correct_return_count']==gain['correct_returns']
    rate_text=' '.join(f"IMAGE−{a}: {ru(v['delta_pooled_HOTA'])}п.п.,поправленныйДИ[{ru(v['ci_corrected'][0])};{ru(v['ci_corrected'][1])}]." for a,v in image_rates.items())
    geo=supplement['comparisons']['IMAGE-BBOX'];rate_text+=f" IMAGE−геометрия: {ru(geo['delta_pooled_HOTA'])}п.п.,поправленныйДИ[{ru(geo['ci_corrected'][0])};{ru(geo['ci_corrected'][1])}] (runs/identity_evidence_20261108/image_geometry_comparison.json)."
    interpretation=('Заранее заданный короткий вариант IMAGE прошёл критерии полного пилота.' if accepted else 'Заранее заданный вариант IMAGE не прошёл условия полного пилота.')+' '+rate_text+' Уменьшение счётчика правильных возвратов на1не означает потерю этого правильного продолжения: единственный исчезнувший случай b23b2649-8349d2a1,кадр544,ID100 стал непрерывной выдачей того же публичного номера через нативныйID496. Счётчик событий возврата немонотонен при устранении разрыва; исходные критерии не меняются (runs/identity_evidence_20261108/return_count_diagnostic.json). Даже без этого ограничения прибавкаHOTAи преимущество изображения над согласованными контролями не доказаны. Польза всей политики, причина эффекта изображения и добавка FlyVis — отдельные выводы. Согласование частоты случайных контролей выполнено наfit; на20valчастоты не подгонялись. Ноль дополнительных физических рамок обеспечивается сохранением штатной выдачи, но сам по себе не доказывает сохранение личности.'
    decision=('Сохранить положительный результат короткого пилота, но первую проблему целиком не закрывать: давность2означает один промежуточный пропущенный кадр.' if accepted else 'Не принимать эту версию как решение первой проблемы; не повторять законченный полный пилот.')+' Проблемы2–4отложены. Перед следующим опытом отдельно разобрать сохранённые правильные/чужие продолжения и разницу с согласованным случайным отказом; пороги по этим20valне менять.'
    card['state']=state;card['done']=date;change('obtained',observed);change('interpretation',interpretation);change('decision',decision);change('cannot_claim','Исторические20val,не независимый итоговый тест. Пилот охватывает давность2—один промежуточный пропуск. Его результат не закрывает все длительные перекрытия, конкуренцию двух уже рабочих ID или все способы пространственного считывания. Относительные координаты рамок не гарантируют совпадение физических деталей. Положительная условная связь двух нативных фрагментов не доказывает личность сырой рамки в каждый неразмеченный момент. Ядро FlyVis не менялось; пользу всей системы нельзя автоматически приписать ядру.')
    card['acceptance'][2].update(met=accepted,evidence=comparison+'; '+independent)
    for n,c in [(comparison,'Полный зарегистрированный результат и все контроли'),(independent,'Независимая проверка физических выходов, метрик и возвратов')]:
        if not any(e['ref']==n for e in card['evidence']):card['evidence'].append(dict(kind='file',ref=n,claim=c))
    write(ROOT/'experiments/project_history.json',cat)
    plan=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';text=plan.read_text(encoding='utf-8');entry=f'\n\n**{stamp.strftime("%d.%m.%Y, %H:%M MSK")}** — Первая проблема: полный результат относительного выравнивания и отдельного подтверждения личности.\n\n*Получено (`bbox_singleton_normalized_maps_v1_20261108`).* {observed}\n\n*Вывод.* {interpretation}\n\n*Решение.* {decision} Готовые траектории, карты и метрики не повторять. Научная постановка не изменена по20val; исходники и неудачные прежние варианты сохранены.\n';marker='## Журнал изменений';i=text.index(marker)+len(marker);plan.write_text(text[:i]+entry+text[i:],encoding='utf-8')
    write(RUN/'normalized_full_record_receipt.json',dict(passed=True,recorded=stamp.isoformat(),full_report_sha256=sha(full/'comparison_report.json'),full_audit_sha256=sha(full/'independent_audit.json'),catalog_sha256=sha(ROOT/'experiments/project_history.json'),plan_sha256=sha(plan),card_state=state,short_pilot_accepted=accepted,first_problem_closed=False,other_problems_deferred=[2,3,4]),new=True)


if __name__=='__main__':main()
