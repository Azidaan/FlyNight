"""Record authorized recovery and its eventual outcome, preserving prior conclusions."""
from pathlib import Path
from datetime import datetime
import sys,json,subprocess,hashlib
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'runs/z13_distribution_quality_20261015'
NEW=ROOT/'runs/z13_candidates_20261016'
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    finish='--finish' in sys.argv
    marker=NEW/('resume_finish_recorded.json' if finish else 'resume_launch_recorded.json')
    assert not marker.exists()
    now=datetime.now().astimezone()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    catpath=ROOT/'experiments/project_history.json'; cat=read(catpath)
    def replace(card,field,value):
        if field in card and card[field]!=value:
            card.setdefault('revisions',[]).append(dict(date=now.date().isoformat(),commit=commit,field=field,old=card[field],reason='Возобновление после технического сбоя, разрешённое автором; старый итог сохранён'))
        card[field]=value
    if not finish:
        assert read(OLD/'recovery_verified.json')['passed']
        assert read(NEW/'resume_quality_result.json')['exit_code']==0
        import torch
        r=read(OLD/'result.json')
        for arm in 'BCD':
            ck=torch.load(OLD/arm/'policy_model.pt',map_location='cpu',weights_only=False)
            assert ck['tau']['tau']==r['arms'][arm]['policy']['tau']
            assert all(torch.isfinite(t).all() for key in ('readout','quality','backbone') for t in ck[key].values())
            assert sha(OLD/arm/'policy_model.pt')==read(OLD/'recovery_verified.json')['policies'][arm]
        card=next(c for c in cat['experiments'] if c['id']=='z13_distribution_quality_v1')
        card['state']='infrastructure'
        for a in card['acceptance'][1:]: a.update(met=True,evidence='runs/z13_distribution_quality_20261015/recovery_verified.json')
        replace(card,'obtained','runs/z13_distribution_quality_20261015/result.json: '+json.dumps(r,ensure_ascii=False))
        replace(card,'interpretation','Техническая подготовка завершена после явного переноса оценки качества на CPU. Среди известных исходов верхнего 1% полезны B: 0/67, C: 14/40, D: 16/38 (result.json → arms); неизвестных 17/44/46 соответственно. Это повторно использованные train26, не HOTA и не доказательство вклада ядра. Правило 90%/50 не пройдено ни одним вариантом.')
        replace(card,'decision','Сохранить модели и исходный журнал ошибки. HOTA не запускать автоматически. Разрешённые отдельные функциональные пробы продолжены до 07:40 16.10.2026.')
        card['evidence'].append(dict(kind='json',ref='runs/z13_distribution_quality_20261015/recovery_verified.json',claim='Проверенное восстановление CPU-этапа без повторного обучения ядер/считывателей'))
    report=read(NEW/'report.json') if (NEW/'report.json').exists() else None
    if finish: assert read(NEW/'resume_controller_result.json')['own_worker_ended']
    for mode in ('foreground','size','adaptive'):
        card=next(c for c in cat['experiments'] if c['id']=='z13_candidate_'+mode+'_v1')
        if not finish:
            card['state']='running'; card.pop('done',None)
            replace(card,'obtained','Возобновлён после технического исправления: runs/z13_candidates_20261016/resume_controller.json. Научных результатов пока нет.')
            replace(card,'interpretation','Предыдущая остановка была до GPU; она не является отрицательным итогом узла.')
            replace(card,'decision','Продолжить прежний протокол в разрешённом окне; сохранение 07:35, окончание до 07:40 16.10.2026.')
        else:
            card['done']=now.date().isoformat()
            if report:
                card['state']='completed_positive' if report['passed'][mode] else 'completed_negative'
                for a,met in zip(card['acceptance'],report['criteria'][mode]): a.update(met=met,evidence='runs/z13_candidates_20261016/report.json → criteria.'+mode)
                replace(card,'obtained','runs/z13_candidates_20261016/report.json: '+json.dumps(dict(criteria=report['criteria'][mode],stats={k:v for k,v in report['stats'].items() if k.endswith('_'+mode)},comparisons=report['comparisons']),ensure_ascii=False))
                replace(card,'interpretation','Функциональный пилот на прежних train26; результат не является приростом HOTA или независимым подтверждением.')
                replace(card,'decision','Сохранить результат конкретной реализации; научные настройки не менять по исходу. Следующий дорогой этап отдельно.')
            else:
                card['state']='inconclusive'
                replace(card,'obtained','runs/z13_candidates_20261016/resume_controller_result.json: '+json.dumps(read(NEW/'resume_controller_result.json')))
                replace(card,'interpretation','Полный набор сравнений не завершён; эффективность не установлена.')
                replace(card,'decision','Сохранить готовое и контрольные точки; не продолжать за пределом разрешённого окна.')
    catpath.write_text(json.dumps(cat,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    plan=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md'; text=plan.read_text(encoding='utf-8'); anchor='## Журнал изменений\n\n'; assert text.count(anchor)==1
    detail=('CPU-этап восстановлен в отдельном процессе без изменения замороженного кода и научных настроек; сохранённое GPU-обучение не повторялось. Проверенные модели и пороги: runs/z13_distribution_quality_20261015/recovery_verified.json и result.json. Новая очередь продолжена после успешной проверки CPU-этапа, отдельный контроллер runs/z13_candidates_20261016/resume_controller.json. По требованию автора в AGENTS.md, раздел 9, записаны самостоятельное устранение технических сбоев в разрешённом окне, проверяемое возобновление и явная проверка перехода CPU/CUDA. Простой после установленной причины был ошибкой агента. Прежние коды завершения и тексты сохранены.' if not finish else 'Итог возобновлённой очереди: runs/z13_candidates_20261016/resume_controller_result.json. '+('Полные сравнения: report.json.' if report else 'Полного отчёта нет, записан реальный остаток.'))
    entry=f'**{now:%d.%m.%Y, %H:%M} MSK** — '+('итог возобновления' if finish else 'исправление CPU/CUDA и продолжение без простоя')+'\n\n'+detail+'\n\nHOTA не измерялась; процент к стенду не определён.\n\n'
    plan.write_text(text.replace(anchor,anchor+entry,1),encoding='utf-8')
    marker.write_text(json.dumps(dict(at=now.isoformat(),finish=finish),indent=2),encoding='utf-8')
    print('Recorded recovery',finish)
if __name__=='__main__': main()
