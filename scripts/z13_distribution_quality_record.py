"""Record launch or verified end of the quality preparation, never HOTA claims."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import json
import sys
import hashlib

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/z13_distribution_quality_20261015'
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def main():
    now=datetime.now(timezone(timedelta(hours=3)))
    stage='launch' if '--launch' in sys.argv else 'finish'
    marker=OUT/f'{stage}_recorded.json'
    assert not marker.exists(), 'Already recorded'
    catpath=ROOT/'experiments/project_history.json'; cat=read(catpath)
    c=next(c for c in cat['experiments'] if c['id']=='z13_distribution_quality_v1')
    if stage=='launch':
        controller=read(OUT/'controller.json'); state=read(OUT/'status.json')
        assert state.get('checkpoint_verified') and state['stage']=='oof_training'
        c['state']='running'
        text=(f"Автор разрешил один подготовленный запуск до двух часов. Старт {controller['started']}, мягкая остановка {controller['save_deadline']}, "
              f"предел {controller['deadline']}; независимый контроллер, один GPU-процесс. Первая запись проверенного сохранения: "
              f"{state['arm']} {state['fold']}, шаг {state['step']}. Источники: `runs/z13_distribution_quality_20261015/authorization.json`, "
              "`controller.json`, `status.json`. Обучаются считыватели разделённых выборок; полные локализаторы не переобучаются. HOTA/dev/отложенные видео не запускаются.")
    else:
        end=read(OUT/'controller_result.json'); assert end['own_worker_ended']
        result=OUT/'result.json'
        if result.exists() and end['exit_code']==0:
            import torch
            r=read(result); p=read(OUT/'protocol.json')
            verified={}
            for j in p['jobs']:
                folder=OUT/j['arm']/j['name']; done=read(folder/'completed.json')
                assert done['step']==500 and done['backbone_unchanged']
                assert hashlib.sha256((folder/'predictions.json').read_bytes()).hexdigest()==done['prediction_sha256']
            for arm in 'BCD':
                ck=torch.load(OUT/arm/'policy_model.pt',map_location='cpu',weights_only=False)
                assert ck['format']=='z13-spatial-hota-policy-v1' and ck['tau']['tau']==r['arms'][arm]['policy']['tau']
                assert all(torch.isfinite(t).all() for key in ('readout','quality','backbone') for t in ck[key].values())
                verified[arm]=r['arms'][arm]
            (OUT/'final_verification.json').write_text(json.dumps(verified,ensure_ascii=False,indent=2),encoding='utf-8')
            c['state']='infrastructure'
            for a in c['acceptance'][1:]: a.update(met=True,evidence='runs/z13_distribution_quality_20261015/final_verification.json')
            c['obtained']='runs/z13_distribution_quality_20261015/result.json: '+json.dumps(r,ensure_ascii=False)
            c['interpretation']='Завершена техническая подготовка трёх согласованных моделей и фиксированных порогов. Доли полезных поправок относятся только к прежним train26; это не HOTA и не независимое подтверждение.'
            c['decision']='Сохранить модели. Следующий этап — отдельное согласование применения и бюджета HOTA, без автоматического запуска.'
        else:
            state=read(OUT/'status.json') if (OUT/'status.json').exists() else {}
            c['state']='inconclusive'
            c['obtained']='runs/z13_distribution_quality_20261015/controller_result.json: '+json.dumps(end)+'; status.json: '+json.dumps(state)
            c['interpretation']='Подготовка не завершена полностью; частичные результаты не доказывают полезность.'
            c['decision']='Сохранить остаток. Не продолжать вне разрешённого окна и не менять настройки по частичным результатам.'
        c['done']=now.date().isoformat()
        text=c['obtained']+' '+c['interpretation']+' '+c['decision']
    catpath.write_text(json.dumps(cat,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    plan=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md'; old=plan.read_text(encoding='utf-8'); anchor='## Журнал изменений\n\n'; assert old.count(anchor)==1
    entry=f'**{now:%d.%m.%Y, %H:%M} MSK** — {stage}: оценка качества пространственной цели З13\n\n`z13_distribution_quality_v1`. {text}\n\nНовой HOTA нет; процент изменения стенда и доля оракульного запаса неизвестны.\n\n'
    plan.write_text(old.replace(anchor,anchor+entry,1),encoding='utf-8')
    marker.write_text(json.dumps(dict(recorded=now.isoformat(),state=c['state']),indent=2),encoding='utf-8')
    print(text)

if __name__=='__main__': main()
