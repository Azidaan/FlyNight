"""Record a verified complete functional pilot or the actual saved deadline state."""
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
OUT=ROOT/'runs/z13_distribution_pilot_20261015'

def read(p): return json.loads(p.read_text(encoding='utf-8'))
def main():
    from tracking.z13_checkpoint import atomic_json
    marker=OUT/'recorded.json'
    assert not marker.exists(),'Already recorded'
    now=datetime.now(timezone(timedelta(hours=3)))
    path=ROOT/'experiments/project_history.json'; cat=read(path)
    c=next(x for x in cat['experiments'] if x['id']=='z13_distribution_pilot_v1')
    result_path=OUT/'report.json'
    if result_path.exists():
        import torch
        report=read(result_path)
        verified={}
        for arm in 'BCD':
            for loss in ('hard','spatial'):
                folder=OUT/arm/loss
                checkpoint=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False)
                assert checkpoint['step']==500 and checkpoint['optimizer']['state']
                assert read(folder/'completed.json')['backbone_unchanged']
                verified[arm+'_'+loss]=dict(step=checkpoint['step'],prediction_rows=len(read(folder/'predictions.json')),
                    checkpoint_rng_saved=all(k in checkpoint for k in ('generator','rng_python','rng_numpy','rng_torch','rng_cuda')))
        atomic_json(OUT/'final_verification.json',verified)
        source='runs/z13_distribution_pilot_20261015/report.json'
        for i,a in enumerate(c['acceptance']): a.update(met=report['criteria'][i],evidence=f'{source}: criteria[{i}]')
        c['state']='completed_positive' if report['all_gates_passed'] else 'completed_negative'
        c['done']=now.date().isoformat()
        d=report['comparisons']['D_spatial-D_hard']; b=report['comparisons']['D_spatial-B_spatial']
        c['obtained']=f"{source}: {json.dumps({'D_hard':report['arms']['D_hard'],'D_spatial':report['arms']['D_spatial'],'D_spatial-D_hard':d,'D_spatial-B_spatial':b,'criteria':report['criteria']},ensure_ascii=False)}. Все шесть продолжений по 500 шагов и прогнозы на прежних 26 train завершены; final_verification.json."
        c['interpretation']='Функциональный пилот на данных разработки; '+('критерии выполнены, нужна отдельная проверка применения и HOTA.' if report['all_gates_passed'] else 'все заранее записанные критерии вместе не выполнены; этот масштаб распределённой цели не получает автоматического продолжения.')
        c['decision']='Сохранить результат без изменения стенда, порогов и отложенных данных. Не перебирать ширину по этому результату. HOTA и подтверждение вклада FlyVis этим пилотом не установлены; окончание разрешённого окна не продлевается.'
        c['evidence'].extend([dict(kind='json',ref=source,claim='Парные функциональные результаты на train26'),dict(kind='json',ref='runs/z13_distribution_pilot_20261015/final_verification.json',claim='Шаги, сохранения и полнота всех шести вариантов')])
        text=f"Завершены все шесть вариантов по 500 шагов. D spatial−hard: доля полезных положений {d['useful_fraction_delta']:+.6f}, ДИ {d['ci95']}; разность среднего медиан относительной ошибки {d['mean_video_median_error_delta']:+.6f}. D spatial−B spatial: {b['useful_fraction_delta']:+.6f}, ДИ {b['ci95']}. Критерии {report['criteria']}. Источник — `{source}`."
    else:
        assert now>=datetime.fromisoformat('2026-10-15T19:55:00+03:00') or '--failed' in sys.argv,'Do not prematurely record a running pilot'
        state=read(OUT/'status.json') if (OUT/'status.json').exists() else {}
        completed=[str(x.relative_to(OUT)) for x in OUT.glob('*/*/completed.json')]
        atomic_json(OUT/'saved_window_state.json',dict(status=state,completed=completed,recorded=now.isoformat()))
        c['state']='inconclusive'
        c['obtained']='runs/z13_distribution_pilot_20261015/saved_window_state.json: '+json.dumps(dict(status=state,completed=completed),ensure_ascii=False)
        c['interpretation']='Полное парное сравнение не завершено; промежуточная ошибка обучения не является результатом полезности.'
        c['decision']='Состояние сохранено. Не сокращать число шагов и выборку для объявления итога; не продолжать после 20:00 без нового разрешения.'
        c['evidence'].append(dict(kind='json',ref='runs/z13_distribution_pilot_20261015/saved_window_state.json',claim='Фактический остаток и состояние'))
        text='Полное парное сравнение не завершено. Сохранённое состояние и готовые варианты — `runs/z13_distribution_pilot_20261015/saved_window_state.json`. Критерии полезности не выставлены по неполным результатам.'
    path.write_text(json.dumps(cat,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    plan=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md'; old=plan.read_text(encoding='utf-8'); anchor='## Журнал изменений\n\n'; assert old.count(anchor)==1
    entry=f"**{now:%d.%m.%Y, %H:%M} MSK** — итог окна пространственной цели З13\n\n*Постановка.* `z13_distribution_pilot_v1`: одинаковые продолжения обучения считывателей B/C/D с обычной и распределённой целью; ядра фиксированы; прежние train104/26; независимые видео и HOTA не открывались.\n\n*Получено.* {text}\n\n*Вывод.* {c['interpretation']}\n\n*Решение.* {c['decision']} Новая HOTA не измерена; процент к стенду и доля оракульного запаса неизвестны. Существующий `.git/index.lock` не удаляется.\n\n"
    plan.write_text(old.replace(anchor,anchor+entry,1),encoding='utf-8')
    atomic_json(marker,dict(recorded=now.isoformat(),state=c['state']))
    print(c['state'],text)

if __name__=='__main__': main()
