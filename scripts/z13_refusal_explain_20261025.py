"""Explain saved refusal outcomes using original labels and one exact box example."""
from pathlib import Path
from collections import Counter
import sys
import subprocess
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_refusal_causes_20261025 import RUN,OLD,CACHE,IDENT,read,write,sha,now,journal


def main():
    render_only='--render-only' in sys.argv
    assert render_only or not (RUN/'explanation.json').exists()
    p=read(OLD/'protocol.json');labels=read(CACHE/'evaluate/labels.json')
    decision={r['label']:r for r in read(RUN/'rows.json') if r['arm']=='D'}
    selected_kind=Counter();negative_reasons=Counter();details=[];base_episode=0
    for video in p['evaluate26']:
        source=ROOT/f'runs/z13_data/{video}.npz'
        assert sha(source)==p['sha256'][source.relative_to(ROOT).as_posix()]
        with np.load(source) as z:
            ep=z['lab_ep'];pos=z['lab_pos'];kind=z['lab_kind'];center=z['lab_center'];crop=z['lab_crop'];n=len(z['ep_tid'])
        lookup={(int(e),int(t)):i for i,(e,t) in enumerate(zip(ep,pos))}
        for l in [x for x in labels if x['video']==video]:
            row=decision[l['label']]
            if not row['selected']:continue
            i=lookup[(l['episode']-base_episode,l['pos'])]
            assert int(kind[i])==row['kind']
            selected_kind[str(row['kind'])]+=1
            reason='unknown' if row['kind']<0 else 'target_in_eye' if row['kind']==1 else None
            if row['kind']==0:
                if not np.isfinite(center[i]).all():
                    reason='target_not_annotated_in_frame'
                else:
                    x,y=center[i];a,b,c,d=crop[i]
                    reason='target_outside_square_crop' if not(a<=x<=c and b<=y<=d) else 'target_outside_hexagonal_eye_inside_square'
                negative_reasons[reason]+=1
            details.append(dict(uid=row['uid'],video=video,label=l['label'],kind=row['kind'],reason=reason,score=row['score'],tau=row['tau']))
        base_episode+=n
    assert sum(selected_kind.values())==84 and sum(negative_reasons.values())==62
    trace_report=read(RUN/'trace/report.json')
    # Deterministic example: first harmful rank0 with a correctly labelled high detection.
    options=sorted([r for r in trace_report['old_center_useful_harmful'] if r['rank']==0 and r['correct_high_detection_indices']],key=lambda r:r['uid'])
    example=options[0]
    job=read(RUN/'trace/videos'/(example['video']+'.json'))
    row=next(r for r in job['rows'] if r['uid']==example['uid']);tr=row['trace']
    label=next(l for l in labels if l['label']==row['label'])
    center=np.array(tr['candidate_centers']['D'][0]);mean=np.array(tr['query_mean']);gt=np.array(label['gt'])
    detection=tr['high_detections'][example['correct_high_detection_indices'][0]]
    example.update(gt_center=gt.tolist(),predicted_center=mean[:2].tolist(),proposed_center=center.tolist(),
        distance_gt_before=float(np.linalg.norm(mean[:2]-gt)),distance_gt_after=float(np.linalg.norm(center-gt)),
        correct_cost_before=tr['baseline_cost_row'][example['correct_high_detection_indices'][0]],
        correct_cost_after=tr['candidate_cost_rows']['D'][0][example['correct_high_detection_indices'][0]],
        match_threshold=tr['match_threshold'],fixed_policy_selected=decision[row['label']]['selected'],
        correct_detection=detection)
    output=dict(at=now().isoformat(),selected_D_label_kinds=dict(selected_kind),
        selected_D_negative_reasons=dict(negative_reasons),selected_D_details=details,
        exact_box_example=example,source_sha256={
            'rows.json':sha(RUN/'rows.json'),'trace/report.json':sha(RUN/'trace/report.json'),
            (OLD/'protocol.json').relative_to(ROOT).as_posix():sha(OLD/'protocol.json')},
        limits='Target not annotated is not proof of physical absence; square/hexagonal labels mean local visibility only. Harmful box example is a candidate counterfactual rejected by the fixed policy, not an actually applied runtime error. No HOTA claim.')
    if not render_only:
        write(RUN/'explanation.json',output)
    else:
        assert output['exact_box_example']==read(RUN/'explanation.json')['exact_box_example']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig,ax=plt.subplots(figsize=(9,7),layout='constrained')
    pred=np.r_[mean[:2]-mean[2:]/2,mean[2:]]
    alt=np.r_[center-mean[2:]/2,mean[2:]]
    det=np.array(detection['xyxy'])
    for rect,color,labeltext,style in [(pred,'#386cb0','Рамка прогноза до поправки','--'),(alt,'#d95f02','Рамка с предложенным центром','-')]:
        ax.add_patch(Rectangle(rect[:2],rect[2],rect[3],fill=False,edgecolor=color,linewidth=2.7,linestyle=style,label=labeltext))
    ax.add_patch(Rectangle(det[:2],det[2]-det[0],det[3]-det[1],fill=False,edgecolor='#238b45',linewidth=2.7,label='Правильная детекция YOLO'))
    ax.scatter(*gt,c='black',marker='*',s=160,label='Центр по разметке',zorder=6)
    ax.annotate('',xy=center,xytext=mean[:2],arrowprops=dict(arrowstyle='->',lw=2.5,color='#d95f02'))
    ax.scatter(*mean[:2],c='#386cb0',s=50,zorder=5);ax.scatter(*center,c='#d95f02',s=50,zorder=5)
    ax.set(xlim=(-20,65),ylim=(445,315),aspect='equal',xlabel='Координата X, пиксели',ylabel='Координата Y, пиксели')
    ax.axvline(0,color='gray',lw=1);ax.grid(alpha=.18)
    ax.legend(loc='upper right',fontsize=10)
    ax.set_title('Центр ближе к разметке, но правильная детекция теряется\n'
        f'Ошибка центра: {example["distance_gt_before"]:.2f} → {example["distance_gt_after"]:.2f} пикс.\n'
        f'Стоимость: {example["correct_cost_before"]:.4f} → {example["correct_cost_after"]:.4f}; допустимо ≤ {example["match_threshold"]:.1f}',fontsize=13)
    fig.text(.5,-.035,f'{example["video"]}, кадр {example["frame"]}, трек {example["tid"]}. Проверка кандидата; фиксированный порог его отверг.',ha='center',fontsize=9)
    fig.savefig(RUN/'box_counterexample.png',dpi=160,bbox_inches='tight')
    fig.savefig(RUN/'box_counterexample.svg',bbox_inches='tight')
    clean_svg()
    plt.close(fig)
    if render_only:
        return
    cp=ROOT/'experiments/project_history.json';cat=read(cp);card=next(c for c in cat['experiments'] if c['id']==IDENT)
    old=card['obtained'];head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()
    card.setdefault('revisions',[]).append(dict(date=now().strftime('%Y-%m-%d'),commit=head,field='obtained',old=old,reason='Уточнён смысл отрицательной метки и проверен конкретный механизм потери правильной детекции.'))
    card['obtained']=old+' runs/z13_refusal_causes_20261025/explanation.json:выбранныеDпоkind='+str(dict(selected_kind))+';отрицательные='+str(dict(negative_reasons))+'. Контрпример UID'+str(example['uid'])+':ошибкацентра'+str(example['distance_gt_before'])+'→'+str(example['distance_gt_after'])+'px,стоимостьправильнойдетекции'+str(example['correct_cost_before'])+'→'+str(example['correct_cost_after'])+'при пороге0,8. Это отвергнутыйкандидат,нефактическоевмешательствополитики.'
    card['evidence'].append(dict(kind='file',ref='runs/z13_refusal_causes_20261025/explanation.json',claim='Смысл отрицательных меток и точный контрпример геометрии'))
    write(cp,cat)
    journal('уточнение:порог выбирает пики вне видимой цели;контрпример рамки',
        '*Получено.* runs/z13_refusal_causes_20261025/explanation.json:84выбранныхD → '+str(dict(selected_kind))+
        ';62отрицательных → '+str(dict(negative_reasons))+'. Отсутствиеаннотации неравнофизическомуотсутствию;неизвестные20неполучилиисход. '
        'Контрпример UID'+str(example['uid'])+':центрпоевклидовомурасстояниюсталближе,ноиз-зафиксированногоразмерарамкистоимостьправильнойYOLOдетекциипересекла0,8. '
        'Рисунокbox_counterexample.png/svgсобранпокоординатампервичногоtrace. Это отвергнутыйпорогомкандидат,непричинаизмереннойHOTA. '
        '*Вывод.* Высокаяотносительнаявершинакартысама неидентифицируетцелевое присутствиеипользуназначению;размеррамкииизмерениедетектораважныдляпоследствия. '
        'Следующаяправкадолжнаразделитьналичиецели,преимуществонадисходнымпрогнозомипригодностьпредлагаемойдляназначениярамки. Нельзяобъявлятьузелотказаисправленнымпоэтойдиагностике.')
    print(str({k:v for k,v in output.items() if k!='selected_D_details'}))


def clean_svg():
    path=RUN/'box_counterexample.svg'
    path.write_text('\n'.join(line.rstrip() for line in path.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')


if __name__=='__main__':
    if '--trim-svg' in sys.argv:
        clean_svg()
    else:
        main()
