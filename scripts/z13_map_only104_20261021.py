"""Prospective map-only branch using already planned train104 control models.

The geometry ablation removes raw activity vectors, not the FlyVis response
map. This branch is registered before train104 models exist; old train20
comparisons are explicitly posthoc and do not change their failed primary gate.
"""
from __future__ import annotations
import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,sha,verify,now,MSK
from scripts.z13_continue import paired,per_video_useful,coverage_table
from scripts.z13_train import success
RUN=ROOT/'runs/z13_map_only104_20261021';SOURCE=ROOT/'runs/z13_candidate_data104_20261021';OLD=ROOT/'runs/z13_candidate_ranker_20261021'
IDENT='z13_map_only104_v1';ATTEMPT='full';LIMIT=120;T0=time.monotonic()


def journal(title,text):
    path=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';s=path.read_text(encoding='utf-8');a='## Журнал изменений\n';assert s.count(a)==1
    path.write_text(s.replace(a,a+f'\n**{now():%d.%m.%Y, %H:%M} MSK** — {title} `{IDENT}`\n\n'+text+'\n',1),encoding='utf-8')
def update(card,field,value,head,reason):
    old=card.get(field,'')
    if old and old!=value:card.setdefault('revisions',[]).append(dict(date=f'{now():%Y-%m-%d}',commit=head,field=field,old=old,reason=reason))
    card[field]=value


def prepare():
    if RUN.exists():raise RuntimeError('Existing branch must be inspected')
    if (SOURCE/'report.json').exists() or (SOURCE/'C/geometry/seed0/model.pt').exists():raise RuntimeError('This branch must be registered before train104 efficacy/models')
    base=read(SOURCE/'protocol.json');verify(base);post=OLD/'map_only_posthoc.json';assert post.exists()
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();cp=ROOT/'experiments/project_history.json';cat=read(cp)
    parent=next(c for c in cat['experiments'] if c['id']=='z13_candidate_ranker_v1');audit=read(post);fmt=lambda v:f'{100*v:+.3f}'.replace('.',',')
    extra=' Дополнительный вторичный разбор после просмотра исходов: runs/z13_candidate_ranker_20261021/map_only_posthoc.json → C map-only−top1 '+fmt(audit['paired']['C_map_only-top1']['mean_per_video'])+'п.п.; D '+fmt(audit['paired']['D_map_only-top1']['mean_per_video'])+'п.п.; R1 '+fmt(audit['paired']['R1_map_only-top1']['mean_per_video'])+'п.п. Geometry сохраняет пики/высоты карты FlyVis, обнуляются лишь дополнительные векторы активности. Исходный основной критерий не меняется.'
    update(parent,'obtained',parent['obtained']+extra,head,'Вторичный разбор карты как отдельного канала; прежний основной итог сохранён.')
    update(parent,'cannot_claim',parent['cannot_claim']+' Контроль geometry не является полным исключением FlyVis: его кандидаты и высоты пиков получены из карты мухи. Общий эффект считывателя отделяется от вклада мухи сопоставимым R1.',head,'Уточнён смысл абляции без изменения исходного вердикта.')
    parent['evidence'].append(dict(kind='file',ref=post.relative_to(ROOT).as_posix(),claim='Вторичный разбор после просмотра; не первичный критерий'))
    primary_report=OLD/'report.json';digest=sha(post);baselinesha=sha(primary_report)
    for arm in ('C','D'):
        k=arm+'_map_only-top1';obj=audit['paired'][k]
        if obj['ci95'][0]>0:
            parent['confirmed_results'].append(dict(id=arm+'_map_only_vs_top1_posthoc',title='Выбор по карте и геометрии против максимума — '+arm,category='readout',metric='useful_center_fraction',unit='fraction',evidence=dict(source=post.relative_to(ROOT).as_posix(),locator=['paired',k],sha256=digest),baseline=dict(source=primary_report.relative_to(ROOT).as_posix(),locator=['summary',arm+'_top1','useful_mean_video'],sha256=baselinesha),subset='train26; вторичный разбор после просмотра',comparison='Те же три пика; используются высоты/положения карты, дополнительные векторы активности обнулены.',conditions='Фиксированный ансамбль3 сидов. Основной критерий исходной карточки не выполнен.',limitations='Не заранее заданный главный результат. R1 получает такой же считыватель и также улучшается; нельзя весь прирост приписать мухе. Нужна отдельная проверка, не HOTA.'))
    p=dict(id=IDENT,created=now().isoformat(),registration='Before any train104 model training/output. Extra84 TRAIN cache is in progress. No new training, model selection or setting changes are introduced by this branch.',
        action='Evaluate the pre-existing geometry control as a map-reading node: it still receives peak responses/positions from FlyVis, but no extra raw activity vectors. Use only predeclared train104 models and all3 fixed seeds, not choose20 vs104 or best seed.',
        primary='C104 map-only-C original centered top1: mean>=0.02 and paired-video95%lower>0. C104 map-only-R1_104 map-only and C104 map-only-C nearest-prediction candidate: both lower>0. All3 C seed gains over original top1 nonnegative points. All required.',
        secondary='D same contrasts; C104 vs C20 map-only; difference-in-differences of readout effects C versusR1 is reported to separate generic improvement, not hidden if unfavorable. This is exploratory on repeatedly used train26; transfer needed.',
        counterargument='C map-only performance may come from generic geometry; R1 also benefits. Removing raw vectors may discard useful information or fail at larger-data validation. No cause is asserted from the old ablation.',
        budget_seconds=120,save_at='2026-10-21T07:35:00+03:00',hard_stop='2026-10-21T07:39:30+03:00',
        sha256={**base['sha256'],'scripts/z13_map_only104_20261021.py':sha(Path(__file__).resolve()),post.relative_to(ROOT).as_posix():sha(post)})
    write(RUN/'protocol.json',p)
    cat['experiments'].append(dict(id=IDENT,title='З13: считывание пиков карты без дополнительных векторов активности',area='flyvis',state='planned',started=f'{now():%Y-%m-%d}',tags=['fly-problem-P8','engineering-future'],
        question='Даёт ли считывание карты FlyVis без дополнительной сырой активности полезный выбор центра сверх максимума, ближайшего к прогнозу пика и такого же узла на R1?',
        hypothesis='Самой карты отклика может быть достаточно для нового выбора центра; дополнительная ветвь активности не является обязательным каналом вклада FlyVis.',
        expected='Только заранее заданные104train geometry-модели из однофакторного опыта, те же3 сида и критерии до их обучения. Никакого выбора между20/104 по исходам; новых моделей не обучается. Анализ всех26 после полного завершения источника, CPU до120с.',
        acceptance=[dict(criterion='Регистрация до обучения104, те же8326 строк/три сида, исходные хеши и завершение источника проверены.',met=None),dict(criterion='C104 map-only−C top1: средний сдвиг>=2п.п. и нижняя95%повидеоДИ>0; точки3 сидов неотрицательны.',met=None),dict(criterion='C104 map-only лучше R1_104 map-only и ближайшего к прогнозу пика C: обе нижние95%повидеоДИ>0.',met=None)],
        summary='Отдельная ветвь считывания карты зарегистрирована до обучения104; исход не получен.',done='',obtained='',interpretation='',decision='',cannot_claim='Map-only/geometry сохраняет сигнал мухи в карте; это не контроль без FlyVis. Контроль без FlyVis — тот же узел на R1. Вторичные20train сравнения не превращаются в заранее заданный главный результат; HOTA и безопасный отказ не измерены.',metrics=[],commits=[],files=['scripts/z13_map_only104_20261021.py','tracking/z13_candidate_ranker.py'],evidence=[dict(kind='file',ref='runs/z13_map_only104_20261021/protocol.json',claim='Регистрация до обучения104'),dict(kind='file',ref=post.relative_to(ROOT).as_posix(),claim='Вторичный исход, мотивирующий самостоятельную проверку')]))
    write(cp,cat);journal('уточнён канал карты и зарегистрирована ветвь до обучения104','*Проверено.* '+extra+'\n\nЭто не пересмотр непройденного главного критерия старой карточки. Сырые векторы активности и карта ответа — разные каналы; geometry сохраняет второй. Во вторичном разборе20train улучшения C/D сопровождаются улучшением R1, что явно сохранено и не приписывается целиком мухе.\n\n*Постановка.* '+p['action']+'\n\nКритерии до новых исходов: '+p['primary']+'\n\n104train модели ещё не обучались; меняется только заранее заданная интерпретация одного ранее предусмотренного контроля как самостоятельного узла. Настройки, обучение и данные исходного опыта не меняются; сравнения с20 не выбирают победителя на26. Нужен отдельный перенос перед стендом.')


def calculate():
    p=read(RUN/'protocol.json');verify(p);end=read(SOURCE/'train_controller_result.json');assert end['worker_ended'] and end['exit_code']==0 and not end['hard_timeout']
    pv={};rows={};source_sha={}
    for arm in ('C','D','R1'):
        paths={arm+'_map':SOURCE/arm/'geometry/ensemble_rows.json',arm+'_top1':ROOT/f'runs/z13_candidate_identity_20261021/{arm}/top1_rows.json',arm+'_kalman':ROOT/f'runs/z13_candidate_identity_20261021/{arm}/kalman_rows.json',arm+'_map20':OLD/arm/'geometry/ensemble_rows.json'}
        for k,path in paths.items():
            rr=read(path);assert len(rr)==8326;source_sha[path.relative_to(ROOT).as_posix()]=sha(path)
            for r in rr:
                if r['kind']==1:assert r['useful']==bool(success(r))
            rows[k]=rr;pv[k]=per_video_useful(rr)
    pairs=[]
    for arm in ('C','D'):pairs.extend([(arm+'_map',arm+'_top1'),(arm+'_map','R1_map'),(arm+'_map',arm+'_kalman'),(arm+'_map',arm+'_map20')])
    comp={a+'-'+b:paired(pv[a],pv[b]) for a,b in pairs};comp['R1_map-R1_top1']=paired(pv['R1_map'],pv['R1_top1'])
    cg={v:pv['C_map'][v]-pv['C_top1'][v] for v in pv['C_top1']};rg={v:pv['R1_map'][v]-pv['R1_top1'][v] for v in pv['R1_top1']};comp['C_gain-R1_gain']=paired(cg,rg)
    seedpv=read(SOURCE/'seed_metrics.json')['per_video'];seedgain={s:paired(v,pv['C_top1']) for s,v in seedpv['C_geometry'].items()}
    d=comp['C_map-C_top1'];passed=d['mean_per_video']>=.02 and d['ci95'][0]>0 and comp['C_map-R1_map']['ci95'][0]>0 and comp['C_map-C_kalman']['ci95'][0]>0 and all(v['mean_per_video']>=0 for v in seedgain.values())
    return dict(primary_pass=passed,paired=comp,C_seeds_vs_top1=seedgain,source_sha256=source_sha,summary={k:dict(rows=len(v),useful_mean_video=float(np.mean(list(pv[k].values()))),coverage=coverage_table(v)) for k,v in rows.items()},limitations='Exploratory reused train26. Geometry retains the response map, R1 is the no-FlyVis control. Generic readout effect is reported separately; no HOTA or calibrated refusal.')


def finish():
    if (RUN/'finish_recorded.json').exists():raise RuntimeError('Already recorded')
    end=read(RUN/'full_controller_result.json');assert end['worker_ended'] and end['exit_code']==0 and not end['hard_timeout']
    report=read(RUN/'report.json');assert report==calculate();d=report['paired']['C_map-C_top1'];passed=report['primary_pass'];first=d['mean_per_video']>=.02 and d['ci95'][0]>0 and all(v['mean_per_video']>=0 for v in report['C_seeds_vs_top1'].values());second=report['paired']['C_map-R1_map']['ci95'][0]>0 and report['paired']['C_map-C_kalman']['ci95'][0]>0
    state='completed_positive' if passed else ('completed_negative' if d['ci95'][1]<.02 else 'inconclusive');fmt=lambda x:f'{100*x:+.3f}'.replace('.',',')
    def interval(o):return f"{fmt(o['mean_per_video'])} п.п. [{fmt(o['ci95'][0])}; {fmt(o['ci95'][1])}]"
    obtained='runs/z13_map_only104_20261021/report.json → paired: '+'; '.join(k+' '+interval(v) for k,v in report['paired'].items())+'. Полные8326 строк,3 сида и исходники проверены; новых моделей в этой ветви не обучалось.'
    interpretation=('Основные критерии считывания карты выполнены.' if passed else 'Основной критерий этой ветви считывания карты не выполнен.')+' R1 получает такой же узел: общий прирост от чтения карты не приписывается целиком мухе. Это поисковая функциональная проверка, не HOTA.'
    decision='Сохранить фиксированную версию и проверить перенос с теми же контролями; до стенда нужен отдельный покадровый отбор вмешательств.' if passed else 'Не переносить в стенд; не выбирать удачный объём данных или сид по26. Прежний центрированный считыватель сохранён.'
    cp=ROOT/'experiments/project_history.json';cat=read(cp);card=next(c for c in cat['experiments'] if c['id']==IDENT);head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    for field,v in [('summary',interpretation),('obtained',obtained),('interpretation',interpretation),('decision',decision)]:update(card,field,v,head,'Полный итог заранее зарегистрированной ветви.')
    card.update(state=state,done=f'{now():%Y-%m-%d}')
    for i,met in enumerate([True,first,second]):card['acceptance'][i].update(met=bool(met),evidence='runs/z13_map_only104_20261021/report.json')
    card['evidence'].append(dict(kind='file',ref='runs/z13_map_only104_20261021/report.json',claim='Полный проверенный исход'))
    write(cp,cat);base=report['summary']['C_top1']['useful_mean_video'];reserve=read(ROOT/'runs/z13_candidate_reserve_20261021/report.json')['arms']['C']['reserve']['mean_per_video'];share=100*d['mean_per_video']/reserve
    journal('полный итог ветви считывания карты','*Получено.* '+obtained+'\n\n*Вывод.* '+interpretation+'\n\n*Решение.* '+decision+'\n\nСводка к C top1: доля полезных центров '+f'{100*d["mean_per_video"]/base:+.3f}%'.replace('.',',')+'; отношение к измеренному приросту оракула тех же3 пиков '+f'{share:+.3f}%'.replace('.',',')+' (точечное отношение без собственного интервала). HOTA не измерялась. Формулировки проверены на предмет/инструмент, отсутствие лишнего повтора и соответствие выполненному.')
    write(RUN/'finish_recorded.json',dict(state=state,primary_pass=passed,recorded=now().isoformat(),report_sha256=sha(RUN/'report.json')));print(state,interval(d))


def main():
    global ATTEMPT,LIMIT
    a=argparse.ArgumentParser();a.add_argument('--prepare',action='store_true');a.add_argument('--worker',action='store_true');a.add_argument('--finish',action='store_true');a.add_argument('--attempt',default='full');a.add_argument('--budget',type=int,default=120);x=a.parse_args()
    if x.prepare:prepare();return
    if x.finish:finish();return
    ATTEMPT=x.attempt;LIMIT=x.budget
    if not x.worker:raise ValueError('Choose an action')
    if time.monotonic()-T0>LIMIT or now()>=datetime(2026,10,21,7,35,tzinfo=MSK):raise TimeoutError('No work window')
    write(RUN/'status.json',dict(state='running',phase=ATTEMPT,stage='Проверка полного будущего104train результата',updated=now().isoformat()))
    report=calculate();write(RUN/'report.json',report);write(RUN/'status.json',dict(state='completed',phase=ATTEMPT,stage='Ветвь карты сохранена',updated=now().isoformat()))


if __name__=='__main__':main()
