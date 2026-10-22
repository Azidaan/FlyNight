"""Verify saved CPU readout comparison and record bounded outcomes."""
from pathlib import Path
import sys
import argparse
import subprocess
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_center_hota_20261021 import read,write,sha,now
from scripts.z13_record_center_hota_20261021 import revise
from scripts.z13_hota_deadline_guard import WindowsProcesses
from scripts.z13_continue import paired


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);args=ap.parse_args()
    run=args.run.resolve();run.relative_to(ROOT.resolve());p=read(run/'protocol.json');ident=p['id']
    if (run/'finish_recorded.json').exists():raise RuntimeError('Already recorded')
    end=read(run/'controller_result.json');control=read(run/'controller.json');processes=WindowsProcesses()
    assert end['exit_code']==0 and end['worker_ended'] and end['all_owned_ended'] and not end['remaining'] and not end['hard_timeout']
    assert processes.creation(control['pid'])!=control['birth']
    assert not {pid:b for pid,b in end['owned'].items() if processes.creation(int(pid))==b}
    for n,h in p['sha256'].items():assert sha(ROOT/n)==h,n
    r=read(run/'report.json');name='boundary' if ident=='z13_boundary_readout_v1' else p['mode']
    for key,saved in r['paired'].items():
        if key=='D_gain-R1_gain':
            left={v:r['per_video']['D_'+name][v]-r['per_video']['D_raw'][v] for v in r['per_video']['D_raw']}
            right={v:r['per_video']['R1_'+name][v]-r['per_video']['R1_raw'][v] for v in r['per_video']['R1_raw']}
        else:
            a,b=key.split('-',1);left=r['per_video'][a];right=r['per_video'][b]
        assert paired(left,right)==saved,key
    primary=['D_'+name+'-D_'+b for b in ('raw','geometry6','map14')]+['D_'+name+'-R1_'+name]
    passed=all(r['criteria'].values());assert passed==r['primary_pass']
    negative=any(r['paired'][k]['ci95'][1]<=0 for k in primary)
    state='completed_positive' if passed else ('completed_negative' if negative else 'inconclusive')
    cat=read(ROOT/'experiments/project_history.json');card=next(c for c in cat['experiments'] if c['id']==ident)
    head=subprocess.check_output(['git','--no-optional-locks','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    report_rel=(run/'report.json').relative_to(ROOT).as_posix();digest=sha(run/'report.json')
    obtained=report_rel+' → paired: '+str(r['paired'])+'; summary: '+str(r['summary'])+'. Полный train26,3варианта, по84выбранных ответа, исходные центры/баллы воспроизведены; CUDA/dev/HOTA не использовались.'
    interpretation='Все критерии этого поискового сравнения выполнены; это ещё не проверка пользы стенду.' if passed else 'Основной сравнительный критерий не выполнен. Этот механизм отказа/балла не принят; частные сравнения и исправление одного R1примера не заменяют исход D.'
    decision=p['decision']
    card.update(state=state,done=f'{now():%Y-%m-%d}')
    for field,value in [('summary',interpretation),('obtained',obtained),('interpretation',interpretation),('decision',decision)]:revise(card,field,value,head,'Проверенный полный итог CPU-сравнения с прежними контролями.')
    for criterion,value in zip(card['acceptance'],[r['criteria']['reproduction'],r['criteria']['selection'],r['criteria']['fly_control']]):criterion.update(met=bool(value),evidence=report_rel)
    card['commits']=list(dict.fromkeys(card.get('commits',[])+[head]))
    for fname in ('report.json','reproduction.json','controller_result.json','launch.json'):card['evidence'].append(dict(kind='file',ref=(run/fname).relative_to(ROOT).as_posix(),claim='Полный исход, контроль неизменности и фактическое исполнение'))
    confirmed=[]
    for arm in ('C','D'):
        for base in (arm+'_raw',arm+'_geometry6',arm+'_map14','R1_'+name):
            k=arm+'_'+name+'-'+base;obj=r['paired'][k]
            if obj['ci95'][0]<=0:continue
            confirmed.append(dict(id=ident+'_'+k.replace('-','_vs_'),title=arm+' '+name+' против '+base+' при равном отборе',
                category='readout',metric='selected_success_fraction_positive',unit='fraction',comparison=k,subset='Повторно используемые26train-видео,84выбранных ответа из8326',
                conditions='Фиксированный механизм без подбора по train26; прежние карты/центры/ядро; одинаковый отбор1%; исходная геометрия6 и оценка14 как контроли; парный повидео бутстрэп5000,сид1.',
                limitations='Поисковая проверка отбора, не точность среди выбранных, не HOTA и не доказанная калибровка. Идея мотивирована разобранным R1dev-примером; полные критерии карточки сохраняются. Не доказывает преимущество обучения ядра или анатомии.',
                evidence=dict(source=report_rel,sha256=digest,locator=['paired',k])))
    card['confirmed_results']=confirmed
    means={k:float(np.mean(list(v.values()))) for k,v in r['per_video'].items()}
    audit=dict(at=now().isoformat(),report_sha256=digest,verified_comparisons=len(r['paired']),all_owned_ended=True,means=means,scientific_frozen_files_verified=len(p['sha256']),state=state)
    write(run/'review.json',audit)
    card['metrics']=[dict(key='selected_success_fraction_positive',value=means['D_'+name],baseline=means['D_raw'],subset='train26;D '+name,source=(run/'review.json').relative_to(ROOT).as_posix(),locator='means.D_'+name)]
    write(ROOT/'experiments/project_history.json',cat)
    plan=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';text=plan.read_text(encoding='utf-8');anchor='## Журнал изменений\n';assert text.count(anchor)==1
    event=f'\n**{now():%d.%m.%Y, %H:%M} MSK** — полный итог CPU-узла считывания `{ident}`\n\n*Получено.* '+obtained+'\n\n*Вывод.* '+interpretation+'\n\n*Решение.* '+decision+'\n\nHOTA не измерялась; процент к стенду и доля потолка не определены. Формулировки проверены: FlyVis — объект, отбор и метрики — инструменты; вывод относится только к завершённому протоколу. Новая запись сохраняет все прежние отрицательные результаты.\n'
    plan.write_text(text.replace(anchor,anchor+event,1),encoding='utf-8')
    write(run/'finish_recorded.json',dict(at=now().isoformat(),state=state,report_sha256=digest,commit_before=head))
    print(state,r['criteria'])
    for arm in ('C','D','R1'):
        for mode in ('raw',name,'geometry6','map14'):print(arm,mode,r['summary'][arm+'_'+mode])
    print('D primary',{k:r['paired'][k] for k in primary})


if __name__=='__main__':main()
