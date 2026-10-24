"""Wait locally, independently check the maps, record and publish exact files."""
from pathlib import Path
from datetime import datetime,timedelta
import os
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.z13_template_swap_20261025 import RUN,W,IDENT,ARMS,read,write,sha,now,journal


def independent_audit(report):
    import numpy as np
    from scripts.z13_continue import paired
    from tracking.z13_readout import HexGeometry,to_frame
    from tracking.z13_ridge_localizer import map_center
    g=read(ROOT/'runs/z13_local_peak_20261021/geometry.json')
    geom=HexGeometry(np.asarray(g['points'],np.float32),g['eye_size'])
    result={}
    for arm in ARMS:
        labels={r['label']:r for r in read(ROOT/f'runs/z13_query_memory_20261021/{arm}/evaluate/labels.json')}
        values={v:[[],[]] for v in read(RUN/'protocol.json')['evaluate']}
        count=0
        for path in sorted((RUN/arm).glob('chunk_*.json')):
            job=read(path)
            assert sha(path)==report['chunk_sha256'][path.relative_to(ROOT).as_posix()]
            assert sha(path.with_suffix('.npz'))==job['maps_sha256']
            with np.load(path.with_suffix('.npz')) as z:
                maps,indices,eligible=z['maps'],z['labels'],z['eligible']
            assert len(maps)==len(job['rows'])
            for j,row in enumerate(job['rows']):
                label=labels[row['label']]
                assert int(indices[j])==row['label'] and bool(eligible[j])==row['eligible']
                assert row['video']==label['video'] and row['kind']==label['kind']
                useful=[]
                for k,output in enumerate(row['outputs']):
                    c,pres=map_center(maps[j,k],geom)
                    center=np.asarray(to_frame(c,label['crop'],g['eye_size']))
                    assert np.max(np.abs(center-output['center']))<1e-8
                    assert abs(pres-output['p_present'])<1e-8
                    if row['kind']==1:
                        error=np.linalg.norm(center-label['gt'])
                        value=bool(error<=.25*label['scale'] and error<np.linalg.norm(np.asarray(label['pred'])-label['gt']))
                        assert value==output['useful']
                        useful.append(value)
                if row['eligible'] and row['kind']==1:
                    values[row['video']][0].append(useful[0])
                    values[row['video']][1].append(float(np.mean(useful[1:])))
                count+=1
        assert count==8326
        own={v:float(np.mean(x[0])) for v,x in values.items()}
        wrong={v:float(np.mean(x[1])) for v,x in values.items()}
        calculated=paired(own,wrong)
        assert calculated==report['arms'][arm]['paired']
        result[arm]=dict(rows=count,all_output_maps_and_centers_checked=True,paired=calculated)
    write(RUN/'independent_audit.json',dict(at=now().isoformat(),report_sha256=sha(RUN/'report.json'),arms=result))


def record():
    from scripts.z13_hota_deadline_guard import WindowsProcesses
    from scripts.z13_localization_labels_20261023 import verify_sources
    assert not (RUN/'finish_recorded.json').exists()
    p,e=read(RUN/'protocol.json'),read(RUN/'controller_result.json')
    assert e['worker_ended'] and e['all_owned_ended'] and not e['remaining']
    ps=WindowsProcesses()
    assert all(ps.creation(int(pid))!=birth for pid,birth in e['owned'].items())
    verify_sources(p)
    full=e['exit_code']==0 and (RUN/'report.json').exists() and read(RUN/'report.json')['complete']
    source=(RUN/'report.json').relative_to(ROOT).as_posix()
    if full:
        report=read(RUN/'report.json');independent_audit(report)
        results={a:report['arms'][a]['paired'] for a in ARMS}
        depends=results['D']['ci95'][0]>0
        fields=dict(summary='Изолированная подмена полного фильтра проверена для D/C/R1; это диагностика зависимости, не новое улучшение.',
            obtained=source+' → arms: '+str({a:dict(paired=report['arms'][a]['paired'],eligible=report['arms'][a]['eligible'],interpretation=report['arms'][a]['interpretation']) for a in ARMS})+'. Все8326 исходных строк каждого варианта воспроизведены; первичные карты и интервалы проверены независимо.',
            interpretation=('В этом считывателе правильный образец даёт положительный вклад в полезные центры сверх трёх чужих моделей при общей истории поиска. Общее утверждение, что текущий фильтр игнорирует образец, не подтверждается.' if depends else 'Преимущество правильного фильтра по полезным центрам в этом сравнении не подтверждено. Это не доказывает игнорирование образца всей FlyVis: правильная история сохранена в общем состоянии.'),
            decision='Не переносить вспомогательное обучение из статьи только по сходству симптомов. Сопоставить диагностические ответы с ошибками отказа и доступностью правильного кандидата перед выбором одной функциональной правки. Дорогой этап автоматически не запускается.',
            cannot_claim=p['counterargument'])
        state='completed_positive' if depends else 'inconclusive'
    else:
        depends=None
        fields=dict(summary='Сохранена незавершённая изолированная проверка образца.',
            obtained=(RUN/'controller_result.json').relative_to(ROOT).as_posix()+': '+str(e)+'. Сохранённых групп:'+str(sum(len(list((RUN/a).glob('chunk_*.json'))) for a in ARMS)),
            interpretation='Неполный проход не даёт вывода о зависимости от образца или эффективности.',
            decision='Сохранить готовое; проверить служебную причину и остаток исходного бюджета. Не повторять готовые группы и не продлевать бюджет.',cannot_claim=p['counterargument'])
        state='inconclusive'
    cp=ROOT/'experiments/project_history.json';cat=read(cp);card=next(c for c in cat['experiments'] if c['id']==IDENT)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()
    for key,value in fields.items():
        if card.get(key) and card[key]!=value:
            card.setdefault('revisions',[]).append(dict(date=now().strftime('%Y-%m-%d'),commit=head,field=key,old=card[key],reason='Проверен полный либо частичный результат изолированной подмены.'))
        card[key]=value
    card.update(state=state,done=now().strftime('%Y-%m-%d'))
    card['acceptance'][0].update(met=True if full else None,evidence=(RUN/'independent_audit.json').relative_to(ROOT).as_posix() if full else (RUN/'controller_result.json').relative_to(ROOT).as_posix())
    card['acceptance'][1].update(met=depends,evidence=source if full else 'Неполный проход')
    if full:card['evidence'].append(dict(kind='file',ref=source,claim='Свой и три чужих фильтра, одинаковое состояние и геометрия'))
    write(cp,cat)
    journal('итог изолированной подмены образца','*Получено.* '+fields['obtained']+'\n\n*Вывод.* '+fields['interpretation']+'\n\n*Решение.* '+fields['decision']+'\n\nHOTA и процент к стенду не измерены; это не доля достижимого потолка. Нового архитектурного узла не добавлено, пересборка draw.io не требуется.')
    write(RUN/'finish_recorded.json',dict(at=now().isoformat(),complete=full,dependence_detected_D=depends,report_sha256=sha(RUN/'report.json') if (RUN/'report.json').exists() else None,all_owned_ended=True))
    path=W/'HANDOFF.md'
    header='# Актуально — '+now().strftime('%d.%m.%Y %H:%M')+' MSK\n\n'+IDENT+' окончен. '+fields['obtained']+' '+fields['decision']+' Публикацию проверить по runs/z13_template_swap_20261025/publication.json. Это диагностический исход, не улучшение HOTA. Старые источники и57чужихправок сохранять.\n\nНиже прежние передачи сохранены.\n\n'
    path.write_text(header+path.read_text(encoding='utf-8'),encoding='utf-8')


def publish():
    from scripts.z13_publish_native20_final_20261024 import foreign_unchanged,graph_rows
    foreign=foreign_unchanged()
    commands=[['scripts/validate_project_history.py'],['-m','pytest','tests/test_project_history.py','tests/test_z13_template_swap.py','-q']]
    for i,args in enumerate(commands):
        with (RUN/f'checks_{i}.log').open('w',encoding='utf-8') as out:
            subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),*args],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,check=True)
    before={n:sha(ROOT/n) for _,n in graph_rows() if (ROOT/n).is_file() and n not in foreign}
    receipt=RUN/'owned_graph_before.json';write(receipt,before)
    graph_report=RUN/'vault_final.json'
    subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),'scripts/z13_build_vault_preserving_edits.py','--report',str(graph_report),'--refresh-owned',str(receipt)],cwd=ROOT,check=True,capture_output=True)
    foreign_unchanged()
    names=['tracking/z13_template_swap.py','tests/test_z13_template_swap.py','scripts/z13_template_swap_20261025.py','scripts/z13_finish_template_swap_20261025.py','scripts/z13_template_swap_console.ps1','experiments/project_history.json','dissertation/ЕДИНЫЙ_ПЛАН.md','runs/z13_weekend_20261024/HANDOFF.md']
    names += [x.relative_to(ROOT).as_posix() for x in RUN.glob('*.json') if x.name!='publication.json']
    names += [x.relative_to(ROOT).as_posix() for a in ARMS for x in (RUN/a).glob('chunk_*.json')]
    names += [x.relative_to(ROOT).as_posix() for a in ARMS for x in (RUN/a).glob('done.json')]
    for _,name in graph_rows():
        if name not in foreign and (ROOT/name).is_file():names.append(name)
    names=sorted(set(names));assert not set(names)&foreign and '.gitignore' not in names
    staged=set(subprocess.check_output(['git','diff','--cached','--name-only','-z'],cwd=ROOT).decode().split('\0'))-{''}
    assert staged<=set(names) and not (ROOT/'.git/index.lock').exists()
    assert subprocess.check_output(['git','remote','get-url','origin'],cwd=ROOT).decode().strip()=='https://gitlab.com/aziteam-group/drbdiss.git'
    assert subprocess.check_output(['git','branch','--show-current'],cwd=ROOT).decode().strip()=='main'
    for i in range(0,len(names),25):subprocess.run(['git','add','-f','--',*names[i:i+25]],cwd=ROOT,check=True,capture_output=True)
    subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True,capture_output=True)
    subprocess.run(['git','commit','--quiet','-m','Verify isolated template dependence with frozen FlyVis query states','-m','Co-authored-by: Codex <codex@openai.com>'],cwd=ROOT,check=True,capture_output=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()
    subprocess.run(['git','push','origin','main'],cwd=ROOT,check=True)
    foreign_unchanged()
    write(RUN/'publication.json',dict(at=now().isoformat(),commit=commit,push_completed=True,paths=names))


def main():
    p=read(RUN/'protocol.json')
    write(RUN/'completion.json',dict(pid=os.getpid(),started=now().isoformat()))
    while not (RUN/'controller_result.json').exists():
        if now()>datetime.fromisoformat(p['hard_stop'])+timedelta(seconds=90):raise TimeoutError('Controller result missing')
        time.sleep(3)
    record();publish()


if __name__=='__main__':
    try:main()
    except Exception as exc:
        write(RUN/'monitor_failure.json',dict(at=now().isoformat(),error=repr(exc)));raise
