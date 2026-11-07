from pathlib import Path
import sys,subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261029 import RUN,service

def main():
    folder=RUN/'masa_record_fp32_recovery1'
    for name in ['masa_query_fp32_recovery1','masa_audit_fp32_recovery1']:
        c=service.read(RUN/name/'controller_result.json')
        assert c['exit_code']==0 and c['all_owned_ended'] and not any(service.matching(k,v) for k,v in c['owned'].items())
    a=service.read(RUN/'masa_query_audit/report.json');assert a['passed'] and a['complete']
    h=service.read(ROOT/'experiments/project_history.json');c=next(x for x in h['experiments'] if x['id']=='reid_masa_detic_v1')
    updates=dict(obtained='runs/continuation_20261029/masa_audit_fp32_recovery1/report.json: полный пилот20val/5вариантов,всеraw/JSON/события/метрикиповторены;парныеинтервалывpaired.',
        interpretation='Польза признаков MASA-Detic в одинаковом query-сопоставлении '+('подтверждена по двум заранее заданным сравнениям.' if a['main_acceptance'] else 'не подтверждена по всем заранее заданным сравнениям. Частные сравнения опубликованы отдельно.'),
        decision='Сохранить полный результат и ограничения; перед следующим изменением проверить открытые проблемы. Повторять готовые20не нужно.',
        cannot_claim=c['cannot_claim']+' Полныйопытотноситсякнашейquery-адаптации/историческим20val,неофициальномуMASAтрекеру,неMGOR/ScaleTrackинеядруFlyVis.')
    for key,value in updates.items():
        if c.get(key):c.setdefault('revisions',[]).append(dict(date='2026-10-29',commit=None,field=key,old=c[key],reason='Записан новый полный проверенный опыт; прежняя подготовка и её ограничения сохранены.'))
        c[key]=value
    c['state']='completed_positive' if a['main_acceptance'] else 'inconclusive';c['done']='2026-10-29'
    for i,item in enumerate(c['acceptance']):
        item['met']=True if i==0 else a['main_acceptance'];item['evidence']='runs/continuation_20261029/masa_audit_fp32_recovery1/report.json'
    c.setdefault('evidence',[]).append(dict(kind='file',ref='runs/continuation_20261029/masa_audit_fp32_recovery1/report.json',claim='Полный независимый результат нового пилота'))
    service.write(ROOT/'experiments/project_history.json',h)
    from scripts.b123_safe_journal_20261027 import journal
    journal(c['id'],'Проверен полный пилот признаков MASA-Detic',c['obtained']+' '+c['interpretation']+' '+str(a['paired']))
    for i,args in enumerate([['scripts/validate_project_history.py'],['-m','pytest','tests/test_project_history.py','-q'],['scripts/restore_obsidian_sections_20261027.py'],['-m','pytest','tests/test_obsidian_fundamental_overview.py','-q']]):
        with (folder/f'check_{i}.log').open('w',encoding='utf-8') as out:code=subprocess.run([sys.executable,*args],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,timeout=420).returncode
        assert code==0,(i,code)
    service.write(folder/'report.json',dict(complete=True,checks_passed=True,main_acceptance=a['main_acceptance'],updated=service.now()))
    with (RUN/'HANDOFF.md').open('a',encoding='utf-8') as f:f.write('\nПолныйMASAпилотзавершён/оформлен. Прочитатьreport/audit/controller/PIDend,частныеэффектыиограничения,обновитьПроблемы/confirmed_results/авторскуюветкуТ10толькопоправилам. Не переноситьитогнаMGOR/ScaleTrack/ядроFlyVis.\n')

if __name__=='__main__':main()
