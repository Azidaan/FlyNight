"""Record the full-map CPU gate after the independent scientific worker ended."""
import json
from datetime import datetime
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_parallel_next import read,write
from scripts import z13_record_next as recorder
RUN=ROOT/'runs/z13_map_utility_20261018'


def main():
    end=read(RUN/'controller_result.json');assert end['worker_ended']
    if not (RUN/'recorded.json').exists():
        path=ROOT/'experiments/project_history.json';catalog=read(path)
        card=next(c for c in catalog['experiments'] if c['id']=='z13_map_utility_v1')
        if end['exit_code']==0 and not end['hard_timeout'] and (RUN/'report.json').exists():
            report=read(RUN/'report.json');source='runs/z13_map_utility_20261018/report.json'
            obtained=source+': '+json.dumps(report,ensure_ascii=False)
            state='completed_positive' if report['primary_pass'] else 'inconclusive'
            interpretation='Фиксированный критерий полного оценщика '+('выполнен.' if report['primary_pass'] else 'не выполнен. Эта реализация не подтверждена; причина ограничений всего представления не установлена.')
            met=[True,report['primary_pass']];decision=report['decision']
        else:
            source='runs/z13_map_utility_20261018/controller_result.json';obtained=source+': '+json.dumps(end,ensure_ascii=False)
            state='inconclusive';interpretation='Сохранены завершённые варианты/видео; при частичном итоге вердикт об эффективности не делается.'
            met=[None,None];decision='save_partial_without_efficiency_verdict'
        recorder.update(card,dict(state=state,done=datetime.now().date().isoformat(),obtained=obtained,interpretation=interpretation,
          decision=decision,cannot_claim='Нет HOTA, независимой проверкиdev или гарантии риска на неизвестных исходах. Train26 повторно использован; один сид.'),'Завершён CPU-прогон полных карт ответа')
        for item,value in zip(card['acceptance'],met):item.update(met=value,evidence=source)
        card['evidence'].append(dict(kind='json',ref=source,claim='Проверенный полный/частичный первичный итог'))
        path.write_text(json.dumps(catalog,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
        recorder.journal('итог/сохранение z13_map_utility_v1','*Получено.* '+obtained+'\n\n*Вывод.* '+interpretation+'\n\n*Решение.* '+decision+'. HOTA не измерялась.')
        write(RUN/'recorded.json',dict(recorded=True))
    with (RUN/'map_tests.log').open('w',encoding='utf-8') as log:
        result=subprocess.run([sys.executable,'-m','pytest','tests/test_z13_response_utility.py','-q'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=120)
    assert result.returncode==0
    recorder.OWN_FILES += ['tracking/z13_response_utility.py','scripts/z13_map_utility.py',
       'scripts/z13_map_utility_controller.py','scripts/z13_prepare_map_utility.py','scripts/z13_record_map_utility.py',
       'tests/test_z13_response_utility.py']
    recorder.OWN_FILES += [p.relative_to(ROOT).as_posix() for p in RUN.rglob('*') if p.is_file() and p.suffix not in ('.next','.tmp')]
    recorder.OWN_FILES += ['runs/z13_map_utility_20261018/'+x for x in ('validation_0.log','validation_1.log','validation_result.json','vault_preservation.json','commit_scope.json')]
    recorder.validate_and_publish(RUN,True,False)


if __name__=='__main__':main()
