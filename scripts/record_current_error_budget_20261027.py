"""Зафиксировать только проверенный диагностический результат, без научной прибавки модели."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.recovery_research_service_20261027 import RUN,read,write,sha
from scripts.b123_safe_journal_20261027 import journal

def main():
    folder=RUN/'error_budget/recovery1';a=read(folder/'independent_audit.json');r=read(folder/'report.json')
    assert a['passed'] and a['report_sha256']==sha(folder/'report.json') and a['source_sha256']==sha(ROOT/'scripts/audit_current_error_budget_20261027.py')
    path=ROOT/'experiments/project_history.json';before=path.read_bytes();cat=json.loads(before);card=next(c for c in cat['experiments'] if c['id']=='hota_error_budget_current_v1')
    assert not card['obtained'],'Не перезаписывать уже записанный итог'
    source='runs/recovery_research_20261027/error_budget/recovery1/report.json'
    obtained=source+':complete=true;paired='+json.dumps(r['paired'],ensure_ascii=False)+'. Источник независимой сверки: runs/recovery_research_20261027/error_budget/recovery1/independent_audit.json. '
    obtained+='40видео;источник independent_audit.complete_videos; seconds='+str(r['seconds'])+';источник report.seconds. Значения — средние парные разности HOTAпо видео,не pooledHOTAи не прибавка реальной модели.'
    card.update(state='infrastructure',done='2026-10-27',obtained=obtained,
        interpretation='При сохранённых выходах исправление геометрии и присвоения личности с разметкой даёт заметные диагностические изменения. '
        'Это основания для выбора конкретных замен геометрии и сопоставления; это не доказательство обучаемости или достижимой прибавки. '
        'Условное добавление отсутствующих GTпосле исправленияIDне отделяет все причины пропусков и не измеряет отдельно восстановление после перекрытия. '
        'Историческая лестница уже проверяла другие вмешательства в детекции; её результаты не отменяются и не смешиваются с этим опытом.',
        decision='Продолжить подготовку обучаемого допуска с изображениями и числовым контролем. Для следующего крупного действия проверить прежние попытки уточнения рамок и сопоставления, '
        'затем выбрать одну содержательно новую обучаемую поправку с реальными причинными входами, точным выключенным контролем и короткой стоимостью. '
        'После не более3–4диагностик на действие переходить к рабочей HOTAили фиксировать конкретный остаток. Основной базис не менять.')
    for criterion in card['acceptance']:criterion.update(met=True,evidence='runs/recovery_research_20261027/error_budget/recovery1/independent_audit.json:passed=true; '+source)
    card['evidence'] += [dict(kind='file',ref=source,claim='Полные раздельные сравнения на двух20частях исторического val; только диагностика'),
        dict(kind='file',ref='runs/recovery_research_20261027/error_budget/recovery1/independent_audit.json',claim='Источники,поля всех выходов,биекцияID,раздельныевыборки,парныеразности,прежнийсрок проверены')]
    data=next(c for c in cat['experiments'] if c['id']=='recovery_admission_data_v1');data['state']='running'
    assert path.read_bytes()==before;write(path,cat)
    write(RUN/'agent_attention_resolution.json',dict(original_job='010_error_budget.json',resolved_by='error_budget/recovery1',
        original_failure_preserved=True,ready_stage_not_repeated=True,controller_all_owned_ended=True,independent_audit_passed=True))
    text=('*Получено (`hota_error_budget_current_v1`).* '+obtained+'\n\n*Вывод.* '+card['interpretation']+'\n\n*Решение.* '+card['decision']+'\n\n'
          '*Исполнение.* Первый запуск остановлен оценщиком до полного видео: TrackEval BDD100K требует числовыхID. '
          'Отдельная recovery1кодирует те же смысловые личности биективными целыми,политика и рамки не меняются. '
          'Прежние исходник/логи/код1сохранены;исходный срокнепродлён, код0/all_owned_ended. '
          'Первый полныйпроход2,922с,цена347,916с<3457,273с;источник recovery1/price.json. '
          'Проверка сначала сверила441691запись,затем добавлено прямое утверждение о прежнемсроке и повторена только проверка;первыйаудитсохранён отдельно. '
          'Данные130train/35dev продолжаютготовиться в другойCPUступени,первое видео10,468с,цена3553,504<7164,109с;источник admission_dataset/price.json. '
          'Названия признаков координат уточнены до публикации заданий без изменениячисел;прежнийисточник/протоколсохранены. '
          'Каталог послеэтойзаписи требуетобязательнойпроверки. Научнаяприбавкареальнойсистемывэтомшагенеизмерялась.')
    journal('hota_error_budget_current_v1','геометрия и личность дали диагностический резерв на актуальном выходе',text)
    print(json.dumps(dict(recorded=True,state='infrastructure',real_model_gain=False)))

if __name__=='__main__':main()
