"""Экспорт признаков завершён; основная проверка обученного допуска остаётся running до valHOTA."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.recovery_research_service_20261027 import RUN,read,write
from scripts.b123_safe_journal_20261027 import journal

def main():
    features=read(RUN/'appearance_features/report.json');audit=read(RUN/'appearance_features/audit.json');learning=read(RUN/'admission_learning/report.json')
    assert features['complete'] and audit['passed'] and learning['complete']
    for name in ['appearance_features','admission_learning']:
        c=read(RUN/name/'controller_result.json');assert c['exit_code']==0 and c['all_owned_ended']
    path=ROOT/'experiments/project_history.json';before=path.read_bytes();cat=json.loads(before)
    card=next(c for c in cat['experiments'] if c['id']=='recovery_appearance_features_v1');assert not card['obtained']
    card.update(state='infrastructure',done='2026-10-27',obtained='runs/recovery_research_20261027/appearance_features/audit.json: '+json.dumps(audit,ensure_ascii=False)+
        '; source report.seconds='+str(features['seconds'])+' in runs/recovery_research_20261027/appearance_features/report.json.',
        interpretation='Полные причинные признаки подготовлены; числовые входы и метки не изменились. Это готовность данных, не польза полного трекера.',
        decision='Использовать для одинаковых обучаемых голов и проверки реального выхода; готовый экспорт не повторять.')
    for c in card['acceptance']:c.update(met=True,evidence='runs/recovery_research_20261027/appearance_features/audit.json:passed=true')
    card['evidence'].append(dict(kind='file',ref='runs/recovery_research_20261027/appearance_features/audit.json',claim='Конечные эмбеддинги,точные X/y,165видеои конецдеревапроверены'))
    main_card=next(c for c in cat['experiments'] if c['id']=='recovery_admission_learning_v1');main_card['state']='running'
    main_card['acceptance'][0].update(met=True,evidence='runs/recovery_research_20261027/admission_learning/report.json:complete=true;6models;visual_numeric_head_parameters_equal=true')
    main_card['evidence'].append(dict(kind='file',ref='runs/recovery_research_20261027/admission_learning/report.json',claim='Полное обучение двухголоввтрёхсидах;неитоговаяHOTA'))
    assert path.read_bytes()==before;write(path,cat)
    journal('recovery_admission_learning_v1','полное обучение шести голов выполнено, выбор порогов идёт на dev',
        '*Получено.* Полныепризнаки165видео/21145кандидатов,источник appearance_features/audit.json;числовыеXиметкиyпобайтны. '
        'Обучены6голов(2варианта×3сида),100эпохкаждая,одинаковое число '+str(learning['models'][0]['parameters'])+'параметров;источник admission_learning/report.json.models. '
        'CheckpointкаждогосидапоdevBCE,непоval;невзвешеннаядвоичнаяперекрёстнаяэнтропия,неизвестныеметкиисключеныизошибки,предсказаниядлянихсохранены. '
        'Ступень050код0/all_owned_ended. Источникцены admission_learning/price.json:3эпохи0,766с,консервативныйостаток424,868с<1793,261с. '
        'Это обученныеголовы,ноHOTAпользыещёнет.\n\n*Следующее действие.* 060_dev_thresholdвыбираетпорогтольконаполной35devHOTAиззаранеесетки[0.5,0.7,0.9],сохраняявсе3сида. '
        'Послефактическоговыбора—frozen_for_validation.json и полныйинференснапрежних40valсточнымбазисом/правилом,безподборапотесту. '
        'Карточкаосновногоопытаrunning,критерийполнойHOTAостаётсянепроверенным. '
        'Запретпроверятьобобщениедетекторанаtrain_cacheсохраняется:devHOTAздесьлишькритерийвыбораполитикинаданныхразработки. '
        'Каталогпослеэтойзаписитребуетобязательныхпроверок.')
    print('Features recorded; full learned-tracker experiment remains running')

if __name__=='__main__':main()
