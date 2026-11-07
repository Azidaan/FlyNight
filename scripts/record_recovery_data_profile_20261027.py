"""Записать подготовленные данные и технический профиль после полного аудита и конца деревьев."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.recovery_research_service_20261027 import RUN,read,write
from scripts.b123_safe_journal_20261027 import journal

def main():
    data=read(RUN/'admission_dataset/report.json');audit=read(RUN/'admission_dataset/audit.json')
    profile=read(RUN/'appearance_gpu_profile/report.json')
    assert data['complete'] and audit['passed'] and profile['complete']
    for name in ['admission_dataset','admission_dataset_audit','appearance_gpu_profile']:
        controller=read(RUN/name/'controller_result.json');assert controller['exit_code']==0 and controller['all_owned_ended']
    path=ROOT/'experiments/project_history.json';before=path.read_bytes();cat=json.loads(before)
    for ident,source,obtained,interpretation,decision in [
        ('recovery_admission_data_v1','runs/recovery_research_20261027/admission_dataset/report.json',
         'Данные готовы; источник runs/recovery_research_20261027/admission_dataset/audit.json: '+json.dumps({k:audit[k] for k in ['videos','candidates','labels','train_dev_disjoint','unknown_not_negative','strict_past_all_records','finite_features']},ensure_ascii=False),
         'Раздельные данные для обучения и разработки подготовлены и проверены; неизвестная метка сохранена отдельно. Это готовность входов, не полезность модели.',
         'Использовать те же данные и кандидаты для числового контроля и признаков внешнего вида. Исторический val не использовать для обучения/выбора модели. Следующее действие — реальное обучение и полная HOTA после измеренной цены.'),
        ('recovery_appearance_profile_v1','runs/recovery_research_20261027/appearance_gpu_profile/report.json',
         'Технический профиль: источник runs/recovery_research_20261027/appearance_gpu_profile/report.json: '+json.dumps(profile,ensure_ascii=False),
         'Проверены причинный фиксированный режим извлекателя и техническая обучаемость новой головы; веса головы не сохранялись. Это не полное обучение и не HOTA.',
         'Цена допускает подготовку отдельной ступени полного извлечения и обучения с заранее фиксированными условиями.' if profile['cost_gate_for_next_feature_stage'] else 'Цена не прошла заданный гейт следующего извлечения; сохранить результат и выбрать более дешёвую реализацию, не продлевать профиль.')]:
        card=next(c for c in cat['experiments'] if c['id']==ident);assert not card['obtained'],'Итог уже записан; не повторять recorder'
        card.update(state='infrastructure',done='2026-10-27',obtained=obtained,interpretation=interpretation,decision=decision)
        for item in card['acceptance']:item.update(met=True,evidence=source+'; admission_dataset/audit.json:passed=true')
        card['evidence'].append(dict(kind='file',ref=source,claim='Проверенный инфраструктурный результат, не польза полного трекера'))
    assert path.read_bytes()==before;write(path,cat)
    journal('recovery_data_and_profile_20261027','данные допуска и технический GPUпрофиль завершены',
        '*Получено.* Данные: runs/recovery_research_20261027/admission_dataset/audit.json, '+json.dumps({k:audit[k] for k in ['videos','candidates','labels']},ensure_ascii=False)+'. '
        'Профиль: runs/recovery_research_20261027/appearance_gpu_profile/report.json, стоимость признаков='+str(profile['feature_seconds'])+'с,чтения/подготовки='+str(profile['input_seconds'])+'с; '
        'оценкаследующейступени='+str(profile['estimated_full_feature_seconds_conservative'])+'с,гейт='+str(profile['cost_gate_for_next_feature_stage'])+'. '
        'Контроллеры020/025/030код0/all_owned_ended;неизвестныеметкиисключеныизтехническойошибки,BNбуферынеизменны,пакет/поодномусогласованы. '
        'Новыхобученныхвесовпрофильнесохранял,этонеполноеобучениеинеHOTA.\n\n*Решение.* Подготовитьотдельныйфиксированныйпротоколполногоизвлечения/обученияиHOTA,еслиценадопускает;нетратитьокнонаповторпрофиля. '
        'Основнойбазиснеизменён,историческийvalневподгонку,ядроFlyVisнеиспользовалось. Каталогпослеэтойзаписитребуетобязательныхпроверок.')
    print(json.dumps(dict(recorded=True,next_feature_stage_cost_gate=profile['cost_gate_for_next_feature_stage'])))

if __name__=='__main__':main()
