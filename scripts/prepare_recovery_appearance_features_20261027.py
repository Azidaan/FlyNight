"""Полное извлечение допускается только после измеренной цены и проверенного конца профиля."""
from pathlib import Path
from datetime import datetime,timezone
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.recovery_research_service_20261027 import RUN,read,write,sha,matching
from scripts.b123_safe_journal_20261027 import journal

def main():
    folder=RUN/'appearance_features';assert not folder.exists()
    profile=read(RUN/'appearance_gpu_profile/report.json');controller=read(RUN/'appearance_gpu_profile/controller_result.json')
    assert controller['all_owned_ended'] and controller['exit_code']==0 and profile['cost_gate_for_next_feature_stage']
    assert not any(matching(p,b) for p,b in controller['owned'].items())
    source=['scripts/recovery_appearance_features_20261027.py','tracking/busca_runtime.py','weights/busca_official_20261027/model_busca.pth',
        'runs/recovery_research_20261027/admission_dataset/report.json','runs/recovery_research_20261027/admission_dataset/audit.json','runs/recovery_research_20261027/appearance_gpu_profile/report.json']
    p=dict(id='recovery_appearance_features_v1',created=datetime.now(timezone.utc).isoformat(),budget_seconds=3600,
        author_stop=read(RUN/'authorization.json')['save_at'],device='cuda',feature_dim=profile['features_per_image'],
        question='Можно ли получить полные причинные признаки внешнего вида для фиксированных данных и следующего обучения?',
        hypothesis='Измеренная стоимость позволяет подготовить признаки той же пары областей; это готовность входов,не их польза.',
        action='Те же JPEGобласти и режим извлекателя,что в прошедшем коротком профиле. Пакеты16изображений,FP32,BNзаморожена,авторскаяBGRнормировкапотомRGBCHW. '
               'Сохранить признаки каждой прошлой/текущейобласти,копии числовыхXи неизменныхy. Ни обучение,ни новыеобласти,ниподборпоvalне выполняются.',
        counterargument='Техническая полнота не доказывает различимость или HOTA; внешний вид может не обобщаться на наш домен. Нужны равные контроли и реальный обученный выход.',
        cost_gate='После первого полного видео:2*его стоимость/числопар*оставшиесяпары+120 должны помещаться в остаток3600с. Число оставшихся пар известно из полного аудита. Не продлевать бюджет.',
        acceptance=['Все165видеосовпадаютсаудированнымразделением,неизменныечисловыепризнакииметки',
                    'ФиксированныевесаиBN,конечныепризнаки и хешикаждоговыхода,тожепреобразованиечтовпрофиле',
                    'Измереннаяполнаястоимостьипроверенныйконецдерева;этоещёнеобучение/HOTA'],
        limitation='Инфраструктурныйэкспорт,непользамодели;ядроFlyVisнеиспользуется. СобственныйфиксированныйрежимReID,неточноеприменениеполнойBUSCA. '
                   'Изображенияпоследнегоназначениятрекеранеявляютсягарантированнойистиннойличностью;unknownметкиостаютсянеизвестными.',
        sha256={s:sha(ROOT/s) for s in source})
    write(folder/'protocol.json',p)
    path=ROOT/'experiments/project_history.json';before=path.read_bytes();cat=json.loads(before);assert not any(c['id']==p['id'] for c in cat['experiments'])
    cat['experiments'].append(dict(id=p['id'],title='Области объекта во времени — полные признаки для обучения допуска',area='flyvis',state='ready',started='2026-10-27',
        question=p['question'],hypothesis=p['hypothesis'],expected=p['action'],summary=p['counterargument'],obtained='',interpretation='',decision='',cannot_claim=p['limitation'],
        acceptance=[dict(criterion=x,met=None) for x in p['acceptance']],metrics=[],commits=[],files=source[:2],
        evidence=[dict(kind='file',ref='runs/recovery_research_20261027/appearance_features/protocol.json',claim='Полноеизвлечениепослезамера,не обучение и не польза')]))
    assert path.read_bytes()==before;write(path,cat)
    job=dict(name='040_appearance_features',folder='runs/recovery_research_20261027/appearance_features',script=source[0],script_sha256=sha(ROOT/source[0]),protocol_sha256=sha(folder/'protocol.json'))
    write(RUN/'appearance_features_job_prepared.json',job)
    journal(p['id'],'полные признаки областей поставлены после измеренной GPUцены',
        '*Получено до ступени.* Данные165видео/21145кандидатов;метки unknown12137,negative2745,positive6263;источник admission_dataset/audit.json. '
        'Профиль128изображений: чтение/подготовка1,047с,GPU1,890с,признак512,пик328434176байт,консервативнаястоимостьполногоэкспорта2240,714531с<3600с; '
        'источник appearance_gpu_profile/report.json. Код0/all_owned_ended и реальныйконецCUDAпроверены. '
        '*Постановка.* '+p['action']+' '+p['cost_gate']+' '+p['limitation']+' '
        'Дальнейшаяступень—реальноеобучениечисловогоконтроляиголовывнешнеговида;подготовитьеёпокаработаетэкспорт,неждатьнового«давай». '
        'Источник runs/recovery_research_20261027/appearance_features/protocol.json. До публикацииобязательныепроверкикаталога.')
    print('Prepared full feature stage after passed cost; validate before publication')

if __name__=='__main__':main()
