"""Зарегистрировать замену извлекателя; короткая техническая ступень до полной серии."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.recovery_research_service_20261027 import RUN,read,write,sha,now,matching
from scripts.b123_safe_journal_20261027 import journal

def main():
    assert read(RUN/'fixed_composition/independent_audit.json')['passed']
    for sub in ['fixed_composition','fixed_composition_audit']:
        c=read(RUN/sub/'controller_result.json');assert c['exit_code']==0 and c['all_owned_ended'] and not any(matching(k,v) for k,v in c['owned'].items())
    folder=RUN/'frame_masa_profile';assert not folder.exists();old=read(RUN/'lazy_appearance_dev/protocol.json');source=old['sources'][0]
    files=['tracking/frame_masa_appearance.py','scripts/profile_frame_masa_association_20261027.py','tracking/masa_r50.py','tracking/lazy_appearance_association.py','weights/masa/masa_r50.pth',source['detections'],source['video_path']]
    p=dict(id='frame_masa_association_v1',created=now(),budget_seconds=600,device='cuda',author_stop=read(RUN/'authorization.json')['save_at'],source=source,frames=120,config=old['config'],
        action='Заменить извлекательMOTлюдей на существующий MASA-R50,сохраняя прежнее правило спорного первого назначения и реальное обновление состояния. '
            'Признаки256вычисляютсяизтекущего полного кадра и реальной области. Все наблюдения текущего кадра получают вектор до перехода кследующему,прошлые изображения не пересчитываются. '
            'Замороженныевеса/BN,FP32,прежнийкадровыйMASApreprocess1024иROIAlign,безновойзагрузки/обучения/изменениядетектора.',
        hypothesis='Более подходящий разным классам извлекатель может помочь реальным назначениям лучше прежнего извлекателя людей;это рабочая гипотеза,не установленная причина малого эффекта.',
        counterargument='Сильный старый MASA использовал офлайн-сшивку:его польза может происходить от другой задачи и временного охвата,а не качества признаков для соседнихкадров. '
            'Геометрическое окно/ошибочная прошлая область сохраняются. Полный кадр дороже вырезок;дешёвый профиль не доказывает цену полного опыта.',
        papers=[dict(url='https://arxiv.org/abs/2406.04221',role='ПервичнаяработаMASA,классонезависимые соответствияобластей;её результатне наш'),dict(url='https://github.com/siyuanliii/masa',role='Официальныеисходники;местныйизвлекатель уже существует')],
        acceptance=['Техническийпрофиль120кадров,все необходимые ключивесовзагружены,finite/строгоепрошлое/неизменнаяBN/нетGT;это ещёнеHOTA',
            'Полнаяследующая35devсетка[0,.1,.3]толькопослеуспехапрофиляифактическогоконцадерева;первоеПОЛНОЕвидео2*time*34+180<остаток7200с,иначестопбезпродления',
            'Полные40историческихvalпослефиксированиянаdev,триnull256итотжеисходныйтрекер. Польза—нижняя95%граница>0кбазе,прежнемуReIDи3null;однапрофильнаяработакоданеуспех'],
        limitation='Новаяинтеграциявпервуюассоциацию,не повтор старойофлайнсшивки,неполнаяMASA/BUSCA. Вычисляютсяи сохраняются все реальные наблюдения текущего кадра,это другойрежимстоимости,не только ленивые вырезки. '
            'Историческийvalненовыйнезависимыйтест,новые веса/данныене скачиваются. ЯдроFlyVisнеиспользуется,Б1–Б3сохраняютпрежниекритерии;основнойbaselineне меняется.',sha256={f:sha(ROOT/f) for f in files})
    write(folder/'protocol.json',p)
    path=ROOT/'experiments/project_history.json';before=path.read_bytes();cat=json.loads(before);assert not any(c['id']==p['id'] for c in cat['experiments'])
    cat['experiments'].append(dict(id=p['id'],title='Проверить классонезависимые признаки MASA в реальных назначениях',area='flyvis',state='ready',started='2026-10-27',
        question='Поможет ли замена извлекателя при том же причинном правиле назначения?',hypothesis=p['hypothesis'],expected=p['action'],summary=p['counterargument'],
        obtained='',interpretation='',decision='',cannot_claim=p['limitation'],acceptance=[dict(criterion=c,met=None) for c in p['acceptance']],metrics=[],commits=[],files=files[:4],
        evidence=[dict(kind='file',ref=folder.relative_to(ROOT).as_posix()+'/protocol.json',claim='Новыйизвлекатель,первичнаялитература,прежнеедействиеипроверкидо120кадров')]))
    assert path.read_bytes()==before;write(path,cat)
    write(RUN/'160_frame_masa_profile_prepared.json',dict(name='160_frame_masa_profile',folder=folder.relative_to(ROOT).as_posix(),script=files[1],script_sha256=sha(ROOT/files[1]),protocol_sha256=sha(folder/'protocol.json')))
    journal(p['id'],'поставлена замена извлекателя внутри прежнего правила назначений',
        '*Постановка.* '+p['action']+'\n\n*Обоснование и возражение.* '+p['counterargument']+' ПервичнаяMASA2406.04221иофициальныйрепозиторийпроверены;PDF34МБнеоткрылсявweb,абстрактиофициальнаястраницадоступны. '
        'Локальнаяtracking/masa_r50.pyполностьюпрочитанапоузламизагрузкевесов. Новая интеграция не воспроизводит старую сшивку.\n\n*Приёмка.* '+json.dumps(p['acceptance'],ensure_ascii=False)+' '+p['limitation']+
        ' Только600сGPUпрофильпослезелёныхпроверок,одинGPU/прежнийсрок. НикакойновойHOTAилигарантииприбавкипоканет.')
    print('Prepared one bounded MASA profile; full scientific series remains to implement')

if __name__=='__main__':main()
