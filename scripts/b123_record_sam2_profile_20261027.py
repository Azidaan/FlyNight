"""Записать технический профиль отдельно от будущего HOTA."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
def main():
    source='runs/b123_20261027/sam2_runtime_profile/report.json'
    run=ROOT/'runs/b123_20261027/sam2_runtime_profile';r=json.loads((ROOT/source).read_text(encoding='utf-8'))
    c=json.loads((run/'controller_result.json').read_text());assert c['exit_code']==0 and c['all_owned_ended'] and r['complete']
    assert r['all_image_requests_causal'] and r['late_second_object_added'] and r['returned_object_removed']
    path=ROOT/'experiments/project_history.json';before=path.read_bytes();cat=json.loads(before)
    card=next(x for x in cat['experiments'] if x['id']=='sam2_mask_runtime_profile_v1');assert card['state']=='ready'
    card.update(state='infrastructure',done='2026-10-27',
        obtained=source+': coverage.state_keys=471, all_image_requests_causal=true, initialized=2, propagated=21, '
            'late_second_object_added=true, returned_object_removed=true, peak_allocated_bytes=745392128, seconds=11.406. '
            'Профиль30кадров с искусственно отключёнными назначениями, не оценка точности.',
        interpretation='Причинный ввод и официальный перенос маски работают. Профиль не измеряет пользу HOTA и не воспроизводит LTR.',
        decision='Перейти к отдельно зарегистрированному полному видеопроходу с неизменным XYWH и допуском стоимости до всей серии.')
    for item in card['acceptance']:item.update(met=True,evidence=source)
    card['evidence'].append(dict(kind='file',ref=source,claim='Технический профиль, не научный результат'))
    assert path.read_bytes()==before;path.write_text(json.dumps(cat,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    from scripts.b123_safe_journal_20261027 import journal
    journal(card['id'],'причинный масочный профиль прошёл, HOTA ещё не измерена',
        '*Получено.* '+card['obtained']+'\n\n*Вывод.* '+card['interpretation']+'\n\n*Решение.* '+card['decision']+' '
        'Все ключи официальных весов загружены строго. Основное окружение не менялось, расширениеCUDAне собиралось. '
        'Сеть получает только уже декодированные изображения; обращение к будущему запрещено проверяемым хранилищем. '
        'max_frame_num_to_track=0 даёт ровно текущий кадр, так как конец диапазона в официальном API включителен. '
        'Сравнение с baseline в процентах пока невозможно: HOTA нет.')
    with (ROOT/'runs/b123_20261027/HANDOFF.md').open('a',encoding='utf-8') as f:
        f.write('\n\n## '+datetime.now(timezone(timedelta(hours=3))).strftime('%H:%M')+' МСК — собственная масочная ветвь технически готова\n\n'
            'sam2_runtime_profile/report.json complete=true:471ключ,30кадров,2объекта,21перенос,11.406с,peak745392128байт. '
            'Причинный ввод/late-add/remove подтверждены,2теста причинности/IoU passed. '
            'Это НЕLTR и НЕSGAR: SAM2.1tiny, нативныйobject_score>0,маска>0,IoUотказ0.9,нет27масок/селектора/DeAOT. '
            'Источники вtools/sam2-runtime-20261027,commit2b90b9f5ceec907a1c18123530e92e794ad901a4, '
            'веса156008466байт,SHA7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69. '
            'Hydra/OmegaConf/ANTLR изолированывtools/sam2-deps-20261027,основное окружение не менялось. '
            'Готов scripts/b123_sam2_output_pilot_20261027.py; после постановки/зелёныхпроверок публиковать120. '
            'Пилот7200с,первое полное видео решаетцену2*time*19+120; не подбиратьпороги/неповторятьготовое. '
            'БазовыйXYWHнеизменный,сравнениеbaseline/simple/BUSCA,контрольвсехrawиHOTA. '
            'Графобновлёнтолькофундаментальнойзадачей4послеотрицательнойBUSCA,40видимых/0удалений/гипотезыпобайтны. '
            'Ночнойсрок07:30сохраняется,одинGPU.\n')
    print('Technical SAM2 result recorded')
if __name__=='__main__':main()
