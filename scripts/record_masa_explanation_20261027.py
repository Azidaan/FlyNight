"""Закончить учёт ценового стопа и уточнить смысл уже измеренной связки с MASA."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.recovery_research_service_20261027 import RUN,read,write,sha,now,matching
from scripts.b123_safe_journal_20261027 import journal

def main():
    r=read(RUN/'masa_saved_explanation/report.json');assert r['complete'] and len(r['per_video'])==20
    p=read(RUN/'frame_masa_fp16_dev/price.json');assert not p['proceed']
    for f in ['frame_masa_fp16_dev','frame_masa_fp16_dev_audit']:
        c=read(RUN/f/'controller_result.json');assert c['all_owned_ended'] and not any(matching(k,v) for k,v in c['owned'].items())
    path=ROOT/'experiments/project_history.json';before=path.read_bytes();cat=json.loads(before)
    fp=next(c for c in cat['experiments'] if c['id']=='frame_masa_fp16_association_v1');assert fp['state']=='ready' and not fp['obtained']
    fp.update(state='inconclusive',done='2026-10-27',obtained='runs/recovery_research_20261027/frame_masa_fp16_dev/price.json: '+json.dumps(p,ensure_ascii=False)+
        ';одно полное видео,остальные34не допущены,полного35reportнет. Контроллеры frame_masa_fp16_dev/controller_result.json и frame_masa_fp16_dev_audit/controller_result.json:all_owned_ended=true;фактический конец проверен в frame_masa_fp16_cost_stop_audit.json.',
        interpretation='Смешанная FP16 прошла короткую численную пробу,но плотная интеграция не прошла цену полной серии. Качество на35/40видео не измерено;это не отрицательный научный результат MASA.',
        decision='По прямому уточнению автора закончить учёт,не продолжать новые плотные разновидности MASA. Основной приоритет Б1→Б2→Б3;отдельно разобрать источник прибавки готовой сшивки по сохранённым данным.')
    fp['acceptance'][0].update(met=True,evidence='runs/recovery_research_20261027/frame_masa_fp16_profile/report.json:passed=true,numeric_pairs')
    fp['acceptance'][1].update(met=False,evidence='runs/recovery_research_20261027/frame_masa_fp16_dev/price.json:proceed=false')
    # Последний критерий качества остаётся неизвестным: полной серии не было.
    assert fp['acceptance'][2]['met'] is None
    original=next(c for c in cat['experiments'] if c['id']=='z13_val20_native_masa_v1')
    extra=' Дополнительная проверка сохранённых выходов: менялись толькоID,а не кадры/рамки/классы/баллы/число. Источник runs/recovery_research_20261027/masa_saved_explanation/report.json. '
    extra+='Проверенная связка включала замороженную логистическую регрессию по геометрии,миниатюре,гистограмме,входу глаза и MASA. Отдельный вклад признака MASA не изолирован;весь прирост нельзя приписать одной нейросети.'
    field='interpretation';old=original[field];original.setdefault('revisions',[]).append(dict(date='2026-10-27',commit='',field=field,old=old,reason='Прямое поручение автора уточнить,что именно дало прибавку связки с MASA.'))
    original[field]=old+extra
    original['evidence'].append(dict(kind='file',ref='runs/recovery_research_20261027/masa_saved_explanation/report.json',claim='Все20выходовизменяюттолькоID;фактическийсоставголовы;условнаясогласованностьобъединений'))
    assert path.read_bytes()==before;write(path,cat)
    write(RUN/'masa_hypothesis_authorization.json',dict(created=now(),user_request='Запиши в гипотезы MASA, и надо выяснить, почему оно вообще прирост дало',
        subsequent_scope='Б1–Б3 и подробнее разобрать причину:что именно дало прирост от MASA. Плотные новые MASAне продолжать,закончитьоформление.',
        allowed_changes=['ДобавитьодиндочернийузелБ1.3пропризнакиMASA','ДобавитьвнутреннийлучБ1→Б1.3','Двустороннийстелспереходксуществующейопорнойзадаче1'],
        preserve='Все прежние узлы гипотез побайтно,кроме блока дочернихподходовБ1;основнойтекст/статусы/цветаБ1–Б3не менять'))
    journal('frame_masa_fp16_association_v1','ценовой стоп FP16 записан,приоритет возвращён Б1–Б3',
        '*Получено.* '+fp['obtained']+'\n\n*Вывод.* '+fp['interpretation']+'\n\n*Решение.* '+fp['decision']+
        ' Карточкаисправленасreadyнаinconclusive;непровереннаяHOTAнепомеченапроваленной. Нужныкаталог/тесты/обзор.')
    journal('z13_val20_native_masa_v1','уточнён источник прибавки связки и разрешена одна запись MASA под Б1',
        '*Проверено.* runs/recovery_research_20261027/masa_saved_explanation/report.json:20видео/'+str(r['totals']['frame_count'])+'кадров/'+str(r['totals']['box_count'])+'рамок,во всех случаяхизменялисьтолькоID. '
        'ВидимыхисходныхID'+str(r['totals']['raw_ids'])+',послесшивки'+str(r['totals']['stitched_ids'])+'. УсловногруппсоднойувереннойGTличностью47,сконфликтом27,неизвестных/смешанных271;'
        'это только поддержанные опорные наблюдения,не точнаядельтаHOTAкаждойсвязи,не все принятые30Гцсвязи. Unknownнеотрицательныйпример. '
        'Фактическаяголоваbase_masaимеет10признаков,включаягеометрию/миниатюру/гистограмму/входглаза/MASA. '+r['unresolved']+
        '\n\n*Решение автора.* ОднапонятнаяMASAзаписьвдеревеБ1какпровереннаясвязка/сильныйконтроль;не новаяБ4,неподтверждениеБ1. Внешниепереходытолькостелс. '
        'Новыхобучений/GPU/TrackEvalне запускалось,ядроFlyVisнеиспользовалось. Б1–Б3основнойприоритет. Согласование сохранено в masa_hypothesis_authorization.json.')
    print('Recorded FP16 cost outcome and actual saved MASA mechanism boundaries')

if __name__=='__main__':main()
