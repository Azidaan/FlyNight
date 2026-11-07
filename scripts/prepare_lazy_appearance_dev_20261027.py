"""Новое действие: внешний вид влияет на онлайн-назначение и последующее состояние."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.recovery_research_service_20261027 import RUN,read,write,sha,now
from scripts.b123_safe_journal_20261027 import journal

def main():
    folder=RUN/'lazy_appearance_dev';assert not folder.exists();assert read(RUN/'pixel_box_numeric_control/independent_audit.json')['passed']
    sources=[];bound=['tracking/lazy_appearance_association.py','scripts/run_lazy_appearance_association_20261027.py','scripts/lazy_appearance_dev_20261027.py',
        'tracking/xywh_bytetrack.py','tracking/recovery_admission_features.py','tracking/busca_runtime.py','weights/busca_official_20261027/model_busca.pth']
    baseline=read(RUN/'pixel_box_dev/report.json')['per_video']['baseline']
    for v in read(RUN/'pixel_box_dataset/protocol.json')['videos']['dev']:
        s=dict(video=v,baseline=f'runs/recovery_research_20261027/admission_dataset/dev/{v}/baseline.json',alignment=f'runs/train_cache/{v}/alignment.json',detections=f'runs/train_cache/{v}/detections/raw_video.npz',video_path=f'data/bdd100k/videos/train/{v}.mov',baseline_metrics=baseline[v]);sources.append(s)
        bound += [s[k] for k in ['baseline','alignment','detections','video_path']]
    p=dict(id='lazy_appearance_association_v1',created=now(),phase='dev',budget_seconds=7200,author_stop=read(RUN/'authorization.json')['save_at'],sources=sources,
        config=read(RUN/'admission_dataset/protocol.json')['config'],weights_grid=[0.,.1,.3],ambiguity_margin=.15,
        action='Добавить проверку внешнего вида внутрь первого назначения XYWHByteTrack. Прежняя геометрическая стоимость и её кандидаты сохраняются. '
               'Только при двух близких кандидатах (разрыв<=.15) или конкуренции нескольких треков за детекцию стоимость допустимого ребра получает weight*(1-cosine)/2. '
               'Сеть считает признаки по текущей детекции и последней реальной прошлой детекции трека; решение действительно меняет назначение и дальнейшее состояние. '
               'Нет новых рёбер вне прежнего порога,нет нового class-гейта,вторая/неподтверждённаяассоциацияне меняются. Нулевой вес точно даёт исходный трекер.',
        hypothesis='Внешний вид спорных реальных назначений может помочь сохранению личности; это ещё не измеренная польза.',
        counterargument='ГотовыйReIDобучалсяпреимущественнонаMOTлюдях,аBDDсодержитразныеклассы/масштабы;области128×384могутискажатьвнешнийвид. '
            'Ошибка прошлых назначений портит память;геометрическийгейтможетвообщенеиметьправильногокандидата. Добавочныйштрафможетлишьменятьчислоотказов,поэтому нужны3контролясбессмысленнымипризнаками. '
            'ИзмеренныйGTрезервличностинеобещаетегообучаемость/достижимость.',
        difference_from_previous='Не сшивка завершённых обрывковMASA,не class_gate после30кадров,не сопоставление трёх пиков FlyVis,не выходное восстановление BUSCA. '
            'МеняютсяреальныеназначенияиKalman-состояниявпотокекадров. ПолнаяавторскаяBUSCA/DeepSORT/DeepOCSORTне воспроизводятся;используется только существующий извлекатель.',
        papers=[dict(url='https://arxiv.org/abs/1703.07402',role='Обоснование связи внешнего вида с реальными назначениями'),dict(url='https://arxiv.org/abs/2302.11813',role='Обоснование избирательного применения внешнего вида; метод не копируется целиком')],
        feature_regime='Полные379готовыхключейcheckpointстрогозагружаются,используетсятолькоReID512;замороженнаяBN,use_domain_adaptation=False,eval,FP32,batch16,L2. '
            'Та жеавторскаяBGRнормировка,нашcropбезJPEG/авторскогоpadding. Признакиленивые,толькоконкретныепрошлые/текущиедетекции;общийкэшнеобмениваетсяназначениями. '
            'СчётчикиIDкаждоговариантанезависимы. Случайныеконтролипосид42/43/44получаюттежеобластииштраф,нодетерминированныебессмысленныевекторы.',
        acceptance=['Полный35dev,нулевойпутьточноповторяетбазу;выборweight[0,.1,.3]толькопоdevHOTA,приравенствеменьше,затемфиксированныетеже40val',
                    'Причиннаяпамять,неизменныедетекции/порогидетектора,замороженнаяBN/веса;реальныеизмененияназначений/состояниянепутатьсправкойвыдачи',
                    'ПолныеHOTA/AssA/IDF1/FP/FN/IDSw;положительнаянижняя95%границакбазеинавсех3бессмысленныхконтроляхна40. Приweight0нулевойрезультатнеуспех.'],
        cost_gate='ПервоеПОЛНОЕdev-видеосо3полнымитрекерами/признаками/оценкой:2*time*34+180<остаток7200с;иначестопбезпродления. Первое видео не повторять. Затемвалотдельный7200сигейтпервогополного,толькопослеуспешногодев-аудита.',
        limitation='Извлекательнепереобучается,триnullseedнеявляютсятремяобученияминастоящейсети. Devвиделдетектор,это выборполитики,неегообобщение. '
            '40valисторические,не новыйнезависимыйтест. MASAофлайннапервых20—отдельнаясильнаясправка,несмешиватьеёчислаиусловияс40/онлайном. '
            'Система всей работы,неядро/модульFlyVisи не научная проверкаБ1–Б3. Основнойbaselineне меняется.',
        sha256={f:sha(ROOT/f) for f in set(bound)})
    write(folder/'protocol.json',p)
    path=ROOT/'experiments/project_history.json';before=path.read_bytes();cat=json.loads(before);assert not any(c['id']==p['id'] for c in cat['experiments'])
    cat['experiments'].append(dict(id=p['id'],title='Проверять внешний вид спорных назначений внутри трекера',area='flyvis',state='ready',started='2026-10-27',
        question='Помогает ли внешний вид реальных спорных назначений при обновлении основного трекера?',hypothesis=p['hypothesis'],expected=p['action'],summary=p['counterargument'],
        obtained='',interpretation='',decision='',cannot_claim=p['limitation'],acceptance=[dict(criterion=c,met=None) for c in p['acceptance']],metrics=[],commits=[],files=bound[:3],
        evidence=[dict(kind='file',ref='runs/recovery_research_20261027/lazy_appearance_dev/protocol.json',claim='Новое действие, первичная литература, контроли, цена и критерии до серии')]))
    assert path.read_bytes()==before;write(path,cat)
    write(RUN/'lazy_appearance_dev_job_prepared.json',dict(name='130_lazy_appearance_dev',folder=folder.relative_to(ROOT).as_posix(),script='scripts/lazy_appearance_dev_20261027.py',script_sha256=sha(ROOT/'scripts/lazy_appearance_dev_20261027.py'),protocol_sha256=sha(folder/'protocol.json')))
    journal(p['id'],'поставлена проверка внешнего вида внутри реальных назначений',
        '*Постановка.* '+p['action']+'\n\n*Отличие.* '+p['difference_from_previous']+'\n\n*Возражение.* '+p['counterargument']+
        '\n\n*Контроли/стоимость.* '+p['feature_regime']+' '+p['cost_gate']+' '+p['limitation']+
        ' ПервичныеработыDeepSORT(1703.07402),DeepOCSORT(2302.11813)иBoTSORT(2206.14651)прочитаны,ихчисланенашрезультат. '
        'Сверены старые association_log/report.json, class_gate/report.json и карточки сшивки/пиков:такаяинтеграциявпервыйпроходотличаетсяотних. '
        'Дватеста:отключённыйпуть/полныеKalman-состояния100кадровточны,искусственныйпримерпроверяетизменениестоимоститолькодопустимыхпариспрошлойобластью. '
        'ПолнойновойHOTAещёнет. Публикацияпослеобязательныхпровероккаталога,одинGPU,тотжесрок28.10 07:40МСК.')
    print('Prepared actual online assignment intervention and first full-video price gate')

if __name__=='__main__':main()
