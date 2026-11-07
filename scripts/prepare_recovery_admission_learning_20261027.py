"""Протокол реального обучения и дальнейшего принятия по HOTA, до обучения."""
from pathlib import Path
from datetime import datetime,timezone
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.recovery_research_service_20261027 import RUN,read,write,sha
from scripts.b123_safe_journal_20261027 import journal

def main():
    folder=RUN/'admission_learning';assert not folder.exists()
    sources=['scripts/train_recovery_admission_20261027.py','tracking/learned_recovery_admission.py',
        'runs/recovery_research_20261027/admission_dataset/audit.json','runs/recovery_research_20261027/appearance_features/protocol.json']
    p=dict(id='recovery_admission_learning_v1',created=datetime.now(timezone.utc).isoformat(),budget_seconds=1800,
        author_stop=read(RUN/'authorization.json')['save_at'],device='cuda',arms=['numeric','appearance'],seeds=[42,43,44],
        epochs=100,batch_size=256,lr=.001,weight_decay=.0001,projection_seed=271027,
        action='Обучить одинаковые головы допуска на раздельныхtrain/dev,не меняя кандидатов/трекера. '
               'Числовой контроль: нормированные признаки24 и фиксированная нелинейная проекция tanhв2048размеров. '
               'Внешнийвид: те же24числовыхпризнака,нормированныевекторы прошлой/текущейобласти512,ихмодульразности и произведение. '
               'Обе головы имеют один размер2072,LayerNorm/Linear32/ReLU/Linear1,одинаковое числообучаемыхпараметров. '
               'Нормировки толькоиз trainсизвестнымиметками,unknownвошибкуневходит;предсказаниядляunknownсохраняются. '
               'Три одинаковыхсида/порядкапримеров,100эпох,невзвешеннаядвоичнаяперекрёстнаяэнтропия. Лучшийcheckpointкаждогосида выбираетсятолькопоdevBCE.',
        hypothesis='Реальные области могут помочь отбору больше,чем числовые признаки при одинаковой голове; полезность полного трекера неизвестна.',
        counterargument='Прежние считыватели и числовые признаки не обеспечили пользу. Новыйизвлекательможетбытьнепригоденкдомену,меткинеполныинеравныдельтеHOTA. '
                        'ВыборпоBCEсампособенеподтверждаетпользуHOTA;нуженполныйвыходиотдельныйконтрольправила.',
        cost_gate='Первые3эпохипервогоконтроля:2*измеренноевремя/3*оставшиесяэпохивсех6моделей+120с должныпоместитьсявостаток1800с. Иначеостановить безпродления.',
        next_threshold_protocol='После фиксациивесов выбратьпорогкаждоймодели тольконаполнойdevHOTAиззаранеезаданных[0.5,0.7,0.9]. '
                                'Потомзафиксироватьпорогивеса,оценитьтеже40историческихvalбезподбора. Сохранитьвсе3сида. '
                                'Контроли: исходныйбазис,фиксированноеправило,числоваяголова,головасвнешнимвидом. Самостоятельногоизменениякандидатов/пороговнаvalнет.',
        acceptance=['Полноеобучение3сидовобоихвариантовнаодинаковыхtrain/dev,равноечислопараметров,unknownнеnegative,фиксированныеисточники',
                    'Выборcheckpointтолькопоdev;последующийвыборпорогатолькозаранеезаданнойсеткойdevHOTA,безподгонапоval',
                    'ПодтверждениеполезностипоследующейполнойHOTA:положительнаянижняяграницакбазеикфиксированномуправилу;длявкладавнешнеговида—такжекчисловомуконтролю,сразбросомсидов'],
        limitation='ЭтообучениеещёнеполныйэкспериментHOTA;выводпопризнакам/ошибкеклассификации не заменяеттрекер. '
                   'Всеvalимеютисториюпроекта,нельзяназыватьновымнезависимымитоговымтестом. Проекциячисловогоконтроляне добавляетновыхнаблюдений,ноуравниваетразмерголовы. '
                   'Сравнениевсегопроекта,неядраFlyVis;GTнедоступенвыбирающеймоделипривыводе.',
        sha256={s:sha(ROOT/s) for s in sources})
    write(folder/'protocol.json',p)
    path=ROOT/'experiments/project_history.json';before=path.read_bytes();cat=json.loads(before);assert not any(c['id']==p['id'] for c in cat['experiments'])
    cat['experiments'].append(dict(id=p['id'],title='Обучаемый допуск продолжений — числовой контроль и реальные области во времени',area='flyvis',state='ready',started='2026-10-27',
        question='Поможет ли обучаемый отбор продолжений полному трекеру и добавит ли пользу внешний вид реальных областей?',hypothesis=p['hypothesis'],expected=p['action'],summary=p['counterargument'],
        obtained='',interpretation='',decision='',cannot_claim=p['limitation'],acceptance=[dict(criterion=c,met=None) for c in p['acceptance']],metrics=[],commits=[],files=sources[:2],
        evidence=[dict(kind='file',ref='runs/recovery_research_20261027/admission_learning/protocol.json',claim='Архитектура,3сида,ошибка,допускстоимости,выборпорогаHOTAикритериидосерии')]))
    assert path.read_bytes()==before;write(path,cat)
    job=dict(name='050_admission_learning',folder='runs/recovery_research_20261027/admission_learning',script=sources[0],script_sha256=sha(ROOT/sources[0]),protocol_sha256=sha(folder/'protocol.json'))
    write(RUN/'admission_learning_job_prepared.json',job)
    journal(p['id'],'поставлено полное обучение допуска с равными головами и тремя сидами',
        '*Постановка.* '+p['action']+'\n\n*Сильное возражение.* '+p['counterargument']+'\n\n*Стоимость.* '+p['cost_gate']+'\n\n*Принятие.* '+p['next_threshold_protocol']+' '
        'КритерииполнойHOTAсохраненывкарточке,доинференсаониостаютсянепроверенными;classifierBCEнеуспех. '
        'Источник runs/recovery_research_20261027/admission_learning/protocol.json. Публикацияпослеполного040игринпровероккаталога,одинGPU,прежнийсрок28.10 07:40МСК.')
    print('Prepared complete matched heads training, not a HOTA claim')

if __name__=='__main__':main()
