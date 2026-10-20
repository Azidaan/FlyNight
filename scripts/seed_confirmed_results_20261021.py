"""Register already measured effects without changing their original experiment verdicts."""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.obsidian_confirmed_results import collect
from scripts.z13_record_next import journal


def ref(source,*locator):
    return dict(source=source,locator=list(locator),sha256=hashlib.sha256((ROOT/source).read_bytes()).hexdigest())


def main():
    hp=ROOT/'experiments/project_history.json'; gp=ROOT/'experiments/metrics_glossary.json'
    h=json.loads(hp.read_text(encoding='utf-8')); g=json.loads(gp.read_text(encoding='utf-8'))
    g['metrics']['useful_center_fraction']=dict(title='Доля полезных центров, среднее по видео',
        ru='Для каждого видео: доля положительных случаев, в которых оценённый центр точнее прогноза и проходит фиксированный порог ошибки; затем среднее по видео.',
        better='higher',unit='доля',range=[0,1],tool='scripts/z13_continue.py:per_video_useful; scripts/z13_train.py:success',
        caveat='Это локализация заданной потерянной цели, не HOTA и не точность среди выбранных вмешательств; неизвестные исходы отдельно.',
        source='scripts/z13_train.py:success')
    cards={c['id']:c for c in h['experiments']}
    train='runs/z13_target_center_20261018/report.json'
    dev='runs/z13_target_center_dev_20261018/report.json'
    conditions='Одни и те же эпизоды внутри сравнения; фиксированные сохранённые ядра и условия считывания; парный бутстрэп по видео; веса ядра здесь не переобучались.'
    def add(cid,rid,title,category,source,contrast,baseline,subset,comparison,limitations,baseline_ref=None):
        item=dict(id=rid,title=title,category=category,metric='useful_center_fraction',unit='fraction',
            evidence=ref(source,'paired',contrast),baseline=baseline_ref or ref(source,'summary',baseline,'useful_mean_video'),
            subset=subset,comparison=comparison,conditions=conditions,limitations=limitations)
        card=cards[cid]
        assert not any(x['id']==rid for x in card.get('confirmed_results',[])),rid
        card.setdefault('confirmed_results',[]).append(item)
    exploratory='Поисковая train26 многократно использована; это не независимая итоговая проверка, не HOTA и не доказательство надёжного отбора поправок.'
    reused='dev35 ранее использовался в проекте; это не слепая оценка, не HOTA и не доказательство надёжного отбора поправок.'
    for arm in ('C','D'):
        add('z13_target_center_v1',arm+'_center_vs_R1_train26',arm+' с центрированной целью против входа глаза — train26',
            'representation',train,arm+'_center-R1_center','R1_center','train26, 26 видео; поисковая проверка',
            arm+' с центрированной регрессионной целью против R1 с той же целью; контроль использует вход глаза без мухи.',
            exploratory+' Преимущество именно обучения ядра D над штатной C этим сравнением не устанавливается.')
    add('z13_target_center_v1','C_center_vs_binary_train26','Центрированная цель вместо маски — C, train26',
        'readout',train,'C_center-C_binary','C_binary','train26, 26 видео; поисковая проверка',
        'На том же ядре C центрированная регрессионная цель считывания против плоской маски объекта.',exploratory)
    for arm in ('C','D'):
        add('z13_target_center_dev_v1',arm+'_center_vs_binary_dev35','Центрированная цель вместо маски — '+arm+', dev35',
            'readout',dev,arm+'_center-'+arm+'_binary',arm+'_binary','dev35, 35 видео; повторно использованная выборка',
            'На том же ядре '+arm+' центрированная регрессионная цель считывания против плоской маски объекта.',reused)
        add('z13_target_center_dev_v1',arm+'_center_vs_R1_dev35',arm+' с центрированной целью против входа глаза — dev35',
            'representation',dev,arm+'_center-R1_center','R1_center','dev35, 35 видео; повторно использованная выборка',
            arm+' против входа глаза R1 с тем же способом считывания.',reused+' Это не доказательство преимущества обучения ядра D над штатной C.')
    add('z13_target_center_dev_v1','D_center_vs_prior_dev35','Новое считывание против прежнего на той же D — dev35',
        'readout',dev,'D_center-D_prior','D_prior','dev35, 35 видео; повторно использованная выборка',
        'Онлайн-модель цели с центрированной целью против прежнего считывателя при неизменном ядре D.',reused)
    source='runs/z13_context_transfer_20261020/report.json'
    item=dict(id='C_directional_vs_R1_dev35',title='Пространственное считывание C против входа глаза — dev35',
        category='representation',metric='useful_center_fraction',unit='fraction',
        evidence=ref(source,'localization','C_directional-R1_directional'),
        baseline=ref('runs/z13_context_transfer_20261020/recovery_summary_audit.json','levels_mean_per_video','R1_directional'),
        subset='dev35, 35 видео; повторно использованная выборка',comparison='Направленные пространственные признаки C против аналогичного считывания входа глаза R1.',
        conditions=conditions,limitations=reused+' Составные критерии улучшения относительно прежней C и отбора поправок НЕ пройдены.')
    cards['z13_context_transfer_v1'].setdefault('confirmed_results',[]).append(item)
    results=collect(ROOT,h,g)
    assert len(results)==9
    hp.write_text(json.dumps(h,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    gp.write_text(json.dumps(g,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    journal('создана подборка лучших подтверждённых сравнений FlyVis',
        'По просьбе автора создан блок11 Лучшие подтверждённые результаты; правила — experiments/CONFIRMED_RESULTS.md и AGENTS.md. '
        'В исходные карточки добавлены9ссылок confirmed_results на уже измеренные сравнения; ни одно итоговое поле, критерий или общий исход опыта не пересмотрен. '
        'Числа и интервалы читаются из первичных JSON с проверкой SHA256; новые расчёты ради графа не запускались. '
        'Есть группы улучшения считывания, мухи против входа и пользы стенду. Порог10п.п. используется только как визуальная метка крупных сдвигов. '
        'Каждый узел ведёт на первоначальную карточку с гипотезой и критериями, показывает выборку и ограничения; непройденные составные критерии не скрываются.')
    print('Registered',len(results),'source-verified comparisons')


if __name__=='__main__': main()
