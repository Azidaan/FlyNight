"""Record a supplementary exact explanation, without changing the experiment."""
import study_v2 as s
from study_v2 import *
QA=ROOT/'runs/historical_assignment_context_20261109'
def main():
    assert read(QA/'final_checks.json')['passed'];verify(read(OUT/'attribution_protocol.json'));d=read(OUT/'attribution.json');r=read(OUT/'summary.json');assert d['maximum_accounting_error']<1e-10;fail=[]
    for row in d['records']:
        c=row['variants']['C34'];by={v['kind']:v for v in c['candidates']};delta=np.array(c['own_minus_foreign_by_channel']);assert abs(delta.sum()-c['own_minus_foreign'])<1e-10
        if c['own_minus_foreign']>=0:continue
        t=float(delta[4:12].sum());other=float(np.r_[delta[:4],delta[12:]].sum());foreign=by['foreign']['rectangle_membership'];fail.append(dict(uid=row['uid'],margin=c['own_minus_foreign'],T4T5_contribution=t,other26_contribution=other,other26_dominates_wrong_margin=bool(other<0 and abs(other)>abs(t)),foreign_query_rectangle_counts={k:v['count'] for k,v in foreign.items()}))
    assert len(fail)==5 and sum(x['other26_dominates_wrong_margin'] for x in fail)==4
    assert sum(x['foreign_query_rectangle_counts']['foreign_only']>0 for x in fail)==4
    write('attribution_summary.json',dict(passed=True,errors=fail,wrong_orders=5,errors_with_foreign_only_sample_points=4,errors_other26_dominate=4,maximum_accounting_error=d['maximum_accounting_error'],no_new_winner_search=True,interpretation='Common bbox overlap cannot explain every wrong order. Four errors are arithmetically dominated by non-T4/T5 channels at saved C34 winners; that does not justify selecting/deleting channels on these cases. Point location is not neural ownership or a causal intervention.'))
    paragraphs='''
5. Где возникли пять неверных порядков

Выполнено только разложение уже выбранных максимумов. Новый поиск сдвига и вычисление карт не проводились. Сумма вкладов каналов и ячеек воспроизводит готовую оценку с ошибкой не более 3,331·10⁻¹⁶ (`attribution.json`, `attribution_summary.json`).

В запросах 4982, 7796, 11705 и 12206 перевес чужой рамки в основном создают 26 каналов вне T4/T5. В запросе 17832 перевес создаёт T4/T5, а остальные каналы немного помогают своей рамке. Поэтому утверждение «достаточно оставить T4/T5» не подтверждено этими данными. В зафиксированном вторичном варианте T4/T5 исправлен только один из пяти неверных выборов C34 — 7796; основной результат эксперимента остаётся прежним.

Для чужого кандидата в 4982 имеется 10 точек только внутри чужой размеченной рамки и 5 в пересечении. В 7796 — 8 только в чужой и 2 только в своей. В 11705 все 97 точек лежат в пересечении рамок. В 12206 — 23 только в чужой и 1 вне обеих. В 17832 — 33 только в чужой и 16 в пересечении. Следовательно, общее пересечение не объясняет все ошибки. При этом рамка разметки не является маской пикселей, а активность FlyVis зависит от окружающего пространства и прошлых входов; принадлежность сигнала одной машине здесь не доказана.

Разложение является учётом слагаемых формулы, а не причинным вмешательством в нейроны. Пороги, веса каналов, состояние ядра и решения трекера не изменены. Отбор каналов по этим пяти ошибкам означал бы подгонку на уже просмотренных случаях.
'''
    report=OUT/'ИТОГ.md';text=report.read_text(encoding='utf8');assert '5. Где возникли' not in text;report.write_text(text+'\n'+paragraphs.strip()+'\n',encoding='utf8');journal('Разложены пять ошибок C34 без повторного поиска',f'*Получено (`{OUT.name}`).* На 12 готовых запросах разложены уже выбранные максимумы C34 и двух обычных 34-канальных контролей; ни карты, ни 19-сдвиговый поиск не повторялись. Пять неверных порядков C34: в четырёх отрицательный перевес в основном дают 26 каналов вне T4/T5; в одном — T4/T5. У чужого кандидата четыре из пяти ошибок содержат точки только в чужой размеченной рамке; одно пересечение не объясняет всё. Ошибка суммы вкладов ≤3,331·10⁻¹⁶. Источники `runs/{OUT.name}/attribution_protocol.json`, `attribution_summary.json`, `ИТОГ.md`.\n\n*Граница и решение.* Это алгебраическое разложение на просмотренных данных, не причинное влияние отдельных нейронов и не улучшение ID. Не отбирать каналы по пяти ошибкам. Разметка использована лишь после оценки, рамка не является пиксельной маской. Итог и статус основного эксперимента не меняются.')
    print('Attribution recorded without changing catalogue, channels or criteria',flush=True)
if __name__=='__main__':main()
