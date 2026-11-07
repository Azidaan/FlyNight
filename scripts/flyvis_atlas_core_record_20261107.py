"""Write the Russian result and verify that the supervised process tree ended."""
import json
import hashlib
from pathlib import Path
import psutil

BASE=Path('runs/flyvis_atlas_core_20261107')


def record():
    def read(path):
        return json.loads(path.read_bytes())
    a=read(BASE/'independent_audit.json')
    p=read(BASE/'protocol.json')
    finished=read(BASE/'inference_completed.json')
    controller=read(Path(finished['guard']['supervisor_report']))
    assert a['passed'] and controller['returncode']==0 and controller['reason'] is None
    for path,expected in p['source_sha256'].items():
        h=hashlib.sha256()
        with Path(path).open('rb') as f:
            for b in iter(lambda:f.read(1<<20),b''):h.update(b)
        assert expected==h.hexdigest(),path
    ended={}
    for pid,birth in controller['owned'].items():
        try: alive=psutil.Process(int(pid)).create_time()==birth
        except psutil.NoSuchProcess: alive=False
        assert not alive,pid
        ended[pid]=birth
    conditions={c['name']:c for c in a['conditions']};n=a['pairs'];original=conditions['native']['correct']
    sets=[conditions['0.90/'+t] for t in p['types']]
    assert len(sets)==65 and n==172
    positive=[c for c in sets if c['ci_family65_pp'][0]>0]
    negative=[c for c in sets if c['ci_family65_pp'][1]<0]
    active=[c for c in sets if c['choice_changes']];score_only=[c for c in sets if not c['choice_changes'] and c['max_abs_margin_change']>1e-7]
    def num(x): return f'{x:.3f}'.replace('.',',')
    def names(items):return ', '.join(c['name'].split('/')[1] for c in items) if items else 'нет'
    text=f"""# Однократное изменение внутреннего состояния FlyVis перед выбором рамки

Измерены все {a['units']} типов клеток в 172 исторических парах из {a['videos']} видео. Каждая пара имеет две допустимые рамки и одну известную верную метку. В исходном состоянии считывание 34 прямых каналов по полным рамкам дало {original:g}/{n} верных выборов ({num(original/n*100)}%).

Сеть получила архивную последовательность изображений глаза до предпоследнего кадра включительно. Для каждого типа по отдельности все его значения в копии состояния умножены на 0,90. Следующий реальный кадр и два внутренних шага обновления прошли через прежние связи. Симметричный множитель 1,10 проверен на тех же типах. Коэффициенты двух оценок рамок сохранены из истории цели. Масштабирование значений на десять процентов даёт разную абсолютную энергию вмешательства для типов разной численности и исходной активности.

При ослаблении хотя бы один выбор изменился у {len(active)} типов. Ещё у {len(score_only)} типов менялся отрыв между рамками более чем на 10⁻⁷ без смены выбора. Установленный допуск ничьей составляет 10⁻¹⁰. Положительная нижняя граница парного интервала с поправкой на 65 сравнений обнаружена у {len(positive)} типов ({names(positive)}), отрицательная верхняя граница — у {len(negative)} ({names(negative)}). Для краткости ниже показан один заранее фиксированный знак импульса. Значения для усиления, интервалы и число ничьих доступны в [независимом аудите](independent_audit.json).

Это проверка немедленного эффекта одного импульса в конкретном состоянии сети. Для более длинного пути между типами может потребоваться больше двух внутренних шагов. Сырые значения активности меняются до передачи по связям. Измерены баллы соперничающих рамок, а длительную идентичность трека и поведение на новых видео следует оценивать отдельным опытом. Набор уже использовался в предшествующих исследованиях. Выбор вмешательства по наблюдаемой здесь точности требует новой выборки для проверки.

| Тип | Изменили исход | Стало лучше | Стало хуже | Верных в итоге | Изменение, п.п. | Интервал с поправкой, п.п. |
|---|---:|---:|---:|---:|---:|---:|
"""
    for c in sets:
        name=c['name'].split('/')[1];lo,hi=c['ci_family65_pp']
        text+=f'| {name} | {c["choice_changes"]} | {c["gains"]} | {c["losses"]} | {c["correct"]:g} | {num(c["paired_delta_pp"])} | {num(lo)} … {num(hi)} |\n'
    text+=f'\nМаксимальная абсолютная разность исходной нормированной карты и архивной — {a["archive_max_abs_map"]:.5g}, тождественной копии и нетронутого вычисления — {a["identity_max_abs_map"]:.5g}. Контроллер завершился с кодом 0. При выполнении оставалось не менее {controller["minimum_available_bytes"]/2**30:.2f} ГиБ системной памяти. [Протокол](protocol.json), [контрольные суммы выходов](inference_completed.json).\n'
    (BASE/'РЕЗУЛЬТАТ.md').open('x',encoding='utf-8').write(text)
    receipt=dict(**controller,all_owned_ended=True,ended_identities=ended,input_sha256_reverified=len(p['source_sha256']))
    (BASE/'controller_verified.json').open('x',encoding='utf-8').write(json.dumps(receipt,indent=2)+'\n')
    print('Baseline',original,'types changing choice',len(active),'improvements',names(positive),'worse',names(negative))


if __name__=='__main__': record()
