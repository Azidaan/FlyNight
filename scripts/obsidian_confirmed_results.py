"""Source-checked individual effects; an effect is not the whole experiment's verdict."""
import hashlib
import json
import math

HUB='11 Лучшие результаты/Лучшие подтверждённые результаты'
GROUPS={'readout':'Улучшения считывания FlyVis',
        'representation':'Муха против входа глаза',
        'tracker':'Польза полному стенду'}
FINAL={'completed_positive','completed_negative','inconclusive','closed','superseded'}


def source_value(root, reference):
    root=root.resolve()
    path=(root/reference['source']).resolve()
    path.relative_to(root)
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=reference['sha256']:
        raise ValueError('Changed confirmed-result source: '+reference['source'])
    value=json.loads(raw)
    for part in reference['locator']:
        value=value[part]
    return value


def collect(root, catalog, glossary):
    results=[]; seen=set()
    for card in catalog['experiments']:
        for entry in card.get('confirmed_results',[]):
            if card['state'] not in FINAL:
                raise ValueError('Unfinished experiment in confirmed results: '+card['id'])
            if entry['id'] in seen:
                raise ValueError('Duplicate confirmed-result id: '+entry['id'])
            seen.add(entry['id'])
            if entry['category'] not in GROUPS or entry['metric'] not in glossary['metrics']:
                raise ValueError('Unknown category or metric: '+entry['id'])
            for name in ('title','comparison','subset','conditions','limitations'):
                if not entry.get(name):
                    raise ValueError('Missing '+name+': '+entry['id'])
            if entry['unit'] not in ('fraction','points'):
                raise ValueError('Unsupported effect unit: '+entry['id'])
            metric_range=glossary['metrics'][entry['metric']].get('range')
            if (entry['unit']=='fraction') != (metric_range==[0,1]):
                raise ValueError('Effect unit does not match the metric range: '+entry['id'])
            effect=source_value(root,entry['evidence'])
            mean=float(effect['mean_per_video']); lo,hi=map(float,effect['ci95'])
            if not all(math.isfinite(v) for v in (mean,lo,hi)) or not 0<lo<=mean<=hi:
                raise ValueError('Unconfirmed effect or invalid interval: '+entry['id'])
            if entry['unit']=='fraction' and hi>1:
                raise ValueError('Fraction effect outside its range: '+entry['id'])
            if not isinstance(effect['n'],int) or effect['n']<2:
                raise ValueError('Missing independent video count: '+entry['id'])
            baseline=float(source_value(root,entry['baseline'])) if entry.get('baseline') else None
            if baseline is not None and not math.isfinite(baseline):
                raise ValueError('Invalid baseline: '+entry['id'])
            results.append(dict(entry,card_id=card['id'],card_state=card['state'],
                parent_has_failed_criteria=any(c.get('met') is False for c in card.get('acceptance',[])),
                mean=mean,ci95=[lo,hi],n=effect['n'],baseline_value=baseline,
                relative_percent=100*mean/baseline if baseline else None))
    return results


def number(value, signed=False):
    return format(value,'+.2f' if signed else '.2f').replace('.',',')


def build(V, root, catalog, glossary, card_path, ref, fm, safe):
    results=collect(root,catalog,glossary)
    intro=('Это подборка отдельных измеренных сравнений, а не рейтинг разных метрик и не список готовых улучшений трекера. '
           'Доверительный интервал каждого сравнения выше нуля; условия, контроль и ограничения указаны отдельно. '
           'Большой прирост по полезным центрам не означает прирост HOTA.\n\n')
    body='# Лучшие подтверждённые результаты\n\n↑ [[00 Главная|Главная]]\n\n'+intro
    for group,title in GROUPS.items():
        items=[r for r in results if r['category']==group]
        group_path='11 Лучшие результаты/'+title
        body+=f'[[{group_path}|{title}]] — {len(items)} сравнений.\n\n'
        contents=f'# {title}\n\n↑ [[{HUB}|Лучшие результаты]]\n\n'+intro
        if not items:
            contents+='В подборке пока нет записей этой категории; это не оценка всей истории проекта.\n'
        for item in sorted(items,key=lambda x:(x['metric'],x['unit'],-x['mean'],x['id'])):
            scale=100 if item['unit']=='fraction' else 1
            unit='п.п.' if item['unit']=='fraction' else 'пункта'
            mean=number(item['mean']*scale,True)
            interval='['+number(item['ci95'][0]*scale,True)+'; '+number(item['ci95'][1]*scale,True)+']'
            path='11 Лучшие результаты/Сравнения/'+safe(item['id'])
            large=' · сдвиг от 10 п.п.' if item['unit']=='fraction' and item['mean']>=.1 else ''
            contents+=f'[[{path}|{item["title"]}]] — **{mean} {unit}**, 95% ДИ {interval}, {item["subset"]}{large}.\n\n'
            original=ref('11',card_path[item['card_id']],'Исходная карточка: гипотеза, критерии и полный итог')
            metric=glossary['metrics'][item['metric']]['title']
            note=f'# {item["title"]}\n\n↑ [[{group_path}|{title}]]\n\n{original}\n\n'
            note+=f'**Результат:** {mean} {unit}; 95% доверительный интервал {interval}; видео: {item["n"]}.\n\n'
            note+=f'**Метрика:** {metric}.\n\n**Сравнение:** {item["comparison"]}\n\n**Выборка:** {item["subset"]}\n\n'
            note+=f'**Условия:** {item["conditions"]}\n\n**Ограничения:** {item["limitations"]}\n\n'
            state_ru={'completed_positive':'положительный','completed_negative':'отрицательный',
                      'inconclusive':'неопределённый','closed':'закрыт','superseded':'заменён'}[item['card_state']]
            note+=f'**Исход полной карточки:** {state_ru}.\n\n'
            if item['parent_has_failed_criteria']:
                note+='В полном эксперименте есть непройденные критерии; это отдельное положительное сравнение, а не успех всего эксперимента.\n\n'
            ev=item['evidence']
            note+=f'**Первичный источник:** `{ev["source"]}`, ключ `'+'.'.join(ev['locator'])+'`.\n\n'
            if item['relative_percent'] is not None:
                note+='**Относительно указанного контроля:** '+number(item['relative_percent'],True)+'%; это та же метрика, не HOTA.\n\n'
            else:
                note+='Относительный процент не вычислен: абсолютная база не указана либо равна нулю.\n\n'
            note+='Доля измеренного потолка не определена: соответствующий потолок для этого сравнения не записан.\n'
            V.write(path+'.md',fm({'tags':['лучший-результат','результат/'+group]})+note)
        V.write(group_path+'.md',fm({'tags':['лучшие-результаты','группа']})+contents)
    body+='При последующих итогах агент добавляет проверенное сравнение в поле confirmed_results исходной карточки; раздел обновляется генератором. Правила: `experiments/CONFIRMED_RESULTS.md`.\n'
    V.write(HUB+'.md',fm({'tags':['центр','лучшие-результаты']})+body)
