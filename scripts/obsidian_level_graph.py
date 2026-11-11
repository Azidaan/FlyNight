"""Generate isolated, numbered navigation trees from the project's primary records."""
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import quote
import hashlib
import json
import re
from datetime import date, datetime, timedelta, timezone
import calendar

from scripts.build_obsidian_vault import ROOT, safe
from scripts.obsidian_confirmed_results import GROUPS as RESULT_GROUPS, collect
from scripts.obsidian_fly_problems import GROUPS, PROBLEMS, status_of
from scripts.obsidian_system_problems import load as system_problems

FOLDER = 'Карта проекта'
ACTIVE = 'граф-проекта/видно'
SECTION_NAMES = ('Гипотезы', 'Эксперименты', 'Термины', 'Журнал', 'План',
                 'Исследования', 'Проблемы мухи', 'Лучшие результаты')
COLORS = {'ожидает': 0xe74c3c, 'в-работе': 0xf2c744, 'выполнено': 0x39ab61,
          'без-положительного-итога': 0x9b59b6}
STATES = {'planned':'запланирован', 'ready':'готов к запуску', 'running':'выполняется',
          'unknown':'статус неизвестен', 'completed_positive':'завершён с положительным результатом',
          'completed_negative':'завершён с отрицательным результатом', 'inconclusive':'неопределённый результат',
          'closed':'закрыт', 'superseded':'заменён другой постановкой', 'infrastructure':'инфраструктура'}
LINK = re.compile(r'\[\[([^|\]#]+)(?:#[^|\]]*)?(?:\|([^\]]+))?\]\]')

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def plain(text):
    # The source note remains accessible. Its cross-links must not join the trees.
    text = LINK.sub(lambda m:m[2] or Path(m[1]).name, str(text or ''))
    text = re.sub(r'!?\[([^\]\n]*)\]\((?!https?://|obsidian://|file:///)[^)\n]+\)',r'\1',text)
    return text.replace('[[','［［').replace(']]','］］')

def source_note(path):
    return '[Открыть исходную заметку](obsidian://open?vault=ObsidianFlyVis&file='+quote(path, safe='')+')'

def local_file(name,label):
    return '['+label+']('+(ROOT/name).as_uri()+')'

def original(vault,path):
    text=(vault/(path+'.md')).read_text(encoding='utf8')
    content=re.sub(r'\A---\n.*?\n---\n','',text,flags=re.S)
    # Keep the actual method, limitations and evidence, rather than only the
    # first line (often a status). Original graph edges stay in the archive.
    content=re.sub(r'^# [^\n]*\n?', '', content, count=1, flags=re.M)
    content=re.sub(r'^↑[^\n]*?(?: · |$)', '', content, flags=re.M)
    content=plain(content.strip())
    if not re.sub(r'\[[^\]]*\]\([^)]*\)', '', content).strip(' \n:').removeprefix('источник').strip(' \n:'):
        content='В архивной заметке сохранено только название. Подробная постановка в ней отсутствует.\n\n'+content
    return content+'\n\n'+source_note(path)

def card_color(c):
    if c['state']=='running':return 'в-работе'
    if c['state']=='completed_positive':return 'выполнено'
    if c['state'] in ('completed_negative','inconclusive','closed','superseded'):
        return 'без-положительного-итога'
    if c['state']=='infrastructure':return None
    return 'ожидает'

def semantic_parent(tree,root,topics):
    """The same operation for every section: one meaning, one parent, one ray."""
    parent=root
    for topic in topics:
        key='topic:'+parent+':'+topic
        if key not in tree.nodes:
            tree.add(key,topic,parent,'Группа внутри раздела «'+tree.nodes[parent]['title']+'». Здесь собраны записи по теме «'+topic+'».')
        parent=key
    return parent

def archive_topic(title):
    text=title.lower()
    if any(w in text for w in ('детектор','yolo','пересвет','уверенност','возраст')):
        method='Обучение и адаптация' if any(w in text for w in ('обуч','аугмент','адаптац')) else 'Признаки и уверенность'
        return ['Детектор и качество обнаружений',method]
    if any(w in text for w in ('движени','поток','столкнов','точки отсчёта')):
        return ['Движение и пространственное положение']
    if any(w in text for w in ('коннектом','flywire','ансамбль','мушиност')):
        return ['Устройство и обучение зрительного модуля']
    if any(w in text for w in ('трек','сопостав','сшив','узнаван','объект','памят')):
        return ['Сопровождение и память объектов']
    return ['Методы обучения и проверки']

def research_topic(group,title):
    code=re.match(r'^([A-Z])(\d+) ',title)
    if not code:return []
    family,num=code[1],int(code[2])
    if group=='Фундаментальные проблемы':
        if num in (1,2,19):return ['Новая и доступная информация']
        if num in (3,4,5,6,7,8,18):return ['Подключение модулей и обучение']
        if num in (9,10):return ['Узкое место и запас улучшения']
        if num in (11,12,13):return ['Устойчивость и биологические ограничения']
        return ['Достоверность измерений и контролей']
    if group=='Модернизация мухи и линия B':
        if family=='C':
            if num in (1,2,12,14):part='Анатомия и ограничения коннектома'
            elif num in (4,5,9,13):part='Обучение и адаптация входа'
            elif num in (6,7,8,10,11):part='Выходные признаки и клеточные пути'
            elif num in (3,15):part='Ансамбль и условия съёмки'
            else:part='Стоимость вычислений'
            return ['Устройство и адаптация FlyVis',part]
        return [{'A':'Внешность и повторное узнавание','B':'Геометрия и правила трекера',
                 'C':'Устройство и адаптация FlyVis','D':'От признака к качеству трекера'}[family]]
    if group=='Детектор с мухой':
        if num in (1,2,3,13):return ['Биологические признаки для обнаружения']
        if num in (4,5,6,7):return ['Временная информация и слияние признаков']
        if num in (8,9,10):return ['Слабое освещение и смена условий']
        return ['Обучение и сильная база детектора']
    return []

def result_topic(result):
    text=result['title'].lower()
    for tokens,title in [
        (('balanced','взвеш','односторон','hinge'),'Разделение цели и фона'),
        (('центрирован','центра'),'Обучение на центре объекта'),
        (('коэффициент','модели цели'),'Представление конкретной цели'),
        (('карт','окно','пик'),'Чтение пространственной карты'),
        (('masa','внешн'),'Внешний вид и сопоставление'),
        (('сшив','обрыв'),'Объединение траекторий'),
        (('движ','поток'),'Прогноз движения'),
        (('жив','цепоч','фильтр'),'Жизнь и удаление треков'),
        (('задержк',),'Переходная реакция ядра'),
        (('контекст','соответствие'),'История и соответствие наблюдений'),
        (('метка','частота примеров'),'Метки и состав обучающих примеров'),
        (('рамок','калибровк','назначени'),'Геометрия рамок и назначений'),
        (('обнаружени','маска','калман'),'Восстановление с поддержкой обнаружений'),
        (('допуск',),'Допуск восстановления объекта'),
        (('б1',),'Временная поправка к сильной базе'),
        (('hota','стенд','систем'),'Зрительный модуль в полном сопровождении')]:
        if any(token in text for token in tokens):return [title]
    return ['Замена пространственного считывателя']

class Tree:
    def __init__(self,vault):
        self.vault=vault
        self.nodes={}
        self.children=defaultdict(list)
        self.stealth=defaultdict(list)
        manifest=vault/'.graph-visible.json'
        self.previous=read(manifest) if manifest.exists() else {}
        self.old=self.previous.get('nodes',{})
        # Preserve an explicit rename of a generated root in Obsidian.
        for key,node in self.old.items():
            if node['parent'] is not None or (vault/node['path']).exists():continue
            candidates=[]
            for p in (vault/FOLDER/node['section']).glob('*.md'):
                text=p.read_text(encoding='utf8')
                if 'пункт: "'+node['number']+'"' in text.split('---',2)[1] and re.search(r'^# '+re.escape(node['number'])+r'\. '+re.escape(node['title'])+r'$',text,re.M):
                    candidates.append(p.relative_to(vault).as_posix())
            if len(candidates)==1:node['path']=candidates[0]
        self.used=defaultdict(int)
        for node in self.old.values():
            self.used[node['parent']]=max(self.used[node['parent']],int(node['number'].split('.')[-1]))

    def add(self,key,title,parent=None,body='',color=None):
        # Section numbers belong to the generated tree, not to the title.
        title=re.sub(r'^\d+[а-я]?(?:\.\d+)*\.\s+', '', title)
        if key in self.nodes:raise ValueError('Duplicate item '+key)
        if parent is None:
            number=str(SECTION_NAMES.index(title)+1)
            section=title
        else:
            previous=self.old.get(key,{})
            if previous.get('parent')==parent and previous['number'].rsplit('.',1)[0]==self.nodes[parent]['number']:
                number=previous['number']
            else:
                self.used[parent]+=1
                number=self.nodes[parent]['number']+'.'+str(self.used[parent])
            section=self.nodes[parent]['section']
            self.children[parent].append(key)
        path=f'{FOLDER}/{section}/{number}. {safe(title,85).rstrip(". ")}.md'
        if parent is None and key in self.old and (self.vault/self.old[key]['path']).exists():
            path=self.old[key]['path']
        self.nodes[key]=dict(number=number,title=title,section=section,parent=parent,path=path,body=body,color=color)
        return key

    def save(self):
        # New thematic groups replace obsolete direct leaves. Number them from
        # one instead of inheriting the largest old leaf number (e.g. 121).
        roots={n['parent'] for k,n in self.nodes.items() if k.startswith('topic:') and not n['parent'].startswith('topic:')}
        if any(k.startswith('hyp-month:') for k in self.nodes):roots.add('Гипотезы')
        def number_children(parent):
            for ordinal,key in enumerate(self.children.get(parent,[]),1):
                node=self.nodes[key]
                node['number']=self.nodes[parent]['number']+'.'+str(ordinal)
                node['path']=f'{FOLDER}/{node["section"]}/{node["number"]}. {safe(node["title"],85).rstrip(". ")}.md'
                number_children(key)
        for root in roots:number_children(root)
        rendered={}
        for key,node in self.nodes.items():
            tags=[ACTIVE]+(['граф-проекта/'+node['color']] if node['color'] else [])
            depth=len(node['number'].split('.'))
            date='дата: "'+node['date']+'"\n' if node.get('date') else ''
            body='---\nпункт: "'+node['number']+'"\nуровень: '+str(depth)+'\n'+date+'tags: ['+', '.join(tags)+']\n---\n\n# '+node['number']+'. '+plain(node['title'])+'\n\n'+node['body']
            for child in sorted(self.children.get(key,[]),key=lambda k:tuple(map(int,self.nodes[k]['number'].split('.')))):
                item=self.nodes[child]
                body+='\n\n[['+item['path'][:-3]+'|'+item['number']+'. '+safe(item['title'],140)+']]'
            for target,label in self.stealth.get(key,[]):
                item=self.nodes[target]
                body+='\n\n['+label+'](obsidian://open?vault=ObsidianFlyVis&file='+quote(item['path'][:-3],safe='')+')'
            rendered[node['path']]='\n'.join(line.expandtabs(4).rstrip() for line in body.strip().splitlines())+'\n'
        assert len(rendered)==len(self.nodes)
        for path,body in rendered.items():
            dest=self.vault/path
            dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists():
                old=dest.read_text(encoding='utf8')
                if old==body:continue
                if 'граф-проекта/' not in old.split('---\n',2)[1]:raise ValueError('Occupied path: '+path)
            dest.write_bytes(body.encode())
        for path in {n['path'] for n in self.old.values()}-set(rendered):
            dest=self.vault/path
            if dest.is_file():
                text=dest.read_text(encoding='utf8')
                dest.write_bytes(text.replace(ACTIVE,'граф-проекта/архив',1).encode())
        meta={k:{f:v for f,v in n.items() if f!='body'} for k,n in self.nodes.items()}
        query=f'path:"{FOLDER}/" tag:#{ACTIVE}'
        config=self.vault/'.obsidian/graph.json'
        graph=read(config) if config.exists() else {}
        graph.update(search=query,hideUnresolved=True,showAttachments=False,showTags=False,showOrphans=True,
                     colorGroups=[dict(query='tag:#граф-проекта/'+k,color=dict(a=1,rgb=v)) for k,v in COLORS.items()])
        config.parent.mkdir(exist_ok=True)
        for p,obj in [(config,graph),(self.vault/'.graph-visible.json',dict(version=2,files=sorted(rendered),nodes=meta,search=query))]:
            p.write_bytes((json.dumps(obj,ensure_ascii=False,indent=2)+'\n').encode())

def hypothesis_week(day):
    first=day.replace(day=1)
    number=(day.day+first.weekday()-1)//7+1
    start=max(first,day-timedelta(days=day.weekday()))
    end=min(day+timedelta(days=6-day.weekday()),day.replace(day=calendar.monthrange(day.year,day.month)[1]))
    return number,start,end

def hypotheses(tree,vault,as_of=None):
    today=as_of or datetime.now(timezone(timedelta(hours=3))).date()
    registry=read(ROOT/'experiments/hypothesis_calendar.json')['entries']
    months=('Январь','Февраль','Март','Апрель','Май','Июнь','Июль','Август','Сентябрь','Октябрь','Ноябрь','Декабрь')
    # Navigation groups are not hypotheses. Each real hypothesis appears once.
    available={p.relative_to(vault).with_suffix('').as_posix()
               for folder in ['Гипотезы — подходы','Гипотезы — трекинг','Гипотезы — ядро FlyVis']
               for p in (vault/folder).glob('*.md') if not p.stem.startswith('Трекинг —')}
    available.update(m[1] for m in LINK.finditer((vault/'Гипотезы.md').read_text(encoding='utf8')) if m[1].startswith('09 Запланировано/Б'))
    assert available==set(registry),'Every hypothesis needs an explicit evidenced registration date'
    def day_parent(day):
        month='hyp-month:'+day.strftime('%Y-%m')
        if month not in tree.nodes:
            tree.add(month,months[day.month-1]+' '+str(day.year),'Гипотезы')
        number,start,end=hypothesis_week(day)
        week=month+':week:'+str(number)
        if week not in tree.nodes:
            current=start<=today<=end
            tree.add(week,'Неделя №'+str(number),month,
                     'Период: '+start.strftime('%d.%m.%Y')+' — '+end.strftime('%d.%m.%Y')+'. Неделя месяца, понедельник–воскресенье.'+
                     (' Текущая неделя.' if current else ''),'в-работе' if current else None)
            tree.nodes[week].update(week_start=start.isoformat(),week_end=end.isoformat(),current_week=current)
        key='hyp-day:'+day.isoformat()
        if key not in tree.nodes:
            tree.add(key,day.strftime('%d.%m.%Y'),week,'Дата документированной регистрации гипотез и подходов.')
            tree.nodes[key]['date']=day.isoformat()
        return key
    # Include the current week even when it has no new hypotheses; do not
    # invent an empty day or a scientific hypothesis for it.
    grouped=defaultdict(list)
    for path,item in registry.items():grouped[date.fromisoformat(item['date'])].append((path,item))
    for day,items in sorted(grouped.items()):
        parent=day_parent(day)
        for path,item in sorted(items):
            text=(vault/(path+'.md')).read_text(encoding='utf8')
            front=text.split('---',2)[1]
            color='выполнено' if 'проверка-гипотезы/завершена' in front else 'ожидает' if 'проверка-гипотезы/не-завершена' in front else None
            if path.startswith('Гипотезы — трекинг/Т10 '):color='без-положительного-итога'
            body='Дата регистрации: '+day.strftime('%d.%m.%Y')+'. Основание: '+item['locator']+'.\n\n'+local_file(item['source'],'Источник даты')+'\n\n'
            body+='Связанная тематическая ветвь: '+item['thematic_parent']+'.\n\n'+original(vault,path)
            body=body.replace('Зелёный означает завершение проверки, а не подтверждение пользы.',
                              'Фиолетовый означает завершённую проверку без подтверждённого положительного результата.')
            key=tree.add('hyp:'+path,Path(path).name,parent,body,color)
            tree.nodes[key].update(date=day.isoformat(),date_source=item['source'],date_locator=item['locator'])
    current_month='hyp-month:'+today.strftime('%Y-%m')
    current_number,start,end=hypothesis_week(today)
    current_week=current_month+':week:'+str(current_number)
    if current_week not in tree.nodes:
        if current_month not in tree.nodes:tree.add(current_month,months[today.month-1]+' '+str(today.year),'Гипотезы')
        tree.add(current_week,'Неделя №'+str(current_number),current_month,
                 'Текущая неделя: '+start.strftime('%d.%m.%Y')+' — '+end.strftime('%d.%m.%Y')+
                 '. Новых зарегистрированных гипотез пока нет.','в-работе')
        tree.nodes[current_week].update(week_start=start.isoformat(),week_end=end.isoformat(),current_week=True)

def experiments(tree,catalog):
    cards=catalog['experiments']
    topics=read(ROOT/'experiments/obsidian_experiment_topics.json')['topics']
    for area in sorted(catalog['areas'],key=lambda a:a['order']):
        items=[c for c in cards if c['area']==area['id']]
        if not items:continue
        parent=tree.add('area:'+area['id'],area['title'],'Эксперименты',plain(area.get('description')))
        for state,title in STATES.items():
            group=[c for c in items if c['state']==state]
            if not group:continue
            folder=tree.add(parent+':'+state,title.capitalize(),parent)
            for c in sorted(group,key=lambda c:(c['title'].casefold(),c['id'])):
                target=folder
                if len(group)>12:
                    route=topics[c['id']]
                    if not route:continue
                    target=semantic_parent(tree,folder,route)
                body='Состояние: '+title+'. Код: `'+c['id']+'`.'
                for key in ('question','obtained','interpretation','decision','cannot_claim'):
                    if c.get(key):body+='\n\n'+plain(c[key])
                for e in c.get('evidence',[]):
                    if isinstance(e,dict) and e.get('kind')=='file' and (ROOT/e['ref']).is_file():
                        body+='\n\n'+local_file(e['ref'],safe(e.get('claim') or Path(e['ref']).name))
                color=card_color(c)
                # The author excluded the obsolete DRB project from this change.
                old=tree.old.get('exp:'+c['id'])
                if 'drb' in (c['area']+' '+c['title']).lower() and old:
                    color=old['color']
                tree.add('exp:'+c['id'],c['title'],target,body,color)

def journal(tree,plan):
    from scripts.build_obsidian_vault import split_journal
    journal=plan.split('## Журнал изменений',1)[1]
    # One historical heading kept its closing bold marker after the topic.
    journal=re.sub(r'^\*\*(\d\d\.\d\d\.\d{4}) — (.+?)\*\*',r'**\1** — \2',journal,flags=re.M)
    entries=split_journal(journal)
    names=read(ROOT/'experiments/obsidian_journal_titles.json')
    if not entries:raise ValueError('No dated journal entries')
    months=defaultdict(lambda:defaultdict(list))
    for e in entries:months[e['date'][:7]][e['date']].append(e)
    for month,days in sorted(months.items()):
        parent=tree.add('month:'+month,names['months'].get(month,'Новые проверки и решения проекта'),'Журнал',
                        'Период: '+month+'.')
        for day,rows in sorted(days.items()):
            body='Дата: '+day+'.\n\n'+'\n\n'.join('## '+e['time']+' '+plain(e['topic'])+'\n\n'+plain(e['body']) for e in rows)
            key=tree.add('day:'+day,names['days'].get(day,'Решения и результаты по '+plain(rows[0]['topic'])[:65]),parent,body)
            tree.nodes[key]['date']=day
    return len(entries)

def plan_nodes(tree,vault,plan,catalog):
    folder=tree.add('plan:current','Текущая задача','План')
    section=plan.split('### 7.8.',1)[1].split('**Решение автора 04.10.',1)[0]
    tree.add('plan:identity','Сохранить номер после пропуска при одной допустимой рамке',folder,
             plain(section.partition('\n')[2].strip())+'\n\n'+local_file('dissertation/ЕДИНЫЙ_ПЛАН.md','Единый план'),'в-работе')
    folder=tree.add('plan:document','Разделы Единого плана','План')
    heads=list(re.finditer(r'^## (\d+[а-я]?\. [^\n]+)',plan.split('## Журнал изменений')[0],re.M))
    for i,heading in enumerate(heads):
        title=heading[1]
        body=plan[heading.end():heads[i+1].start() if i+1<len(heads) else plan.index('## Журнал изменений')]
        parent=tree.add('plan:section:'+title.split('.')[0],title,folder,local_file('dissertation/ЕДИНЫЙ_ПЛАН.md',title))
        subs=list(re.finditer(r'^### (.+)$',body,re.M))
        tree.nodes[parent]['body']+='\n\n'+plain(body[:subs[0].start()] if subs else body).strip()
        for j,sub in enumerate(subs):
            content=body[sub.end():subs[j+1].start() if j+1<len(subs) else len(body)]
            tree.add(parent+':'+str(j),sub[1],parent,plain(content.strip()))
    folder=tree.add('plan:ideas','Идеи и прежние постановки','План',
                    'Архив предложений. Актуальная работа указана в ветви «Текущая задача».')
    cards={c['id']:c for c in catalog['experiments']}
    topics=read(ROOT/'experiments/obsidian_experiment_topics.json')['topics']
    recovered={i['id']:i for i in read(ROOT/'experiments/obsidian_archive_review.json')['ideas']}
    for d in sorted((vault/'09 Запланировано').glob('**/*.md')):
        src=d.relative_to(vault).with_suffix('').as_posix()
        c=cards.get(d.stem)
        recovery=recovered.get(d.stem.split(' ',1)[0]) if d.parent.name=='Идеи из плана' else None
        route=topics.get(c['id'])[:2] if c and topics.get(c['id']) else archive_topic(d.stem)
        if recovery:
            route={'Д1':['Детектор и качество обнаружений','Диагностика потерь и стыка модулей'],'Д2':['Сопровождение и память объектов'],
                   'Д3':['Методы обучения и проверки'],'Д4':['Движение и пространственное положение'],
                   'Д5':['Устройство и обучение зрительного модуля'],'Д6':['Детектор и качество обнаружений','Диагностика потерь и стыка модулей'],
                   'Д7':['Детектор и качество обнаружений','Диагностика потерь и стыка модулей']}[recovery['id']]
        parent=semantic_parent(tree,folder,route)
        if recovery:
            text='Архивная проверка '+recovery['id']+'.\n\n'+recovery['mechanism']+'\n\n'+recovery['finding']+'\n\n'+recovery['decision']
            if recovery['card']:text+='\n\nКарточка: `'+recovery['card']+'`.'
            text+='\n\n'+local_file(recovery['source'],'Подтверждающий источник')+'\n\n'+local_file(recovery['origin'],'Исходная постановка')+'\n\n'+source_note(src)
            tree.add('plan:idea:'+src,recovery['id']+' — '+recovery['title'],parent,text,recovery['color'])
        else:
            tree.add('plan:idea:'+src,c['title'] if c else d.stem,parent,
                     ('Архивная постановка. Код: `'+c['id']+'`.\n\n' if c else '')+original(vault,src))

def problems(tree,vault):
    for title,keys in GROUPS:
        branch=tree.add('flygrp:'+title,title,'Проблемы мухи')
        for key in keys:
            p=PROBLEMS[key]; status,reason=status_of(key)
            name=key+' — '+p['title']
            tree.add('fly:'+key,name,branch,plain(p['essence'])+'\n\n'+plain(reason)+'\n\n'+source_note('10 Проблемы мухи/'+name),
                     {'решена':'выполнено','частично':'в-работе','открыта':'ожидает'}[status])
    registry=system_problems()
    for typ,title in [('problems','Проблемы авторской системы'),('patterns','Наблюдаемые повторения')]:
        branch=tree.add('sys:'+typ,title,'Проблемы мухи')
        for item in registry[typ]:
            body=plain(item.get('fact') or item.get('observation'))
            for case in item.get('cases',[])[-3:]:
                body+='\n\n'+case['date']+' — '+plain(case['fact'])
            body+='\n\n'+plain(item['next_check'])
            for e in item['evidence']:
                body+='\n\n'+local_file(e['source'],safe(e['claim']))
            tree.add('sys:'+item['id'],item['id']+' — '+item['title'],branch,body,
                     {'open':'ожидает','partial':'в-работе','resolved':'выполнено'}.get(item['status']))

def used_literature(tree):
    registry=ROOT/'experiments/useful_literature.json'
    if not registry.exists():return
    for source in read(registry)['sources']:
        parent=semantic_parent(tree,'Исследования',['Использованная литература',source['problem_group'],source['method_group']])
        body=source['publication']+'\n\nDOI `'+source['doi']+'`.\n\n[Первичная статья]('+source['url']+')\n\n'
        for label,field in [('Проверенный раздел','sections'),('Что установили авторы','established'),
                            ('Связь с нашей проблемой','problem'),('Как использовали','application'),
                            ('Граница вывода','limitation')]:
            body+='## '+label+'\n\n'+source[field]+'\n\n'
        for evidence in source['evidence']:
            body+=local_file(evidence,'Первичный материал проекта')+'\n\n'
        key=tree.add('literature:'+source['id'],source['title'],parent,body)
        for target in source['related']:
            assert target in tree.nodes,target
            tree.stealth[key].append((target,tree.nodes[target]['title']))
            tree.stealth[target].append((key,'Литературное основание — '+source['title']))


def build(vault):
    tree=Tree(vault)
    for name in SECTION_NAMES:tree.add(name,name)
    catalog=read(ROOT/'experiments/project_history.json')
    glossary=read(ROOT/'experiments/metrics_glossary.json')
    plan=(ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md').read_text(encoding='utf8')
    hypotheses(tree,vault)
    experiments(tree,catalog)
    for kind in ('terms','metrics'):
        for k,g in sorted(glossary[kind].items(),key=lambda i:i[1]['title'].casefold()):
            tree.add(kind+':'+k,g['title'],'Термины',plain(g['ru'])+'\n\n'+plain(g.get('caveat')))
    count=journal(tree,plan)
    plan_nodes(tree,vault,plan,catalog)
    for group in sorted((vault/'07 Исследования').iterdir()):
        if not group.is_dir():continue
        branch=tree.add('research:'+group.name,group.name,'Исследования')
        for p in sorted(group.glob('**/*.md')):
            path=p.relative_to(vault).with_suffix('').as_posix()
            if p.stem==group.name:
                tree.nodes[branch]['body']=original(vault,path)
            else:
                parent=semantic_parent(tree,branch,research_topic(group.name,p.stem))
                tree.add('research:'+path,p.stem,parent,original(vault,path))
    problems(tree,vault)
    used_literature(tree)
    results=collect(ROOT,catalog,glossary)
    for group,title in RESULT_GROUPS.items():
        branch=tree.add('resultgrp:'+group,title,'Лучшие результаты')
        for r in sorted((x for x in results if x['category']==group),key=lambda x:x['title']):
            body=plain(r['comparison'])+'\n\n'+plain(r['subset'])+'\n\n'+plain(r['limitations'])
            scale=100 if r['unit']=='fraction' else 1
            body+=f"\n\nРазность {r['mean']*scale:+.4f}, 95% интервал [{r['ci95'][0]*scale:.4f}, {r['ci95'][1]*scale:.4f}] "+('п.п.' if scale==100 else 'пункта')+'.'
            body+='\n\n'+local_file(r['evidence']['source'],'Первичный результат')
            parent=semantic_parent(tree,branch,result_topic(r))
            tree.add('result:'+r['id'],r['title'],parent,body,'выполнено')
    # Archive files stay on disk; the author excludes obsolete DRB from view.
    hidden={k for k,n in tree.nodes.items() if re.search(r'\bDRB\b',n['title'],re.I)}
    while True:
        more={k for k,n in tree.nodes.items() if n['parent'] in hidden}
        if more<=hidden:break
        hidden|=more
    tree.nodes={k:n for k,n in tree.nodes.items() if k not in hidden}
    tree.children={k:[c for c in cs if c not in hidden] for k,cs in tree.children.items() if k not in hidden}
    tree.save()
    return dict(nodes=len(tree.nodes),experiments=sum(k.startswith('exp:') for k in tree.nodes),journal_entries=count)
