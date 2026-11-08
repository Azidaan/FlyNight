"""Generate isolated, numbered navigation trees from the project's primary records."""
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import quote
import hashlib
import json
import re

from scripts.build_obsidian_vault import ROOT, safe
from scripts.obsidian_confirmed_results import GROUPS as RESULT_GROUPS, collect
from scripts.obsidian_fly_problems import GROUPS, PROBLEMS, status_of
from scripts.obsidian_system_problems import load as system_problems

FOLDER = 'Карта проекта'
ACTIVE = 'граф-проекта/видно'
SECTION_NAMES = ('Гипотезы', 'Эксперименты', 'Термины', 'Журнал', 'План',
                 'Исследования', 'Проблемы мухи', 'Лучшие результаты')
COLORS = {'ожидает': 0xe74c3c, 'в-работе': 0xf2c744, 'выполнено': 0x39ab61}
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
    intro=next((line.strip() for line in content.splitlines() if len(line.strip())>36
                and not line.lstrip().startswith(('#','↑','[[','[','-','|','<'))),'')
    return plain(intro)+'\n\n'+source_note(path)

def card_color(c):
    if c['state']=='running':return 'в-работе'
    if c.get('done') or c['state'] in ('completed_positive','completed_negative','closed'):
        return 'выполнено'
    return 'ожидает'

class Tree:
    def __init__(self,vault):
        self.vault=vault
        self.nodes={}
        self.children=defaultdict(list)
        manifest=vault/'.graph-visible.json'
        self.previous=read(manifest) if manifest.exists() else {}
        self.old=self.previous.get('nodes',{})
        self.used=defaultdict(int)
        for node in self.old.values():
            self.used[node['parent']]=max(self.used[node['parent']],int(node['number'].split('.')[-1]))

    def add(self,key,title,parent=None,body='',color=None):
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
        self.nodes[key]=dict(number=number,title=title,section=section,parent=parent,path=path,body=body,color=color)
        return key

    def save(self):
        rendered={}
        for key,node in self.nodes.items():
            tags=[ACTIVE]+(['граф-проекта/'+node['color']] if node['color'] else [])
            depth=len(node['number'].split('.'))
            body='---\nуровень: '+str(depth)+'\ntags: ['+', '.join(tags)+']\n---\n\n# '+node['number']+'. '+plain(node['title'])+'\n\n'+node['body']
            for child in sorted(self.children[key],key=lambda k:tuple(map(int,self.nodes[k]['number'].split('.')))):
                item=self.nodes[child]
                body+='\n\n[['+item['path'][:-3]+'|'+item['number']+'. '+safe(item['title'],140)+']]'
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

def hypotheses(tree,vault):
    parent='Гипотезы'
    paths={p.relative_to(vault).with_suffix('').as_posix() for folder in ['Гипотезы — подходы','Гипотезы — трекинг','Гипотезы — ядро FlyVis']
           for p in (vault/folder).glob('*.md')}
    root_targets=[m[1] for m in LINK.finditer((vault/'Гипотезы.md').read_text(encoding='utf8'))]
    paths.update(root_targets)
    seen=set()
    def visit(path,up):
        if path in seen:return
        seen.add(path)
        text=(vault/(path+'.md')).read_text(encoding='utf8')
        if path.startswith('Гипотезы — ядро'):
            up=tree.add('core-hub','Ядро FlyVis',up)
        front=text.split('---',2)[1]
        color='выполнено' if 'проверка-гипотезы/завершена' in front else 'ожидает' if 'проверка-гипотезы/не-завершена' in front else None
        k=tree.add('hyp:'+path,Path(path).name,up,original(vault,path),color)
        for m in LINK.finditer(text):
            if m[1] in paths and m[1] not in root_targets:visit(m[1],k)
    for path in root_targets:visit(path,parent)
    assert seen==paths

def experiments(tree,catalog):
    cards=catalog['experiments']
    for area in sorted(catalog['areas'],key=lambda a:a['order']):
        items=[c for c in cards if c['area']==area['id']]
        if not items:continue
        parent=tree.add('area:'+area['id'],area['title'],'Эксперименты',plain(area.get('description')))
        for state,title in STATES.items():
            group=[c for c in items if c['state']==state]
            if not group:continue
            folder=tree.add(parent+':'+state,title.capitalize(),parent)
            for c in sorted(group,key=lambda c:(c['title'].casefold(),c['id'])):
                body='Состояние: '+title+'. Код: `'+c['id']+'`.'
                for key in ('question','obtained','interpretation','decision','cannot_claim'):
                    if c.get(key):body+='\n\n'+plain(c[key])
                for e in c.get('evidence',[]):
                    if isinstance(e,dict) and e.get('kind')=='file' and (ROOT/e['ref']).is_file():
                        body+='\n\n'+local_file(e['ref'],safe(e.get('claim') or Path(e['ref']).name))
                tree.add('exp:'+c['id'],c['title'],folder,body,card_color(c))

def journal(tree,plan):
    from scripts.build_obsidian_vault import split_journal
    journal=plan.split('## Журнал изменений',1)[1]
    # One historical heading kept its closing bold marker after the topic.
    journal=re.sub(r'^\*\*(\d\d\.\d\d\.\d{4}) — (.+?)\*\*',r'**\1** — \2',journal,flags=re.M)
    entries=split_journal(journal)
    if not entries:raise ValueError('No dated journal entries')
    months=defaultdict(lambda:defaultdict(list))
    for e in entries:months[e['date'][:7]][e['date']].append(e)
    for month,days in sorted(months.items()):
        parent=tree.add('month:'+month,month,'Журнал')
        for day,rows in sorted(days.items()):
            body='\n\n'.join('## '+e['time']+' '+plain(e['topic'])+'\n\n'+plain(e['body']) for e in rows)
            tree.add('day:'+day,day,parent,body)
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
    groups={}
    cards={c['id']:c for c in catalog['experiments']}
    areas={a['id']:a['title'] for a in catalog['areas']}
    for d in sorted((vault/'09 Запланировано').glob('**/*.md')):
        src=d.relative_to(vault).with_suffix('').as_posix()
        c=cards.get(d.stem)
        category=areas[c['area']] if c else ('Идеи из плана' if d.parent.name=='Идеи из плана' else 'Исторические указатели и подходы')
        if category not in groups:groups[category]=tree.add('plan:archive-group:'+category,category,folder)
        tree.add('plan:idea:'+src,c['title'] if c else d.stem,groups[category],
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
            else:tree.add('research:'+path,p.stem,branch,original(vault,path))
    problems(tree,vault)
    results=collect(ROOT,catalog,glossary)
    for group,title in RESULT_GROUPS.items():
        branch=tree.add('resultgrp:'+group,title,'Лучшие результаты')
        for r in sorted((x for x in results if x['category']==group),key=lambda x:x['title']):
            body=plain(r['comparison'])+'\n\n'+plain(r['subset'])+'\n\n'+plain(r['limitations'])
            scale=100 if r['unit']=='fraction' else 1
            body+=f"\n\nРазность {r['mean']*scale:+.4f}, 95% интервал [{r['ci95'][0]*scale:.4f}, {r['ci95'][1]*scale:.4f}] "+('п.п.' if scale==100 else 'пункта')+'.'
            body+='\n\n'+local_file(r['evidence']['source'],'Первичный результат')
            tree.add('result:'+r['id'],r['title'],branch,body,'выполнено')
    tree.save()
    return dict(nodes=len(tree.nodes),experiments=len(catalog['experiments']),journal_entries=count)
