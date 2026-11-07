"""Обновить обычный граф, не записывая отдельную группу и настройки цветов."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import argparse,hashlib,json,re,sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
VAULT=ROOT/'ObsidianFlyVis'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--report-dir')
    parser.add_argument('--authorized-stealth-links', help='Файл прямого поручения только на добавление стелс-раздела')
    parser.add_argument('--update-hypothesis-tree', action='store_true', help='Разрешённые автором дочерние подходы; основные карточки сохраняются')
    parser.add_argument('--archive-obsolete-generated', action='store_true', help='Переместить старые производные заметки в проверяемый архив вместо удаления')
    options=parser.parse_args()
    if options.archive_obsolete_generated:
        raise RuntimeError('Автор запретил перемещение файлов ради графа. Используйте build_obsidian_vault.py: скрытие через .graph-visible.json.')
    suffix=datetime.now(timezone(timedelta(hours=3))).strftime('%Y%m%d_%H%M%S')
    OUT=ROOT/(options.report_dir or 'runs/b123_20261027/graph_sync_preserved_'+suffix)
    OUT.resolve().relative_to(ROOT.resolve())
    OUT.mkdir(exist_ok=False)
    import shutil
    if options.archive_obsolete_generated:
        # Полный снимок сохраняет в том числе незарегистрированные старые копии.
        for note in VAULT.rglob('*.md'):
            destination=OUT/'vault-before'/note.relative_to(VAULT)
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(note,destination)
    hub='Гипотезы.md'
    body=(VAULT/hub).read_text(encoding='utf-8')
    targets=re.findall(r'\[\[([^|\]#]+)',body)
    assert len(targets)==3 and len(set(targets))==3
    foreign='01 Муха/Эксперименты/z13_distribution_pilot_v1.md'
    protected=[hub,*[t+'.md' for t in targets]]
    if (VAULT/foreign).exists():protected.append(foreign)
    assert all((VAULT/p).is_file() for p in protected)
    history=json.loads((ROOT/'experiments/project_history.json').read_text(encoding='utf-8'))
    branch_tags={'HY26-1','HY26-2','HY26-3'}
    branches=[c for c in history['experiments'] if branch_tags.intersection(c.get('tags',[]))]
    assert len(branches)==3
    for card in branches:
        # A final state would relocate a note. Stop before any vault write until
        # a separate preservation path handles that relocation without new nodes.
        assert card['state'] in {'planned','ready','running'}, 'Protected hypothesis relocation needs a preservation update before rebuilding'
    hashes={p:digest(VAULT/p) for p in protected}
    authorized={}
    assert not (options.authorized_stealth_links and options.update_hypothesis_tree)
    if options.authorized_stealth_links:
        grant=(ROOT/options.authorized_stealth_links).resolve()
        grant.relative_to(ROOT.resolve())
        authorization=json.loads(grant.read_text(encoding='utf-8'))
        assert authorization['scope']=='append_b123_literature_stealth_links_only'
        assert authorization['user_request'] and authorization['term_definition']
        assert set(authorization['before_sha256'])==set(protected[1:4])
        assert all(hashes[p]==h for p,h in authorization['before_sha256'].items()), 'Authorization snapshot changed or already used'
        authorized={p:(VAULT/p).read_bytes() for p in protected[1:4]}
    settings=VAULT/'.obsidian/graph.json'
    settings_hash=digest(settings)
    for rel in [*protected,'.obsidian/graph.json']:
        backup=OUT/'before'/rel;backup.parent.mkdir(parents=True,exist_ok=True)
        backup.write_bytes((VAULT/rel).read_bytes())
    (OUT/'before.json').write_text(json.dumps(hashes,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    from scripts import build_obsidian_vault as generator
    original_write=generator.Vault.write
    tree_original={p:(VAULT/p).read_bytes() for p in protected[1:4]} if options.update_hypothesis_tree else {}
    if authorized or tree_original:
        from scripts.obsidian_b123_literature import START,branch_section,branch_tag
        def authorized_write(self,rel,text):
            if rel in tree_original:
                from scripts.obsidian_hypothesis_tree import START as TREE_START, END as TREE_END, section
                old=tree_original[rel].decode('utf-8')
                pattern=re.escape(TREE_START)+r'.*?'+re.escape(TREE_END)+r'\s*'
                original=re.sub(pattern,'',old,flags=re.S).rstrip()
                card=next(c for c in branches if c['title']==Path(rel).stem)
                (self.path/rel).write_bytes((original+'\n\n'+section(card)).encode('utf-8'))
                self.written.append(rel.replace('\\','/'))
                return
            if rel in authorized:
                assert START.encode('utf-8') not in authorized[rel], 'Stealth section already exists'
                card=next(c for c in branches if c['title']==Path(rel).stem)
                assert branch_tag(card)
                section=branch_section(card,generator.ref)
                (self.path/rel).write_bytes(authorized[rel]+b'\n'+section.encode('utf-8'))
                self.written.append(rel.replace('\\','/'))
                return
            original_write(self,rel,text)
        generator.Vault.write=authorized_write
    # The four protected notes use Vault.write(), which already supports exclusion.
    # Keep current live graph settings too: no colour or filter change in this run.
    generator.graph_settings=lambda path:None
    original_unlink=Path.unlink
    original_rmdir=Path.rmdir
    owned=set(json.loads((VAULT/'.generated.json').read_text(encoding='utf-8')))
    archived=[]
    def archive_file(path):
        absolute=path.resolve();relative=absolute.relative_to(VAULT.resolve()).as_posix()
        assert relative not in protected[:4] and relative!='ИНСТРУКЦИЯ — как вести записи.md'
        target=(OUT/'archived-notes'/relative).resolve();target.relative_to(OUT.resolve())
        target.parent.mkdir(parents=True,exist_ok=True);assert not target.exists()
        before=digest(absolute);absolute.rename(target);assert digest(target)==before
        archived.append(dict(from_path=relative,to_path=target.relative_to(ROOT).as_posix(),sha256=before))
    def forbid_delete(path,*args,**kwargs):
        if options.archive_obsolete_generated:
            relative=path.resolve().relative_to(VAULT.resolve()).as_posix()
            assert relative in owned,relative
            archive_file(path)
            return
        raise RuntimeError('Deletion blocked during protected graph synchronization: '+str(path))
    Path.unlink=forbid_delete
    original_archive=generator.archive_obsolete_note
    if options.archive_obsolete_generated:
        generator.archive_obsolete_note=lambda path,vault:archive_file(path)
    Path.rmdir=lambda path,*a,**k:None
    args=['build_obsidian_vault.py']
    for rel in protected:args+=['--preserve-note',rel]
    sys.argv=args
    try:code=generator.main()
    finally:
        Path.unlink=original_unlink
        Path.rmdir=original_rmdir
        generator.Vault.write=original_write
        generator.archive_obsolete_note=original_archive
    assert code==0
    def retained_digest(relative):
        path=VAULT/relative
        if path.exists():return digest(path)
        archived_path=OUT/'archived-notes'/relative
        assert archived_path.is_file(),relative
        return digest(archived_path)
    assert all(retained_digest(p)==h for p,h in hashes.items() if p not in authorized and p not in tree_original)
    if tree_original:
        from scripts.obsidian_hypothesis_tree import START as TREE_START, END as TREE_END
        def without_tree(raw):
            return re.sub(re.escape(TREE_START)+r'.*?'+re.escape(TREE_END)+r'\s*','',raw.decode('utf-8'),flags=re.S).rstrip()
        assert all(without_tree((VAULT/p).read_bytes())==without_tree(old) for p,old in tree_original.items())
    if options.archive_obsolete_generated:
        # Старые копии карточек, которые прежний генератор уже потерял из manifest.
        current=set(json.loads((VAULT/'.generated.json').read_text(encoding='utf-8')))
        by_id={c['id']:c for c in history['experiments']}
        for folder in ('09 Запланировано','01 Муха/Эксперименты','02 Эксперименты'):
            for note in (VAULT/folder).glob('*.md'):
                relative=note.relative_to(VAULT).as_posix()
                if relative in current or relative in protected or note.stem not in by_id:continue
                body=note.read_text(encoding='utf-8')
                assert '- эксперимент' in body and f'`{note.stem}`' in body, 'Unknown note; do not archive: '+relative
                archive_file(note)
        for note in VAULT.glob('*.md'):
            # Две пустые случайные заметки, названные автором; никаких пользовательских текстов.
            if note.stat().st_size==0 and re.fullmatch(r'[0-9eE., +\[\]-]+',note.stem):archive_file(note)
        for note in (VAULT/'06 Журнал').rglob('*.md'):
            archive_file(note)
    for rel,old in authorized.items():
        assert (VAULT/rel).read_bytes().startswith(old), 'Pre-existing hypothesis content changed'
    assert digest(settings)==settings_hash
    result=dict(passed=True,at=datetime.now(timezone(timedelta(hours=3))).isoformat(),
        protected_note_hashes=hashes,protected_hypothesis_nodes=4,authorized_stealth_sections=len(authorized),
        graph_settings_unchanged=True,foreign_note_content_preserved=True,authorized_tree_sections=len(tree_original),archived_notes=archived,
        generated_files=len(json.loads((VAULT/'.generated.json').read_text(encoding='utf-8'))),
        scope='Fundamental overview generated. Existing core hypothesis text preserved; authorized child-link sections may change. Obsolete derived notes moved to an audited archive, never deleted. Graph settings not written.')
    (OUT/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=True,generated_files=result['generated_files'],archived_notes=len(archived),authorized_tree_sections=len(tree_original),report=str(OUT/'report.json')),ensure_ascii=False))


if __name__=='__main__':main()
