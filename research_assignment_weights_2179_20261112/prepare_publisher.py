from common import *
s=(ROOT/'research_identity_holdout_20261112/publish.py').read_text(encoding='utf8')
s=s.replace("path.match('queue7*')","path.match('queue3*')")
s=s.replace("or rel.as_posix()=='correspondence/result.json'",'')
start=s.index('for dest in [baseline,live,pub]:');end=s.index('\nclass Proxy:',start)
s=s[:start]+'''# Two compact text-only copies. Baseline is measured in memory, then its
# work tree is reused for the live build; no third vault snapshot on disk.
def compact_copy(dest):
    assert not dest.exists();dest.mkdir()
    for source in vault.rglob('*.md'):
        if any(part in {'.trash','plugins'} for part in source.relative_to(vault).parts):continue
        target=dest/source.relative_to(vault);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    for rel in ['.graph-visible.json','.obsidian/graph.json']:
        target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(vault/rel,target)
for dest in [live,pub]:compact_copy(dest)
''' + s[end:]
s=s.replace('gen.build(baseline)','gen.build(live);baseline_hashes={f.relative_to(live).as_posix():sha(f) for f in live.rglob(\'*.md\')}')
s=s.replace("target.read_bytes()==(baseline/rel).read_bytes()","sha(target)==baseline_hashes.get(rel)")
needle="('identity_repair_holdout_recovery_20261112','без-положительного-итога')"
assert needle in s;s=s.replace(needle,needle+",('association_weights_2179_copy_20261112','без-положительного-итога')")
receipt="""put('publication_storage_receipt.json',dict(passed=True,calculation_artifact_bytes=read(D/'storage_summary.json')['calculation_artifact_bytes'],task_bytes_before_frozen_archive=sum(f.stat().st_size for f in D.rglob('*') if f.is_file()),graph_copy_count=2,graph_copy_format='Markdownandmanifest/settingsonly; noimages/binaryattachments; baselinekeptinhashmap, not athirdsnapshot.',own_outputs_drive='E:',temp_drive='E:',cleanup_performed=False,scope='TaskfilesandrequiredrecordkeepingonE; not wholeappC history orphysicalallocation.'))
"""
s=s.replace('selected={}\n',receipt+'selected={}\n',1)
(OUT/'publish.py').write_bytes((s.rstrip()+'\n').encode())
s=(ROOT/'research_identity_holdout_20261112/verify_publication.py').read_text(encoding='utf8')
s=s.replace("queue7_execution.json","queue3_execution.json")
s=s.replace("read(OUT/'audit.json')['scientific_artifacts']","read(OUT/'initial_state.json')['prior_science']")
(OUT/'verify_publication.py').write_bytes((s.rstrip()+'\n').encode())
(OUT/'.gitignore').write_bytes(b'*.npz\nbefore/\ngraph_*/\npublication_text/\npublication*.json\npublication_plan.md\n__pycache__/\ntemp/\n*.pending\n')
print('TWO_COMPACT_GRAPH_COPIES_NO_BINARY_ATTACHMENTS_READY')
