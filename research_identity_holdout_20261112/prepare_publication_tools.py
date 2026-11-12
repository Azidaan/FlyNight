from common import *
s=(ROOT/'research_identity_repairs_20261112/publish.py').read_text(encoding='utf8')
s=s.replace("path.match('queue8*')","path.match('queue7*')")
s=s.replace("if path.match('queue7*') or path.match('publication*'):continue","if path.match('queue7*') or path.match('publication*') or rel.as_posix()=='correspondence/result.json':continue")
s=s.replace("['.py','.json','.md','.txt','.log','.csv','.png','.svg','.sha256']","['.py','.json','.md','.txt','.log','.csv','.png','.svg','.sha256','.zip']")
a=s.index("for name in ['tracking/census_correspondence.py'");b=s.index('\naud.ROOT=ROOT',a)
s=s[:a]+s[b:]
(OUT/'publish.py').write_bytes((s.rstrip()+'\n').encode())
s=(ROOT/'research_identity_repairs_20261112/verify_publication.py').read_text(encoding='utf8').replace("queue8_execution.json","queue7_execution.json")
(OUT/'verify_publication.py').write_bytes((s.rstrip()+'\n').encode())
print('HOLDOUT_PUBLICATION_TOOLS_READY')
