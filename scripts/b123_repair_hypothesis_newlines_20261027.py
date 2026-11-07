"""Исправить только собственную ошибку повторного перевода строк, сверив прежний вывод."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.obsidian_hypothesis_tree import section
VAULT=ROOT/'ObsidianFlyVis';RUN=ROOT/'runs/b123_20261027/graph_fundamental_rebuild'
def main():
    history=json.loads((ROOT/'experiments/project_history.json').read_text(encoding='utf-8'))
    for card in history['experiments']:
        if not any(t in {'HY26-1','HY26-2','HY26-3'} for t in card.get('tags',[])):continue
        rel=Path('09 Запланировано')/(card['title']+'.md')
        original=(RUN/'before'/rel).read_bytes()
        expected=(original.decode('utf-8').rstrip()+'\n\n'+section(card)).replace('\n','\r\n').encode('utf-8')
        actual=(VAULT/rel).read_bytes()
        assert actual==expected, 'File differs from our own failed output; preserve before editing: '+str(rel)
        backup=RUN/'failed-newline-output'/rel;backup.parent.mkdir(parents=True,exist_ok=True);backup.write_bytes(actual)
        (VAULT/rel).write_bytes(original)
    print('Original hypothesis bytes restored; separate failed output retained.')
if __name__=='__main__':main()
