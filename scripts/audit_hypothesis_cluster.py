"""Проверить реальные вики-ссылки группы «Гипотезы» после сборки."""
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
VAULT = ROOT / 'ObsidianFlyVis'


def legacy_main():
    hub = 'Гипотезы'
    body = (VAULT / (hub + '.md')).read_text(encoding='utf-8')
    targets = re.findall(r'\[\[([^|\]#]+)', body)
    assert len(targets) == 3 and len(set(targets)) == 3, targets
    component = {hub, *targets}
    aliases = {name: name for name in component}
    history = json.loads((ROOT / 'experiments/project_history.json').read_text(encoding='utf-8'))
    for name in component:
        aliases[Path(name).name] = name
    for card in history['experiments']:
        if card['title'] in aliases:
            aliases[card['id']] = aliases[card['title']]
    edges, external = set(), []
    for path in VAULT.rglob('*.md'):
        source = path.relative_to(VAULT).with_suffix('').as_posix()
        text = path.read_text(encoding='utf-8', errors='replace')
        for target in re.findall(r'\[\[([^|\]#]+)', text):
            target = aliases.get(target.removesuffix('.md'), target.removesuffix('.md'))
            if source in component or target in component:
                if source not in component or target not in component:
                    external.append([source, target])
                else:
                    edges.add(tuple(sorted([source, target])))
    expected = {tuple(sorted([hub, target])) for target in targets}
    assert edges == expected and not external, (edges, external)
    report = dict(passed=True, nodes=sorted(component), internal_edges=[list(x) for x in sorted(edges)],
                  external_edges=external, node_count=4, ray_count=3,
                  scope='Generated Markdown links, including card aliases; no live UI inspection')
    output = ROOT / 'runs/flyvis_temporal_residual_20261026/hypotheses_component_audit.json'
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))


def main():
    from scripts.audit_fundamental_obsidian import audit
    result=audit(VAULT)
    output=ROOT/'runs/flyvis_temporal_residual_20261026/hypotheses_component_audit.json'
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('passed','hypothesis_node_count','hypothesis_edge_count','root_rays','external_hypothesis_edges')},ensure_ascii=False))


if __name__ == '__main__':
    main()
