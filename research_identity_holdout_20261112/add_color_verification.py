from common import *
p=OUT/'publish.py';s=p.read_text(encoding='utf8');needle='selected={}\n';assert s.count(needle)==1
extra="""color_manifest=read(pub/'.graph-visible.json')
color_checks={}
for identity,expected in [('identity_repair_observed_episode_20261112','выполнено'),('identity_repair_combined_assignment_20261112','выполнено'),('identity_repair_holdout_correspondence_20261112','без-положительного-итога'),('identity_repair_holdout_recovery_20261112','без-положительного-итога')]:
    node=color_manifest['nodes']['exp:'+identity];assert node['color']==expected
    assert 'граф-проекта/'+expected in (pub/node['path']).read_text(encoding='utf8')
    color_checks[identity]=dict(path=node['path'],color=expected)
put('graph_science_color_verification.json',dict(passed=True,checks=color_checks,success_rgb='#39ab61',scope='Confirmed five-case successes remaingreen. New54-case transfer is inconclusive andpurple. No manual circle edits or closureofallidentityproblems.'))
"""
s=s.replace(needle,extra+needle);p.write_bytes(s.encode())
print('GENERATOR_COLOR_VERIFICATION_ADDED')
