"""The validation must preserve frozen choices and video pairing."""
from scripts.z13_target_center_dev import frozen_lambdas,canonical


def test_frozen_lambdas_preserve_each_mode_without_dev_selection():
    choices={'D':{'binary':{'lambda_value':.1},'center':{'lambda_value':1.}},'C':{'binary':{'lambda_value':.001},'center':{'lambda_value':.01}}}
    assert frozen_lambdas('D',choices)=={'binary':[.1],'center':[1.]}
    assert frozen_lambdas('C',choices)=={'binary':[.001],'center':[.01]}


def test_video_local_indices_do_not_collide_and_remain_pairable():
    rows=[dict(label=0,kind=1,useful=True),dict(label=1,kind=0,useful=False)]
    a=canonical(rows,'video_a');b=canonical(rows,'video_b')
    assert {r['label'] for r in a}.isdisjoint(r['label'] for r in b)
    assert [(r['label'],r['kind']) for r in a]==[(r['label'],r['kind']) for r in canonical(rows,'video_a')]
    assert rows[0]['label']==0
