import re
from scripts import obsidian_tracking_hypotheses as T
from scripts import build_obsidian_vault as G


def test_tracking_candidates_are_numbered_and_unverified_with_transfer_limits():
    class Vault:
        def __init__(self):self.notes={}
        def write(self,name,body):self.notes[name]=body
    v=Vault();T.build(v,G);data=T.load()
    assert len(v.notes)==13 and len(T.paths())==13
    hub=v.notes[T.HUB+'.md']
    assert len(re.findall(r'\[\[',hub))==13  # parent + twelve author-requested candidates
    for m in data['methods']:
        text=v.notes[T.method_path(m)+'.md']
        if m['status']=='completed':
            assert 'проверка-гипотезы/завершена' in text
            assert m['result']['source'] in text
            assert m['result']['verdict'] in text
        else:
            assert 'проверка-гипотезы/не-завершена' in text
        assert re.findall(r'\[\[([^|\]]+)',text)==[T.HUB]
        assert m['paper'] in text and m['limitation'] in text
    assert 'другой обучаемой архитектуры' in v.notes[T.method_path(data['methods'][3])+'.md']
    assert 'после обработки видео' in v.notes[T.method_path(data['methods'][6])+'.md']
    assert 'без доступа к будущим кадрам' in v.notes[T.method_path(data['methods'][6])+'.md']
    assert [m['id'] for m in data['methods'][-3:]]==['T10','T11','T12']
    assert all(m['status']=='planned' for m in data['methods'][-2:])
