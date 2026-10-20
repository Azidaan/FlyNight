import copy
import hashlib
import json
import pytest
from scripts.obsidian_confirmed_results import collect,build


def fixture(tmp_path,lo=.09,state='inconclusive',unit='fraction'):
    raw=json.dumps({'paired':{'test':{'mean_per_video':.13,'ci95':[lo,.17],'n':35}}}).encode()
    (tmp_path/'source.json').write_bytes(raw)
    entry=dict(id='test',title='Проверенное сравнение',category='representation',metric='useful',unit=unit,
        evidence=dict(source='source.json',locator=['paired','test'],sha256=hashlib.sha256(raw).hexdigest()),
        comparison='Муха против входа',subset='dev35',conditions='Фиксированные условия',limitations='Не HOTA')
    card=dict(id='card',state=state,acceptance=[{'met':False}],confirmed_results=[entry])
    return {'experiments':[card]},{'metrics':{'useful':{'title':'Доля полезных центров','range':[0,1]}}}


def test_individual_positive_does_not_change_inconclusive_parent_and_keeps_the_original_link(tmp_path):
    h,g=fixture(tmp_path); before=copy.deepcopy(h)
    class Vault:
        def __init__(self): self.notes={}
        def write(self,path,text): self.notes[path]=text
    v=Vault()
    build(v,tmp_path,h,g,{'card':'01 Муха/Эксперименты/card'},
          lambda block,path,label:f'[{label}]({path})',lambda data:'',lambda value:value)
    text=v.notes['11 Лучшие результаты/Сравнения/test.md']
    assert '01 Муха/Эксперименты/card' in text and 'непройденные критерии' in text
    assert '+13,00 п.п.' in text and h==before


def test_interval_touching_zero_is_not_a_confirmed_effect(tmp_path):
    h,g=fixture(tmp_path,lo=0)
    with pytest.raises(ValueError,match='Unconfirmed'):
        collect(tmp_path,h,g)


def test_changed_primary_file_requires_rechecking_the_reference(tmp_path):
    h,g=fixture(tmp_path)
    (tmp_path/'source.json').write_text('{}',encoding='utf-8')
    with pytest.raises(ValueError,match='Changed'):
        collect(tmp_path,h,g)


def test_running_experiment_cannot_enter_the_results_section(tmp_path):
    h,g=fixture(tmp_path,state='running')
    with pytest.raises(ValueError,match='Unfinished'):
        collect(tmp_path,h,g)


def test_source_path_cannot_escape_the_project(tmp_path):
    h,g=fixture(tmp_path)
    h['experiments'][0]['confirmed_results'][0]['evidence']['source']='../outside.json'
    with pytest.raises(ValueError):
        collect(tmp_path,h,g)


def test_point_units_are_not_silently_multiplied_by_one_hundred(tmp_path):
    h,g=fixture(tmp_path,unit='points')
    g['metrics']['useful']['range']=[0,100]
    assert collect(tmp_path,h,g)[0]['unit']=='points'
    h['experiments'][0]['confirmed_results'][0]['unit']='percent_guess'
    with pytest.raises(ValueError,match='unit'):
        collect(tmp_path,h,g)


def test_hota_range_cannot_be_presented_as_fractional_percentage_points(tmp_path):
    h,g=fixture(tmp_path)
    g['metrics']['useful']['range']=[0,100]
    with pytest.raises(ValueError,match='metric range'):
        collect(tmp_path,h,g)
