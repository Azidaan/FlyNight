"""Materialize separately reviewable full-pilot modules before registration."""
from pathlib import Path
root=Path(__file__).resolve().parents[1];target=Path(__file__).parent;old=root/'research_native_confirm_full_20261108'
for name in ['evaluate.py','additional.py','catalog_check.py','statistics.py','audit.py','features.py']:
    text=(old/name).read_text(encoding='utf-8').replace('research_native_confirm_full_20261108.common','research_bbox_normalized_full_20261108.common')
    if name in ['statistics.py','audit.py']:text=text.replace('2.5/16','2.5/20').replace('family=16','family=20').replace('sixteen-comparison','twenty-comparison')
    if name=='statistics.py':
        text=text.replace("accepted[p['selected_system']]","accepted.get(p['selected_system'],False)").replace("accepted['FLYVIS']","accepted.get('FLYVIS',False)")
        text=text.replace("secondary_preregistered_systems=accepted,","secondary_preregistered_systems=accepted,any_preregistered_system_accepted=any(accepted.values()),primary_failed_conditional_screen=p['selected_system'] not in p['eligible_systems'],")
    if name=='features.py':
        needle="for e in event['edges']:\n                    row=e['row_data']";assert text.count(needle)==1;text=text.replace(needle,"for e in event['edges']:\n                    if e.get('unavailable'):continue\n                    row=e['row_data']")
        text=text.replace("for arm in ['RANDOM','IMAGE','R1','DIS','FLYVIS','PERM']:","for arm in ['SUPPORT','RANDOM','IMAGE','R1','DIS','FLYVIS','PERM']:")
        text=text.replace("arm in ['RANDOM','FLYVIS','PERM']","arm in ['SUPPORT','RANDOM','FLYVIS','PERM']")
        needle="            files={n:h for rep in reps.values()";assert needle in text;text=text.replace(needle,"            for event in metadata['events']:\n                for e in event['edges']:\n                    if e.get('unavailable'):scores[f\"{event['frame']}|{e['old_tid']}|{e['new_tid']}\"]=dict(scores={a:None for a in ['SUPPORT','RANDOM','IMAGE','R1','DIS','FLYVIS','PERM']},support=0)\n"+needle)
        name='features_logic.py'
    if name=='audit.py':
        text=text.replace(" and e['support']>=3",'')
        needle="            for e in event['edges']:\n                score=";assert text.count(needle)==1;text=text.replace(needle,"            for e in event['edges']:\n                if e.get('unavailable'):\n                    assert all(v is None for v in saved_scores[f\"{event['frame']}|{e['old_tid']}|{e['new_tid']}\"]['scores'].values());continue\n                score=")
        text=text.replace("for arm in ['RANDOM','IMAGE','R1','DIS','FLYVIS','PERM']:","for arm in ['SUPPORT','RANDOM','IMAGE','R1','DIS','FLYVIS','PERM']:")
        text=text.replace("accepted[p['selected_system']]","accepted.get(p['selected_system'],False)").replace("accepted['FLYVIS']","accepted.get('FLYVIS',False)")
        text=text.replace("assert accepted==report['accepted'];","assert report['any_preregistered_system_accepted']==any(accepted.values());assert accepted==report['accepted'];")
    with (target/name).open('x',encoding='utf-8') as f:f.write(text)
text=(root/'research_native_confirm_full_recovery1_20261108/prepare_views.py').read_text(encoding='utf-8').replace('research_native_confirm_full_recovery1_20261108.common','research_bbox_normalized_full_20261108.common')
text=text.replace("for old in [PARENT/'retinal'/video/dest.name,BASE/'full_precision/retinal'/video/dest.name]:","for old in [PARENT/'retinal'/video/dest.name,ROOT/'runs/singleton_verify_20261108/full/retinal'/video/dest.name,BASE/'full_precision/retinal'/video/dest.name]:")
with (target/'prepare_views.py').open('x',encoding='utf-8') as f:f.write(text)
