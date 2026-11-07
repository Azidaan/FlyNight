from scripts.obsidian_fundamental_overview import resolved_tasks,TASKS


def test_unfinished_transfer_does_not_change_graph_result():
    before=repr(TASKS)
    assert resolved_tasks({'experiments':[dict(id='detection_support_transfer_v1',state='ready',obtained='')]})==TASKS
    assert repr(TASKS)==before


def test_complete_transfer_updates_one_existing_task_without_accumulating_nodes():
    for state in ['completed_positive','completed_negative','inconclusive']:
        history={'experiments':[dict(id='detection_support_transfer_v1',state=state,obtained='primary measured result')]}
        first=resolved_tasks(history);second=resolved_tasks(history)
        assert first==second and len(first)==len(TASKS)==5
        task=next(t for t in first if t['title'].startswith('Задача 4'))
        assert task['ids'].count('detection_support_transfer_v1')==1
        assert all('detection_support_transfer_v1' not in t['ids'] for t in TASKS)
