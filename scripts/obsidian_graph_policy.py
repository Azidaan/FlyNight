"""Обзор — опорные результаты; подробные проверки не становятся отдельными узлами."""
DETAILS='Подробности/Проверки и первичные источники'
CURRENT='09 Запланировано/Сейчас — гипотезы и восстановление потерянных объектов'
FUNDAMENTAL={
    'infra_bdd_baseline','tracker_zoo','xywh_bytetrack','wavelet_motion',
    'flyvis_standalone','flyvis_stress','fly_input_norm_hota_v1','motion_scheme_search_v1',
    'fly_signature_reid_v1','fly_matched_controls_v1','fly_connectome_null_v1',
    'fly_tracker_lifecycle_v3','z13_spatial_set_readout_v1',
    'z13_box_supervision_followup_v1','z13_information_ceiling_v1',
    'z13_temporal_identity_readout_v1','z13_hybrid_fly_confirm_v1',
    'z13_val20_native_masa_v1','recovery_motion_output_control_v1'}
PLANS={'birth_weighting','qmatch_candidate_shift',
    'fly_future_kys_state_v1','fly_future_keeptrack_candidates_v1',
    'fly_future_dimp_readout_v1','fly_future_stmtrack_memory_v1',
    'flyvis_temporal_residual_association_v1','flyvis_predetection_temporal_branch_v1',
    'flyvis_connectome_object_memory_v1'}
OVERVIEW_PATHS=set()


def overview_path(path):
    return (path in OVERVIEW_PATHS or path.startswith('10 Проблемы мухи/')
            or path.startswith('11 Лучшие результаты/'))


def build_current(vault,history,paths,ref,fm,hub):
    cards={c['id']:c for c in history['experiments']}
    body='# Сейчас — гипотезы и восстановление потерянных объектов\n\n'
    body+=f'↑ [[{hub}|Запланировано]]\n\n'
    body+='Здесь направление текущей работы. Служебные проверки, ошибки запуска и сохранения не создают дополнительных узлов. Статус карточки каталога сам по себе не доказывает, что процесс сейчас работает.\n\n'
    body+='## Три основные гипотезы\n\n'
    body+=ref('','Гипотезы','Открыть отдельное дерево «Гипотезы → Б1–Б3 → подходы»')+'\n\n'
    for ident in ('flyvis_temporal_residual_association_v1','flyvis_predetection_temporal_branch_v1','flyvis_connectome_object_memory_v1'):
        if ident in cards and ident in paths:
            c=cards[ident];body+=ref('',paths[ident],c['title'])+'\n\n'
    body+='Б1: первая обученная временная поправка сохранена; преимущество на сохранённых запросах не подтверждено, полная HOTA этой версии не измерена. Б2 и Б3: прошли только технические проверки обучения, научная эффективность ещё не проверена. Источники — первичные отчёты в `runs/b123_20261027/`.\n\n'
    body+='## Следующее действие — восстановление потерянного объекта\n\n'
    from scripts.obsidian_b123_literature import SOURCES,path
    for key in ('busca','ltr'):
        body+=ref('',path(key),SOURCES[key]['title'])+'\n\n'
    body+='По поручению автора: сначала BUSCA, затем восстановление по маскам LTR; готовность реализации, весов и стоимость проверяются до полного запуска. Сами методы на нашем стенде ещё не подтвердили пользу. Их подходы показаны дочерними узлами внутри отдельного дерева гипотез.\n\n'
    ident='recovery_motion_output_control_v1'
    if ident in paths:
        body+='Общий контроль уже показал вред продолжения без надёжного отказа: '+ref('',paths[ident],cards[ident]['title'])+'. Это не отрицательный результат BUSCA или LTR.\n\n'
    body+='## Где подробности\n\n'+ref('',DETAILS,'Каталог подробных проверок и первичных файлов')+'\n'
    vault.write(CURRENT+'.md',fm({'tags':['текущая-работа','обзор-графа']})+body)
