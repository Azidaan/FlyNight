import json
from pathlib import Path
H=json.loads(Path('experiments/project_history.json').read_text(encoding='utf8'))
def topic(c):
 i=c['id'];a=c['area'];t=c['title'].lower()
 def p(*x):return list(x)
 A='Сохранение личности'
 Z='Пространственное считывание FlyVis'
 M='Движение и состояние FlyVis'
 L='Жизнь и рождение треков'
 D='Детектор и точность рамок'
 I='Стенд и вычисления'
 if a=='drb':return None
 if i.startswith(('sam2_','ltr_')):return p(A,'Восстановление потерянного объекта','Маски объекта')
 if i.startswith('busca_'):return p(A,'Восстановление потерянного объекта','Проверка истории сетью BUSCA')
 if i.startswith(('detection_supported','detection_support','online_detection_support')):return p(A,'Восстановление потерянного объекта','Поддержка текущей детекцией')
 if i.startswith(('recovery_admission','recovery_appearance')):return p(A,'Восстановление потерянного объекта','Обучаемый допуск продолжений')
 if i=='recovery_motion_output_control_v1':return p(A,'Восстановление потерянного объекта','Геометрический контроль')
 if i.startswith(('bbox_','singleton_','native_confirm','identity_assignment','spatial_identity_maps')):
  if any(x in i for x in ['birth','two_frames','box_history']):return p(A,'Выбор продолжения среди рамок','История рамок и предупреждения')
  if any(x in i for x in ['id_fate','followup','native_confirm']):return p(A,'Сохранение старого ID','Подтверждение и дальнейшая судьба номера')
  if any(x in i for x in ['map','readout','template','spatial','calibration']):return p(A,'Выбор продолжения среди рамок','Пространственное сравнение своих и чужих объектов')
  if any(x in i for x in ['cost','geometry','recency']):return p(A,'Выбор продолжения среди рамок','Допуск и стоимость без зрительного сигнала')
  return p(A,'Выбор продолжения среди рамок','Разбор отказов и доступных данных')
 if i.startswith(('flyvis_pair_support','flyvis_exclusive','flyvis_atlas','flyvis_separate','flyvis_two_frame','flyvis_cached_readout','flyvis_local_shift','flyvis_full_spatial_capture')):
  return p(A,'Выбор продолжения среди рамок','Уникальные колонки и признаки FlyVis')
 if i.startswith(('stitch_','fly_stitch','lost_link','buffer_extend','association_log','class_gate','masa_feature')):
  if 'oracle' in i or 'audit' in i or 'diag' in i:return p(A,'Сшивка обрывков траекторий','Запас и причины ошибок')
  if any(x in i for x in ['appearance','models','feature','realness']):return p(A,'Сшивка обрывков траекторий','Признаки внешнего вида')
  return p(A,'Сшивка обрывков траекторий','Окна поиска и назначение связей')
 if i.startswith(('history_','query_')):return p(A,'Память объекта', 'Представления истории' if i.startswith('history_') else 'Обновление и применение памяти')
 if i.startswith(('reid_','tracking_scaletrack','frame_masa','lazy_appearance')):return p(A,'Признаки внешнего вида','Сравнение извлекателей и использование в назначениях')
 if i.startswith(('b1_','flyvis_temporal_residual')):return p(A,'Временная поправка к сопоставлению','Обучение и полный трекер')
 if i.startswith(('b3_','flyvis_connectome_object')):return p(A,'Память объекта','Обучаемые связи памяти')
 if i.startswith(('b2_','flyvis_predetection')):return p(D,'Временная ветвь перед детектором','Данные и обучение')
 if i.startswith(('pixel_box','fixed_association_geometry','box_smoothing')):return p(D,'Уточнение рамок','Изображение и геометрический контроль')
 if i.startswith(('z13_triton','z13_cache','z13_shared')):return p(I,'Ускорение зрительного модуля','Triton и общие признаки')
 if i.startswith('z13_val20') or i.startswith('z13_matched'):return p(I,'Одинаковые условия сравнения','Исходные трекеры и общие детекции')
 if i.startswith('z13_') or i in ['fly_object_readout_z13_v1','fly_spatial_quality_z13_v2','fly_spatial_hota_policy_z13_v1']:
  if any(x in i for x in ['temporal','observed_memory','history','anchor']):return p(Z,'История пространственного образа','Межкадровые наблюдения и состояние')
  if 'identity_gate' in i or 'hybrid' in i:return p(Z,'Узнавание цели','Различение личности и отказ от вмешательства')
  if any(x in i for x in ['region','spatial_bins']):return p(Z,'Чтение реальных рамок','Сходство цели, области и соседей')
  if any(x in i for x in ['full_box','box_supervision','unmatched_box','box_proposal','selected_box','proposal_geometry']):return p(Z,'Предложение полной рамки','Геометрия и последствия назначения')
  if any(x in i for x in ['candidate','semantic','association','competing']):return p(Z,'Выбор среди кандидатов','Метки, обучение и полезность выбора')
  if any(x in i for x in ['utility','quality','confidence','uncertainty','margin','null','refusal','value']):return p(Z,'Доверие к ответу','Отбор полезных поправок и отказ')
  if any(x in i for x in ['target','ridge','hinge','filter','score_swap','aligned','readout_overlap']):return p(Z,'Модель конкретной цели','Цель обучения и способ сравнения')
  if any(x in i for x in ['context','specificity','contrast','template','spatial','presence']):return p(Z,'Расположение сигналов','Карта цели, фон и соседние объекты')
  if any(x in i for x in ['effect','assignment','activation','ceiling','math','research']):return p(Z,'Проверка полезности для трекера','Потолок и реальные последствия')
  return p(Z,'Обучение считывателя','Базовый путь и его замены')
 if i.startswith(('flyvis_y1','flyvis_tau','flyvis_history_state','flyvis_partial_state','flyvis_state_subgroups','flyvis_internal_signal','flyvis_t4t5')):return p(M,'Динамика ядра','Память, разворот и внутренние вмешательства')
 if i.startswith(('flyvis_offset','flyvis_still','flyvis_robust','flyvis_local_','flyvis_brightness','flyvis_order','flyvis_counterfactual','flyvis_equal')):return p(M,'Отделение движения от фона','Смещение, неподвижность и освещение')
 if i.startswith(('flyvis_system','flyvis_repairs')):return p(I,'Надёжность исполнения','Состояние, время и происхождение данных')
 if i.startswith(('life_','newval_')):
  if any(x in i for x in ['oracle','error','diag','harm','overlap']):return p(L,'Удаление ложных жизней','Запас и цена ошибочного удаления')
  if any(x in i for x in ['context','appearance','history']):return p(L,'Удаление ложных жизней','История, внешность и соседи')
  return p(L,'Удаление ложных жизней','Обучаемый фильтр и правило применения')
 if i.startswith(('chain_','output_filter')):return p(L,'Фильтрация целых цепочек','Голова удаления и проверка признаков')
 if i.startswith(('birth_','fly_birth','fly_tracker_lifecycle','fly_p5')):return p(L,'Подтверждение рождения' if 'birth' in i else 'Продолжить или удалить трек','Правило решения и контроли')
 if i.startswith(('fly_persistent','fly_efference','fovea_target')):return p(M,'Глаз сопровождаемого объекта','Перемещение, фовеация и перенос состояния')
 if i.startswith(('fly_motion','fly_finetune','fly_input','fly_dynamic','fly_gain','fly_scale','fly_noise','fly_reference','fly_stage','fly_wide','fly_decoder','flyvis_fovea_motion','motion_','e2e_motion','dis_')):
  if any(x in i for x in ['norm','noise','reference','retina']):return p(M,'Вход и отсчёт движения','Контраст, шум и фоновый отклик')
  if any(x in i for x in ['trust','confidence','reliability','photometric']):return p(M,'Доверие к оценке движения','Контроль по изображению и ошибка прогноза')
  if 'finetune' in i:return p(M,'Дообучение движения','Вход, декодер и ядро')
  return p(M,'Качество поля движения','Масштаб, считывание и полный трекер')
 if i.startswith(('fly_signature','fly_matched','fly_connectome','fly_ensemble','fly_p10','fly_flywire','flywire')):return p('Вклад и устройство FlyVis','Информативность и устойчивость признаков','Вход, коннектом, ансамбль и режим данных')
 if i.startswith(('flyvis_retina','fly_eye_frontend','flyvis_standalone','flyvis_stress','flyvis_internal_readout','flyvis_foveation','flyvis_readout','flyvis_event','fly_assoc','fly_future')):
  return p('Вклад и устройство FlyVis','Подключение зрительного модуля','Вход, считывание и первые архитектуры')
 if a in ['confidence','age','occlusion','baseline'] or i in ['detection_oracle_ladder_v1','res960_probe_v1','past_view_oracle_v1']:
  return p(D,'Обнаружение объектов','Потери, уверенность и обучение')
 if a in ['infra','video30'] or i.startswith(('train_','bench_','event_dataset','hota_error')):return p(I,'Данные и воспроизводимость','Видео, кэши и контрольная точка')
 if a in ['qmatch','event_confidence']:return p(L,'Событийные решения','Признаки, выборки и головы')
 if a in ['tracker10','thresholds','zoo','own_tracker','motion']:return p(A,'Геометрическая основа трекера','Модель рамки и правила сопоставления')
 if a=='frequency':return p('Архивные методы обработки','Частотное представление','Проверенный фиксированный фильтр')
 if i=='line_b_combined_v1':return p(A,'Сборка системы','Совместное действие модулей')
 if i=='o2_masa_residual_v1':return p(A,'Восстановление потерянного объекта','Остаточный запас после MASA')
 raise ValueError((i,c['title']))
mapping={c['id']:topic(c) for c in H['experiments']}
overrides={
 'Обучение ядра и считывателя':['fly_object_readout_z13_v1','z13_distribution_pilot_v1','z13_core_continue_v1','z13_next_differentiable_v1','z13_unperturbed_readout_v1'],
 'Извлечение положения из карты':['z13_local_peak_v1','z13_map_only104_v1','z13_motion_window_v1','z13_search_conditioned_v1'],
 'Перенос считывателя на другие видео':['z13_map_transfer_v1','z13_map_transfer104_v1','z13_center_hota_v1'],
 'Причины неверного назначения':['z13_threshold_diagnostic_v1','z13_first_divergence_v1','z13_pairwise_map_v1','z13_boundary_readout_v1','z13_localization_labels_v1'],
 'Контроли памяти и отрицательных ответов':['z13_query_memory_v1','z13_observed_negatives_gate_v1','z13_activity_amplitude_v1','z13_default_class_readout_v1']}
for title,ids in overrides.items():
 for i in ids:mapping[i]=['Пространственное считывание FlyVis','Обучение и применение считывателя',title]
for i in list(mapping):
 if mapping[i] and mapping[i][-1]=='Вход, считывание и первые архитектуры':
  if 'retina' in i or 'frontend' in i:sub='Фоторецептор и ламина'
  elif 'future' in i:sub='Подходы к памяти и модели цели'
  elif 'readout' in i:sub='Первые считыватели активности'
  else:sub='Первое подключение к трекеру'
  mapping[i]=['Вклад и устройство FlyVis','Подключение зрительного модуля',sub]
Path('experiments/obsidian_experiment_topics.json').write_text(json.dumps(dict(scope='Навигация по задаче и механизму; не изменение научных статусов',topics=mapping),ensure_ascii=False,indent=2)+'\n',encoding='utf8')
from collections import Counter
for k,v in Counter(tuple(x) for x in mapping.values() if x).most_common():print(v,' > '.join(k))
