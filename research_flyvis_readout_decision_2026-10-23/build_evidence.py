"""Snapshot existing evidence; never runs an experiment or imports torch."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def main():
    specs = {
        'z13_target_center_20261018': ['paired'],
        'z13_target_center_dev_20261018': ['paired'],
        'z13_candidate_reserve_20261021': ['arms'],
        'z13_candidate_identity_20261021': ['paired'],
        'z13_candidate_data104_20261021': ['paired'],
        'z13_map_only104_20261021': ['paired'],
        'z13_observed_memory_20261021': ['paired'],
        'z13_query_memory_20261021': ['paired'],
        'z13_aligned_core_20261018': ['stages', 'decision'],
        'z13_balanced_ridge_20261022': ['paired', 'summary'],
        'z13_hinge_target_20261023': ['paired', 'summary'],
        'z13_first_divergence_20261022': ['conclusion', 'limits', 'counterfactual'],
        'z13_center_hota_20261021/eval': ['pooled', 'diff'],
    }
    records = []
    for run, keys in specs.items():
        rel = f'runs/{run}/report.json'
        path = ROOT / rel
        obj = json.loads(path.read_text(encoding='utf-8-sig'))
        records.append({'source': rel, 'sha256': sha(path),
                        'excerpts': [{'locator': [key], 'value': obj[key]} for key in keys]})
    codes = ['scripts/z13_candidate_ranker_20261021.py',
             'scripts/z13_candidate_data104_20261021.py',
             'tracking/z13_candidate_ranker.py', 'tracking/z13_candidate_identity.py',
             'tracking/z13_readout.py', 'tracking/z13_aligned_loss.py',
             'scripts/z13_aligned_core.py', 'tracking/z13_target_center.py',
             'tracking/z13_observed_memory.py', 'tracking/z13_query_memory.py',
             'tracking/z13_candidate_modules.py', 'scripts/z13_data.py',
             'scripts/z13_continue.py']
    save('evidence.json', {'date_msk': '2026-10-23', 'kind': 'existing_primary_artifacts_only',
                           'new_experiments': False, 'records': records,
                           'code_sources': [{'source': p, 'sha256': sha(ROOT / p)} for p in codes]})
    sources = [
        ('S1', 'Learning Target Candidate Association to Keep Track of What Not to Track', '2021', '2103.16556v2', '3.2–3.7; Appendix B; Table 11', 'Обучаемое соответствие кандидатов и отсутствие пары.', 'Ручное сопоставление проекта не воспроизводит обучаемый модуль.'),
        ('S2', 'SuperGlue: Learning Feature Matching with Graph Neural Networks', '2019/2020', '1911.11763v1', '3.2–3.3', 'Совместное сопоставление и обучение отсутствию соответствия.', 'Ключевые точки и геометрия камер; не FlyVis и не HOTA.'),
        ('S3', 'Siam R-CNN: Visual Tracking by Re-Detection', '2019/2020', '1911.12836v1', '3.1–3.3', 'Различение экземпляров, трудные отрицательные объекты, история траекторий.', 'Готовые богатые признаки и детектор нельзя считать вкладом мухи.'),
        ('S4', 'TAPIR: Tracking Any Point with per-frame Initialization and temporal Refinement', '2023', '2306.08637v2', '3.1; Equation 1', 'Отдельные ошибки положения, видимости и точности.', 'Точки, не рамки; точность не равна полезности поправки.'),
        ('S5', 'Know Your Surroundings: Exploiting Scene Information for Object Tracking', '2020', '2003.11014v2', '3; Table 1', 'Перенос локализованного состояния сцены по соответствиям.', 'Нужны обученные соответствия; простое состояние само по себе не помогает.'),
        ('S6', 'Transforming Model Prediction for Tracking', '2022', '2203.11192v1', '3.2–3.3; Table 1', 'Предсказание модели цели с учётом наблюдений и текущего поиска.', 'Не эквивалент смешивания готовых карт; фон не универсально полезен.'),
        ('S7', 'Distractor-aware Siamese Networks for Visual Object Tracking', '2018', '1808.06048v1', '3.2; Table 2', 'Отрицательные пары похожих объектов, а не только лёгкий фон.', 'Отрицательные примеры уже есть в нашем обучении.'),
        ('S8', 'Towards Efficient Training with Negative Samples in Visual Tracking', '2023', '2309.02903v1', '3.3–3.4; Appendix E', 'Поисковые области без целевого объекта.', 'Не решает автоматически длительное удержание идентичности; статус публикации отдельно не проверен.'),
        ('S9', 'Distractor-Aware Memory-Based Visual Object Tracking', '2025', '2509.13864v1', '3.2; 5.7; 6', 'Память наблюдений для различения конкурентов с контролем записи.', 'Зависимость от SAM2; расширение исходной работы не независимое подтверждение.'),
        ('S11', 'Learning Discriminative Model Prediction for Tracking', '2019', '1904.07220v2', 'method; target model learning and ablations', 'Обучение модели цели с учётом фона и применяемого считывания.', 'Ряд элементов уже проверялся у нас; чужие AUC не переводятся в HOTA.'),
    ]
    entries = [{'id': i, 'title': title, 'year': year,
                'url': f'https://arxiv.org/html/{arxiv}', 'checked_sections': sections,
                'scope': 'selected_full_text_methods_and_ablations', 'mechanism': mechanism,
                'transfer_limit': limit, 'tested_on_our_project': False}
               for i, title, year, arxiv, sections, mechanism, limit in sources]
    entries.append({'id': 'S10', 'title': 'When Trackers Fail: VLM-Guided Verification and Recovery for Robust Video Object Segmentation',
                    'year': '2026', 'url': 'https://arxiv.org/abs/2610.15929',
                    'scope': 'abstract_and_metadata_only', 'used_for_quantitative_recommendation': False,
                    'transfer_limit': 'Тяжёлый внешний механизм; полный метод не проверен.'})
    save('sources.json', {'date_msk': '2026-10-23', 'coverage': 'targeted_not_systematic', 'sources': entries})
    print(f'Saved {len(records)} primary reports, {len(codes)} code hashes, {len(entries)} literature entries.')

if __name__ == '__main__':
    main()
