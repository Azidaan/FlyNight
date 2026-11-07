"""Фиксировать цену и источник перед загрузкой, срок не продлевается."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,hashlib,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
def main():
    root=ROOT/'runs/b123_20261027/sam2_weights_download';assert not root.exists();root.mkdir()
    source=json.loads((ROOT/'tools/sam2-runtime-20261027/provenance.json').read_text(encoding='utf-8'))
    p=dict(created=datetime.now(timezone(timedelta(hours=3))).isoformat(),budget_seconds=300,
        author_stop='2026-10-27T07:20:00+03:00',device='cpu',url=source['weight_url'],bytes=source['weight_bytes'],
        target='weights/sam2_official_20261027/sam2.1_hiera_tiny.pt',
        purpose='Собственная масочная адаптация после отрицательного выходного BUSCA. SAM2.1tiny вместо SAM+DeAOT+селекторLTR: не называть воспроизведением статьи. Новыхданныхнет.',
        source_sha256=hashlib.sha256((ROOT/'scripts/b123_sam2_download_20261027.py').read_bytes()).hexdigest())
    (root/'protocol.json').write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    from scripts.b123_safe_journal_20261027 import journal
    journal('sam2_mask_adaptation_preparation','начата подготовка собственной масочной ветви после BUSCA',
        '*Действие.* Продолжать потерянный объект по переносимой маске, а не только по прямоугольному прогнозу. '
        'Собственная адаптация использует компактную SAM2.1tiny и её оценку наличия объекта. '
        'Это не точная LTR: нет её27масок, обученного селектора и DeAOT; SGAR-MOT также не подменяется. '
        'Самое сильное возражение: маска может перейти на соседа, а уверенность наличия не доказывает личность. '
        'Нужны тот же неизменный XYWH, ложные продолжения, FP/FN/IDSw и полная HOTA, с заранее фиксированным отказом.\n\n'
        '*Подготовка.* Официальные43текстовыхфайла SAM2,410746байт, commit2b90b9f5ceec907a1c18123530e92e794ad901a4; '
        'проверена доступность официального checkpoint156008466байт. Источник:tools/sam2-runtime-20261027/provenance.json. '
        'Загрузка только весов ограничена300с отдельным контроллером; исходные меньшие бюджеты не продлеваются. '
        'Протокол:runs/b123_20261027/sam2_weights_download/protocol.json. Новые наборы данных не загружаются, основное окружение не меняется. '
        'Технический GPU-профиль и научный пилот регистрируются отдельно до расчёта.')
    print('Prepared bounded official checkpoint download')
if __name__=='__main__':main()
