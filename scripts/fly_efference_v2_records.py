"""Append-only chronology and separate experiment card for the authorized revision."""
from __future__ import annotations
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/fly_efference_eye_v2'
ID='fly_efference_eye_v2'


def main(mode):
    now=datetime.now(timezone(timedelta(hours=3)))
    hp=ROOT/'experiments/project_history.json'
    original=hp.read_text(encoding='utf-8'); h=json.loads(original)
    if mode=='format':
        hp.write_text(json.dumps(h,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
        return
    pp=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md'; plan=pp.read_text(encoding='utf-8')
    if mode=='start':
        if any(e['id']==ID for e in h['experiments']): raise ValueError('Card already exists')
        old=json.loads((ROOT/'runs/fly_efference_eye/calib.json').read_text(encoding='utf-8'))
        selection=json.loads((ROOT/'runs/train_cache/selection.json').read_text(encoding='utf-8'))['videos']['train']
        used={x[0] for x in old['calib_images']+old['test_images']}
        pool=sorted(set(selection)-used)
        held=np.random.default_rng(12026).choice(pool,4,replace=False).tolist()
        protocol={'id':ID,'registered_msk':now.isoformat(),'deadline_msk':'2026-10-12T07:40:00+03:00',
            'calibration_images':old['calib_images'],'regression_images':old['test_images'],
            'holdout_images':[[v,900] for v in held], 'seed':12026,
            'sizes':[60,90,120],'speeds':[1,2,4],'steps':list(range(1,13)),
            'calibration':'2x2 least squares on accumulated 12-step displacement; no intercept; same procedure all arms',
            'variants':['old','legacy','new','ctrl'],
            'reference':'birth activity transported exactly like live state, then decoded and subtracted; boundary nearest neighbor as v1',
            'gate':'T1 drift <=0.25; T2,T3 cos >=0.8 and length 0.6..1.4; T4 error <=12px. Both old two-image group and new four-video train group must pass averages.',
            'stage1':'Only if gate passes: fixed dev35, new-old/new-none/new-ctrl mean per-video errors all negative with CI95 entirely below zero, 5000 paired video bootstrap',
            'stop':'No tuning after gate; no HOTA/MASA tonight; stop by deadline; incomplete set is not a conclusion',
            'counterargument':'Global signal scaling may not transfer between scenes; DIS may remain much more accurate. Technical success alone does not prove benefit on video.'}
        OUT.mkdir(parents=True,exist_ok=True)
        (OUT/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
        parent=next(e for e in h['experiments'] if e['id']=='fly_efference_eye_v1')
        e={'id':ID,'title':'Согласованный начальный отклик и калибровка накопленного движения фовеального глаза',
          'area':parent['area'],'state':'running','started':now.date().isoformat(),
          'question':'Проходит ли одна исправленная реализация техническую проверку и уменьшает ли накопленную ошибку на dev35?',
          'hypothesis':'Перенос начального состояния вместе с активностью и подбор считывания по накопленному движению устраняют несогласованность ссылки и цели калибровки. Это гипотеза, не установленная причина провала v1.',
          'expected':'Разрешение автора в чате 12.10 до 07:40 MSK. Одна версия, критерии и новые train-видео зафиксированы до запуска: runs/fly_efference_eye_v2/protocol.json. Без изменения lcn, детектора, трекера и MASA; HOTA этой ночью не запускается.',
          'summary':'Продолжение fly_efference_eye_v1 с отдельными исходниками и результатами. Начальный отклик определяется как декодирование перенесённого состояния рождения; пространственная интерполяция та же, что для активности. Калибровка 2x2 по суммарному перемещению шагов 1–12, общий протокол для всех вариантов. Старый глаз, прежняя стабилизация и исправленная — отдельные варианты; DIS — контроль.',
          'acceptance':[{'criterion':protocol['gate'],'met':None},{'criterion':protocol['stage1'],'met':None}],
          'tags':['fly-problem-P2','fly-problem-P9'], 'files':['tracking/efference_eyes_v2.py','scripts/fly_efference_v2.py','scripts/fly_efference_stage1_v2.py'],
          'runs':['runs/fly_efference_eye_v2'],'commits':[],
          'evidence':[{'kind':'json','ref':'runs/fly_efference_eye_v2/protocol.json','locator':'gate','claim':'Критерии до запуска'}]}
        h['experiments'].append(e)
        body='*Постановка.* Автор разрешил Codex выполнить прогон до 07:40 MSK. Отдельная реализация `fly_efference_eye_v2`: перенос начального состояния и активности одинаковым оператором, затем декодирование ссылки; калибровка по суммарному смещению за шаги 1–12. Один GPU-процесс. Прежние файлы и результаты v1 не перезаписываются.\n\n*Критерии.* `runs/fly_efference_eye_v2/protocol.json`: прежние T1–T4 должны пройти и на двух прежних изображениях, и на четырёх новых train-видео (seed 12026); новые кадры не участвуют в подборе. При успехе — ступень 1 на всех dev35; новая ошибка ниже старой, без поправки и DIS с ДИ ниже нуля. При неудаче — остановка реализации, без перенастройки. Ступень HOTA и MASA не разрешена этой ночью.\n\n*Ограничение.* Перенос ссылки — выбранная инженерная гипотеза. Причина провала v1 не установлена; синтетические сдвиги всей сцены не доказывают движение объекта на реальном фоне.'
    else:
        raise ValueError(mode)
    if hp.read_text(encoding='utf-8') != original or pp.read_text(encoding='utf-8') != plan:
        raise RuntimeError('Concurrent edit detected')
    hp.write_text(json.dumps(h,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    heading='## Журнал изменений'
    entry=f'\n\n**{now:%d.%m.%Y, %H:%M} MSK** — `{ID}`: {"постановка повторной технической проверки"}\n\n{body}\n'
    pp.write_text(plan.replace(heading,heading+entry,1),encoding='utf-8')
    print(ID,mode,now.isoformat())


if __name__=='__main__': main(sys.argv[1])
