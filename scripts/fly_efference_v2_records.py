"""Append-only chronology and separate experiment card for the authorized revision."""
from __future__ import annotations
import json
import sys
import hashlib
import subprocess
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
    pp=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md'; plan=pp.read_text(encoding='utf-8'); original_plan=plan
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
    elif mode=='finish':
        e=next(x for x in h['experiments'] if x['id']==ID)
        if e.get('done'): raise ValueError('Result already recorded')
        gate=json.loads((OUT/'gate.json').read_text(encoding='utf-8'))
        night=json.loads((OUT/'night_result.json').read_text(encoding='utf-8'))
        stage=json.loads((OUT/'stage1_report.json').read_text(encoding='utf-8')) if (OUT/'stage1_report.json').exists() else None
        def number(x): return f'{x:.3f}'.replace('.',',')
        chunks=[]
        for g,label in [('regression_images','2 прежних изображения'),('holdout_images','4 новых для калибровки train-видео')]:
            a=gate['agg'][g]
            chunks.append(f"{label}: новая T1 {number(a['new']['T1_drift_share'])}, T1s {number(a['new']['T1s_drift_share'])}, T2 cos {number(a['new']['T2_cos'])} длина {number(a['new']['T2_ratio'])}, T3 cos {number(a['new']['T3_cos'])} длина {number(a['new']['T3_ratio'])}; T4 новая {number(a['new']['T4_err_px'])} px, прежняя стабилизация {number(a['legacy']['T4_err_px'])}, старый глаз {number(a['old']['T4_err_px'])}, DIS {number(a['ctrl']['T4_err_px'])}, без поправки 24 px. Условия: {gate['passed'][g]}.")
        gate_text=' '.join(chunks)+' Источник: runs/fly_efference_eye_v2/gate.json (agg, passed).'
        if not gate['all_passed']:
            state='completed_negative'; decision='Технический критерий не выполнен. Ступень 1 и HOTA/MASA не запускались. Остановлена эта реализация; коэффициенты и выборка после результата не менялись.'
        elif stage and stage['complete']:
            state='completed_positive' if stage['criterion'] else 'completed_negative'
            decision=f"Ступень 1 завершена на {stage['videos_done']} видео, критерий {stage['criterion']}. HOTA/MASA не запускались согласно ночному ограничению."
        else:
            state='inconclusive'; decision='Технический критерий выполнен, но полный набор dev35 не завершён в разрешённое окно. Промежуточные числа не являются итогом ступени 1. HOTA/MASA не запускались.'
        e.update(state=state,done=now.date().isoformat(),obtained=gate_text,
                 interpretation='Новая реализация меняет перенос начального отклика и цель калибровки. Сравнение с legacy при одинаковой калибровке выделяет перенос ссылки. Синтетика проверяет перенос всего изображения; результат не доказывает выделение объекта на движущемся фоне. Причина межсценового разброса не установлена.',
                 decision=decision,cannot_claim='Нельзя заявлять улучшение HOTA, бесполезность эфферентной копии вообще или доказанную причину провала v1. Проверочные train-кадры отложены только от калибровки; это не независимые новые val-видео.')
        a=gate['agg']['holdout_images']
        worse=sum(r['new']['T4_err_px']>r['legacy']['T4_err_px'] for group in gate['checks'].values() for r in group.values())
        e['interpretation']+=f" Гипотеза Codex о пользе переноса начального состояния не подтвердилась: T4 хуже legacy на {worse} из 6 изображений; на четырёх отложенных кадрах {number(a['legacy']['T4_err_px'])} → {number(a['new']['T4_err_px'])} px при той же калибровке. T1s вырос {number(a['legacy']['T1s_drift_share'])} → {number(a['new']['T1s_drift_share'])}. Это недостаток испытанного переноса ссылки, не доказательство пользы неподвижной ссылки вообще."
        e['acceptance'][0].update(met=gate['all_passed'],evidence='runs/fly_efference_eye_v2/gate.json: passed, all_passed')
        e['acceptance'][1].update(met=(bool(stage['criterion']) if stage and stage['complete'] else None),evidence=('runs/fly_efference_eye_v2/stage1_report.json' if stage else 'Не запускалась: технический критерий не выполнен'))
        e['commits']=['7ec9d52']
        e['evidence'] += [{'kind':'json','ref':'runs/fly_efference_eye_v2/gate.json','locator':'agg','claim':'Техническая проверка на обоих наборах'},
                          {'kind':'json','ref':'runs/fly_efference_eye_v2/night_result.json','locator':'finished','claim':'Завершение ограниченного запуска'}]
        if stage:
            e['obtained']+=' Ступень 1: '+json.dumps(stage,ensure_ascii=False)+'; runs/fly_efference_eye_v2/stage1_report.json.'
        manifest={'finished_msk':now.isoformat(),'head_before_result':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                  'sha256':{str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'tracking/efference_eyes_v2.py',ROOT/'tracking/efference_eyes.py',ROOT/'tracking/foveated_eyes.py',ROOT/'scripts/fly_efference_eye.py',ROOT/'scripts/fly_efference_v2.py',ROOT/'scripts/fly_efference_stage1_v2.py',OUT/'protocol.json',OUT/'calibration_raw.npz',OUT/'checks_raw.npz',OUT/'gate.json']},
                  'model':'flow/0000/000','deadline':'2026-10-12T07:40:00+03:00','night':night}
        (OUT/'provenance.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
        body=f'*Получено (`{ID}`).* {gate_text}\n\n*Вывод.* {e["interpretation"]}\n\n*Решение.* {decision} Статусы P2 и P9 не повышены: критерии решения этих проблем не измерены.\n\n*Сводка.* HOTA не измерялась; доля оракульного запаса не вычисляется, поскольку синтетическая ошибка и HOTA — разные метрики.'
        appendix=f'### 3.10. Согласованный фовеальный глаз, проверка 12.10 (`{ID}`)\n\n{gate_text}\n\n{decision}\n\n'
        plan=plan.replace('## 4. Что закрыто, и по какому основанию',appendix+'## 4. Что закрыто, и по какому основанию',1)
        (OUT/'summary.md').write_text(f'# {ID}\n\n{body}\n',encoding='utf-8')
    else:
        raise ValueError(mode)
    if hp.read_text(encoding='utf-8') != original or pp.read_text(encoding='utf-8') != original_plan:
        raise RuntimeError('Concurrent edit detected')
    hp.write_text(json.dumps(h,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    heading='## Журнал изменений'
    title='постановка повторной технической проверки' if mode=='start' else 'итог ограниченного прогона'
    entry=f'\n\n**{now:%d.%m.%Y, %H:%M} MSK** — `{ID}`: {title}\n\n{body}\n'
    pp.write_text(plan.replace(heading,heading+entry,1),encoding='utf-8')
    print(ID,mode,now.isoformat())


if __name__=='__main__': main(sys.argv[1])
