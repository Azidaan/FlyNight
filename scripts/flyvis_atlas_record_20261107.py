"""Publish the independently evaluated record in the local atlas and project chronology."""
from pathlib import Path
import sys,json
from datetime import datetime
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import flyvis_pair_support_overlap_20261107 as io
from scripts.flyvis_atlas_readout_20261107 import R, WEB
C=Path('runs/flyvis_atlas_core_20261107')

def main():
    import psutil
    a=io.read(C/'independent_audit.json'); assert a['passed'] and (a['units'],a['pairs'],a['videos'])==(65,172,23)
    q=io.read(C/'protocol.json');inf=io.read(C/'inference_completed.json');assert a['raw_manifest_sha256']==io.sha(C/'inference_completed.json')
    assert inf['protocol_sha256']==io.sha(C/'protocol.json')
    ctrl_path=Path(io.read(C/'progress.json')['guard']['supervisor_report']);ctrl=io.read(ctrl_path)
    assert ctrl['returncode']==0 and not ctrl['stop_errors'] and ctrl['reason'] is None
    assert all(not psutil.pid_exists(int(pid)) or psutil.Process(int(pid)).create_time()!=birth for pid,birth in ctrl['owned'].items())
    rd=io.read(R/'report.json');cond={r['name']:r for r in a['conditions']};assert len(cond)==132 and cond['native']['correct']==rd['baseline']['credit']
    assert cond['identity']['choice_changes']==0
    results=[{**cond['0.90/'+t],'type':t} for t in q['types']];pos=[r['type'] for r in results if r['ci_family65_pp'][0]>0];neg=[r['type'] for r in results if r['ci_family65_pp'][1]<0]
    changes=[r for r in results if r['choice_changes']>0]
    types=[dict(name=r['type'],credit=r['correct'],delta=r['paired_delta_pp'],changes=r['choice_changes'],ci=r['ci95_pp'],
                adjustedCI=r['ci_family65_pp'],gains=r['gains'],losses=r['losses'],maxAbsMarginChange=r['max_abs_margin_change']) for r in results]
    msg=f"Ослабление одного типа на 10% перед последним кадром: {len(changes)} из 65 типов изменили хотя бы один выбор. После поправки за 65 сравнений положительных интервалов: {len(pos)}, отрицательных: {len(neg)}."
    data=dict(status='complete',n=172,videoCount=23,message=msg,types=types,baseCredit=cond['native']['correct'])
    (WEB/'core.js').write_text('window.FLY_CORE='+json.dumps(data,ensure_ascii=False,separators=(',',':'))+';\n',encoding='utf-8')
    io.dump(C/'record_verified.json',dict(passed=True,owned_ended=True,controller=ctrl,audit_sha256=io.sha(C/'independent_audit.json'),positive=pos,negative=neg),True)
    cat=io.read(io.CAT);card=next(x for x in cat['experiments'] if x['id']=='flyvis_atlas_core_pulse_v1')
    card.update(state='completed_positive' if pos else 'inconclusive',done=datetime.now().date().isoformat(),
         obtained=msg+' Источник: '+(C/'independent_audit.json').as_posix()+'.',interpretation='Чувствительность данного считывания к короткому импульсу. Измерение не проверяет длительную роль клетки, идентичность после окклюзии или HOTA.',
         cannot_claim='Повторно использованы исторические пары. Одна модель, один горизонт. Один тип представлен разным числом нейронов. Отсутствие смены победителя не означает отсутствие изменения баллов.',
         decision='Сохранить все сравнения и контроль с усилением. Не выбирать исправление по этим же 172 парам.')
    for item in card['acceptance']:item.update(met=True,evidence=(C/'independent_audit.json').as_posix())
    card['evidence'].append(dict(kind='file',ref=(C/'independent_audit.json').as_posix(),claim='Все условия и перебор 65 типов в рекуррентном состоянии.'))
    io.dump(io.CAT,cat);io.journal('аудит 65 внутренних вмешательств на 172 парах (`flyvis_atlas_core_pulse_v1`). '+msg+' '+(C/'independent_audit.json').as_posix()+'.')
    with Path('FLYVIS_TRACKER_RESEARCH_ROADMAP.md').open('a',encoding='utf-8') as f:f.write('\nВнутри ядра: '+msg+' Источник: '+(C/'independent_audit.json').as_posix()+'.\n')
    def fr(x):return f'{x:.2f}'.replace('.',',')
    tab=['| Тип клеток | Правильных из 172 | Изменилось решений | Исправлено / испорчено | Изменение, п.п. | Интервал с поправкой, п.п. |','|---|---:|---:|---:|---:|---|']
    for r in results:
        tab.append(f"| {r['type']} | {r['correct']:g} | {r['choice_changes']} | {r['gains']} / {r['losses']} | {fr(r['paired_delta_pp'])} | [{fr(r['ci_family65_pp'][0])}; {fr(r['ci_family65_pp'][1])}] |")
    doc='# Что меняется при ослаблении одного типа клеток FlyVis\n\nНа прежних 172 парах из 23 исторических видео без обучения проверены две разные операции.\n\n'
    doc+=f"**Чтение выхода.** На целых рамках фиксированные восемь каналов T4/T5 выбрали верное продолжение {rd['t4t5']['credit']:g} раз, а все 34 прямых канала с прежними весами — {rd['baseline']['credit']:g}. Разность {fr(rd['comparison']['delta'])} п.п.; 95%-й интервал по видео [{fr(rd['comparison']['ci'][0])}; {fr(rd['comparison']['ci'][1])}]. После поправки Холма ни одно из 755 отдельных ослаблений выхода не установило устойчивое изменение качества.\n\n"
    doc+='**Изменение состояния.** Непосредственно перед последним кадром все нейроны одного типа получают множитель 0,90 на одну запись внутреннего состояния. Исходные веса, предыдущие кадры, рамки и следующие два шага сети одинаковы. Для каждого типа также проверено усиление 1,10. Сравнение ведётся с исходным 34-канальным считывателем. '+msg+'\n\n'+'\n'.join(tab)+'\n\n'
    doc+='Таблица описывает ослабление; симметричное усиление, медианы и максимумы изменения разницы баллов, обычные и скорректированные интервалы сохранены в [независимом расчёте](independent_audit.json). '+f"Предел расхождения исходной нормированной карты с архивом по наблюдениям: {a['archive_max_abs_map']:.8g}, контроль тождественной копии: {a['identity_max_abs_map']:.8g}. "+'Нет постоянного выключения типа и оценки качества полных траекторий. Число нейронов у разных типов различается, величина суммарного вмешательства не выровнена.\n'
    with (C/'СВОДКА_ДВУХ_ПРОВЕРОК.md').open('x',encoding='utf-8') as f:f.write(doc)
    print(msg)

if __name__=='__main__':main()
