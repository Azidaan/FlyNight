"""Post-hoc explanation from immutable saved rows, without another tracker run."""
from pathlib import Path
from collections import Counter
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
from scripts.z13_continue import paired
RUN=ROOT/'runs/z13_box_proposal_20261025'
GATE=ROOT/'runs/z13_proposal_geometry_20261025/recovery1'


def main():
    assert not (RUN/'final_summary.json').exists()
    r=read(RUN/'report.json');g=read(GATE/'report.json');p=read(RUN/'protocol.json')
    assert read(RUN/'finish_recorded.json')['report_sha256']==sha(RUN/'report.json')
    zero={v:0. for v in r['per_video']['D_box']}
    standalone={a:paired(x,zero) for a,x in r['per_video'].items()}
    differences=Counter();selection=Counter();eligible=0
    for video in p['evaluate26']:
        op=GATE/'videos'/(video+'.json');np=RUN/'videos'/(video+'.json')
        assert sha(op)==g['video_sha256'][video] and sha(np)==r['video_sha256'][video]
        old={x['uid']:x for x in read(op)['rows']}
        for row in read(np)['rows']:
            if row['kind']<0:continue
            index=read(RUN/'choice.json')['indices']['D_box']
            a,b=row['arms']['D_center'][index],row['arms']['D_box'][index]
            outcome=lambda x:'beneficial' if x['beneficial'] else 'harmful' if x['harmful'] else 'neutral'
            differences[outcome(a)+' -> '+outcome(b)]+=1
            if row['uid'] not in old:continue
            parent=old[row['uid']]
            if not any(x['beneficial'] for x in parent['arms']['boxes_eye']):continue
            eligible+=1
            chosen=row['proposal_choices']['D']['index']
            options=[x['index'] for x in parent['geometry_meta']['proposals'] if x['inside_eye']]
            if chosen<0:kind='NULL_selected_despite_helpful_box'
            else:
                assert chosen in options
                full=parent['arms']['boxes_eye'][1+options.index(chosen)]
                if full['beneficial']:
                    kind='helpful_box_chosen_and_alpha025_helpful' if b['beneficial'] else 'helpful_full_box_chosen_but_alpha025_not_helpful'
                else:kind='selected_other_box_despite_helpful_box'
            selection[kind]+=1
    assert sum(selection.values())==eligible and sum(differences.values())==5822
    write(RUN/'final_summary.json',dict(at=now().isoformat(),posthoc_explanation_only=True,no_new_fit_threshold_or_tracker_run=True,
        report_sha256=sha(RUN/'report.json'),gate_report_sha256=sha(GATE/'report.json'),
        prototype_vs_baseline=standalone,center_to_box_outcome_cross=dict(differences),
        rows_with_any_helpful_observable_full_box=eligible,D_selection_among_these=dict(selection),
        scientific_status='The originally registered result stays negative. Selection and interpolation counts explain this implementation on reused baseline states; they do not identify a core/representation cause or constitute HOTA.',
        gate_metadata_note='geometry_meta.original_xywh is the helper-entry snapshot after legacy R1 center alternatives; its XY must not be interpreted as the original tracker prediction. Legacy alternatives change only XY; WH is original. Gate counts use proposal indices/support and oracle_box, never this XY field. Original baseline prediction is in immutable reserve labels; exact raw replay was checked independently.'))
    from scripts import z13_box_proposal_20261025 as m
    m.journal('объяснение отрицательного выхода рамки по сохранённым строкам','*Проверено после основного исхода.* Дополнительных прогонов и подгонки нет. По полным прежним строкам сопоставлены выбор реальной области, геометрический оракул и действие с заранее выбранным весом0,25. Источник runs/z13_box_proposal_20261025/final_summary.json.\n\n*Получено.* В '+str(eligible)+' ситуациях доступна полезная полная рамка; разбор выбораD: '+str(dict(selection))+'. Переходы исходовцентр→рамка: '+str(dict(differences))+'.\n\n*Ограничение.* Это разбор после просмотра исхода, основной отрицательный критерий не меняется. Выбор неправильной области не доказывает отсутствие информации в ядре. Поле original_xywh в геометрическом gate — снимок на входе после последней прежней альтернативыR1; егоXYне является исходным прогнозом. В числах gate это поле не используется, исходный прогноз сохранён отдельно в reserveметках. ПрежниеJSONне переписаны.')


if __name__=='__main__':main()
