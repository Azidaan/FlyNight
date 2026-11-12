from common import *
import shutil,subprocess
ag=read(ROOT/'research_agreement_readout_20261112/protocol.json');inputs={}
for n in ['common.py','register.py','probe.py','audit.py']:inputs[(OUT/n).relative_to(ROOT).as_posix()]=sha(OUT/n)
for n in ['research_agreement_readout_20261112/protocol.json','research_agreement_readout_20261112/common.py','research_agreement_readout_20261112/features.json','research_agreement_readout_20261112/predictions.json','tracking/full_template_correspondence.py','tracking/observed_spatial_reader.py','tracking/hex_linear_interpolation.py','tracking/channel_agreement_correspondence.py','tracking/z13_readout.py','runs/z13_local_peak_20261021/geometry.json']:inputs[n]=sha(ROOT/n)
for r in ag['train']+ag['screen']:
    for key in r['keys']:
        for item in ag['snapshots'][key]['files']:
            n=item['path'];inputs[n]=sha(ROOT/n);assert inputs[n]==ag['inputs'][n]
save('protocol.json',dict(registered=now(),head=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),authorization='Continue user-authorized exact-cause diagnosis; no full tracker and no threshold changes. This is not promotion of failed agreement variant.',inputs=inputs,
 question='Is the saved shift chooser sufficient to explain the wrong FlyVis decision in own232, and what occurs in all other archived pairs?',
 hypothesis='Different selected shifts can confound the comparison between FlyVis and ordinary-image heads. Exchange only the three already selected shifts while keeping maps, support, geometry and each fitted head fixed.',
 action='Determine whether changing the existing shift selection alone is supported as a repair, before any implementation. No further search for a favorable shift.',
 design='For all26 train-development rows and5 known screen cases, evaluate all3representations at each of3saved per-arm winning shifts. The diagonal is the identity control; two off-diagonals are prespecified alternatives. Copy query tensors; maps/support/weights unchanged. Original leave-video-out fit per26row; FINAL fit per5case. Agreement heads only, fixed threshold0.',
 primary='For232, record whether C34 sign changes under either image-selected shift and whether ordinary heads retain positive signs under C34 shift. Both alternatives reported; no winner adopted.',
 acceptance=['All93diagonal scores and saved feature vectors reproduced within1e-10; original maps and sources unchanged.','All279raw feature vectors and scores saved; independent interpolation/formula/score audit agrees within1e-9.','A shift-only explanation for232 requires at least one prespecified replacement to change its C34decision. Failure rejects only these substitutions, not all possible physical correspondences.'],
 limitations='Known development cases, no independent test or population claim. Off-diagonal features may be out of training distribution. No physical-correctness claim for any saved shift. No recurrent/input/core intervention. No per-case adoption, new fits, shifts, thresholds or tracker calls.',deadline='2026-11-12T07:40:00+03:00'))
(OUT/'before').mkdir()
for n in ['experiments/project_history.json','experiments/obsidian_experiment_topics.json','experiments/system_problems.json','dissertation/ЕДИНЫЙ_ПЛАН.md']:shutil.copy2(ROOT/n,OUT/'before'/Path(n).name)
path=ROOT/'experiments/project_history.json';raw=path.read_bytes();d=json.loads(raw);assert not any(c['id']==ID for c in d['experiments'])
p=read(OUT/'protocol.json');d['experiments'].append(dict(id=ID,title='Одинаковый сдвиг сравнения — отделение выбора координат от зрительных признаков',area='flyvis',state='running',started='2026-11-12',question=p['question'],hypothesis=p['hypothesis'],expected='Проверить достаточность замены выбранного сдвига для объяснения отказа232, сохранив все остальные случаи и ограничения.',acceptance=[dict(criterion=s,met=None) for s in p['acceptance']],evidence=[dict(kind='file',ref='research_saved_shift_probe_20261112/protocol.json')]))
assert raw==path.read_bytes();path.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
path=ROOT/'experiments/obsidian_experiment_topics.json';d=read(path);d['topics'][ID]=['Сохранение личности','Выбор продолжения среди рамок','Причины неверного сравнения карт'];path.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
journal('Постановка (`'+ID+'`): все 26 прежних связей и пять случаев; перекрёстно проверить три заранее сохранённых сдвига при прежних картах, поддержке и обученных слоях. Нового поиска сдвига, обучения, порога и трекера нет. Это проверка достаточности конкретной причины, не физическая разметка третьего этапа. Условия: research_saved_shift_probe_20261112/protocol.json.')
print('REGISTERED',len(inputs))
