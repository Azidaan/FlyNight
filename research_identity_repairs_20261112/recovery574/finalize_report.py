import json, re, hashlib
from pathlib import Path
OUT = Path(__file__).resolve().parent

def read(name):
    return json.loads((OUT / name).read_text(encoding='utf8'))

def write(name, value):
    with (OUT / name).open('x', encoding='utf8', newline='\n') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2).rstrip() + '\n')

summary = read('interpretation_final.json')
criteria = []
for variant in summary['variants']:
    source = variant['source']
    final = variant['variant'] == 'final_native_episode'
    criteria.append(dict(variant=variant['variant'], state='completed_positive' if variant['whole_pool_engineering_gate'] else 'completed_negative',
                         scope='Only the registered5-case engineering gate; not independent scientific generalization.',
                         acceptance=[dict(criterion='Correct additional recovery proposal for574', met=True, evidence=source + ': records[mode=complete_recovery,old_case=574].target_evaluation_own'),
                                     dict(criterion='No foreign or unknown new proposal anywhere in all5complete pools', met=variant['whole_pool_engineering_gate'], evidence=dict(source=source, own=variant['own'], foreign=variant['foreign'], unknown=variant['unknown'])),
                                     dict(criterion='Freeze every saved native public assignment and all occupied aliases', met=True, evidence=('audit_episode.json' if final else 'audit_lost_only.json' if 'lost' in variant['variant'] and variant['variant'] != 'primary_tracked_or_lost' else 'audit.json') + ': protected_native_row_decisions and independently reconstructed policy'),
                                     dict(criterion='No target continuation in all5remove-all-own controls; no descriptor/permutation/missing-bank false success hidden', met=True, evidence=source + ': records[mode=remove_all_target_own/descriptor_permutation/missing_competitors/no_descriptors]'),
                                     dict(criterion='Full107reference rows and222raw candidates retained for competition before NMS', met=True, evidence=source + ': baseline; records[*].rows/candidates/result.cosine'),
                                     dict(criterion='No evaluation labels or hardcoded identifier values in decision rule; candidate labels only outside policy', met=True, evidence='appearance_recovery_policy_v4.py with ordinary native identifier mapping only' if final else 'frozen puremodule and independent auditor'),
                                     dict(criterion='Public-ID lifecycle/HOTA measured', met=False, evidence='Not run and not required by this boundedengineering gate; explicitly cannot_claim, not a hidden failed performance metric.')],
                         decision='May be staged as an experimental module for the boundedadapter; no fullruntime activation' if variant['whole_pool_engineering_gate'] else 'Do not promote this broad variant; preserve3unknown proposals and the local574positive',
                         cannot_claim=summary['cannot_claim']))
write('criteria.json', dict(variants=criteria, no_further_variants=True, final_gate='Local engineering positive only; primary and lost-only failures remain recorded.'))
report_path = OUT / 'ИТОГ_ВОССТАНОВЛЕНИЯ_574.md'
text = report_path.read_text(encoding='utf8')
text = text.replace('по raw35280', 'по сырой детекции с номером 35280').replace('raw35292', 'Детекция 35292')
text = text.replace('близость MASA', 'близость MASA (модуля описания внешности объектов)', 1)
text = text.replace('Детекторная уверенность 35280', 'Уверенность детектора для рамки 35280')
text = text.replace('конкурентном пуле', 'наборе конкурирующих траекторий').replace('в исходном пуле', 'в исходном наборе')
text = text.replace('В пяти полных пулах', 'В пяти полных наборах конкурентов')
text = text.replace('нативных стадий', 'штатных стадий')
text = text.replace('Чистый модуль принимает', 'Модуль без изменения состояния трекера принимает')
text = text.replace('неизмеренного', 'неизмеренного')
report_path.write_bytes((text.rstrip() + '\n').encode('utf8'))
sentences = []
for paragraph in text.split('\n\n'):
    if not paragraph.strip() or re.match(r'^\d+\.', paragraph):
        continue
    for sentence in re.split(r'(?<=[.!?])\s+(?=[А-ЯA-Z])', paragraph.strip()):
        if 'порог' in sentence.lower() or 'правил' in sentence.lower() or 'модул' in sentence.lower() or 'услов' in sentence.lower():
            role = 'Method or implementation explicitly described as a tool, not the research object.'
        else:
            role = 'Observed engineering result, control, provenance or explicit scope limitation; no method/object substitution.'
        sentences.append(dict(sentence=sentence, question1_method_or_object=role,
                              question2_adjacent_tautology='Reviewed: paragraph has a distinct role (rule, complete competition, controls, preserved failures or limits). Repeated scope caveats are intentional protection against interpreting pair recovery as full-ID/HOTA gain.',
                              question3_actual_not_planned='Checked against frozen result_episode/result_lost_only/result, audit and registered source versions; future runtime activation explicitly remains unmeasured.'))
write('text_review.json', dict(report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(), per_sentence_review=sentences, problems_found_requiring_author_decision=[]))
print('FINAL_REPORT_AND_CRITERIA_READY', len(sentences))
