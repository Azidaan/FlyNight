from pathlib import Path
root=Path(__file__).resolve().parents[1];source=root/'research_bbox_normalized_signals_20261108';target=Path(__file__).parent
for name in ['setup_data.py','inputs.py','features.py','score.py','verify.py']:
    text=(source/name).read_text(encoding='utf-8').replace('bbox_normalized_signals_20261108','bbox_fit_maps_20261108')
    if name=='setup_data.py':text=text.replace("['identity_assignment_20261108','singleton_verify_20261108','native_confirm_policy_20261108']","['bbox_normalized_signals_20261108','identity_assignment_20261108','singleton_verify_20261108','native_confirm_policy_20261108']")
    with (target/name).open('x',encoding='utf-8') as f:f.write(text)
