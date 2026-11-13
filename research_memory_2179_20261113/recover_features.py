from common import *
failure=read(OUT/'science2_queue_execution.json'); assert not failure['passed'] and failure['jobs'][-1]['entry'].endswith('extract.py')
source=(OUT/'extract.py').read_text(encoding='utf8')
start=source.index('        boxes=plan[')
end=source.index('        receipts.append',start)
replacement='''        boxes=plan['frames'][key]['boxes']; image_tensor,scale=preprocess(image,'cuda'); feature=model.features(image_tensor); parts=[]
        template=old_plan['frames'].get(key,{}).get('boxes',boxes)
        for box in boxes:
            # Match the original ROI-head batch shape and position. For a raw
            # counterpart substitute only its own filtered box in that batch.
            match=box if box in template else None
            if match is None:
                for obs in plan['observations']:
                    if obs['frame']==frame and obs['raw']==box and obs['filtered'] in template:
                        match=obs['filtered']; break
            position=template.index(match) if match is not None else 0
            offset=(position//16)*16; batch=np.asarray(template[offset:offset+16],np.float32).copy(); local=position-offset
            batch[local]=np.asarray(box,np.float32)
            values=torch.nn.functional.normalize(model.embed_boxes(feature,torch.from_numpy(batch).to('cuda'),scale),dim=1)
            a=values[local].cpu().numpy().copy(); parts.append(a)
            if key in old_plan['frames'] and box in old_plan['frames'][key]['boxes']:
                k=old_plan['frames'][key]['boxes'].index(box)
                overlap_error=max(overlap_error,float(abs(a-old_bank[key][k]).max())); overlap_count+=1
        a=np.stack(parts); assert a.shape==(len(boxes),256) and np.isfinite(a).all(); arrays[key]=a
'''
source=source[:start]+replacement+source[end:]
source=source.replace("OUT/'feature_status.json'", "OUT/'feature_status_v2.json'")
with (OUT/'extract_v2.py').open('x',encoding='utf8',newline='\n') as f:f.write(source)
save('feature_recovery_protocol.json',dict(registered=now(),reason='Original head inference used different ROI batch shapes. Overlapping descriptors differed by 0.00010757893323898315, exceeding unchanged 1e-5 reproduction threshold. Preserve failed extraction. Recompute with exact original batch layout and position, substituting only one observed box per head call. No weights, runtime thresholds or acceptance changes.',failure=failure,original_source_sha256=sha(OUT/'extract.py'),recovery_source_sha256=sha(OUT/'extract_v2.py'),new_gpu_job_is_technical_recovery=True,original_overlap_limit=1e-5))
q=read(OUT/'science2_queue.json'); q['jobs']=q['jobs'][1:];q['jobs'][0]['entry']=(OUT/'extract_v2.py').relative_to(ROOT).as_posix();q['inputs'][(OUT/'extract_v2.py').relative_to(ROOT).as_posix()]=sha(OUT/'extract_v2.py');save('science3_queue.json',q)
journal('Воспроизводимость MASA2179: изменение размера пакета рамок дало максимальное расхождение признаков0,00010757893323898315 при допуске0,00001, поэтому результат не принят. Отдельная extract_v2.py воспроизводит размер/позицию исходного пакета и подменяет одну рамку. Допуск не ослаблен, критерии прежние, сбой сохранён. Источник research_memory_2179_20261113/feature_recovery_protocol.json.')
