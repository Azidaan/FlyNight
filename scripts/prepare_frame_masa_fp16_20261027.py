"""Отдельная численная версия после цены FP32, в прежнем абсолютном окне165."""
from pathlib import Path
from datetime import datetime
import sys,json,time,runpy
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.recovery_research_service_20261027 import RUN,read,write,sha,matching,now
from scripts.b123_safe_journal_20261027 import journal

def save(path,text):
    p=ROOT/path;assert not p.exists();p.write_text(text,encoding='utf-8');return p

def main():
    for f in ['frame_masa_dev/recovery1','frame_masa_dev_audit/recovery1']:
        c=read(RUN/f/'controller_result.json');assert c['all_owned_ended'] and not any(matching(k,v) for k,v in c['owned'].items())
    price=read(RUN/'frame_masa_dev/price.json');assert not price['proceed']
    end=datetime.fromisoformat(read(RUN/'frame_masa_dev/execution_deadline.json')['deadline']).timestamp();assert end-time.time()>2400
    save('tracking/frame_masa_fp16_appearance.py',"\"\"\"Отдельный смешанныйFP16 MASA; косинусная стоимость остаётсяFloat32.\"\"\"\nimport numpy as np\nfrom tracking.frame_masa_appearance import FrameMasaServer as Original\nclass FrameMasaServer(Original):\n    def features(self,tokens,null_seed=None):\n        if null_seed is not None:return super().features(tokens,null_seed)\n        import torch\n        with torch.autocast(device_type='cuda',dtype=torch.float16):\n            super().features(tokens,None)\n        for t in tokens:\n            if t.vector.dtype!=np.float32:t.vector=t.vector.astype(np.float32)\n        return np.stack([t.vector for t in tokens])\n")
    profile=(ROOT/'scripts/profile_frame_masa_association_20261027.py').read_text(encoding='utf-8-sig').replace('frame_masa_','frame_masa_fp16_')
    profile=profile.replace('build,load_state_dict,_map_key','build,load_state_dict,_map_key,preprocess').replace('samples=[]','samples=[];numeric_pairs=[]')
    line="            if frame in [0,1,119]:samples.append"
    assert line in profile
    profile=profile.replace(line,"            if frame in [0,119] and current:\n                with torch.inference_mode():\n                    image_tensor,scale=preprocess(image,'cuda');boxes=torch.from_numpy(np.stack([t.crop for t in current[:2]])).to('cuda')\n                    reference=torch.nn.functional.normalize(model.embed_boxes(model.features(image_tensor),boxes,scale),dim=1).cpu().numpy()\n                computed=np.stack([t.vector for t in current[:2]]);cos=(reference*computed).sum(1)/(np.linalg.norm(reference,axis=1)*np.linalg.norm(computed,axis=1))\n                numeric_pairs.append(dict(frame=frame,cosine=cos.tolist(),max_absolute_difference=float(np.max(np.abs(reference-computed)))))\n                assert np.isfinite(cos).all() and float(cos.min())>=.99\n"+line)
    profile=profile.replace('finite=True,batchnorm_unchanged=True','finite=True,numeric_pairs=numeric_pairs,numeric_precision=\'mixedFP16_float32_similarity\',batchnorm_unchanged=True')
    save('scripts/profile_frame_masa_fp16_association_20261027.py',profile)
    for old,new in [('run_frame_masa_association_20261027.py','run_frame_masa_fp16_association_20261027.py'),('audit_frame_masa_association_20261027.py','audit_frame_masa_fp16_association_20261027.py'),('prepare_frame_masa_chain_20261027.py','prepare_frame_masa_fp16_chain_20261027.py')]:
        text=(ROOT/'scripts'/old).read_text(encoding='utf-8-sig').replace('frame_masa_','frame_masa_fp16_')
        text=text.replace('existing_frame_masa_fp16_fp32_eval','existing_frame_masa_mixed_fp16_eval')
        if new.startswith('prepare_'):
            text=text.replace('import sys','import sys,time\nfrom datetime import datetime')
            text=text.replace("    p=dict(parent_experiment=", "    old_end=datetime.fromisoformat(read(RUN/'frame_masa_dev/execution_deadline.json')['deadline']).timestamp()\n    budget=int(old_end-time.time()-960) if dev else 7200\n    assert budget>300\n    p=dict(parent_experiment=")
            text=text.replace('budget_seconds=7200','budget_seconds=budget')
            text=text.replace('165_frame_masa_fp16_dev','190_frame_masa_fp16_dev').replace('170_frame_masa_fp16_dev_audit','195_frame_masa_fp16_dev_audit').replace('175_frame_masa_fp16_validation','200_frame_masa_fp16_validation').replace('180_frame_masa_fp16_validation_audit','205_frame_masa_fp16_validation_audit')
            text=text.replace('FP32,BNзаморожена','смешаннаяFP16,косинусFloat32,BNзаморожена')
        save('scripts/'+new,text)
    header="from pathlib import Path\nimport sys\nsys.path.insert(0,str(Path(__file__).resolve().parents[1]))\n"
    stubs=[]
    for phase in ['dev','validation']:
        stubs.append(save('scripts/frame_masa_fp16_'+phase+'_20261027.py',header+"from scripts.run_frame_masa_fp16_association_20261027 import run\nif __name__=='__main__':run('frame_masa_fp16_"+phase+"')\n"))
        after="\n    from scripts.prepare_frame_masa_fp16_chain_20261027 import prepare_validation\n    prepare_validation()" if phase=='dev' else ''
        stubs.append(save('scripts/frame_masa_fp16_'+phase+'_audit_20261027.py',header+"from scripts.audit_frame_masa_fp16_association_20261027 import audit\nif __name__=='__main__':\n    audit('frame_masa_fp16_"+phase+"')"+after+'\n'))
    for p in stubs:runpy.run_path(str(p),run_name='prelaunch_import_check')
    oldp=read(RUN/'frame_masa_profile/protocol.json');p=dict(oldp);p.update(id='frame_masa_fp16_association_v1',created=now(),budget_seconds=180,
        original_165_end=end,numeric_precision='mixedFP16,similarity_float32',
        action='Отдельная численная версия той же интеграции MASA:автоматическая смешаннаяFP16внутриизвлекателя,готовыевектора/косинуснаястоимостьFloat32. Веса/вход1024/реальныеобласти/правилоназначенияне меняются. '
            'СравнитьпризнакиFP16иFP32насамихтехжеобластяхкадров0/119;предварительныйчисленныйпорогcosine>=.99—толькоисправностьчисленнойверсии,неэквивалентностьHOTA.',
        hypothesis='Смешанная точность может снизить стоимость полного кадра,не потеряв полезность назначений;обе части требуют своих измерений.',
        counterargument='Численныеотклонениямогутменятьназначенияу границ решения;малоеотклонениевекторовнеобещаетсохраненияHOTA. Профиль120кадровне цена35полныхвидео.',
        limitation=oldp['limitation']+' Это отдельнаясмешаннаяFP16версия,непродолжениеFP32и не обещаниечисленнойэквивалентности. Проба180с/dev/аудитпомещаютсявпрежнийабсолютный165срок;старыйFP32ценовойстопсохранён.')
    p['acceptance']=['120кадров,finite/strictpast/BN,всевеса;двапарныхFP32/FP16сравнениятехжеROIсcosine>=.99. Это техническаяприёмка,неHOTA',
        'Полные35dev[0,.1,.3]ипоследующийаудитВНУТРИстарого165абсолютногосрока,никакого нового7200с. ПервоеПОЛНОЕвидео2*time*34+180<остатоккороткогобюджета,иначестоп',
        oldp['acceptance'][2]]
    files=['tracking/frame_masa_fp16_appearance.py','tracking/frame_masa_appearance.py','tracking/masa_r50.py','tracking/lazy_appearance_association.py',
        'scripts/profile_frame_masa_fp16_association_20261027.py','weights/masa/masa_r50.pth',oldp['source']['detections'],oldp['source']['video_path'],
        'runs/recovery_research_20261027/frame_masa_dev/price.json','runs/recovery_research_20261027/frame_masa_profile/report.json']
    p['sha256']={f:sha(ROOT/f) for f in files};folder=RUN/'frame_masa_fp16_profile';write(folder/'protocol.json',p)
    path=ROOT/'experiments/project_history.json';before=path.read_bytes();cat=json.loads(before)
    original=next(c for c in cat['experiments'] if c['id']=='frame_masa_association_v1');assert not original['obtained']
    original.update(state='inconclusive',done='2026-10-27',obtained='runs/recovery_research_20261027/frame_masa_dev/price.json: '+json.dumps(price,ensure_ascii=False)+
        ';одноПОЛНОЕвидеоготово,остальные34не допущены. 166/171завершены,контроллерыframe_masa_dev/recovery1иframe_masa_dev_audit/recovery1сохранены;полного35reportнет.',
        interpretation='FP32полныйкадрMASAвреальныхназначенияхнепрошёлпредварительнуюценув прежнембюджете. Это не отрицательныйнаучныйрезультатHOTAи не бесполезностьMASA.',
        decision='Сохранитьполноепервоевидео/цену/остатокбезповтора. ОтдельнопроверитьсмешаннуюFP16версиювнутрипрежнего165абсолютногосрока;настройкиFP32не менять.')
    original['acceptance'][0].update(met=True,evidence='runs/recovery_research_20261027/frame_masa_profile/report.json:passed=true')
    original['acceptance'][1].update(met=False,evidence='runs/recovery_research_20261027/frame_masa_dev/price.json:proceed=false')
    assert not any(c['id']==p['id'] for c in cat['experiments'])
    cat['experiments'].append(dict(id=p['id'],title='Проверить MASA в назначениях со смешанной числовой точностью',area='flyvis',state='ready',started='2026-10-27',
        question='Умещается ли отдельная численная версия полной системы в измеренный бюджет и помогает ли HOTA?',hypothesis=p['hypothesis'],expected=p['action'],summary=p['counterargument'],
        obtained='',interpretation='',decision='',cannot_claim=p['limitation'],acceptance=[dict(criterion=c,met=None) for c in p['acceptance']],metrics=[],commits=[],files=files[:5],
        evidence=[dict(kind='file',ref=folder.relative_to(ROOT).as_posix()+'/protocol.json',claim='Численнаяверсияи критериидоCUDAпробы'),dict(kind='file',ref='runs/recovery_research_20261027/frame_masa_dev/price.json',claim='ПрежнийценовойстопFP32не скрыт')]))
    assert path.read_bytes()==before;write(path,cat)
    schedule=RUN/'frame_masa_fp16_schedule';write(schedule/'protocol.json',dict(budget_seconds=120,original_165_end=end,parent_experiment=p['id'],author_stop=p['author_stop']))
    for name,sub,script in [('185_frame_masa_fp16_profile','frame_masa_fp16_profile','scripts/profile_frame_masa_fp16_association_20261027.py'),('187_frame_masa_fp16_schedule','frame_masa_fp16_schedule','scripts/prepare_frame_masa_fp16_chain_20261027.py')]:
        write(RUN/(name+'_prepared.json'),dict(name=name,folder=(RUN/sub).relative_to(ROOT).as_posix(),script=script,script_sha256=sha(ROOT/script),protocol_sha256=sha(RUN/sub/'protocol.json')))
    journal(p['id'],'FP32остановленпостоимости,поставленаотдельнаясмешаннаяточность',
        '*ПолученоFP32.* '+original['obtained']+'\n\n*Вывод.* '+original['interpretation']+'\n\n*Новаяпостановка.* '+p['action']+' '+p['counterargument']+
        ' ОфициальныеMASAdemo/video_demo_with_text.py(--fp16)иmasa/apis/masa_inference.py(autocast)проверенычерезпервичныйрепозиторийhttps://github.com/siyuanliii/masa;'
        'нашлокальныйtorchvisionпутьпроверяетсясамостоятельно,основноеокружениене меняется. Четыреобёрткиновойверсииполностьюимпортированыдоочереди,безвызоваCUDA. '
        'НовыйGPUпрофиль180сиполнаяdev/аудитвнутристарого165абсолютногосрока21:56:02UTC,не новая7200ссессия. '+p['limitation']+' Публикацияпослезелёныхпровероккаталога.')
    print('Recorded FP32 cost stop, prepared distinct FP16 profile and full transitions')

if __name__=='__main__':main()
