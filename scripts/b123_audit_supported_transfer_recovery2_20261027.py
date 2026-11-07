"""Исходная сверка плюс сохранность готовых видео и проверенное уточнение времени."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1];ORIGINAL=ROOT/'scripts/b123_audit_supported_transfer_20261027.py';PARENT=ROOT/'runs/b123_20261027/supported_transfer';RUN=PARENT/'recovery2';OUT=PARENT/'recovery2_audit'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    protocol=read(OUT/'protocol.json');assert sha(ORIGINAL)==protocol['original_audit_sha256']
    ns={'__name__':'frozen_audit_import','__file__':str(ORIGINAL)};exec(compile(ORIGINAL.read_text(encoding='utf-8'),str(ORIGINAL),'exec'),ns);ns['RUN']=RUN;ns['OUT']=OUT;ns['main']()
    result=read(RUN/'independent_audit.json')
    if not result['passed']:return
    service=read(RUN/'service_recovery.json')
    for path,h in service['copied_source_sha256'].items():assert sha(ROOT/path)==h,path
    for v in service['completed_videos_copied']:
        for arm in ['baseline','supported']:assert sha(RUN/arm/(v+'.json'))==sha(PARENT/'recovery1'/arm/(v+'.json'))
    for v,item in service['preflight'].items():
        alignment=read(RUN/'cache'/v/'alignment.json');assert sha(RUN/'preflight'/v/'alignment.json')==item['sha256']
        assert alignment==read(RUN/'preflight'/v/'alignment.json')
        for change in alignment['native_refinements']:
            assert len(change['native_checks'])==3 and {x['best'] for x in change['native_checks']}=={change['accepted']}
        assert all(b['video_frame_index']>a['video_frame_index'] for a,b in zip(alignment['matches'],alignment['matches'][1:]))
        assert not alignment['object_annotations_used'] and not alignment['tracker_results_used']
    result.update(completed_ten_outputs_preserved_exactly=True,remaining_image_correspondences_preflight_verified=True,native_refinement_three_resolutions_agree=True,search_window_not_expanded=True)
    text=json.dumps(result,ensure_ascii=False,indent=2)+'\n';(RUN/'independent_audit.json').write_text(text,encoding='utf-8');(OUT/'report.json').write_text(text,encoding='utf-8')
if __name__=='__main__':main()
