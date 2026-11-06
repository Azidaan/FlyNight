"""Retrospective engineering check; never select a new runtime calibration."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[2]


def main():
    source=ROOT/'runs/flyvis/motion_calibration.json'
    raw=source.read_bytes(); data=json.loads(raw)
    results=[]
    for fit in data['fits']:
        rows=[r for r in data['records'] if r['aggregation']==fit['aggregation'] and
              r['temporal_resampling']==fit['temporal_resampling']]
        old=np.asarray([r['predicted_xy'] for r in rows],np.float64)
        target=np.asarray([r['true_xy'] for r in rows],np.float64)
        gain=fit['fitted_gain']
        # Full-frame test source is 1280 x 720; both retinal dimensions are 391.
        # Positive per-axis scaling commutes with the recorded component median.
        changed=old*np.asarray([1280/720,1.])
        refit=float(np.sum(changed*target)/np.sum(changed*changed))
        rmse=lambda value:float(np.sqrt(np.mean((value-target)**2)))
        original_error=rmse(old*gain)
        assert np.isclose(original_error,fit['rmse_pixels_after_gain'],rtol=0,atol=1e-12)
        results.append(dict(resampling=fit['temporal_resampling'],aggregation=fit['aggregation'],
            samples=len(rows),old_gain=gain,old_rmse=original_error,
            resized_axes_old_gain_rmse=rmse(changed*gain),
            diagnostic_refitted_gain=refit,diagnostic_refitted_rmse=rmse(changed*refit),
            coordinate_only_vectors_and_rmse_unchanged=True))
    assert source.read_bytes()==raw
    report=dict(source=str(source.relative_to(ROOT)),source_sha256=hashlib.sha256(raw).hexdigest(),
        input_size=[1280,720],retinal_size=[391,391],results=results,
        scope='Recompute existing full-frame synthetic calibration records; no new neural inference, gain adoption, training or HOTA',
        finding='Blind transfer of legacy scalar gains to resize_v2 is unsafe; several modes worsen even after scalar refit',
        limitation='Correct coordinate-center remapping is independent of this vector scaling. Exact calibration equality does not prove ROI or tracker-quality equality.',
        decision='Do not publish automatic anisotropic vector rescaling with old calibration. Keep the counterevidence and separate proven coordinate mapping from an unvalidated learned-output scaling hypothesis.')
    out=ROOT/'runs/flyvis_repairs_20261106/geometry_calibration_counterevidence.json'
    if out.exists():raise FileExistsError(out)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
