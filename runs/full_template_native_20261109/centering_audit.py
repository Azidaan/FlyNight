"""Check that additive attribution does not mistake a constant baseline for signal."""
import sys
from common import *
start('cpu',timeout=600)
import numpy as np
mode=sys.argv[1]
if mode=='register':
    files=[Path(__file__),OUT/'integration_audit.json',OUT/'deployment.json',TRAIN/'features.json',
        ROOT/'tracking/channel_identity_head.py']
    write('centering_protocol.json',dict(registered=datetime.now(MSK).isoformat(),
        reason='Raw additive decomposition found positive channel terms and negative intercept shifts. Such signs are coordinate dependent: c->c-mu,b->b+w*mu leaves every prediction and the penalized loss identical because the intercept is unpenalized. Do not label raw positive terms beneficial or deploy an uncentered additive score.',
        fixed_coordinates='Report both ordinary91training-row mean and the EXACT class-balanced training mean(.5meanown+.5meanforeign). The latter matches the frozen loss weighting; neither usesval labels. Reexpress all213scores in both coordinates, unchanged z. Separate geometry-slope shift from centered intercept and centered channel term. No retraining or threshold test.',
        fixed_interpretive_controls='At the same frozen vectors show G7+uncenteredchannelterm versus G7+TRAINconstantmeanterm and G7+centeredchannelterm. These are algebraic no-FlyVis/global-shift explanations, not deployed methods or evaluated ID improvements. No coordinate chosen by validation outcome.',
        decision='If a raw positive channel contribution disappears/reverses after moving the fixed training baseline into the intercept, do not pursue blind geometry/intercept preservation as an identity fix. First diagnose why own instance has lower-than-background correspondence. Original real pilot stays negative.',
        sources={x.relative_to(ROOT).as_posix():sha(x) for x in files}))
    print('Training-only centered attribution registered',flush=True)
elif mode=='run':
    p=read(OUT/'centering_protocol.json');verify(p)
    model=read(OUT/'deployment.json');cc=np.array(model['models']['C34']['theta']);gc=np.array(model['models']['G7']['theta'])
    rows=read(TRAIN/'features.json')['records'];matrix=np.array([r['variants']['C34']['full']['channels'] for r in rows])
    own=np.array([r['label']=='own' for r in rows]);assert matrix.shape==(91,34) and own.sum()==15
    means={'ordinary_training_mean':matrix.mean(0),'class_balanced_training_mean':.5*(matrix[own].mean(0)+matrix[~own].mean(0))}
    original=read(OUT/'integration_audit.json')['records'];result={};maxerror=0.
    for name,mean in means.items():
        constant=float(mean@cc[7:-1]);bias=float(cc[-1]-gc[-1]);records=[]
        for row in original:
            if row.get('fallback'):continue
            centered_channel=row['channel_term']-constant
            centered_intercept=bias+constant;geometry_slope=row['changed_geometry_and_intercept']-bias
            reconstructed=row['geometry_logit']+geometry_slope+centered_intercept+centered_channel
            maxerror=max(maxerror,abs(reconstructed-row['actual_logit']))
            records.append(dict(video=row['video'],frame=row['frame'],kind=row['kind'],actual_logit=row['actual_logit'],
                geometry_logit=row['geometry_logit'],geometry_slope_change=geometry_slope,
                centered_intercept_change=centered_intercept,centered_channel_term=centered_channel,
                constant_without_signal=row['geometry_logit']+constant,
                frozen_geometry_centered_channel_logit=row['geometry_logit']+centered_channel))
        result[name]=dict(training_channel_mean=mean,training_constant_term=constant,
            centered_intercept_change=bias+constant,records=records)
    assert maxerror<1e-12
    verify(p);write('centering_audit.json',dict(passed=True,coordinates=result,max_reconstruction_error=maxerror,
        original_real_predictions_unchanged=True,no_fits_maps_thresholds_or_rollouts=True,
        protocol_sha256=sha(OUT/'centering_protocol.json'),
        limits='Component signs depend on the declared reference level. No algebraic counterfactual establishes future IDs, solves known foreign confusion or justifies a global logit/cost shift.'))
    print('CENTERED',json.dumps({n:dict(constant=r['training_constant_term'],bias=r['centered_intercept_change'],
        known=[x for x in r['records'] if x['kind']!='unknown']) for n,r in result.items()}),flush=True)
else:raise ValueError(mode)
