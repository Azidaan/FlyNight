import numpy as np
from tracking.z13_template_swap import choose_donors, project_models, interpretation
from tracking.z13_ridge_localizer import fit_ridge, ridge_response, map_center
from tracking.z13_readout import HexGeometry


def test_donors_exclude_same_identity_video_geometry_and_repeats():
    r=dict(video='recipient',episode=8,identity='target',mask='same')
    templates=[dict(video=v,episode=i,identity=identity,mask=mask,split='fit20')
               for i,(v,identity,mask) in enumerate([
                   ('recipient','other','same'),('other','target','same'),
                   ('other','a','different'),('other','','same'),
                   ('other','a','same'),('other','a','same'),
                   ('other','b','same'),('other','c','same')])]
    selected=choose_donors(r,templates)
    assert len(selected)==3 and {x['identity'] for x in selected}=={'a','b','c'}
    assert selected==choose_donors(r,list(reversed(templates)))
    assert not choose_donors({**r,'identity':''},templates)
    assert not choose_donors(r,templates[:-1])


def test_omitting_ridge_intercept_preserves_actual_center_and_standardized_score():
    rng=np.random.default_rng(17)
    points=np.array([[x,y] for x in range(5) for y in range(5)],np.float32)
    geometry=HexGeometry(points,5)
    exemplar=rng.normal(size=(25,6));query=rng.normal(size=(25,6))
    model=fit_ridge(exemplar,np.arange(25)%3==0,.1)
    full=ridge_response(model,query)
    projected=project_models(query,model[1][None])[0]
    assert np.ptp(full-projected)<1e-12
    a,sa=map_center(full,geometry);b,sb=map_center(projected,geometry)
    np.testing.assert_allclose(a,b,atol=1e-12,rtol=0)
    assert abs(sa-sb)<1e-12


def test_foreign_filter_recomputes_peak_over_the_entire_search_map():
    features=np.array([[10,0],[0,12],[4,4]],float)
    original=features.copy()
    scores=project_models(features,[[1,0],[0,1]])
    assert np.argmax(scores[0])==0 and np.argmax(scores[1])==1
    np.testing.assert_array_equal(features,original)


def test_interval_with_zero_is_not_proof_of_no_template_use():
    assert interpretation({'ci95':[-.01,.03]})=='inconclusive'
    assert interpretation({'ci95':[-.001,.01]})=='useful_center_advantage_below_2pp_in_this_test'
    assert interpretation({'ci95':[.001,.04]})=='own_template_advantage_detected'
