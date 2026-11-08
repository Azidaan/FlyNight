import numpy as np
import pytest
from tracking.channel_balanced_correspondence import channel_balanced_similarity,centered_contributions,channel_difference_attribution


def test_additive_identity_and_independent_scalar_pearson():
    rng=np.random.default_rng(54);a=rng.normal(size=(7,8));b=rng.normal(size=(7,8))
    q=centered_contributions(a,b);x=a-a.mean(0);y=b-b.mean(0)
    assert abs(q['score']-float(x.ravel()@y.ravel()/np.linalg.norm(x)/np.linalg.norm(y)))<1e-12
    independent=[]
    for c in range(8):
        xc=[float(v-a[:,c].mean()) for v in a[:,c]];yc=[float(v-b[:,c].mean()) for v in b[:,c]]
        independent.append(sum(u*v for u,v in zip(xc,yc))/(sum(u*u for u in xc)*sum(v*v for v in yc))**.5)
    assert np.allclose(q['channel_cosine'],independent,rtol=0,atol=1e-12)
    assert abs(q['score']-q['channel'].sum())<1e-12
    assert abs(q['score']-sum(q['channel_cosine']*q['channel_amplitude_weight']))<1e-12
    assert q['channel_amplitude_weight'].sum()<=1+1e-12
    assert abs(channel_balanced_similarity(a,b)-np.mean(independent))<1e-12
    assert abs(q['balanced_cell'].sum()-q['balanced_score'])<1e-12


def test_channel_gain_offset_invariance_and_equal_energy_limit():
    rng=np.random.default_rng(91);a=rng.normal(size=(11,4));b=rng.normal(size=(11,4))
    score=channel_balanced_similarity(a,b)
    assert abs(score-channel_balanced_similarity(a*[.01,2,50,.3]+[4,1,2,8],b*[8,.02,3,1]+[1,7,2,3]))<1e-12
    x=a-a.mean(0);x/=np.linalg.norm(x,axis=0);y=b-b.mean(0);y/=np.linalg.norm(y,axis=0)
    assert abs(channel_balanced_similarity(x,y)-centered_contributions(x,y)['score'])<1e-12


def test_reference_only_denominator_zero_padding_and_disappearing_channel():
    a=np.array([[-1.,-1.],[0,0],[1,1]]);b=a.copy();b[:,1]=.3
    assert abs(channel_balanced_similarity(a,b)-.5)<1e-12
    assert abs(channel_balanced_similarity(np.pad(a,((0,0),(0,26))),np.pad(b,((0,0),(0,26))))-.5)<1e-12
    assert channel_balanced_similarity(np.full((7,8),.3),np.ones((7,8))) is None
    assert channel_balanced_similarity(a,np.full_like(a,.3)) is None


def test_channel_duplication_permutation_and_invalid_input():
    rng=np.random.default_rng(17);a=rng.normal(size=(9,8));b=rng.normal(size=(9,8));p=rng.permutation(8)
    expected=channel_balanced_similarity(a,b)
    assert abs(channel_balanced_similarity(a[:,p],b[:,p])-expected)<1e-12
    assert abs(channel_balanced_similarity(np.tile(a,2),np.tile(b,2))-expected)<1e-12
    assert abs(channel_balanced_similarity(a,a)-1)<1e-12
    with pytest.raises(ValueError):channel_balanced_similarity(a,b[:3])
    with pytest.raises(ValueError):channel_balanced_similarity(a,np.full_like(a,np.nan))


def test_spatial_and_amplitude_terms_sum_to_the_exact_channel_difference():
    rng=np.random.default_rng(37);patches=[rng.normal(size=(n,8)) for n in [7,7,6,6]]
    own=centered_contributions(*patches[:2]);foreign=centered_contributions(*patches[2:]);d=channel_difference_attribution(own,foreign)
    assert np.allclose(d['total'],foreign['channel']-own['channel'],rtol=0,atol=1e-12)
    assert np.allclose(d['spatial']+d['amplitude'],d['total'],rtol=0,atol=1e-12)
    assert abs(d['total'].sum()-(foreign['score']-own['score']))<1e-12
