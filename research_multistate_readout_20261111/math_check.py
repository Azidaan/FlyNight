"""Synthetic algebra checks before real-data recovery; no fitting."""
from common import *
start()
import numpy as np

if (OUT / 'math_checks.json').exists():
    assert read(OUT / 'math_checks.json')['passed']
    print('Ready mathematical result reused')
    raise SystemExit(0)
rng = np.random.default_rng(20261111)
a, b = rng.normal(size=(2, 40, 34))
v = correlations(a, b)
duplicate = (v + v) / 2
np.testing.assert_array_equal(v, duplicate)
# A common-shift sum must not combine incompatible per-state optima.
s1 = np.array([1., -1.]); s2 = np.array([-1., 1.])
assert max((s1+s2)/2) == 0 and (max(s1)+max(s2))/2 == 1
scores = rng.uniform(-1, 1, (2, 19))
assert np.max(scores.mean(0)) <= np.max(scores, axis=1).mean() + 1e-15
# Additional state can contain complementary spatial evidence even with
# identical last-state evidence: identity/anti-identity at the first state.
own = (correlations(a, a) + correlations(a, b)) / 2
foreign = (correlations(a, -a) + correlations(a, b)) / 2
np.testing.assert_allclose(own - foreign, 1., atol=1e-14)
# Interpolation of ordinary EMA/transient maps is linear and has the same
# 34 channels; it is not presented as a second Euler update of the old EMA.
h0, h1 = rng.normal(size=(2, 40))
e0, e1 = rng.normal(size=(2, 40, 17))
r0 = np.c_[e0, h0[:, None]-e0]
r1 = np.c_[e1, h1[:, None]-e1]
em = (e0+e1)/2; hm = (h0+h1)/2
np.testing.assert_allclose((r0+r1)/2, np.c_[em, hm[:, None]-em], atol=1e-14)
# Freezing the old head means equal output dimension and no added parameters.
theta = rng.normal(size=42); geom = rng.normal(size=7)
z = np.r_[geom, v, 1.] @ theta
np.testing.assert_allclose(z, np.r_[geom, duplicate, 1.] @ theta, atol=1e-14)
write('math_checks.json', dict(passed=True, duplicate_identity=True, common_shift_inequality=True,
    complementary_evidence_constructed=True, ordinary_interpolation_identity=True,
    output_channels=34, new_trainable_parameters=0,
    scope='Two numerical integration states, not graph-depth JK. Equal evidence fusion; frozen previously trained heads. No empirical advantage implied.'))
print('MATH_PASS')
