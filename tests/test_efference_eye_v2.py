"""CPU checks for the new reference convention and path calibration."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from scripts.fly_efference_v2 import fit_path_matrix, verdict
from tracking.efference_eyes_v2 import ReferencedEfferenceEyeBank


def test_path_calibration_recovers_known_two_axis_transform():
    rng = np.random.default_rng(6)
    truth = rng.normal(size=(24, 12, 2))
    forward = np.array([[.2, .03], [-.05, .6]])
    raw = truth @ forward.T
    matrix, quality = fit_path_matrix(raw, truth)
    np.testing.assert_allclose(raw.sum(1) @ matrix.T, truth.sum(1), atol=1e-12)
    assert quality['rank'] == 2


def test_reference_and_state_use_same_spatial_mapping():
    bank = ReferencedEfferenceEyeBank.__new__(ReferencedEfferenceEyeBank)
    bank.torch = torch
    bank._activity = {1: torch.tensor([1., 2., 4.])}
    bank._reference_activity = {1: torch.tensor([1., 2., 4.])}
    bank._baseline = {}; bank._baseline_flow = {}; bank.decode_flow = True
    bank._groups = [{'nodes': torch.tensor([[0,1,2]])}]
    bank._weights = lambda *a: (np.array([[0,1,2],[0,1,2],[0,1,2]]), np.array([[.5,.5,0.],[0.,.5,.5],[0.,0.,1.]]))
    bank._by_type = lambda a: a.numpy()
    bank._decode = lambda a: torch.stack((a[..., :3], a[..., :3]), dim=-2).mean(1)
    bank._remap(1, (0,0,10,10), (1,0,11,10))
    torch.testing.assert_close(bank._activity[1], torch.tensor([1.5,3.,4.]))
    torch.testing.assert_close(bank._reference_activity[1], bank._activity[1])
    decoded = bank._decode(bank._activity[1][None,None])[0]
    torch.testing.assert_close(decoded-bank._baseline_flow[1], torch.zeros_like(decoded))


def test_gate_rejects_bad_closed_loop_even_if_direction_is_correct():
    a = {'new': dict(T1_drift_share=.05,T1s_drift_share=.05,T2_cos=.95,T2_ratio=1.,T3_cos=.95,T3_ratio=1.,T4_err_px=12.01)}
    assert verdict(a) == {'T1': True,'T2':True,'T3':True,'T4':False}
