"""An inverted model carries its own Christiansen-Auken depth of investigation.

``make_inversion_outputs`` writes ``doi_upper`` / ``doi_lower`` [m below the surface] onto
the model, and the cumulative sensitivity as a layer table, from one Jacobian at the model.
The inversion itself is not run: a stand-in ``inv`` holds a real misfit (survey, simulation,
data, weights from the system) and a chosen model, which is all the outputs read. The
middle sounding has no data, so the system filters it out and its DOI must come back NaN.
"""

import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import libaarhusxyz

from simpeg.electromagnetics.utils.doi_utils import doi_christiansen_auken
from simpeg.electromagnetics.utils.static_instrument import base, dual


N_SOUNDINGS, N_GATES = 3, 8
THICKNESSES = 2.0 * 1.4 ** np.arange(11)          # 12 layers [m]
N_LAYER = len(THICKNESSES) + 1
KEEP = np.array([True, False, True])              # sounding 1 has no data


class FakeGEX:
    def __init__(self):
        two_point = np.array([[-1e-3, 1.0], [0.0, 0.0]])
        self.General = {"TxLoopArea": 342.0, "RxCoilPosition": np.array([-13.25, 0.0, -2.0]),
                        "WaveformLMPoint": two_point, "WaveformHMPoint": two_point}
        self.gex_dict = {"General": self.General}
        for ch, (moment, dipole) in {1: ("LM", 2736.0), 2: ("HM", 451440.0)}.items():
            self.gex_dict["Channel%d" % ch] = {
                "TransmitterMoment": moment, "ReceiverPolarizationXYZ": "Z",
                "ApproxDipoleMoment": dipole, "GateFactor": 1.0, "NoGates": N_GATES}

    def gate_times(self, channel):
        return np.column_stack([np.logspace(-5, -2.5, N_GATES)] * 3)


def make_xyz():
    xyz = libaarhusxyz.XYZ()
    xyz.flightlines = pd.DataFrame({
        "x": 20.0 * np.arange(N_SOUNDINGS), "y": np.zeros(N_SOUNDINGS),
        "alt": np.full(N_SOUNDINGS, 30.0), "line_no": np.zeros(N_SOUNDINGS, dtype=int),
        "tilt_x": np.zeros(N_SOUNDINGS), "tilt_y": np.zeros(N_SOUNDINGS)})
    data = np.full((N_SOUNDINGS, N_GATES), -1e-9)
    data[~KEEP] = np.nan
    for ch in (1, 2):
        xyz.layer_data["dbdt_ch%dgt" % ch] = pd.DataFrame(data.copy())
        xyz.layer_data["dbdt_std_ch%dgt" % ch] = pd.DataFrame(np.full((N_SOUNDINGS, N_GATES), 0.05))
    return xyz


def system(**opts):
    class System(dual.DualMomentTEMXYZSystem):
        pass
    System.gex = FakeGEX()
    s = System(make_xyz(), validate=False, simulation__parallel=False, simulation__n_cpu=1, **opts)
    s.options.update(gate_filter__start_lm=0, gate_filter__end_lm=N_GATES,
                     gate_filter__start_hm=0, gate_filter__end_hm=N_GATES)
    return s


def model_of(res_layer):
    """Log conductivity [ln S/m] of the inverted soundings: 100 ohm.m with a res_layer ohm.m layer at 20-50 m."""
    top = np.r_[0.0, np.cumsum(THICKNESSES)]
    res = np.where((top >= 20) & (top < 50), res_layer, 100.0)
    return np.tile(np.log(1 / res), KEEP.sum())


def run_outputs(irls=False, **opts):
    """make_inversion_outputs on a stand-in inversion; returns (system, misfit, final, smooth model)."""
    s = system(**opts)
    dmis = s.make_misfit(THICKNESSES)
    final, smooth = model_of(10.0), model_of(30.0)
    inv_prob = SimpleNamespace(dmisfit=dmis, model=final, dpred=dmis.simulation.dpred(final))
    if irls:
        inv_prob.l2model, inv_prob.l2dpred = smooth, dmis.simulation.dpred(smooth)
    s.inv = SimpleNamespace(invProb=inv_prob)
    s.make_inversion_outputs()
    return s, dmis, final, smooth


def direct(s, dmis, model, weighted=False):
    """The DOI by a direct call, with the relative uncertainty from the system's own arrays."""
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = s.uncert_array / np.abs(s.data_array_nan)
    rel[~np.isfinite(rel)] = np.inf
    doi, S, _ = doi_christiansen_auken(dmis.simulation, model, rel, weighted=weighted)
    return doi, S


def assert_carries(model_xyz, doi, S):
    fl = model_xyz.flightlines
    np.testing.assert_allclose(fl["doi_upper"].values[KEEP], doi[:, 0])
    np.testing.assert_allclose(fl["doi_lower"].values[KEEP], doi[:, 1])
    assert fl.loc[~KEEP, ["doi_upper", "doi_lower"]].isna().all().all()
    table = model_xyz.layer_data["cumulative_sensitivity"].values.astype(float)
    assert table.shape == (N_SOUNDINGS, N_LAYER)
    np.testing.assert_allclose(table[KEEP], S)
    assert np.isnan(table[~KEEP]).all()


# ── The model carries its DOI, unfiltered back to every sounding ─────────────

def test_model_carries_the_doi_of_a_direct_call():
    s, dmis, final, _ = run_outputs()
    doi, S = direct(s, dmis, final)
    assert_carries(s.l2, doi, S)
    assert s.sparse is None
    upper, lower = doi.T
    assert ((0 < upper) & (upper < lower)).all()


def test_weighted_uses_the_inversions_own_uncertainties():
    s, dmis, final, _ = run_outputs(doi__weighted=True)
    doi, S = direct(s, dmis, final, weighted=True)
    assert_carries(s.l2, doi, S)


def test_thresholds_are_parameters():
    s, dmis, final, _ = run_outputs(doi__upper=3.0, doi__lower=0.3)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = s.uncert_array / np.abs(s.data_array_nan)
    rel[~np.isfinite(rel)] = np.inf
    doi, _, _ = doi_christiansen_auken(dmis.simulation, final, rel, thresholds=(3.0, 0.3))
    np.testing.assert_allclose(s.l2.flightlines["doi_upper"].values[KEEP], doi[:, 0])
    np.testing.assert_allclose(s.l2.flightlines["doi_lower"].values[KEEP], doi[:, 1])


def test_smooth_and_sparse_models_each_carry_their_own():
    s, dmis, final, smooth = run_outputs(irls=True)
    assert_carries(s.sparse, *direct(s, dmis, final))
    assert_carries(s.l2, *direct(s, dmis, smooth))
    assert not np.allclose(s.l2.flightlines["doi_lower"].values[KEEP],
                           s.sparse.flightlines["doi_lower"].values[KEEP])


# ── Switches, and a DOI that fails ───────────────────────────────────────────

def test_disabled_writes_nothing():
    s, *_ = run_outputs(doi__enable=False)
    assert "doi_upper" not in s.l2.flightlines and "doi_lower" not in s.l2.flightlines
    assert "cumulative_sensitivity" not in s.l2.layer_data


def test_sensitivity_table_is_optional():
    s, *_ = run_outputs(doi__write_sensitivity=False)
    assert s.l2.flightlines["doi_upper"].notna().sum() == KEEP.sum()
    assert "cumulative_sensitivity" not in s.l2.layer_data


def test_a_failing_doi_warns_and_the_model_survives(monkeypatch):
    def broken(*args, **kw):
        raise RuntimeError("out of memory")
    monkeypatch.setattr(base, "doi_christiansen_auken", broken)
    with pytest.warns(UserWarning, match="DOI not written: out of memory"):
        s, *_ = run_outputs()
    assert "resistivity" in s.l2.layer_data
    assert "doi_upper" not in s.l2.flightlines
