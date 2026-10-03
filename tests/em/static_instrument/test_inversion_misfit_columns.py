"""An inverted model carries its own misfit, and never someone else's.

``smooth_model`` used to leave ``resdata`` / ``restotal`` / ``numdata`` empty - or,
worse, carry them through from the input file, where Aarhus Workbench writes its
own values under exactly those names. A model from here then showed the
contractor's fit as its own. Meanwhile the synthetic computed ``resdata`` into a
frame it *shared* with the input, writing into the caller's data.

Now: both outputs are built from a copy; an input's misfit columns are renamed
``input_<name>``; the synthetic computes all three; the model gets the same
three copied on. Regression tests for YmerFlow/Ymerflow#35.

The inversion itself is not run. ``make_inversion_outputs`` reads a handful of
attributes off ``self.inv``; a stand-in with those attributes, holding a known
predicted-data vector, is enough to check the bookkeeping exactly.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import libaarhusxyz

from SimPEG.electromagnetics.utils.static_instrument import dual


N_SOUNDINGS, N_GATES = 3, 4


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
        return np.column_stack([np.logspace(-5, -3, N_GATES)] * 3)


def make_xyz(with_input_misfit):
    xyz = libaarhusxyz.XYZ()
    fl = {"x": np.arange(N_SOUNDINGS, dtype=float), "y": np.zeros(N_SOUNDINGS),
          "alt": np.full(N_SOUNDINGS, 30.0), "line_no": np.zeros(N_SOUNDINGS, dtype=int),
          "tilt_x": np.zeros(N_SOUNDINGS), "tilt_y": np.zeros(N_SOUNDINGS)}
    if with_input_misfit:
        # What a Workbench dat export carries: the contractor's own fit.
        fl["resdata"] = np.full(N_SOUNDINGS, 0.66)
        fl["restotal"] = np.full(N_SOUNDINGS, 0.40)
        fl["numdata"] = np.full(N_SOUNDINGS, 51)
    xyz.flightlines = pd.DataFrame(fl)
    for ch in (1, 2):
        xyz.layer_data["dbdt_ch%dgt" % ch] = pd.DataFrame(np.full((N_SOUNDINGS, N_GATES), 1e-9))
        xyz.layer_data["dbdt_std_ch%dgt" % ch] = pd.DataFrame(np.full((N_SOUNDINGS, N_GATES), 0.05))
    return xyz


def system(with_input_misfit=False):
    class System(dual.DualMomentTEMXYZSystem):
        pass
    System.gex = FakeGEX()
    s = System(make_xyz(with_input_misfit), validate=False)
    s.options.update(gate_filter__start_lm=0, gate_filter__end_lm=N_GATES,
                     gate_filter__start_hm=0, gate_filter__end_hm=N_GATES)
    return s


def fake_inversion(s, residual_per_sounding):
    """A stand-in for ``s.inv`` after a run.

    Every datum of sounding i is predicted off by ``residual_per_sounding[i]``
    normalized units, so the per-sounding RMS is exactly that number and the
    total RMS is the RMS of the vector.
    """
    n_data = N_SOUNDINGS * 2 * N_GATES
    dobs = np.full(n_data, 1e-9)
    std = 0.05 * dobs
    W = 1.0 / std
    per_datum = np.repeat(residual_per_sounding, 2 * N_GATES)   # data are ordered sounding-major
    dpred = dobs - per_datum / W
    thicknesses = np.array([10.0, 30.0])
    n_layers = len(thicknesses) + 1
    model = np.log(1 / 50.0) * np.ones(N_SOUNDINGS * n_layers)
    dmisfit = SimpleNamespace(data=SimpleNamespace(dobs=dobs),
                              W=SimpleNamespace(diagonal=lambda: W),
                              simulation=SimpleNamespace(thicknesses=thicknesses))
    return SimpleNamespace(invProb=SimpleNamespace(dmisfit=dmisfit, model=model, dpred=dpred))


def run_outputs(with_input_misfit=False, residual=(0.5, 1.0, 2.0)):
    s = system(with_input_misfit)
    s.inv = fake_inversion(s, np.asarray(residual, dtype=float))
    s.make_inversion_outputs()
    return s


# ── The synthetic computes the three Workbench misfit columns ─────────────────

def test_synthetic_carries_per_sounding_rms_and_count_and_total():
    s = run_outputs(residual=(0.5, 1.0, 2.0))
    fl = s.l2pred.flightlines
    np.testing.assert_allclose(fl["resdata"].values, [0.5, 1.0, 2.0])
    assert fl["numdata"].tolist() == [2 * N_GATES] * N_SOUNDINGS
    expected_total = np.sqrt(np.mean(np.repeat([0.5, 1.0, 2.0], 2 * N_GATES) ** 2))
    np.testing.assert_allclose(fl["restotal"].values, expected_total)


# ── The model carries the same three ─────────────────────────────────────────

def test_model_carries_the_misfit_of_the_data_it_was_fit_to():
    s = run_outputs(residual=(0.5, 1.0, 2.0))
    for col in ("resdata", "restotal", "numdata"):
        np.testing.assert_array_equal(s.l2.flightlines[col].values, s.l2pred.flightlines[col].values)
    assert not s.l2.flightlines["resdata"].isna().any()


# ── An input's own misfit columns never masquerade as the inversion's ──────────

def test_input_misfit_columns_are_renamed_not_carried():
    s = run_outputs(with_input_misfit=True, residual=(0.5, 1.0, 2.0))
    for out in (s.l2, s.l2pred):
        fl = out.flightlines
        np.testing.assert_allclose(fl["input_resdata"].values, 0.66)
        np.testing.assert_allclose(fl["input_restotal"].values, 0.40)
        assert fl["input_numdata"].tolist() == [51] * N_SOUNDINGS
        np.testing.assert_allclose(fl["resdata"].values, [0.5, 1.0, 2.0])   # ours, not theirs


def test_the_input_is_not_written_to():
    """Two outputs built from the input's frame used to share it - and overwrite it."""
    s = run_outputs(with_input_misfit=True)
    fl_in = s._xyz.flightlines
    assert "input_resdata" not in fl_in.columns
    np.testing.assert_allclose(fl_in["resdata"].values, 0.66)
    assert "numdata" in fl_in.columns and fl_in["numdata"].tolist() == [51] * N_SOUNDINGS


def test_no_input_misfit_means_no_input_columns():
    s = run_outputs(with_input_misfit=False)
    assert not any(c.startswith("input_") for c in s.l2.flightlines.columns)
