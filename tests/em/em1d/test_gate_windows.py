"""Gate values averaged over each gate's open-close window in the 1D TEM simulation.

AarhusInv averages a gate over its window when the open and close times are
given, and Aarhus Workbench's forward does. Sampling the gate center instead
leaves the response 1-1.5% low at late SkyTEM gates and up to 6-7% at the
earliest kept ones. With ``gate_windows`` on a receiver, the simulation builds
the time matrix at Gauss-Legendre nodes inside each window and averages the
rows back to one per gate, so forward and Jacobian cost the same as before.
"""

import numpy as np
import pytest
from scipy.integrate import simpson

from simpeg import maps
from simpeg.electromagnetics import time_domain as tdem

H = 30.0  # transmitter altitude [m]
RX = np.array([[-13.25, 0.0, H + 2.0]])  # SkyTEM-like offset receiver [m]
CENTERS = np.logspace(np.log10(10e-6), np.log10(1e-3), 12)  # gate centers [s]
WINDOWS = np.c_[0.9 * CENTERS, 1.1 * CENTERS]  # 20% wide, like SkyTEM gates [s]
WAVE = np.array([[-1e-3, 0.0], [-0.8e-3, 1.0], [0.0, 1.0], [4e-6, 0.0]])  # [s], [A/A]
THICKNESSES = np.array([10.0, 40.0])  # [m]
RHO = np.array([100.0, 10.0, 300.0])  # [ohm.m]


def simulate(times, windows=None, nodes=5, rho=RHO):
    """dB/dt [T/s per unit moment] at ``times`` [s], with coil and board filters."""
    rx = tdem.receivers.PointMagneticFluxTimeDerivative(
        RX, times, "z", lp_cutoff_frequency_1=200e3, lp_power_1=0.99,
        lp_cutoff_frequency_2=1e6, lp_power_2=1, gate_windows=windows,
        gate_window_nodes=nodes)
    src = tdem.sources.MagDipole(
        [rx], location=np.r_[0.0, 0.0, H], orientation="z",
        waveform=tdem.sources.PiecewiseLinearWaveform(WAVE[:, 0], WAVE[:, 1]))
    sim = tdem.Simulation1DLayered(
        survey=tdem.Survey([src]), thicknesses=THICKNESSES,
        sigmaMap=maps.ExpMap(nP=len(RHO)))
    return sim, np.log(1 / rho)


def test_zero_width_windows_are_center_samples():
    sim_c, m = simulate(CENTERS)
    sim_w, _ = simulate(CENTERS, np.c_[CENTERS, CENTERS])
    np.testing.assert_allclose(sim_w.dpred(m), sim_c.dpred(m), rtol=1e-12)


def test_window_average_matches_a_fine_simpson_rule():
    fine = np.concatenate([np.linspace(a, b, 33) for a, b in WINDOWS])
    sim_f, m = simulate(fine)
    width = WINDOWS[:, 1] - WINDOWS[:, 0]
    reference = simpson(sim_f.dpred(m).reshape(len(CENTERS), 33),
                        x=fine.reshape(len(CENTERS), 33), axis=1) / width
    sim_w, _ = simulate(CENTERS, WINDOWS)
    np.testing.assert_allclose(sim_w.dpred(m), reference, rtol=1e-5)
    # The center sample is measurably off at late gates, so the check above
    # is not one that center sampling would also pass.
    sim_c, _ = simulate(CENTERS)
    assert np.abs(sim_c.dpred(m)[-4:] / reference[-4:] - 1).min() > 5e-3


def test_one_node_samples_the_window_midpoint():
    sim_w, m = simulate(CENTERS, np.c_[0.8 * CENTERS, 1.4 * CENTERS], nodes=1)
    sim_c, _ = simulate(1.1 * CENTERS)
    np.testing.assert_allclose(sim_w.dpred(m), sim_c.dpred(m), rtol=1e-12)


def test_jacobian_with_windows_matches_finite_differences():
    sim, m = simulate(CENTERS, WINDOWS)
    v = np.array([1.0, -0.5, 0.3])
    jv = sim.Jvec(m, v)
    eps = 1e-4
    fd = (simulate(CENTERS, WINDOWS)[0].dpred(m + eps * v)
          - simulate(CENTERS, WINDOWS)[0].dpred(m - eps * v)) / (2 * eps)
    np.testing.assert_allclose(jv, fd, rtol=1e-4, atol=1e-6 * np.abs(fd).max())


@pytest.mark.parametrize("windows, match", [
    (WINDOWS[:-1], r"must be \(12, 2\)"),
    (np.c_[1.1 * CENTERS, 0.9 * CENTERS], "close at or after"),
])
def test_bad_windows_are_refused(windows, match):
    with pytest.raises(ValueError, match=match):
        tdem.receivers.PointMagneticFluxTimeDerivative(RX, CENTERS, "z", gate_windows=windows)


def test_zero_nodes_are_refused():
    with pytest.raises(ValueError):
        tdem.receivers.PointMagneticFluxTimeDerivative(RX, CENTERS, "z", gate_window_nodes=0)
