"""The transmitter of an offset-receiver system as the GEX's segmented loop.

By default ``MeasuredTEMXYZSystem`` models a SkyTEM-style transmitter as a
point dipole at the frame center. With ``simulation__tx_loop`` it is the
polygon ``TxLoopPoint1..N`` as a line current of unit moment, which is what
Workbench models: 2-3% less at the earliest gates, the same late.
"""

import numpy as np
import pandas as pd
import pytest

import libaarhusxyz

import simpeg.electromagnetics.time_domain as tdem
from simpeg.electromagnetics.utils.static_instrument import dual

WAVE = np.array([[-1e-3, 0.0], [-0.8e-3, 1.0], [0.0, 1.0], [4e-6, 0.0]])  # [s], [A/A]
CORNERS = np.array([[-12.55, -2.10], [-6.03, -8.63], [6.03, -8.63], [11.34, -3.31],
                    [11.34, 3.31], [6.03, 8.63], [-6.03, 8.63], [-12.55, 2.10]])  # [m]
CENTERS = np.logspace(-5, -2.5, 8)  # [s]


class FakeGEX:
    """A SkyTEM-like LM/HM GEX with the SkyTEM frame's loop corners."""

    def __init__(self, corners=CORNERS, rx=(-13.25, 0.0, -2.0)):
        self.General = {"TxLoopArea": 342.0, "RxCoilPosition": np.array(rx),
                        "WaveformLMPoint": WAVE, "WaveformHMPoint": WAVE}
        if corners is not None:
            self.General["TxLoopPoint"] = corners
        self.gex_dict = {"General": self.General}
        for ch, moment in ((1, "LM"), (2, "HM")):
            self.gex_dict["Channel%d" % ch] = {
                "TransmitterMoment": moment, "ReceiverPolarizationXYZ": "Z",
                "GateFactor": 1.0, "NoGates": len(CENTERS), "ApproxDipoleMoment": 1.0}
        self.number_channels = 2

    def gate_times(self, channel):
        return np.c_[CENTERS, CENTERS, CENTERS]


def make_xyz(n=1):
    xyz = libaarhusxyz.XYZ()
    # The first sounding is not the lowest: the fast-coefficient build anchors
    # at the lowest height, which the first sounding's own height must not mask.
    alt = np.array([40.0, 30.0, 35.0])[:n]
    xyz.flightlines = pd.DataFrame({"x": 100.0 * np.arange(n), "y": np.zeros(n),
                                    "alt": alt, "line_no": np.zeros(n),
                                    "tilt_x": np.zeros(n), "tilt_y": np.zeros(n)})
    xyz.layer_data["resistivity"] = pd.DataFrame(np.tile([100.0, 10.0, 300.0], (n, 1)))
    xyz.layer_data["dep_top"] = pd.DataFrame(np.tile([0.0, 10.0, 50.0], (n, 1)))
    xyz.model_info["scalefactor"] = 1e-12
    return xyz


def system(gex=None, n=1, **options):
    class System(dual.MeasuredTEMXYZSystem):
        pass
    System.gex = gex or FakeGEX()
    s = System(make_xyz(n), validate=False)
    s.options.update(dict(simulation__parallel=False, simulation__n_cpu=1), **options)
    return s


def response(s):
    out = s.forward()
    return np.hstack([out.layer_data["dbdt_ch%dgt" % ch].to_numpy(float) for ch in (1, 2)])


def test_loop_path_is_closed_counter_clockwise_and_centered():
    path = system().tx_loop_path(np.array([100.0, 200.0, 30.0]))
    assert path.shape == (len(CORNERS) + 1, 3)
    np.testing.assert_array_equal(path[0], path[-1])
    np.testing.assert_allclose(path[:-1, :2], CORNERS + [100.0, 200.0])
    np.testing.assert_array_equal(path[:, 2], 30.0)


def test_clockwise_corners_are_reversed():
    path = system(FakeGEX(corners=CORNERS[::-1])).tx_loop_path(np.zeros(3))
    np.testing.assert_allclose(path[:-1, :2], CORNERS)


def test_option_picks_the_source():
    location = np.array([0.0, 0.0, 30.0])
    (src_d, _) = system().make_system(0, location, [CENTERS, CENTERS])
    (src_l, _) = system(simulation__tx_loop=True).make_system(0, location, [CENTERS, CENTERS])
    assert isinstance(src_d, tdem.sources.MagDipole)
    assert isinstance(src_l, tdem.sources.LineCurrent)
    assert src_l.current == pytest.approx(1 / 342.0)


def test_central_loop_systems_are_unaffected():
    s = system(FakeGEX(rx=(0.0, 0.0, 0.0)), simulation__tx_loop=True)
    (src, _) = s.make_system(0, np.array([0.0, 0.0, 30.0]), [CENTERS, CENTERS])
    assert isinstance(src, tdem.sources.CircularLoop)


def test_a_gex_without_corners_is_refused():
    with pytest.raises(ValueError, match="TxLoopPoint"):
        response(system(FakeGEX(corners=None), simulation__tx_loop=True))


def test_loop_is_below_the_dipole_early_and_equal_late():
    ratio = response(system(simulation__tx_loop=True)) / response(system())
    early, late = ratio[0, [0, 8]], ratio[0, [7, 15]]
    assert np.all(early < 0.99) and np.all(early > 0.9)
    np.testing.assert_allclose(late, 1.0, atol=5e-3)


def test_stitched_heights_are_the_loop_altitude():
    s = system(n=3, simulation__tx_loop=True)
    sim = s.make_forward()
    np.testing.assert_allclose(sim.source_locations_for_sounding[:, 2], [40.0, 30.0, 35.0])
    np.testing.assert_allclose(sim.h_vector, [40.0, 30.0, 35.0])


def test_fast_coefficients_match_the_stock_build_with_the_loop():
    stock = response(system(n=3, simulation__tx_loop=True))
    fast = response(system(n=3, simulation__tx_loop=True, simulation__fast_coefficients=True))
    np.testing.assert_allclose(fast, stock, rtol=1e-10)


def test_jacobian_with_the_loop_matches_finite_differences():
    s = system(simulation__tx_loop=True)
    sim = s.make_forward()
    m = np.log(1 / s.xyz.resistivity.values).flatten()
    v = np.array([1.0, -0.5, 0.3])
    jv = sim.Jvec(m, v)
    eps = 1e-4
    fd = (s.make_forward().dpred(m + eps * v) - s.make_forward().dpred(m - eps * v)) / (2 * eps)
    np.testing.assert_allclose(jv, fd, rtol=1e-4, atol=1e-6 * np.abs(fd).max())
