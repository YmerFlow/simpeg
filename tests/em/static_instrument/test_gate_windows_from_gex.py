"""The GEX gate windows reach the simulation's receivers.

A GEX gate table gives each gate's open and close time beside its center.
``MeasuredTEMXYZSystem`` hands them to the receivers, so the 1D simulation
averages each gate over its window, as Workbench's forward does, unless
``simulation__gate_windows`` is off.
"""

import numpy as np
import pandas as pd

import libaarhusxyz

from simpeg.electromagnetics.utils.static_instrument import dual

WAVE = np.array([[-1e-3, 0.0], [-0.8e-3, 1.0], [0.0, 1.0], [4e-6, 0.0]])  # [s], [A/A]
N_GATES = 6
CENTERS = np.logspace(-5, -3, N_GATES)  # [s]


class FakeGEX:
    """A SkyTEM-like LM/HM GEX with 20%-wide gate windows, or a centers-only table."""

    def __init__(self, columns=3):
        self.General = {"TxLoopArea": 342.0,
                        "RxCoilPosition": np.array([-13.25, 0.0, -2.0]),
                        "WaveformLMPoint": WAVE, "WaveformHMPoint": WAVE,
                        "RxCoilLPFilter": np.array([0.99, 202.3e3])}
        self.gex_dict = {"General": self.General}
        for ch, moment in ((1, "LM"), (2, "HM")):
            self.gex_dict["Channel%d" % ch] = {
                "TransmitterMoment": moment, "ReceiverPolarizationXYZ": "Z",
                "GateFactor": 1.0, "NoGates": N_GATES, "ApproxDipoleMoment": 1.0,
                "TiBLowPassFilter": np.array([1.0, 1e6])}
        self.number_channels = 2
        self.columns = columns

    def gate_times(self, channel):
        table = np.c_[CENTERS, 0.9 * CENTERS, 1.1 * CENTERS]
        return table[:, :self.columns]


def make_xyz():
    xyz = libaarhusxyz.XYZ()
    xyz.flightlines = pd.DataFrame({"x": [0.0], "y": [0.0], "alt": [30.0], "line_no": [0],
                                    "tilt_x": [0.0], "tilt_y": [0.0]})
    xyz.layer_data["resistivity"] = pd.DataFrame(np.full((1, 3), 30.0))
    xyz.layer_data["dep_top"] = pd.DataFrame(np.array([[0.0, 10.0, 50.0]]))
    xyz.model_info["scalefactor"] = 1e-12
    return xyz


def system(gex, **options):
    class System(dual.MeasuredTEMXYZSystem):
        pass
    System.gex = gex
    s = System(make_xyz(), validate=False)
    s.options.update(options)
    return s


def receivers(s):
    sources = s.make_system(0, np.array([0.0, 0.0, 30.0]), s.times)
    return [src.receiver_list[0] for src in sources]


def forward(s):
    out = s.forward(simulation__parallel=False, simulation__n_cpu=1)
    return np.hstack([out.layer_data["dbdt_ch%dgt" % ch].to_numpy(float) for ch in (1, 2)])


def test_receivers_get_the_gex_windows():
    for rx in receivers(system(FakeGEX())):
        np.testing.assert_array_equal(rx.gate_windows, np.c_[0.9 * CENTERS, 1.1 * CENTERS])


def test_option_off_samples_the_centers():
    s = system(FakeGEX(), simulation__gate_windows=False)
    assert all(rx.gate_windows is None for rx in receivers(s))


def test_a_table_without_open_and_close_samples_the_centers():
    assert all(rx.gate_windows is None for rx in receivers(system(FakeGEX(columns=1))))


def test_windows_follow_the_times_filter():
    """A culled gate takes its window with it, so the two stay aligned."""
    keep = np.arange(N_GATES) >= 2

    class Culled(dual.MeasuredTEMXYZSystem):
        @property
        def times_filter(self):
            return [keep, keep]
    Culled.gex = FakeGEX()
    s = Culled(make_xyz(), validate=False)
    for rx in receivers(s):
        np.testing.assert_array_equal(rx.times, CENTERS[keep])
        np.testing.assert_array_equal(rx.gate_windows, np.c_[0.9 * CENTERS, 1.1 * CENTERS][keep])


def test_forward_with_windows_off_is_the_center_forward():
    off = forward(system(FakeGEX(), simulation__gate_windows=False))
    centers_only = forward(system(FakeGEX(columns=1)))
    np.testing.assert_array_equal(off, centers_only)
    on = forward(system(FakeGEX()))
    assert np.abs(on / off - 1).max() > 1e-3
