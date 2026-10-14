"""The receiver low-pass filters a GEX declares reach the simulation's receivers.

A GEX can carry up to two first-order low-pass filters, the coil filter in
``General`` and a per-channel board filter, and the simulation has the
receiver properties to apply them. Nothing connected the two, so every forward
ran unfiltered: 20% low at the earliest SkyTEM LM gates against Workbench's
forward of the same model (YmerFlow/Ymerflow#94). The wiring lives in
``MeasuredTEMXYZSystem``, so single-moment central-loop systems and the
dual-moment subclass both get it.

Everything here is built in memory around the smallest GEX that lets
``make_system`` run: a two-point waveform, a coil position, and the channel
blocks the moment resolution needs. Channels are addressed by their GEX channel
number (the N-moment API), not by a contiguous moment index.
"""

import numpy as np
import pandas as pd
import pytest

import libaarhusxyz

from simpeg.electromagnetics.utils.static_instrument import dual

TWO_POINT = np.array([[-1e-3, 1.0], [0.0, 0.0]])
N_GATES = 4


class FakeGEX:
    """A minimal GEX: one central-loop channel, or SkyTEM-style LM and HM."""

    def __init__(self, coil=None, board=None, dual_moment=False):
        if dual_moment:
            self.General = {"TxLoopArea": 342.0,
                            "RxCoilPosition": np.array([-13.25, 0.0, -2.0]),
                            "WaveformLMPoint": TWO_POINT,
                            "WaveformHMPoint": TWO_POINT}
            channels = {1: {"TransmitterMoment": "LM", "ApproxDipoleMoment": 2736.0},
                        2: {"TransmitterMoment": "HM", "ApproxDipoleMoment": 451440.0}}
        else:
            self.General = {"TxLoopArea": 229.66,
                            "RxCoilPosition": np.array([0.0, 0.0, 0.0]),
                            "WaveformPoint": TWO_POINT}
            channels = {1: {"ApproxDipoleMoment": 88000.0}}
        if coil is not None:
            self.General["RxCoilLPFilter"] = coil
        self.gex_dict = {"General": self.General}
        for ch, block in channels.items():
            block.update({"ReceiverPolarizationXYZ": "Z", "GateFactor": 1.0,
                          "NoGates": N_GATES})
            if board is not None:
                block["TiBLowPassFilter"] = board
            self.gex_dict["Channel%d" % ch] = block
        self.number_channels = len(channels)

    def gate_times(self, channel):
        return np.column_stack([np.logspace(-5, -3, N_GATES)] * 3)


def make_xyz(n_moments):
    xyz = libaarhusxyz.XYZ()
    xyz.flightlines = pd.DataFrame({"x": [0.0], "y": [0.0], "alt": [30.0], "line_no": [0],
                                    "tilt_x": [0.0], "tilt_y": [0.0]})
    for ch in range(1, n_moments + 1):
        for name in ("dbdt_ch%dgt", "dbdt_std_ch%dgt"):
            xyz.layer_data[name % ch] = pd.DataFrame(np.full((1, N_GATES), 1e-9))
    # A 50 ohm.m model, so the same dataset can be forward-modelled. Its layer
    # count matches the gate count so libaarhusxyz does not pad the model
    # columns up to the (wider) data columns with a trailing NaN layer, which
    # would desync ``n_layer_used`` (read off the resistivity) from the mesh
    # thicknesses (read off dep_top).
    xyz.layer_data["resistivity"] = pd.DataFrame(np.full((1, N_GATES), 50.0))
    xyz.layer_data["dep_top"] = pd.DataFrame(
        np.cumsum([[0.0]] + [[10.0 * 2 ** i] for i in range(N_GATES - 1)]).reshape(1, N_GATES))
    return xyz


def system(gex, **options):
    base_class = dual.DualMomentTEMXYZSystem if gex.number_channels == 2 else dual.MeasuredTEMXYZSystem

    class System(base_class):
        pass
    System.gex = gex
    s = System(make_xyz(gex.number_channels), validate=False)
    s.options.update(options)
    return s


def receivers(s):
    """The receiver of each channel of one sounding, in channel order."""
    sources = s.make_system(0, np.array([0.0, 0.0, 30.0]), s.times_full)
    return [src.receiver_list[0] for src in sources]


# ── Reading the GEX ──────────────────────────────────────────────────────────

def test_reads_coil_then_board_filter():
    s = system(FakeGEX(coil=np.array([0.99, 210e3]), board=np.array([1.0, 300e3])))
    assert s.receiver_filters(1) == [(0.99, 210e3), (1.0, 300e3)]


def test_dual_moment_reads_each_channel():
    s = system(FakeGEX(coil=np.array([0.99, 210e3]), board=np.array([1.0, 300e3]),
                       dual_moment=True))
    assert s.receiver_filters(1) == s.receiver_filters(2) == [(0.99, 210e3), (1.0, 300e3)]


def test_a_gex_without_filters_declares_none():
    assert system(FakeGEX()).receiver_filters(1) == []


def test_coil_filter_only():
    """The single-moment VTEM-ET case: a coil filter in General, no board filter."""
    assert system(FakeGEX(coil=np.array([1.0, 200e3]))).receiver_filters(1) == [(1.0, 200e3)]


def test_filter_written_as_text_is_read():
    assert system(FakeGEX(coil="1 200E+3")).receiver_filters(1) == [(1.0, 200e3)]


def test_malformed_filter_line_is_refused():
    with pytest.raises(ValueError, match="order"):
        system(FakeGEX(coil=np.array([210e3]))).receiver_filters(1)


# ── Reaching the receivers ───────────────────────────────────────────────────

def test_central_loop_receiver_carries_the_declared_filters():
    (rx,) = receivers(system(FakeGEX(coil=np.array([0.99, 210e3]),
                                     board=np.array([1.0, 300e3]))))
    assert rx.lp_cutoff_frequency_1 == 210e3 and rx.lp_power_1 == 0.99
    assert rx.lp_cutoff_frequency_2 == 300e3 and rx.lp_power_2 == 1.0


def test_dual_moment_receivers_carry_the_declared_filters():
    lm, hm = receivers(system(FakeGEX(coil=np.array([0.99, 210e3]),
                                      board=np.array([1.0, 300e3]), dual_moment=True)))
    for rx in (lm, hm):
        assert rx.lp_cutoff_frequency_1 == 210e3 and rx.lp_power_1 == 0.99
        assert rx.lp_cutoff_frequency_2 == 300e3 and rx.lp_power_2 == 1.0


def test_no_filters_declared_means_no_filtering():
    """An unset ``lp_*`` is simpeg's 'off'; a bare GEX must set nothing."""
    (rx,) = receivers(system(FakeGEX()))
    assert not hasattr(rx, "lp_power_1") and not hasattr(rx, "lp_power_2")


def test_option_switches_the_filters_off():
    (rx,) = receivers(system(FakeGEX(coil=np.array([1.0, 200e3])),
                             simulation__receiver_filters=False))
    assert not hasattr(rx, "lp_power_1") and not hasattr(rx, "lp_power_2")


@pytest.mark.xfail(strict=True, reason=(
    "The feature wires the GEX filters onto the receivers' numbered slots "
    "lp_cutoff_frequency_1/_2 and lp_power_1/_2, byte-identical to the source. "
    "This simpeg's Simulation1DLayered (simulation_1d.py) applies only the "
    "single, un-numbered rx.lp_cutoff_frequency / rx.lp_power, so the wired "
    "numbered slots have no effect on the forward here and the ratio stays 1.0. "
    "The simulation gaining a second low-pass slot is a separate change."))
def test_filters_change_the_early_time_response():
    """End to end on one central-loop sounding over a halfspace.

    A first-order low-pass filter smears the steep early decay, so the
    earliest gate records more signal filtered than unfiltered; late gates
    barely move.
    """
    gex = FakeGEX(coil=np.array([1.0, 200e3]))
    on = system(gex)
    off = system(gex, simulation__receiver_filters=False)
    d_on = on.forward(simulation__parallel=False, simulation__n_cpu=1)
    d_off = off.forward(simulation__parallel=False, simulation__n_cpu=1)
    ratio = (d_on.layer_data["dbdt_ch1gt"].to_numpy()[0]
             / d_off.layer_data["dbdt_ch1gt"].to_numpy()[0])
    assert ratio[0] > 1.02, "earliest gate must see more signal with the filter on"
    assert abs(ratio[-1] - 1) < 0.05, "late gates must be nearly untouched"
