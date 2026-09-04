"""The receiver low-pass filters a GEX declares reach the simulation's receivers.

A SkyTEM GEX carries two first-order low-pass filters - the coil filter in
``General`` and a per-channel board filter - and the simulation has had the
receiver properties to apply them all along. Nothing connected the two, so
every forward ran unfiltered: 20% low at the earliest LM gates against
Workbench's forward of the same model (YmerFlow/Ymerflow#94).

Everything here is built in memory around the smallest GEX that lets
``make_system`` run: a two-point waveform per moment, a coil position, and the
channel blocks the moment resolution needs.
"""

import numpy as np
import pandas as pd
import pytest

import libaarhusxyz

from SimPEG.electromagnetics.utils.static_instrument import dual


class FakeGEX:
    def __init__(self, coil=None, board=None):
        two_point = np.array([[-1e-3, 1.0], [0.0, 0.0]])
        self.General = {
            "TxLoopArea": 342.0,
            "RxCoilPosition": np.array([-13.25, 0.0, -2.0]),
            "WaveformLMPoint": two_point,
            "WaveformHMPoint": two_point,
        }
        if coil is not None:
            self.General["RxCoilLPFilter"] = np.asarray(coil, dtype=float)
        self.gex_dict = {"General": self.General}
        for ch, (moment, dipole) in {1: ("LM", 2736.0), 2: ("HM", 451440.0)}.items():
            block = {"TransmitterMoment": moment, "ReceiverPolarizationXYZ": "Z",
                     "ApproxDipoleMoment": dipole, "GateFactor": 1.0, "NoGates": 4}
            if board is not None:
                block["TiBLowPassFilter"] = np.asarray(board, dtype=float)
            self.gex_dict["Channel%d" % ch] = block

    def gate_times(self, channel):
        return np.column_stack([np.logspace(-5, -3, 4)] * 3)


def make_xyz():
    xyz = libaarhusxyz.XYZ()
    xyz.flightlines = pd.DataFrame({"x": [0.0], "y": [0.0], "alt": [30.0], "line_no": [0],
                                    "tilt_x": [0.0], "tilt_y": [0.0]})
    for name in ("dbdt_ch1gt", "dbdt_ch2gt", "dbdt_std_ch1gt", "dbdt_std_ch2gt"):
        xyz.layer_data[name] = pd.DataFrame(np.full((1, 4), 1e-9))
    # A three-layer 50 ohm.m model, so the same dataset can be forward-modelled.
    xyz.layer_data["resistivity"] = pd.DataFrame(np.full((1, 3), 50.0))
    xyz.layer_data["dep_top"] = pd.DataFrame(np.array([[0.0, 10.0, 40.0]]))
    return xyz


def system(gex, **options):
    class System(dual.DualMomentTEMXYZSystem):
        pass
    System.gex = gex
    s = System(make_xyz(), validate=False)
    s.options.update(options)
    return s


def receivers(s):
    """The LM and HM receiver of one sounding, in that order."""
    sources = s.make_system(0, np.array([0.0, 0.0, 30.0]), s.times_full)
    return [src.receiver_list[0] for src in sources]


# ── Reading the GEX ──────────────────────────────────────────────────────────

def test_reads_coil_then_board_filter():
    s = system(FakeGEX(coil=[0.99, 210e3], board=[1, 300e3]))
    assert s.receiver_filters(1) == [(0.99, 210e3), (1.0, 300e3)]
    assert s.receiver_filters(2) == [(0.99, 210e3), (1.0, 300e3)]


def test_a_gex_without_filters_declares_none():
    assert system(FakeGEX()).receiver_filters(1) == []


def test_board_filter_only():
    assert system(FakeGEX(board=[1, 1e6])).receiver_filters(1) == [(1.0, 1e6)]


def test_malformed_filter_line_is_refused():
    with pytest.raises(ValueError, match="order"):
        system(FakeGEX(coil=[210e3])).receiver_filters(1)


# ── Reaching the receivers ───────────────────────────────────────────────────

def test_receivers_carry_the_declared_filters():
    lm, hm = receivers(system(FakeGEX(coil=[0.99, 210e3], board=[1, 300e3])))
    for rx in (lm, hm):
        assert rx.lp_cutoff_frequency_1 == 210e3 and rx.lp_power_1 == 0.99
        assert rx.lp_cutoff_frequency_2 == 300e3 and rx.lp_power_2 == 1.0


def test_no_filters_declared_means_no_filtering():
    """``lp_power`` 0 is SimPEG's 'off'; a bare GEX must leave it there."""
    lm, hm = receivers(system(FakeGEX()))
    for rx in (lm, hm):
        assert rx.lp_power_1 == 0 and rx.lp_power_2 == 0


def test_option_switches_the_filters_off():
    lm, hm = receivers(system(FakeGEX(coil=[0.99, 210e3], board=[1, 300e3]),
                              simulation__receiver_filters=False))
    for rx in (lm, hm):
        assert rx.lp_power_1 == 0 and rx.lp_power_2 == 0


def test_filters_change_the_early_time_response():
    """The whole point, end to end on one sounding over a halfspace.

    A first-order low-pass filter smears the steep early decay, so the gate
    centred at 10 us records *more* signal filtered than unfiltered - which is
    the direction of the 20% deficit the unfiltered forward showed against
    Workbench. Late gates barely move.
    """
    gex = FakeGEX(coil=[0.99, 210e3], board=[1, 300e3])
    # The fake system has 4 gates per moment; the default gate filter (5..28) would
    # leave none, so open it fully.
    gates = dict(gate_filter__start_lm=0, gate_filter__end_lm=4,
                 gate_filter__start_hm=0, gate_filter__end_hm=4,
                 startmodel__n_layer=3)   # the fixture model has 3 layers
    on = system(gex, **gates)
    off = system(gex, simulation__receiver_filters=False, **gates)
    d_on = on.forward(simulation__parallel=False, simulation__n_cpu=1)
    d_off = off.forward(simulation__parallel=False, simulation__n_cpu=1)
    lm_on = d_on.layer_data["dbdt_ch1gt"].to_numpy()[0]
    lm_off = d_off.layer_data["dbdt_ch1gt"].to_numpy()[0]
    ratio = lm_on / lm_off
    assert ratio[0] > 1.02, "earliest gate must see more signal with the filters on"
    assert abs(ratio[-1] - 1) < 0.05, "late gates must be nearly untouched"
