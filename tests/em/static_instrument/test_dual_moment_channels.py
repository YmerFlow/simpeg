"""Resolving the two moments by physics rather than by position.

``DualMomentTEMXYZSystem`` used to require the low and high moments at channels
1 and 2, because it addressed its arrays and GEX blocks by literal name. Any
dataset where they sat elsewhere had to be rewritten — arrays renamed, GEX
blocks moved — before it could be inverted.

These cover the resolution and the validation that backs it. Everything is built
in memory: the case that motivates the change, a three-component instrument, has
no example file anywhere in the test data.
"""

import re
import warnings

import numpy as np
import pytest

from SimPEG.electromagnetics.utils.static_instrument.dual import (
    _channel_indices,
    _channel_field,
    _declared_rx_orientation,
)


# ── Building GEX dicts ───────────────────────────────────────────────────────

def make_gex(channels):
    """A GEX dict from ``(moment_name, orientation, dipole_moment)`` triples.

    Channels are numbered from 1 in the order given, which is what lets a test
    place the moments somewhere other than positions 1 and 2.
    """
    gex = {"General": {"TxLoopArea": 342.0, "GateTime": np.zeros((30, 3))}}
    for index, (moment, orientation, dipole) in enumerate(channels, start=1):
        gex["Channel%d" % index] = {
            "TransmitterMoment": moment,
            "ReceiverPolarizationXYZ": orientation,
            "ApproxDipoleMoment": float(dipole),
            "GateFactor": 1.0,
            "NoGates": 30,
        }
    return gex


class FakeGEX:
    """Enough of ``libaarhusxyz.GEX`` for the helpers under test.

    Deliberately not the real class: these tests are about channel resolution,
    and constructing a real GEX requires a waveform and gate tables that have
    nothing to do with what is being checked.
    """

    def __init__(self, gex_dict):
        self.gex_dict = gex_dict


def dual_moment(order="lm_first"):
    """A conventional two-channel instrument."""
    lm = ("LM", "Z", 2736.0)
    hm = ("HM", "Z", 451440.0)
    return FakeGEX(make_gex([lm, hm] if order == "lm_first" else [hm, lm]))


def three_component_dual_moment():
    """Six channels: LM/X LM/Y LM/Z HM/X HM/Y HM/Z.

    The vertical pair is channels 3 and 6 — the right two channels, simply not
    numbered 1 and 2. This is the arrangement the change exists for, and no
    example of it exists as a file.
    """
    return FakeGEX(make_gex([
        ("LM", "X", 2736.0), ("LM", "Y", 2736.0), ("LM", "Z", 2736.0),
        ("HM", "X", 451440.0), ("HM", "Y", 451440.0), ("HM", "Z", 451440.0),
    ]))


# ── Enumerating channels ─────────────────────────────────────────────────────

def test_channel_indices_reads_keys_not_a_count():
    """``number_channels`` says how many, not which.

    A GEX holding Channel3 and Channel6 reports 2, so probing
    ``range(1, n + 1)`` looks at 1 and 2 and finds nothing.
    """
    sparse = FakeGEX({"General": {},
                      "Channel3": {"ApproxDipoleMoment": 1.0},
                      "Channel6": {"ApproxDipoleMoment": 2.0}})
    assert _channel_indices(sparse) == [3, 6]


def test_channel_indices_ignores_non_channel_keys():
    gex = FakeGEX({"General": {}, "header": "x", "Channel1": {}, "ChannelInfo": {}})
    assert _channel_indices(gex) == [1]


def test_channel_indices_on_empty_gex():
    assert _channel_indices(FakeGEX({})) == []
    assert _channel_indices(FakeGEX(None)) == []


# ── Reading fields ───────────────────────────────────────────────────────────

def test_channel_field_returns_none_rather_than_raising():
    """Missing channel and missing field both give None.

    ``getattr(gex, "Channel9", None)`` cannot be used for this: ``GEX``
    proxies attribute lookup into its dict, so on the released version a miss
    raises ``KeyError`` and the default is never reached
    (YmerFlow/libaarhusxyz#3).
    """
    gex = dual_moment()
    assert _channel_field(gex, 1, "ApproxDipoleMoment") == 2736.0
    assert _channel_field(gex, 9, "ApproxDipoleMoment") is None
    assert _channel_field(gex, 1, "NoSuchField") is None
    assert _channel_field(FakeGEX(None), 1, "anything") is None


# ── Declared orientation ─────────────────────────────────────────────────────

def test_declared_orientation_when_channels_agree():
    assert _declared_rx_orientation(dual_moment()) == "z"


def test_declared_orientation_none_when_channels_disagree():
    """Two components of one moment. Picking one would be a guess."""
    assert _declared_rx_orientation(three_component_dual_moment()) is None


def test_declared_orientation_none_when_absent():
    gex = FakeGEX({"General": {}, "Channel1": {}, "Channel2": {}})
    assert _declared_rx_orientation(gex) is None


# ── Resolution, exercised through a stand-in for the class ───────────────────

class Resolver:
    """The resolution logic with none of SimPEG's import weight.

    Mirrors ``DualMomentTEMXYZSystem.moment_channels``. Importing the class
    itself pulls in discretize, matplotlib and the whole SimPEG stack, which is
    a heavy price for testing a property that reads a dict.
    """

    def __init__(self, gex, rx_orientation="z"):
        self.gex = gex
        self.rx_orientation = rx_orientation

    @property
    def moment_channels(self):
        from SimPEG.electromagnetics.utils.static_instrument import dual
        return dual.DualMomentTEMXYZSystem.moment_channels.fget(self)


def test_conventional_two_channel_resolves_to_1_2():
    """No behaviour change for any existing dataset."""
    assert Resolver(dual_moment()).moment_channels == (1, 2)


def test_reversed_channel_order_resolves_rather_than_failing():
    """A GEX with its blocks written high moment first now works."""
    assert Resolver(dual_moment(order="hm_first")).moment_channels == (2, 1)


def test_three_component_resolves_to_the_vertical_pair():
    """Channels 3 and 6 — the case the change exists for."""
    assert Resolver(three_component_dual_moment()).moment_channels == (3, 6)


def test_three_component_resolves_x_when_modelling_x():
    """Selection follows the orientation being modelled, not a fixed axis."""
    assert Resolver(three_component_dual_moment(),
                    rx_orientation="x").moment_channels == (1, 4)


def test_sparse_channel_numbering():
    """Channels need not be contiguous or start at 1."""
    gex = FakeGEX({"General": {},
                   "Channel3": {"ReceiverPolarizationXYZ": "Z", "ApproxDipoleMoment": 2736.0},
                   "Channel6": {"ReceiverPolarizationXYZ": "Z", "ApproxDipoleMoment": 451440.0}})
    assert Resolver(gex).moment_channels == (3, 6)


def test_falls_back_to_1_2_with_a_warning_when_fields_absent():
    """The behaviour established for partial and hand-built files."""
    gex = FakeGEX({"General": {}, "Channel1": {}, "Channel2": {}})
    with pytest.warns(UserWarning, match="ApproxDipoleMoment"):
        assert Resolver(gex).moment_channels == (1, 2)


def test_falls_back_when_no_channels_declared():
    with pytest.warns(UserWarning, match="no Channel blocks"):
        assert Resolver(FakeGEX({"General": {}})).moment_channels == (1, 2)


def test_warns_when_no_channel_matches_the_orientation():
    """Considers everything rather than resolving to nothing."""
    gex = FakeGEX(make_gex([("LM", "X", 2736.0), ("HM", "X", 451440.0)]))
    with pytest.warns(UserWarning, match="receiver"):
        assert Resolver(gex, rx_orientation="z").moment_channels == (1, 2)
