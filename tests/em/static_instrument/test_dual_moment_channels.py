"""Resolving the two moments by physics rather than by position.

``DualMomentTEMXYZSystem`` used to require the low and high moments at channels
1 and 2, because it addressed its arrays and GEX blocks by literal name. Any
dataset where they sat elsewhere had to be rewritten — arrays renamed, GEX
blocks moved — before it could be inverted.

These cover the resolution and the validation that backs it, and — from
"Reading the in-use flags" onward — how the two readers of a channel's in-use
array agree about a missing array and about a non-finite flag. Everything is
built in memory: the case that motivates the resolution change, a
three-component instrument, has no example file anywhere in the test data, and
neither in-use path is reachable from a platform run at all.
"""

import re
import warnings

import numpy as np
import pandas as pd
import pytest

from SimPEG.electromagnetics.utils.static_instrument import dual
from SimPEG.electromagnetics.utils.static_instrument.dual import (
    _channel_indices,
    _channel_field,
    _declared_rx_orientation,
    _inuse_mask,
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


def test_raises_when_no_channels_declared():
    """Was: warned and returned (1, 2).

    A GEX with no Channel blocks describes no instrument, so there is nothing
    to resolve. Returning a pair invents one.
    """
    with pytest.raises(ValueError, match="no Channel blocks"):
        Resolver(FakeGEX({"General": {}})).moment_channels


def test_raises_when_no_channel_matches_the_orientation():
    """Was: considered every channel rather than resolving to nothing.

    That returned the pair measured on a *different* component under the
    requested one. Here both channels measure X and Z is asked for; the old
    behavior returned the two X channels as the Z moment pair. It inverts, and
    the model is wrong in a way nothing downstream can detect.

    Refusing is the only answer that does not silently mislabel an axis.
    """
    gex = FakeGEX(make_gex([("LM", "X", 2736.0), ("HM", "X", 451440.0)]))
    with pytest.raises(ValueError, match="receiver"):
        Resolver(gex, rx_orientation="z").moment_channels


# ── Reading the in-use flags ─────────────────────────────────────────────────
#
# ``_usable`` and ``_moment_data`` both read a channel's in-use array, and used
# to disagree about it twice over: one indexed ``layer_data`` directly and
# raised KeyError where the array was absent while the other guarded, and the
# guarding one tested ``flags == 0``, which reads NaN as in use because
# ``NaN == 0`` is False. Neither path is reachable from a platform run, since
# YmerFlow's importer always materializes the array — so nothing else will
# notice if they drift apart again.

N_SOUNDINGS, N_GATES = 3, 4


def gate_frame(values):
    return pd.DataFrame(np.asarray(values, dtype=float))


def layer_data(channel=1, inuse=None, data=None, std=None):
    """``layer_data`` for one channel, with the in-use array optional.

    ``inuse=None`` omits the array entirely rather than filling it with ones —
    the whole point being the dataset that never had one, which is what a
    standalone run or an ALC mapping that names no in-use column produces.
    """
    if data is None:
        data = np.arange(1.0, N_SOUNDINGS * N_GATES + 1).reshape(N_SOUNDINGS, N_GATES)
    if std is None:
        std = np.full((N_SOUNDINGS, N_GATES), 0.05)
    frames = {"dbdt_ch%dgt" % channel: gate_frame(data),
              "dbdt_std_ch%dgt" % channel: gate_frame(std)}
    if inuse is not None:
        frames["dbdt_inuse_ch%dgt" % channel] = gate_frame(inuse)
    return frames


class FakeXYZ:
    """Enough of ``libaarhusxyz.XYZ`` for the two readers under test.

    Tilt is level throughout, so the tilt correction is exactly 1 and
    ``_moment_data`` reduces to a sign flip on the gates it keeps. That leaves
    the in-use handling as the only thing the assertions can be measuring.
    """

    def __init__(self, frames):
        self.layer_data = frames
        self.flightlines = pd.DataFrame({"tilt_x": np.zeros(N_SOUNDINGS),
                                         "tilt_y": np.zeros(N_SOUNDINGS)})
        self.model_info = {}


class Reader:
    """The two in-use readers, borrowed unbound from the real class.

    Same trick as :class:`Resolver`, and for the same reason: constructing a
    ``DualMomentTEMXYZSystem`` needs a real GEX with waveforms and gate tables,
    none of which bears on how an in-use array is read. The methods themselves
    are the real ones, so this cannot pass by re-implementing them.

    ``_usable`` reads ``_xyz`` and ``_moment_data`` reads ``xyz`` — unfiltered
    and filtered respectively in the real class, because ``_usable`` is what
    builds the sounding filter and cannot go through the property that applies
    it. Here both name one object, which is what lets a test hand the same
    dataset to both readers and compare them.
    """

    _gate_key = dual.DualMomentTEMXYZSystem._gate_key
    _usable = dual.DualMomentTEMXYZSystem._usable
    _moment_data = dual.DualMomentTEMXYZSystem._moment_data
    correct_tilt_pitch_for1Dinv = (
        dual.DualMomentTEMXYZSystem.correct_tilt_pitch_for1Dinv)

    def __init__(self, frames, gex=None):
        self._xyz = FakeXYZ(frames)
        self.gex = gex if gex is not None else dual_moment()

    @property
    def xyz(self):
        return self._xyz


def test_inuse_mask_absent_array_is_all_in_use():
    """No flags is no evidence that any gate is bad."""
    mask = _inuse_mask({}, "dbdt_inuse_ch1gt", (N_SOUNDINGS, N_GATES))
    assert mask.dtype == bool
    assert mask.shape == (N_SOUNDINGS, N_GATES)
    assert mask.all()


def test_inuse_mask_reads_zero_and_nonzero():
    frames = layer_data(inuse=[[1, 0, 1, 0]] * N_SOUNDINGS)
    mask = _inuse_mask(frames, "dbdt_inuse_ch1gt", (N_SOUNDINGS, N_GATES))
    assert mask[0].tolist() == [True, False, True, False]


def test_inuse_mask_treats_nan_as_not_in_use():
    """Unknown falls to excluded, not to admitted."""
    frames = layer_data(inuse=[[1, np.nan, 0, 1]] * N_SOUNDINGS)
    mask = _inuse_mask(frames, "dbdt_inuse_ch1gt", (N_SOUNDINGS, N_GATES))
    assert mask[0].tolist() == [True, False, False, True]


def test_usable_without_an_inuse_array():
    """Used to raise KeyError while ``_moment_data`` carried on regardless."""
    data = np.arange(1.0, N_SOUNDINGS * N_GATES + 1).reshape(N_SOUNDINGS, N_GATES)
    data[0, 0] = np.nan
    std = np.full((N_SOUNDINGS, N_GATES), 0.05)
    std[1, 2] = np.nan

    usable = Reader(layer_data(data=data, std=std))._usable(1)

    assert usable.dtype == bool
    # Only the non-finite value and the non-finite uncertainty are excluded;
    # the absent in-use array excludes nothing.
    assert usable.sum() == N_SOUNDINGS * N_GATES - 2
    assert not usable[0, 0] and not usable[1, 2]


def test_moment_data_without_an_inuse_array():
    reader = Reader(layer_data())
    expected = -reader.xyz.layer_data["dbdt_ch1gt"].values
    np.testing.assert_allclose(reader._moment_data(1), expected)


def test_both_readers_agree_when_the_inuse_array_is_absent():
    """The disagreement in the issue, stated directly.

    Neither reader may treat the absent array as a reason to drop a gate, and
    neither may raise.
    """
    reader = Reader(layer_data())
    assert reader._usable(1).all()
    assert np.isfinite(reader._moment_data(1)).all()


def test_usable_treats_nan_flags_as_not_in_use():
    """Used to raise TypeError — ``&`` against a float array holding NaN.

    Loud rather than silent, but still a dataset that could not be inverted.
    """
    frames = layer_data(inuse=[[1, np.nan, 1, 1]] * N_SOUNDINGS)
    usable = Reader(frames)._usable(1)
    assert usable[:, 1].tolist() == [False] * N_SOUNDINGS
    assert usable[:, 0].all() and usable[:, 2].all() and usable[:, 3].all()


def test_moment_data_treats_nan_flags_as_not_in_use():
    """The silent half: ``NaN == 0`` is False, so the datum used to survive."""
    frames = layer_data(inuse=[[1, np.nan, 0, 1]] * N_SOUNDINGS)
    result = Reader(frames)._moment_data(1)
    assert np.isnan(result[:, 1]).all(), "a NaN flag must not admit its gate"
    assert np.isnan(result[:, 2]).all()
    assert np.isfinite(result[:, 0]).all() and np.isfinite(result[:, 3]).all()


def test_both_readers_agree_on_a_nan_flag():
    frames = layer_data(inuse=[[1, np.nan, 0, 1]] * N_SOUNDINGS)
    reader = Reader(frames)
    np.testing.assert_array_equal(reader._usable(1),
                                  np.isfinite(reader._moment_data(1)))
