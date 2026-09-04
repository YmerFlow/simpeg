"""Resolution refuses rather than guesses when the GEX cannot answer.

``moment_channels`` previously fell back to channels 1 and 2 in three places.
Each fallback returned an answer the file does not support, and the answers
invert successfully — the failure is a plausible model, not an exception.

The worst of the three: asked for an orientation no channel declares, it widened
the candidate set to every channel, resolved the pair by dipole moment, and
returned channels measuring a different component under the requested label.

No shipped GEX reaches any of these paths; every file in the corpus declares
both ``ReceiverPolarizationXYZ`` and ``ApproxDipoleMoment`` on both channels.
"""

import pytest

from SimPEG.electromagnetics.utils.static_instrument.dual import (
    DualMomentTEMXYZSystem, _declared_rx_orientation)


class FakeGEX:
    def __init__(self, channels):
        self.gex_dict = {"General": {}}
        for index, fields in channels.items():
            self.gex_dict["Channel%d" % index] = dict(fields)


def _system(gex, orientation="z"):
    system = DualMomentTEMXYZSystem.__new__(DualMomentTEMXYZSystem)
    object.__setattr__(system, "options", {})
    type(system).gex = gex
    type(system).rx_orientation = orientation
    return system


def _channels(system):
    return DualMomentTEMXYZSystem.moment_channels.fget(system)


def _z(moment):
    return {"ReceiverPolarizationXYZ": "Z", "ApproxDipoleMoment": moment}


# ── the normal case still works ──────────────────────────────────────────────

def test_resolves_by_dipole_moment():
    assert _channels(_system(FakeGEX({1: _z(3000.0), 2: _z(150000.0)}))) == (1, 2)


def test_resolves_channels_that_are_not_one_and_two():
    gex = FakeGEX({3: _z(3000.0), 6: _z(150000.0)})
    assert _channels(_system(gex)) == (3, 6)


def test_low_moment_first_regardless_of_index_order():
    gex = FakeGEX({3: _z(150000.0), 6: _z(3000.0)})
    assert _channels(_system(gex)) == (6, 3)


# ── the three former fallbacks ───────────────────────────────────────────────

def test_no_channel_blocks_raises():
    with pytest.raises(ValueError, match="no Channel blocks"):
        _channels(_system(FakeGEX({})))


def test_unsatisfiable_orientation_raises_rather_than_returning_another():
    """The wrong answer this prevents: Z channels returned for an X request."""
    gex = FakeGEX({1: _z(3000.0), 2: _z(150000.0)})
    with pytest.raises(ValueError, match="declares a 'x' receiver|'x' receiver"):
        _channels(_system(gex, orientation="x"))


def test_unsatisfiable_orientation_message_names_what_is_declared():
    gex = FakeGEX({1: _z(3000.0), 2: _z(150000.0)})
    with pytest.raises(ValueError) as excinfo:
        _channels(_system(gex, orientation="x"))
    assert "1='z'" in str(excinfo.value) and "2='z'" in str(excinfo.value)


def test_two_candidates_without_moments_use_index_order_not_one_and_two():
    gex = FakeGEX({3: {"ReceiverPolarizationXYZ": "Z"},
                   6: {"ReceiverPolarizationXYZ": "Z"}})
    with pytest.warns(UserWarning, match="ordering channels 3 and 6 by index"):
        assert _channels(_system(gex)) == (3, 6)


def test_three_candidates_without_moments_raises():
    gex = FakeGEX({i: {"ReceiverPolarizationXYZ": "Z"} for i in (1, 2, 3)})
    with pytest.raises(ValueError, match="cannot be resolved"):
        _channels(_system(gex))


# ── _declared_rx_orientation reads declared indices ──────────────────────────

def test_declared_orientation_reads_channels_that_are_not_one_and_two():
    assert _declared_rx_orientation(FakeGEX({3: _z(1.0), 6: _z(2.0)})) == "z"


def test_declared_orientation_none_when_components_disagree():
    gex = FakeGEX({1: _z(1.0),
                   2: {"ReceiverPolarizationXYZ": "X", "ApproxDipoleMoment": 2.0}})
    assert _declared_rx_orientation(gex) is None


def test_declared_orientation_none_when_no_channels():
    assert _declared_rx_orientation(FakeGEX({})) is None
