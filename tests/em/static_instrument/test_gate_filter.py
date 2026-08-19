"""Building the gate filter from declared channels rather than from array names.

``XYZSystem.gate_filter`` used to read a number out of every ``layer_data`` key
and index the per-moment gate masks with it. That is two assumptions at once:
that a name carrying digits belongs to a channel, and that the channel number is
a moment index. Both fail on real deliveries — the first on an auxiliary array
the instrument description never declares, the second on any instrument whose
moments are not channels 1 and 2.

Everything here is built in memory. The arrangements that motivate the change
are properties of instruments, not of any particular survey.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import libaarhusxyz

from SimPEG.electromagnetics.utils.static_instrument import base, dual


N_GATES_LM = 28
N_GATES_HM = 37


# ── Building datasets and instrument descriptions ────────────────────────────

def make_xyz(arrays, n_soundings=4):
    """An XYZ whose ``layer_data`` holds ``{name: gate count}``."""
    xyz = libaarhusxyz.XYZ()
    xyz.flightlines = pd.DataFrame({
        "x": np.arange(n_soundings, dtype=float),
        "y": np.zeros(n_soundings),
        "z": np.full(n_soundings, 30.0),
        "line_no": np.zeros(n_soundings, dtype=int)})
    for name, n_gates in arrays.items():
        xyz.layer_data[name] = pd.DataFrame(
            np.ones((n_soundings, n_gates)) * 1e-9)
    return xyz


class Stub(base.XYZSystem):
    """The base gate filter with the channel layout stated outright.

    Lets a test place the moments at arbitrary channel numbers without
    describing a whole instrument, which is a GEX, a waveform and two gate
    tables to exercise a property that assembles a dict.
    """

    channels = (1, 2)
    gate_counts = (N_GATES_LM, N_GATES_HM)

    @property
    def moment_channels(self):
        return self.channels

    @property
    def times_full(self):
        return [np.logspace(-5, -3, n) for n in self.gate_counts]


def stub(arrays, channels=(1, 2), gate_counts=(N_GATES_LM, N_GATES_HM)):
    system = Stub(make_xyz(arrays), validate=False)
    system.options.update(channels=channels, gate_counts=gate_counts)
    return system


# ── Reading a channel off an array name ──────────────────────────────────────

@pytest.mark.parametrize("key, channel", [
    ("dbdt_ch1gt", 1),
    ("dbdt_std_ch1gt", 1),
    ("dbdt_inuse_ch2gt", 2),
    ("dbdt_ch6gt", 6),
    ("dbdt_ch10gt", 10),
    ("Gate_Ch01", 1),          # the other naming standard in circulation
    ("STD_Ch02", 2),
    ("Gate_Ch10", 10),         # zero padding must not become channel 1
])
def test_channel_read_from_the_channel_token(key, channel):
    assert base.XYZSystem.layer_data_channel(key) == channel


@pytest.mark.parametrize("key", [
    "resistivity", "dep_top", "dep_bot", "doi_upper",
    "HM_X_G01",                # an array no channel block declares
])
def test_no_channel_where_the_name_names_none(key):
    """The first run of digits is not a channel number.

    ``HM_X_G01`` is the case that matters: reading ``01`` out of it yields
    channel 1, and a low-moment gate mask then lands on an array of
    high-moment width.
    """
    assert base.XYZSystem.layer_data_channel(key) is None


# ── Arrays belonging to no declared channel ──────────────────────────────────

def test_undeclared_array_is_left_out_of_the_filter():
    system = stub({"dbdt_ch1gt": N_GATES_LM,
                   "dbdt_ch2gt": N_GATES_HM,
                   "HM_X_G01": N_GATES_HM})
    assert set(system.gate_filter) == {"dbdt_ch1gt", "dbdt_ch2gt"}


def test_masks_match_the_width_of_the_array_they_filter():
    system = stub({"dbdt_ch1gt": N_GATES_LM, "dbdt_ch2gt": N_GATES_HM})
    filt = system.gate_filter
    assert len(filt["dbdt_ch1gt"]) == N_GATES_LM
    assert len(filt["dbdt_ch2gt"]) == N_GATES_HM


def test_filtering_a_dataset_carrying_an_undeclared_array():
    """The whole point: it filters, rather than raising on a shape mismatch.

    The undeclared array keeps every gate. ``FilteredXYZ`` falls back to a full
    slice for a key the filter does not mention, so leaving it out means
    "unfiltered", not "dropped" — it is still there afterwards, at full width.
    """
    system = stub({"dbdt_ch1gt": N_GATES_LM,
                   "dbdt_ch2gt": N_GATES_HM,
                   "HM_X_G01": N_GATES_HM})
    filtered = system.xyz

    assert filtered.layer_data["dbdt_ch1gt"].shape[1] == N_GATES_LM
    assert filtered.layer_data["HM_X_G01"].shape[1] == N_GATES_HM


def test_narrowed_arrays_come_out_at_the_width_of_their_own_mask():
    """A per-array mask, applied per array — not one mask for the dataset."""

    class Narrowing(Stub):
        @property
        def times_filter(self):
            lm = np.zeros(N_GATES_LM, dtype=bool)
            hm = np.zeros(N_GATES_HM, dtype=bool)
            lm[5:20] = True
            hm[8:30] = True
            return [lm, hm]

    system = Narrowing(make_xyz({"dbdt_ch1gt": N_GATES_LM,
                                 "dbdt_ch2gt": N_GATES_HM,
                                 "HM_X_G01": N_GATES_HM}), validate=False)
    filtered = system.xyz

    assert filtered.layer_data["dbdt_ch1gt"].shape[1] == 15
    assert filtered.layer_data["dbdt_ch2gt"].shape[1] == 22
    assert filtered.layer_data["HM_X_G01"].shape[1] == N_GATES_HM


# ── Moments somewhere other than channels 1 and 2 ────────────────────────────

def test_moment_pair_at_channels_3_and_6():
    """Six channels, one per (moment, component) pair; the vertical two are used.

    Channel 6 gave index 5 into a two-element list, which is the ``IndexError``
    the issue reports. The number is a channel, and its moment is where it sits
    in the pair.
    """
    system = stub({"dbdt_ch3gt": N_GATES_LM, "dbdt_ch6gt": N_GATES_HM},
                  channels=(3, 6))
    filt = system.gate_filter
    assert set(filt) == {"dbdt_ch3gt", "dbdt_ch6gt"}
    assert len(filt["dbdt_ch3gt"]) == N_GATES_LM
    assert len(filt["dbdt_ch6gt"]) == N_GATES_HM


def test_high_moment_first_maps_by_position_not_by_size():
    """``moment_channels`` order is the contract; nothing re-sorts it."""
    system = stub({"dbdt_ch2gt": N_GATES_LM, "dbdt_ch1gt": N_GATES_HM},
                  channels=(2, 1))
    filt = system.gate_filter
    assert len(filt["dbdt_ch2gt"]) == N_GATES_LM
    assert len(filt["dbdt_ch1gt"]) == N_GATES_HM


def test_channel_above_the_pair_no_longer_raises():
    """The reported failure, as a regression check."""
    system = stub({"dbdt_ch1gt": N_GATES_LM,
                   "dbdt_ch2gt": N_GATES_HM,
                   "dbdt_ch3gt": N_GATES_HM})
    with pytest.warns(UserWarning):
        filt = system.gate_filter
    assert "dbdt_ch3gt" not in filt


# ── What gets reported ───────────────────────────────────────────────────────

def test_warns_about_arrays_of_channels_that_are_not_modeled():
    """Measured data the inversion is about to ignore, so say so."""
    system = stub({"dbdt_ch3gt": N_GATES_LM,
                   "dbdt_ch6gt": N_GATES_HM,
                   "dbdt_ch1gt": N_GATES_LM},
                  channels=(3, 6))
    with pytest.warns(UserWarning, match="dbdt_ch1gt"):
        system.gate_filter


def test_the_warning_names_every_array_it_excludes():
    system = stub({"dbdt_ch1gt": N_GATES_LM,
                   "dbdt_ch2gt": N_GATES_HM,
                   "dbdt_ch4gt": N_GATES_LM,
                   "dbdt_ch5gt": N_GATES_HM})
    with pytest.warns(UserWarning) as caught:
        system.gate_filter
    message = str(caught[0].message)
    assert "dbdt_ch4gt" in message and "dbdt_ch5gt" in message


def test_silent_where_the_name_carries_no_channel_at_all():
    """Nothing here can tell an omitted component from a model array.

    ``resistivity``, ``dep_top`` and ``dep_bot`` accompany every inverted file.
    Reporting them alongside a genuinely omitted component would fire on every
    dataset and cost the warning its meaning.
    """
    system = stub({"dbdt_ch1gt": N_GATES_LM,
                   "dbdt_ch2gt": N_GATES_HM,
                   "HM_X_G01": N_GATES_HM,
                   "resistivity": 30,
                   "dep_top": 30})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        system.gate_filter


def test_nothing_reported_when_every_array_belongs_to_a_modeled_channel():
    system = stub({"dbdt_ch1gt": N_GATES_LM, "dbdt_std_ch1gt": N_GATES_LM,
                   "dbdt_ch2gt": N_GATES_HM, "dbdt_std_ch2gt": N_GATES_HM})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        system.gate_filter


# ── The base description, unsubclassed ───────────────────────────────────────

def test_base_describes_one_channel_numbered_one():
    """What ``times_full`` has always assumed, now stated where it can be read."""
    assert base.XYZSystem.moment_channels.fget(None) == (1,)


def test_a_single_channel_system_filters_its_one_channel():
    xyz = make_xyz({"dbdt_ch1gt": 20, "dbdt_std_ch1gt": 20})
    xyz.model_info["gate times for channel 1"] = list(np.logspace(-5, -3, 20))
    system = base.XYZSystem(xyz, validate=False)
    filt = system.gate_filter
    assert set(filt) == {"dbdt_ch1gt", "dbdt_std_ch1gt"}
    assert all(len(mask) == 20 for mask in filt.values())


# ── Through the dual moment system, resolving channels from the description ──

class FakeGEX:
    """Enough of ``libaarhusxyz.GEX`` to resolve channels and fetch gate times."""

    def __init__(self, channels):
        self.General = {"TxLoopArea": 342.0}
        self.gex_dict = {"General": self.General}
        self._gate_times = {}
        for channel, (moment, orientation, dipole, n_gates) in channels.items():
            self.gex_dict["Channel%d" % channel] = {
                "TransmitterMoment": moment,
                "ReceiverPolarizationXYZ": orientation,
                "ApproxDipoleMoment": float(dipole),
                "GateFactor": 1.0,
                "NoGates": n_gates}
            self._gate_times[channel] = np.column_stack([
                np.logspace(-5, -3, n_gates)] * 3)

    def gate_times(self, channel):
        return self._gate_times[channel]


def three_component_system():
    """Six channels, LM/X LM/Y LM/Z HM/X HM/Y HM/Z.

    The vertical pair is 3 and 6 — the arrangement the channel resolution in
    ``dual`` exists for, and the one where indexing a two-element list by the
    channel number ran off the end.
    """
    channels = {1: ("LM", "X", 2736.0, N_GATES_LM),
                2: ("LM", "Y", 2736.0, N_GATES_LM),
                3: ("LM", "Z", 2736.0, N_GATES_LM),
                4: ("HM", "X", 451440.0, N_GATES_HM),
                5: ("HM", "Y", 451440.0, N_GATES_HM),
                6: ("HM", "Z", 451440.0, N_GATES_HM)}

    class System(dual.DualMomentTEMXYZSystem):
        gex = FakeGEX(channels)

    return System


def test_dual_moment_filters_the_resolved_pair():
    System = three_component_system()
    xyz = make_xyz({"dbdt_ch3gt": N_GATES_LM, "dbdt_std_ch3gt": N_GATES_LM,
                    "dbdt_ch6gt": N_GATES_HM, "dbdt_std_ch6gt": N_GATES_HM})
    system = System(xyz, validate=False)

    assert system.moment_channels == (3, 6)
    filt = system.gate_filter
    assert set(filt) == {"dbdt_ch3gt", "dbdt_std_ch3gt",
                         "dbdt_ch6gt", "dbdt_std_ch6gt"}
    # start_lm/end_lm and start_hm/end_hm, applied to the right moment each.
    assert filt["dbdt_ch3gt"].sum() == System.gate_filter__end_lm - System.gate_filter__start_lm
    assert filt["dbdt_ch6gt"].sum() == System.gate_filter__end_hm - System.gate_filter__start_hm


def test_dual_moment_leaves_an_undeclared_array_alone():
    System = three_component_system()
    xyz = make_xyz({"dbdt_ch3gt": N_GATES_LM, "dbdt_ch6gt": N_GATES_HM,
                    "HM_X_G01": N_GATES_HM})
    system = System(xyz, validate=False)
    assert "HM_X_G01" not in system.gate_filter
