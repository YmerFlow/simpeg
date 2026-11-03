"""Two guardrails ported onto the N-moment API.

``gate_filter`` identifies a channel from the ``ch<N>`` token, not from the
first run of digits in an array name: an auxiliary array the GEX does not
describe has digits that are not a channel number, and handing it another
channel's mask applied a wrong-width boolean and raised, so the dataset could
not be processed at all. Such an array is now carried through unfiltered.

``_inuse_mask`` is the single reading of the in-use flags. An absent array means
every gate is in use; a non-finite flag means not in use (``NaN == 0`` is False,
so the old ``== 0`` test read an unknown flag as in use).
"""

import numpy as np

from simpeg.electromagnetics.utils.static_instrument import base
from simpeg.electromagnetics.utils.static_instrument.dual import _inuse_mask


class _FakeXYZ:
    def __init__(self, layer_data):
        self.layer_data = layer_data


def _system(layer_data, masks):
    class Sys(base.XYZSystem):
        validate = False
        times_filter = masks
    return Sys(_FakeXYZ(layer_data))


def _data(n_gates, n_soundings=3):
    return np.ones((n_soundings, n_gates))


# ── gate_filter channel identification ───────────────────────────────────────

def test_declared_channel_gets_its_mask():
    mask = np.array([True, False, True, True, False])
    s = _system({"dbdt_ch1gt": _data(5)}, [mask])
    filt = s.gate_filter
    assert set(filt) == {"dbdt_ch1gt"}
    assert np.array_equal(filt["dbdt_ch1gt"], mask)


def test_array_with_stray_digits_but_no_ch_token_is_carried_unfiltered():
    # "aux_horizontal_01gt" has digits ("01") that are not a channel number.
    # The old first-run-of-digits rule read it as channel 1 and applied that
    # channel's mask; the ch-token rule matches nothing, so it is left out of
    # the filter (FilteredXYZ then keeps every gate).
    s = _system({"dbdt_ch1gt": _data(5), "aux_horizontal_01gt": _data(9)},
                [np.ones(5, dtype=bool)])
    filt = s.gate_filter
    assert "aux_horizontal_01gt" not in filt
    assert "dbdt_ch1gt" in filt


def test_non_channel_arrays_are_ignored():
    s = _system({"dbdt_ch1gt": _data(5), "resistivity": _data(30),
                 "dep_top": _data(30)}, [np.ones(5, dtype=bool)])
    assert set(s.gate_filter) == {"dbdt_ch1gt"}


def test_channel_the_system_does_not_model_is_carried_unfiltered():
    # Only one moment declared; a dbdt_ch2gt array resolves to a channel index
    # past the end of times_filter and is carried through rather than crashing.
    s = _system({"dbdt_ch2gt": _data(5)}, [np.ones(5, dtype=bool)])
    assert s.gate_filter == {}


def test_noncontiguous_channel_numbers():
    # A 3-and-6 instrument: times_filter is positioned by channel number, so the
    # ch token indexes the right mask without assuming channels 1 and 2.
    masks = [np.zeros(4, dtype=bool) for _ in range(6)]
    masks[2] = np.array([True, True, False, False])
    masks[5] = np.array([False, False, True, True])
    s = _system({"dbdt_ch3gt": _data(4), "dbdt_ch6gt": _data(4)}, masks)
    filt = s.gate_filter
    assert np.array_equal(filt["dbdt_ch3gt"], masks[2])
    assert np.array_equal(filt["dbdt_ch6gt"], masks[5])


# ── _inuse_mask ──────────────────────────────────────────────────────────────

def test_absent_inuse_array_means_all_in_use():
    m = _inuse_mask({}, "dbdt_inuse_ch1gt", (2, 4))
    assert m.shape == (2, 4)
    assert m.all()


def test_zero_flag_means_not_in_use():
    flags = np.array([[1.0, 0.0, 1.0]])
    m = _inuse_mask({"dbdt_inuse_ch1gt": flags}, "dbdt_inuse_ch1gt", flags.shape)
    assert np.array_equal(m, [[True, False, True]])


def test_nan_flag_means_not_in_use():
    # NaN is "unknown"; the conservative direction is to exclude it. The old
    # `== 0` test let it through because NaN == 0 is False.
    flags = np.array([[1.0, np.nan, 0.0]])
    m = _inuse_mask({"dbdt_inuse_ch1gt": flags}, "dbdt_inuse_ch1gt", flags.shape)
    assert np.array_equal(m, [[True, False, False]])
