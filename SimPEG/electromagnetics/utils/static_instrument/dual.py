import numpy as np
import os
from matplotlib import pyplot as plt
from discretize import TensorMesh

from SimPEG import maps
from SimPEG.electromagnetics import time_domain as tdem
from SimPEG.electromagnetics.utils.em1d_utils import plot_layer
import libaarhusxyz
import pandas as pd

import numpy as np
from scipy.spatial import cKDTree, Delaunay
import os, tarfile
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from discretize import TensorMesh, SimplexMesh

from SimPEG.utils import mkvc
from SimPEG import (
    maps, data, data_misfit, inverse_problem, regularization, optimization,
    directives, inversion, utils
    )

from SimPEG.utils import mkvc
import SimPEG.electromagnetics.time_domain as tdem
import SimPEG.electromagnetics.utils.em1d_utils
from SimPEG.electromagnetics.utils.em1d_utils import get_2d_mesh,plot_layer, get_vertical_discretization_time
from SimPEG.regularization import LaterallyConstrained, RegularizationMesh

import scipy.stats
from . import base
import typing
import warnings
import re


def _channel_indices(gex):
    """The channel indices a GEX actually declares, ascending.

    ``gex.number_channels`` counts keys containing "Channel"; it does not say
    which. A file holding ``Channel3`` and ``Channel6`` reports 2, so probing
    ``range(1, number_channels + 1)`` looks at channels 1 and 2 and finds
    nothing. The indices have to be read off the keys.
    """
    indices = []
    for key in (getattr(gex, "gex_dict", None) or {}):
        match = re.fullmatch(r"Channel(\d+)", str(key))
        if match:
            indices.append(int(match.group(1)))
    return sorted(indices)


def _channel_field(gex, channel, field):
    """A per-channel GEX field, or None where the file does not declare it.

    Written to tolerate both a missing channel and a missing field, because a
    partial or hand-built GEX legitimately has neither, and a validation helper
    must not be the thing that raises.
    """
    try:
        return gex.gex_dict["Channel%d" % channel][field]
    except (KeyError, AttributeError, TypeError):
        return None


def _declared_rx_orientation(gex):
    """The receiver orientation the GEX declares, lowercased, or None.

    Only returned when every channel agrees. Channels disagreeing means the
    file is not describing a single-orientation instrument, and picking one of
    them would be a guess — :meth:`DualMomentTEMXYZSystem.do_validate` reports
    that case rather than this one resolving it.
    """
    orientations = set()
    for channel in (1, 2):
        value = _channel_field(gex, channel, "ReceiverPolarizationXYZ")
        if value is None:
            return None
        orientations.add(str(value).strip().lower())
    if len(orientations) != 1:
        return None
    orientation = orientations.pop()
    return orientation if orientation in ("x", "y", "z") else None


class DualMomentTEMXYZSystem(base.XYZSystem):
    """Dual moment system, suitable for describing e.g. the SkyTEM
    instruments. This class can not be directly instantiated, but
    instead, instantiable subclasses can created using the class
    method

    ```
    MySurveyInstrument = DualMomentTEMXYZSystem.load_gex(
        libaarhusxyz.GEX("instrument.gex"))
    ```

    which reads a gex file containing among other things the waveform
    description of the instrument.

    See the help for `XYZSystem` for more information on basic usage.
    """
    gate_filter__start_lm=5
    "First LM (low moment) gate to include in the inversion, zero-based index. Early gates contaminated by transmitter on-time ringing or very early induction effects should be excluded. Check the GEX 'RemoveInitialGates' field for the system manufacturer's recommended cutoff."
    gate_filter__end_lm=28
    "Last LM gate to include (exclusive, zero-based). Gates beyond this index are excluded — typically those where signal has decayed below the noise floor. Check late-time gate amplitudes in your data to identify the noise-dominated cutoff."
    gate_filter__start_hm=10
    "First HM (high moment) gate to include in the inversion, zero-based index. Same considerations as start_lm. The HM channel typically has later reliable gates than LM due to its higher transmitter moment."
    gate_filter__end_hm=32
    "Last HM gate to include (exclusive, zero-based). Same considerations as end_lm for the high-moment channel."

    rx_orientation : typing.Literal['x', 'y', 'z'] = 'z'
    "Receiver coil orientation axis. 'z' is vertical (standard for AEM dB/dt measurements). Change only if the system uses a horizontal or tilted receiver coil."
    tx_orientation : typing.Literal['x', 'y', 'z'] = 'z'
    "Transmitter loop orientation axis. 'z' is vertical (standard for horizontal AEM loops). Change only for non-standard transmitter configurations."
    
    @classmethod
    def load_gex(cls, gex):
        """Accepts a GEX file loaded using libaarhusxyz.GEX() and
        returns a new subclass of XYZSystem that can be used to do
        inversion and forward modelling."""

        class GexSystem(cls):
            pass
        GexSystem.gex = gex

        # Take the receiver orientation from the instrument rather than the
        # class default where the GEX declares one. The file knows; assuming
        # 'z' silently models a vertical receiver even when handed data from a
        # horizontal one. Still overridable at instantiation, since this only
        # sets the class attribute.
        declared = _declared_rx_orientation(gex)
        if declared is not None:
            GexSystem.rx_orientation = declared

        return GexSystem
    
    @property
    def moment_channels(self):
        """``(low, high)`` channel indices, resolved by physics rather than position.

        The two moments do not have to sit at channels 1 and 2. Among the
        channels measuring the orientation being modelled, the low moment is the
        one with the smallest ``ApproxDipoleMoment`` and the high moment the
        largest — which is true whatever the file numbers them, and whatever it
        calls them.

        This matters for any instrument declaring one channel per
        (moment, component) pair. A three-component dual-moment system declares
        six channels, ordered LM/X LM/Y LM/Z HM/X HM/Y HM/Z; the two vertical
        ones are 3 and 6. Those are the right channels, simply not numbered 1
        and 2, and without this they would have to be renamed before anything
        could invert them.

        Falls back to ``(1, 2)`` with a warning where the GEX declares neither
        field, which is the behaviour established for partial and hand-built
        files.
        """
        indices = _channel_indices(self.gex)
        if not indices:
            warnings.warn("GEX declares no Channel blocks; assuming channels 1 and 2.")
            return (1, 2)

        # Only channels measuring what is being modelled are candidates. On a
        # multi-component instrument this is what separates "the other moment"
        # from "the same moment on another axis".
        wanted = str(self.rx_orientation).strip().lower()
        candidates = []
        for index in indices:
            declared = _channel_field(self.gex, index, "ReceiverPolarizationXYZ")
            if declared is None or str(declared).strip().lower() == wanted:
                candidates.append(index)
        if not candidates:
            warnings.warn(
                "No channel declares a %r receiver; considering all %d channels."
                % (wanted, len(indices)))
            candidates = indices

        moments = {i: _channel_field(self.gex, i, "ApproxDipoleMoment")
                   for i in candidates}
        known = {i: m for i, m in moments.items() if m is not None}

        if len(known) < 2:
            warnings.warn(
                "GEX does not declare ApproxDipoleMoment for at least two %r "
                "channels; falling back to channels 1 and 2." % wanted)
            return (1, 2)

        ordered = sorted(known, key=lambda i: known[i])
        return (ordered[0], ordered[-1])

    @property
    def lm_channel(self):
        return self.moment_channels[0]

    @property
    def hm_channel(self):
        return self.moment_channels[1]

    def _gate_key(self, channel, kind=""):
        """The ``layer_data`` array name for a channel — ``dbdt_std_ch3gt`` etc."""
        return "dbdt_%sch%dgt" % (kind, channel)

    def _usable(self, channel):
        """Per-datum mask: finite value, finite uncertainty, and flagged in use."""
        data = self._xyz.layer_data[self._gate_key(channel)].values
        std = self._xyz.layer_data[self._gate_key(channel, "std_")].values
        inuse = self._xyz.layer_data[self._gate_key(channel, "inuse_")]
        return np.isfinite(data) & np.isfinite(std) & inuse

    def do_validate(self):
        """Check the scaling, then that the resolved channels are the two moments.

        The pair is chosen by :attr:`moment_channels`, which reads dipole moment
        and receiver orientation rather than assuming positions 1 and 2. These
        checks confirm the pair it found is coherent, so the two are
        complementary: resolution makes the common awkward cases work, and
        validation catches the ones where no sensible pair exists.

        What still fails, correctly:

        - an instrument whose two candidate channels have equal dipole moments,
          so there is no low and high to distinguish
        - a resolved pair whose orientations disagree, meaning they are two
          components rather than two moments
        - a GEX declaring one orientation while the system is configured to
          model another
        - gate-time tables that do not match the data — see
          :meth:`_validate_gate_times`

        Each check uses only fields a GEX already carries. Where a field is
        absent the check is skipped with a warning rather than failing, since a
        partial or hand-built GEX legitimately lacks them.
        """
        super().do_validate()

        lm, hm = self.moment_channels

        # Physics, not labels. The low moment necessarily has the smaller dipole
        # moment, whatever the file calls the channels — so this catches a
        # mislabelled GEX as well as a misordered one, where comparing
        # TransmitterMoment strings would only catch the second.
        lm_moment = _channel_field(self.gex, lm, "ApproxDipoleMoment")
        hm_moment = _channel_field(self.gex, hm, "ApproxDipoleMoment")
        if lm_moment is None or hm_moment is None:
            warnings.warn(
                "GEX does not declare ApproxDipoleMoment for channels %d and %d; "
                "cannot confirm which is the low moment." % (lm, hm))
        else:
            assert lm_moment < hm_moment, (
                "Resolved channels %d and %d have dipole moments %.0f and %.0f A m^2. "
                "The low moment must be the smaller; this instrument does not give "
                "two distinguishable moments on the %r receiver."
                % (lm, hm, lm_moment, hm_moment, self.rx_orientation))

        # Receiver orientation. The pair must measure the same thing; differing
        # orientations mean they are two components rather than two moments.
        lm_rx = _channel_field(self.gex, lm, "ReceiverPolarizationXYZ")
        hm_rx = _channel_field(self.gex, hm, "ReceiverPolarizationXYZ")
        if lm_rx is None or hm_rx is None:
            warnings.warn(
                "GEX does not declare ReceiverPolarizationXYZ for channels %d and %d; "
                "cannot confirm they share a receiver orientation." % (lm, hm))
        else:
            lm_rx, hm_rx = str(lm_rx).strip().lower(), str(hm_rx).strip().lower()
            assert lm_rx == hm_rx, (
                "Resolved channels %d and %d declare different receiver orientations "
                "(%r and %r), so they are two components of one moment rather than "
                "two moments." % (lm, hm, lm_rx, hm_rx))
            assert lm_rx == self.rx_orientation, (
                "GEX declares a %r receiver but the system is configured to model "
                "a %r one. Set rx_orientation to match the data, or correct the "
                "GEX." % (lm_rx, self.rx_orientation))

        self._validate_gate_times(lm, hm)

    def _validate_gate_times(self, lm, hm):
        """Partial cover for the one gap physics-based resolution does not close.

        Channels are resolved by dipole moment and orientation, but
        ``gex.gate_times()`` still resolves its *table* by label — it looks for
        ``GateTime{TransmitterMoment}`` in ``General`` before falling back to a
        shared ``GateTime``. So a wrong ``TransmitterMoment`` selects the right
        channel and fetches the wrong times, and neither check above sees it:
        one trusts the physics, the other trusts the label.

        **What is checked**

        The two resolved channels must carry different ``TransmitterMoment``
        values. Identical labels mean both fetch the same table, which is wrong
        on any system defining one per moment.

        Gate-time count must match the data. Length is set per channel by
        ``NoGates``, not by which table was read, so this only bites when the
        wrongly-selected table is *shorter* than ``NoGates`` — the slice then
        runs out. On a 306HP with the two labels swapped, the high-moment
        channel asks for 41 times from the 25-entry low-moment table and is
        caught; the low-moment channel asks for 25 from the 41-entry table, gets
        25, and is not.

        **What is not checked, and why nothing here can be**

        A single mislabelled channel whose wrong table is long enough passes
        both. There is no invariant left to exploit:

        - Gate times are not positive. Every high-moment channel to hand starts
          before turn-off — −3.7 µs on a 306HP with no shift applied, −41 µs on
          a 312HP — so a negative value is normal rather than a symptom.
        - Monotonicity holds for the wrong table as readily as the right one.
        - Table lengths are not fixed by anything. A 304M defines one shared
          table and a 306HP two separate ones purely because the systems were
          described differently; the same instrument could be written either
          way.

        The tables are arbitrary per-system data, so the only real defence is
        the label being right. This narrows the window rather than closing it.
        """
        lm_label = _channel_field(self.gex, lm, "TransmitterMoment")
        hm_label = _channel_field(self.gex, hm, "TransmitterMoment")
        if lm_label is not None and hm_label is not None:
            assert str(lm_label).strip() != str(hm_label).strip(), (
                "Channels %d and %d both declare TransmitterMoment %r, so both would "
                "read the same gate-time table. On a system defining one table per "
                "moment that is wrong for at least one of them." % (lm, hm, lm_label))

        for channel, name in ((lm, "low"), (hm, "high")):
            key = self._gate_key(channel)
            if key not in self._xyz.layer_data:
                continue
            try:
                n_times = len(self.gex.gate_times(channel))
            except Exception as exc:            # an unusable GEX is base's problem
                warnings.warn("Could not read gate times for channel %d: %s"
                              % (channel, exc))
                continue
            n_gates = self._xyz.layer_data[key].values.shape[1]
            assert n_times == n_gates, (
                "Channel %d (%s moment) has %d gates of data but its gate-time table "
                "yields %d entries. Most likely its TransmitterMoment label points at "
                "another moment's table, which is shorter than its NoGates."
                % (channel, name, n_gates, n_times))

    @property
    def sounding_filter(self):
        lm_key, hm_key = (self._gate_key(c) for c in self.moment_channels)
        if lm_key in self._xyz.layer_data and hm_key in self._xyz.layer_data:
            # Exclude soundings with no usable gates
            lm = self._usable(self.moment_channels[0])
            hm = self._usable(self.moment_channels[1])
            return lm.sum(axis=1) + hm.sum(axis=1) > 0
        elif "resistivity" in self._xyz.layer_data:
            return np.isfinite(self._xyz.resistivity.values).sum(axis=1) > 0
        else:
            return np.ones(len(self._xyz.flightlines))
        
    @property
    def area(self):
        return self.gex.General['TxLoopArea']
    
    @property
    def waveform_hm(self):
        return self.gex.General['WaveformHMPoint']
    
    @property
    def waveform_lm(self):
        return self.gex.General['WaveformLMPoint']

    @property
    def correct_tilt_pitch_for1Dinv(self):
        """Scale amplitudes to what a level transmitter would have measured.

        ``tilt_x`` is **pitch** and ``tilt_y`` is **roll**: x lies along the
        flight direction, so it measures nose-up/down. ``libaarhusxyz`` resolves
        them the same way — ``tilt_pitch_column`` matches ``tilt_x``/``TxPitch``
        and ``tilt_roll_column`` matches ``tilt_y``/``TxRoll``.

        The product is symmetric so the correction is unaffected by which name
        is attached to which axis, but the two are not interchangeable to a
        reader — sustained roll means a turn or a crosswind crab, sustained
        pitch means terrain following.
        """
        cos_pitch = np.cos(self.xyz.flightlines.tilt_x.values/180*np.pi)
        cos_roll = np.cos(self.xyz.flightlines.tilt_y.values/180*np.pi)
        return 1 / (cos_pitch * cos_roll)**2
    
    def _moment_data(self, channel):
        """dB/dt for one channel, tilt-corrected and gate-factored."""
        dbdt = self.xyz.layer_data[self._gate_key(channel)].values
        inuse_key = self._gate_key(channel, "inuse_")
        if inuse_key in self.xyz.layer_data:
            dbdt = np.where(self.xyz.layer_data[inuse_key] == 0, np.nan, dbdt)
        tiltcorrection = self.correct_tilt_pitch_for1Dinv
        tiltcorrection = np.tile(tiltcorrection, (dbdt.shape[1], 1)).T
        gate_factor = _channel_field(self.gex, channel, "GateFactor")
        return (- dbdt * self.xyz.model_info.get("scalefactor", 1)
                * gate_factor * tiltcorrection)

    @property
    def lm_data(self):
        return self._moment_data(self.moment_channels[0])

    @property
    def hm_data(self):
        return self._moment_data(self.moment_channels[1])

    # NOTE: dbdt_std is a fraction, not an actual standard deviation size!
    @property
    def lm_std(self):
        return self.xyz.layer_data[self._gate_key(self.moment_channels[0], "std_")].values
    
    @property
    def hm_std(self):
        return self.xyz.layer_data[self._gate_key(self.moment_channels[1], "std_")].values

    @property
    def data_array_nan(self):
        return np.hstack((self.lm_data, self.hm_data)).flatten()

    @property
    def data_uncert_array(self):
        return np.hstack((self.lm_std, self.hm_std)).flatten()

    @property
    def dipole_moments(self):
        return [_channel_field(self.gex, c, 'ApproxDipoleMoment')
                for c in self.moment_channels]
        
    @property
    def times_full(self):
        return tuple(np.array(self.gex.gate_times(c)[:, 0])
                     for c in self.moment_channels)    

    @property
    def times_filter(self):        
        times = self.times_full
        filts = [np.zeros(len(t), dtype=bool) for t in times]
        filts[0][self.gate_filter__start_lm:self.gate_filter__end_lm] = True
        filts[1][self.gate_filter__start_hm:self.gate_filter__end_hm] = True
        return filts
        
    def make_waveforms(self):
        time_input_currents_hm = self.waveform_hm[:,0]
        input_currents_hm = self.waveform_hm[:,1]
        time_input_currents_lm = self.waveform_lm[:,0]
        input_currents_lm = self.waveform_lm[:,1]

        waveform_hm = tdem.sources.PiecewiseLinearWaveform(time_input_currents_hm, input_currents_hm)
        waveform_lm = tdem.sources.PiecewiseLinearWaveform(time_input_currents_lm, input_currents_lm)
        return waveform_lm, waveform_hm
    
    def make_system(self, idx, location, times):
        # FIXME: Martin says set z to altitude, not z (subtract topo), original code from seogi doesn't work!
        # Note: location[2] is already == altitude
        receiver_location = (location[0] + self.gex.General['RxCoilPosition'][0],
                             location[1],
                             location[2] + np.abs(self.gex.General['RxCoilPosition'][2]))
        waveform_lm, waveform_hm = self.make_waveforms()        

        return [
            tdem.sources.MagDipole(
                [tdem.receivers.PointMagneticFluxTimeDerivative(
                    receiver_location, times[0], self.rx_orientation)],
                location=location,
                waveform=waveform_lm,
                orientation=self.tx_orientation,
                i_sounding=idx),
            tdem.sources.MagDipole(
                [tdem.receivers.PointMagneticFluxTimeDerivative(
                    receiver_location, times[1], self.rx_orientation)],
                location=location,
                waveform=waveform_hm,
                orientation=self.tx_orientation,
                i_sounding=idx)]

    
