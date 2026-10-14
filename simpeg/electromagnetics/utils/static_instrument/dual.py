import numpy as np
import re
import warnings

import simpeg.electromagnetics.time_domain as tdem

from . import base
import typing


# ── GEX channel helpers ──────────────────────────────────────────────────────
# The channels a GEX declares are read off the block names, never assumed to be
# 1..N contiguous or in order. A file may declare Channel3 and Channel6 only, or
# list them out of order; both must work.

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
    them would be a guess.
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


class MeasuredTEMXYZSystem(base.XYZSystem):
    """Measured-data TEM system with an arbitrary number of moments /
    receiver channels, described by a GEX file.

    Each ``[ChannelN]`` section of the GEX is one channel, and every channel
    present in the data is modelled independently and flatly — there is no
    grouping of channels into moments. For each channel the waveform, gate
    times, approximate dipole moment, gate factor and receiver-coil orientation
    are read from the GEX, so the same class handles single-moment
    multi-component systems (e.g. Xcite: a Z and an X receiver on one
    transmitter moment), classic dual-moment systems (e.g. SkyTEM: LM + HM) and
    systems with more moments and/or components, without any code change. A
    three-component dual-moment instrument is simply six flat channels.

    Channels are discovered from the ``dbdt_ch<N>gt`` arrays actually present in
    the data (falling back to the GEX channel blocks for pure forward
    modelling). They do **not** have to be numbered 1..N contiguously or appear
    in order; the channel numbers are read off and sorted.

    This class can not be instantiated directly; create an instantiable
    subclass with the GEX attached via the ``load_gex`` classmethod
    (inherited from ``XYZSystem``)::

        MySurveyInstrument = MeasuredTEMXYZSystem.load_gex(
            libaarhusxyz.GEX("instrument.gex"))

    See the help for ``XYZSystem`` for more information on basic usage.
    """

    rx_orientation : typing.Literal['x', 'y', 'z'] = 'z'
    "Fallback receiver coil orientation, used only for channels whose GEX section has no ReceiverPolarizationXYZ."
    tx_orientation : typing.Literal['x', 'y', 'z'] = 'z'
    "Transmitter loop orientation axis. 'z' is vertical (standard for horizontal AEM loops)."

    @property
    def channels(self):
        """The channel numbers to model, ascending.

        Driven by the ``dbdt_ch<N>gt`` arrays actually present in the data (so a
        dataset that imported only some of the GEX's channels inverts exactly
        those), reading the numbers off the keys rather than counting from 1 —
        non-contiguous and out-of-order numbering both work. Falls back to the
        GEX channel blocks when there is no data yet (pure forward modelling).
        """
        xyz = getattr(self, "_xyz", None)
        if xyz is not None:
            found = set()
            for key in xyz.layer_data.keys():
                match = re.fullmatch(r"dbdt_ch(\d+)gt", str(key))
                if match:
                    found.add(int(match.group(1)))
            if found:
                return sorted(found)
        return _channel_indices(self.gex)

    @property
    def n_moments(self):
        """Number of channels modelled (one flat entry per channel)."""
        return len(self.channels)

    def _channel(self, channel):
        "GEX [ChannelN] dict for the (one-based) channel number."
        return self.gex.gex_dict["Channel%d" % channel]

    def _orientation(self, channel):
        "Receiver orientation for a channel, from the GEX or the fallback."
        pol = _channel_field(self.gex, channel, 'ReceiverPolarizationXYZ')
        return str(pol).strip().lower() if pol else self.rx_orientation

    @property
    def rx_orientations(self):
        return [self._orientation(ch) for ch in self.channels]

    @property
    def gate_factors(self):
        return [(_channel_field(self.gex, ch, 'GateFactor') or 1.0)
                for ch in self.channels]

    @property
    def dipole_moments(self):
        return [_channel_field(self.gex, ch, 'ApproxDipoleMoment')
                for ch in self.channels]

    @property
    def area(self):
        return self.gex.General['TxLoopArea']

    def _waveform_points(self, channel):
        """Waveform (time, current) points for a channel. A per-moment
        Waveform<Moment>Point (e.g. WaveformLMPoint) takes precedence; systems
        with a single shared transmitter waveform fall back to WaveformPoint."""
        general = self.gex.General
        moment_name = self._channel(channel).get('TransmitterMoment', '') or ''
        key = 'Waveform' + moment_name + 'Point'
        if key not in general:
            key = 'WaveformPoint'
        return np.asarray(general[key])

    def make_waveforms(self):
        return [tdem.sources.PiecewiseLinearWaveform(pts[:, 0], pts[:, 1])
                for pts in (self._waveform_points(ch) for ch in self.channels)]

    @property
    def times_full(self):
        """Gate-time arrays, indexed by ``channel number - 1``.

        The base ``gate_filter`` maps a ``dbdt_ch<N>gt`` data key to
        ``times_filter[N-1]``, so this list must be positioned by channel
        number, not by ordinal. Absent channels get an empty array (their slot
        is never looked up, since no data key references them)."""
        chans = self.channels
        if not chans:
            return []
        out = [np.array([]) for _ in range(max(chans))]
        for ch in chans:
            out[ch - 1] = np.array(self.gex.gate_times(ch)[:, 0])
        return out

    # times_filter defaults to "all gates" (inherited from XYZSystem). Cull gates
    # per channel through processing (Disable gates ...) or the InUse flags; the
    # DualMomentTEMXYZSystem subclass below adds the classic LM/HM gate window.

    @property
    def correct_tilt_pitch_for1Dinv(self):
        fl = self.xyz.flightlines
        n = len(fl)
        if 'tilt_x' in fl.columns and 'tilt_y' in fl.columns:
            cos_roll = np.cos(fl.tilt_x.values / 180 * np.pi)
            cos_pitch = np.cos(fl.tilt_y.values / 180 * np.pi)
            return 1 / (cos_roll * cos_pitch) ** 2
        return np.ones(n)

    def _moment_data(self, channel):
        dbdt = self.xyz.layer_data["dbdt_ch%dgt" % channel].values
        inuse_key = "dbdt_inuse_ch%dgt" % channel
        if inuse_key in self.xyz.layer_data:
            dbdt = np.where(self.xyz.layer_data[inuse_key] == 0, np.nan, dbdt)
        tiltcorrection = self.correct_tilt_pitch_for1Dinv
        tiltcorrection = np.tile(tiltcorrection, (dbdt.shape[1], 1)).T
        gate_factor = _channel_field(self.gex, channel, 'GateFactor') or 1.0
        return - dbdt * self.xyz.model_info.get("scalefactor", 1) * gate_factor * tiltcorrection

    # NOTE: dbdt_std is a fraction, not an actual standard deviation size!
    def _moment_std(self, channel):
        return self.xyz.layer_data["dbdt_std_ch%dgt" % channel].values

    @property
    def data_array_nan(self):
        return np.hstack([self._moment_data(ch) for ch in self.channels]).flatten()

    @property
    def data_uncert_array(self):
        return np.hstack([self._moment_std(ch) for ch in self.channels]).flatten()

    @property
    def sounding_filter(self):
        usable = None
        for ch in self.channels:
            dk = "dbdt_ch%dgt" % ch
            sk = "dbdt_std_ch%dgt" % ch
            if dk not in self._xyz.layer_data or sk not in self._xyz.layer_data:
                continue
            m = np.isfinite(self._xyz.layer_data[dk].values) & np.isfinite(self._xyz.layer_data[sk].values)
            ik = "dbdt_inuse_ch%dgt" % ch
            if ik in self._xyz.layer_data:
                m = m & (self._xyz.layer_data[ik].values != 0)
            count = m.sum(axis=1)
            usable = count if usable is None else usable + count
        if usable is not None:
            return usable > 0
        if "resistivity" in self._xyz.layer_data:
            return np.isfinite(self._xyz.resistivity.values).sum(axis=1) > 0
        return np.ones(len(self._xyz.flightlines))

    def do_validate(self):
        """Check the data scaling (base) then every declared channel.

        Modelling is flat, so validation is flat too: for each channel present
        confirm its gate-time table matches its data length — the one genuinely
        load-bearing runtime check, since a channel whose ``TransmitterMoment``
        label points at another moment's (shorter) gate-time table slices out of
        range. Missing orientation / dipole-moment fields are warned about
        rather than failed, because a partial or hand-built GEX legitimately
        lacks them; the fallback orientation then applies.
        """
        super().do_validate()

        for ch in self.channels:
            if _channel_field(self.gex, ch, "ReceiverPolarizationXYZ") is None:
                warnings.warn(
                    "Channel %d declares no ReceiverPolarizationXYZ; using the "
                    "fallback %r." % (ch, self.rx_orientation))
            if _channel_field(self.gex, ch, "ApproxDipoleMoment") is None:
                warnings.warn("Channel %d declares no ApproxDipoleMoment." % ch)

            key = "dbdt_ch%dgt" % ch
            if key not in self._xyz.layer_data:
                continue
            try:
                n_times = len(self.gex.gate_times(ch))
            except Exception as exc:            # an unusable GEX is base's problem
                warnings.warn("Could not read gate times for channel %d: %s"
                              % (ch, exc))
                continue
            n_gates = self._xyz.layer_data[key].values.shape[1]
            assert n_times == n_gates, (
                "Channel %d has %d gates of data but its gate-time table yields "
                "%d entries. Most likely its TransmitterMoment label points at "
                "another channel's table, which is a different length."
                % (ch, n_gates, n_times))

    def make_system(self, idx, location, times):
        # FIXME: Martin says set z to altitude, not z (subtract topo), original code from seogi doesn't work!
        # Note: location[2] is already == altitude
        rx_coil_position = np.asarray(self.gex.General.get('RxCoilPosition', np.zeros(3)), dtype=float)
        receiver_location = (location[0] + rx_coil_position[0],
                             location[1] + rx_coil_position[1],
                             location[2] + np.abs(rx_coil_position[2]))
        horizontal_offset = float(np.hypot(rx_coil_position[0], rx_coil_position[1]))
        waveforms = self.make_waveforms()
        rx_orientations = self.rx_orientations
        area = self.area
        radius = np.sqrt(area / np.pi)
        sources = []
        for pos, channel in enumerate(self.channels):
            receivers = [tdem.receivers.PointMagneticFluxTimeDerivative(
                receiver_location, times[channel - 1], rx_orientations[pos])]
            if horizontal_offset < 1.0:
                # Central-loop geometry — the receiver sits at the loop centre
                # (e.g. Xcite, RxCoilPosition ~ 0). A point MagDipole source is
                # singular at zero transmitter–receiver offset (divide-by-zero in
                # the 1-D kernel), so model the transmitter as a finite
                # CircularLoop of the correct radius. Like the MagDipole branch,
                # the source carries UNIT dipole moment, because the data are
                # dB/dt normalised per unit transmitter moment; unit moment means
                # loop current = 1 / area.
                sources.append(tdem.sources.CircularLoop(
                    location=location,
                    receiver_list=receivers,
                    waveform=waveforms[pos],
                    radius=radius,
                    current=1.0 / area,
                    i_sounding=idx))
            else:
                # Offset receiver (e.g. SkyTEM) — point magnetic dipole source.
                sources.append(tdem.sources.MagDipole(
                    receivers,
                    location=location,
                    waveform=waveforms[pos],
                    orientation=self.tx_orientation,
                    i_sounding=idx))
        return sources


class DualMomentTEMXYZSystem(MeasuredTEMXYZSystem):
    """Backwards-compatible two-moment (LM + HM) system, as used for the
    SkyTEM instruments. Identical to :class:`MeasuredTEMXYZSystem` except that
    the gate window can be set independently for the low- and high-moment
    channels, matching the historical behaviour and saved configurations.

    It also exposes :attr:`moment_channels`, which resolves which two channels
    are the low and high moment by physics (dipole moment among channels of the
    modelled orientation) rather than by position, and validates that the
    resolved pair is coherent."""

    gate_filter__start_lm = 5
    "First LM (low moment) gate to include in the inversion, zero-based index. Early gates contaminated by transmitter on-time ringing or very early induction effects should be excluded. Check the GEX 'RemoveInitialGates' field for the system manufacturer's recommended cutoff."
    gate_filter__end_lm = 28
    "Last LM gate to include (exclusive, zero-based). Gates beyond this index are excluded — typically those where signal has decayed below the noise floor. Check late-time gate amplitudes in your data to identify the noise-dominated cutoff."
    gate_filter__start_hm = 10
    "First HM (high moment) gate to include in the inversion, zero-based index. Same considerations as start_lm. The HM channel typically has later reliable gates than LM due to its higher transmitter moment."
    gate_filter__end_hm = 32
    "Last HM gate to include (exclusive, zero-based). Same considerations as end_lm for the high-moment channel."

    @property
    def moment_channels(self):
        """``(low, high)`` channel numbers, resolved by physics rather than position.

        The two moments do not have to sit at channels 1 and 2. Among the
        channels measuring the orientation being modelled, the low moment is the
        one with the smallest ``ApproxDipoleMoment`` and the high moment the
        largest — true whatever the file numbers them, and whatever it calls
        them.

        This matters for any instrument declaring one channel per
        (moment, component) pair. A three-component dual-moment system declares
        six channels, ordered LM/X LM/Y LM/Z HM/X HM/Y HM/Z; the two vertical
        ones are 3 and 6.

        Falls back to ``(1, 2)`` with a warning where the GEX declares neither
        field, which is the behaviour established for partial and hand-built
        files.
        """
        indices = _channel_indices(self.gex)
        if not indices:
            warnings.warn("GEX declares no Channel blocks; assuming channels 1 and 2.")
            return (1, 2)

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

    @property
    def times_filter(self):
        chans = self.channels
        times = self.times_full
        filts = [np.zeros(len(t), dtype=bool) for t in times]
        lm, hm = chans[0], chans[1]
        filts[lm - 1][self.gate_filter__start_lm:self.gate_filter__end_lm] = True
        filts[hm - 1][self.gate_filter__start_hm:self.gate_filter__end_hm] = True
        return filts

    def do_validate(self):
        """Flat per-channel validation (parent) plus the two-moment coherence
        checks: the resolved pair must have distinct dipole moments and share a
        receiver orientation, otherwise they are not two moments of one
        component."""
        super().do_validate()

        lm, hm = self.moment_channels

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
