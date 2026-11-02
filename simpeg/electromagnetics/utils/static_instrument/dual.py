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


def _inuse_mask(layer_data, key, shape):
    """The in-use flags at ``key`` as a boolean mask, or all-True where absent.

    Both readers of the in-use flags in this module go through here, so a
    dataset behaves the same whichever one looks at it.

    An absent array means every gate is in use: that is what YmerFlow's importer
    materialises anyway, and "no flags" carries no evidence that any gate is bad,
    so the array stays optional for standalone or hand-built datasets.

    A non-finite flag means not in use. NaN in an in-use array is "unknown", and
    the conservative direction is to exclude it. The previous ``== 0`` test fell
    the other way by accident, since ``NaN == 0`` is False, so an unknown flag
    read as in use. Testing finiteness makes this reader safe regardless of what
    convention wrote the array.
    """
    if key not in layer_data:
        return np.ones(shape, dtype=bool)
    flags = np.asarray(layer_data[key], dtype=float)
    return np.isfinite(flags) & (flags != 0)


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

    simulation__receiver_filters: bool = True
    """Apply the receiver low-pass filters the GEX declares.

    A GEX can describe up to two first-order low-pass filters between the
    receiver coil and the recorded gate: the coil filter
    (``General/RxCoilLPFilter1``) and a per-channel board filter
    (``Channel<n>/TiBLowPassFilter``), each written as ``<order> <cutoff Hz>``.
    They shape the earliest gates strongly: on the SkyTEM 304 (210 kHz and
    300 kHz) leaving them out puts the forward 20% low at 8-13 us and 3% low
    at 300 us against Workbench's forward of the same model, which an
    inversion then absorbs as a spurious conductor in the top few metres
    (YmerFlow/Ymerflow#94). The simulation applies them in the frequency domain
    (receiver ``lp_*`` properties); this only wires the GEX values through, for
    every channel. A GEX that declares no filter gets none. Switch off to
    reproduce older results."""

    def receiver_filters(self, channel):
        """The low-pass filters the GEX declares for a channel number.

        Returns ``[(order, cutoff_hz), ...]`` with up to two entries, coil filter
        first; an entry is omitted where the GEX does not declare it.
        libaarhusxyz folds the trailing digit of ``RxCoilLPFilter1`` into the
        key, so the coil filter is read as ``RxCoilLPFilter``.
        """
        filters = []
        coil = self.gex.General.get("RxCoilLPFilter")
        board = _channel_field(self.gex, channel, "TiBLowPassFilter")
        for spec in (coil, board):
            if spec is None:
                continue
            if isinstance(spec, str):
                spec = spec.split()
            spec = np.atleast_1d(np.asarray(spec, dtype=float))
            if spec.size != 2 or spec[1] <= 0:
                raise ValueError(
                    "GEX low-pass filter must be '<order> <cutoff Hz>', got %r" % (spec,))
            filters.append((float(spec[0]), float(spec[1])))
        return filters

    def make_receiver(self, location, times, orientation, channel):
        rx = tdem.receivers.PointMagneticFluxTimeDerivative(location, times, orientation)
        if self.simulation__receiver_filters:
            for slot, (order, cutoff) in enumerate(self.receiver_filters(channel), start=1):
                setattr(rx, "lp_cutoff_frequency_%d" % slot, cutoff)
                setattr(rx, "lp_power_%d" % slot, order)
        return rx

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
    # per channel through processing (Disable gates ...) or the InUse flags.

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
        inuse = _inuse_mask(self.xyz.layer_data,
                            "dbdt_inuse_ch%dgt" % channel, dbdt.shape)
        dbdt = np.where(inuse, dbdt, np.nan)
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
            m = m & _inuse_mask(self._xyz.layer_data,
                                "dbdt_inuse_ch%dgt" % ch, m.shape)
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
            receivers = [self.make_receiver(
                receiver_location, times[channel - 1], rx_orientations[pos], channel)]
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
