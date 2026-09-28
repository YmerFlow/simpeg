import numpy as np

import SimPEG.electromagnetics.time_domain as tdem

from . import base
import typing


class MeasuredTEMXYZSystem(base.XYZSystem):
    """Measured-data TEM system with an arbitrary number of moments /
    receiver channels, described by a GEX file.

    Each ``[ChannelN]`` section of the GEX becomes one moment. For every
    channel the waveform, gate times, approximate dipole moment, gate factor
    and receiver-coil orientation are read from the GEX, so the same class
    handles single-moment multi-component systems (e.g. Xcite: a Z and an X
    receiver on one transmitter moment), classic dual-moment systems (e.g.
    SkyTEM: LM + HM) and systems with more moments, without any code change.

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
    def n_moments(self):
        """Number of moments to model. Driven by the dbdt channels actually
        present in the data (so a dataset that imported only some of the GEX's
        channels inverts exactly those), falling back to the GEX channel count
        when there is no data yet (pure forward modelling)."""
        xyz = getattr(self, "_xyz", None)
        if xyz is not None:
            n = 0
            while ("dbdt_ch%dgt" % (n + 1)) in xyz.layer_data:
                n += 1
            if n > 0:
                return n
        return int(self.gex.number_channels)

    def _channel(self, moment):
        "GEX [ChannelN] dict for the zero-based moment index."
        return self.gex.gex_dict["Channel%d" % (moment + 1)]

    @property
    def rx_orientations(self):
        out = []
        for i in range(self.n_moments):
            pol = self._channel(i).get('ReceiverPolarizationXYZ', None)
            out.append(str(pol).strip().lower() if pol else self.rx_orientation)
        return out

    @property
    def gate_factors(self):
        return [self._channel(i).get('GateFactor', 1.0) for i in range(self.n_moments)]

    @property
    def dipole_moments(self):
        return [self._channel(i)['ApproxDipoleMoment'] for i in range(self.n_moments)]

    @property
    def area(self):
        return self.gex.General['TxLoopArea']

    def _waveform_points(self, moment):
        """Waveform (time, current) points for a moment. A per-moment
        Waveform<Moment>Point (e.g. WaveformLMPoint) takes precedence; systems
        with a single shared transmitter waveform fall back to WaveformPoint."""
        general = self.gex.General
        moment_name = self._channel(moment).get('TransmitterMoment', '') or ''
        key = 'Waveform' + moment_name + 'Point'
        if key not in general:
            key = 'WaveformPoint'
        return np.asarray(general[key])

    def make_waveforms(self):
        return [tdem.sources.PiecewiseLinearWaveform(pts[:, 0], pts[:, 1])
                for pts in (self._waveform_points(i) for i in range(self.n_moments))]

    @property
    def times_full(self):
        return tuple(np.array(self.gex.gate_times(i + 1)[:, 0])
                     for i in range(self.n_moments))

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

    def _moment_data(self, moment):
        dbdt = self.xyz.layer_data["dbdt_ch%dgt" % (moment + 1)].values
        inuse_key = "dbdt_inuse_ch%dgt" % (moment + 1)
        if inuse_key in self.xyz.layer_data:
            dbdt = np.where(self.xyz.layer_data[inuse_key] == 0, np.nan, dbdt)
        tiltcorrection = self.correct_tilt_pitch_for1Dinv
        tiltcorrection = np.tile(tiltcorrection, (dbdt.shape[1], 1)).T
        return - dbdt * self.xyz.model_info.get("scalefactor", 1) * self.gate_factors[moment] * tiltcorrection

    # NOTE: dbdt_std is a fraction, not an actual standard deviation size!
    def _moment_std(self, moment):
        return self.xyz.layer_data["dbdt_std_ch%dgt" % (moment + 1)].values

    @property
    def data_array_nan(self):
        return np.hstack([self._moment_data(i) for i in range(self.n_moments)]).flatten()

    @property
    def data_uncert_array(self):
        return np.hstack([self._moment_std(i) for i in range(self.n_moments)]).flatten()

    @property
    def sounding_filter(self):
        usable = None
        for i in range(self.n_moments):
            dk = "dbdt_ch%dgt" % (i + 1)
            sk = "dbdt_std_ch%dgt" % (i + 1)
            if dk not in self._xyz.layer_data or sk not in self._xyz.layer_data:
                continue
            m = np.isfinite(self._xyz.layer_data[dk].values) & np.isfinite(self._xyz.layer_data[sk].values)
            ik = "dbdt_inuse_ch%dgt" % (i + 1)
            if ik in self._xyz.layer_data:
                m = m & (self._xyz.layer_data[ik].values != 0)
            count = m.sum(axis=1)
            usable = count if usable is None else usable + count
        if usable is not None:
            return usable > 0
        if "resistivity" in self._xyz.layer_data:
            return np.isfinite(self._xyz.resistivity.values).sum(axis=1) > 0
        return np.ones(len(self._xyz.flightlines))

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
        for moment in range(self.n_moments):
            receivers = [tdem.receivers.PointMagneticFluxTimeDerivative(
                receiver_location, times[moment], rx_orientations[moment])]
            if horizontal_offset < 1.0:
                # Central-loop geometry — the receiver sits at the loop centre
                # (e.g. Xcite, RxCoilPosition ~ 0). A point MagDipole source is
                # singular at zero transmitter–receiver offset (divide-by-zero in
                # the 1-D kernel), so model the transmitter as a finite
                # CircularLoop of the correct radius. Like the MagDipole branch
                # (moment=1), the source carries UNIT dipole moment, because the
                # data are dB/dt normalised per unit transmitter moment; unit
                # moment means loop current = 1 / area.
                sources.append(tdem.sources.CircularLoop(
                    location=location,
                    receiver_list=receivers,
                    waveform=waveforms[moment],
                    radius=radius,
                    current=1.0 / area,
                    i_sounding=idx))
            else:
                # Offset receiver (e.g. SkyTEM) — point magnetic dipole source.
                sources.append(tdem.sources.MagDipole(
                    receivers,
                    location=location,
                    waveform=waveforms[moment],
                    orientation=self.tx_orientation,
                    i_sounding=idx))
        return sources


class DualMomentTEMXYZSystem(MeasuredTEMXYZSystem):
    """Backwards-compatible two-moment (LM + HM) system, as used for the
    SkyTEM instruments. Identical to :class:`MeasuredTEMXYZSystem` except that
    the gate window can be set independently for the low- and high-moment
    channels, matching the historical behaviour and saved configurations."""

    gate_filter__start_lm = 5
    "First LM (low moment) gate to include in the inversion, zero-based index. Early gates contaminated by transmitter on-time ringing or very early induction effects should be excluded. Check the GEX 'RemoveInitialGates' field for the system manufacturer's recommended cutoff."
    gate_filter__end_lm = 28
    "Last LM gate to include (exclusive, zero-based). Gates beyond this index are excluded — typically those where signal has decayed below the noise floor. Check late-time gate amplitudes in your data to identify the noise-dominated cutoff."
    gate_filter__start_hm = 10
    "First HM (high moment) gate to include in the inversion, zero-based index. Same considerations as start_lm. The HM channel typically has later reliable gates than LM due to its higher transmitter moment."
    gate_filter__end_hm = 32
    "Last HM gate to include (exclusive, zero-based). Same considerations as end_lm for the high-moment channel."

    @property
    def times_filter(self):
        times = self.times_full
        filts = [np.zeros(len(t), dtype=bool) for t in times]
        filts[0][self.gate_filter__start_lm:self.gate_filter__end_lm] = True
        filts[1][self.gate_filter__start_hm:self.gate_filter__end_hm] = True
        return filts
