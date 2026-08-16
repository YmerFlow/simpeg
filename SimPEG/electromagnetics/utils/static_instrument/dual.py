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
    
    def do_validate(self):
        """Check the scaling, then that channels 1 and 2 really are the two moments.

        This class maps ``dbdt_ch1gt`` to the low moment and ``dbdt_ch2gt`` to
        the high moment by name, and applies each channel's ``GateFactor``,
        ``ApproxDipoleMoment`` and gate times accordingly. Nothing in that
        mapping is derived from the data — so on a dataset whose channels are
        not LM-then-HM it inverts the wrong arrays and returns a model rather
        than an error.

        Two ways that happens in practice:

        - **Channel order.** A GEX whose two channel blocks are written high
          moment first. A text-ordering mistake, not an exotic instrument.
        - **Multi-component receivers.** One channel per (moment, component)
          pair, so a three-component dual-moment instrument declares six.
          Ordered LM/X, LM/Y, LM/Z, HM/X..., channels 1 and 2 are the same
          moment on two different axes.

        Both checks below are cheap and use only fields a GEX already carries.
        Where a field is absent the check is skipped with a warning rather than
        failing, since a partial or hand-built GEX legitimately lacks them.
        """
        super().do_validate()

        # Physics, not labels. The low moment necessarily has the smaller
        # dipole moment, whatever the file calls the channels — so this catches
        # a reversed *and* a mislabelled GEX, where comparing TransmitterMoment
        # strings would only catch the second.
        lm_moment = _channel_field(self.gex, 1, "ApproxDipoleMoment")
        hm_moment = _channel_field(self.gex, 2, "ApproxDipoleMoment")
        if lm_moment is None or hm_moment is None:
            warnings.warn(
                "GEX does not declare ApproxDipoleMoment for both channels; "
                "cannot confirm that channel 1 is the low moment.")
        else:
            assert lm_moment < hm_moment, (
                "Channel1 dipole moment (%.0f A m^2) is not below Channel2's "
                "(%.0f A m^2). This class reads channel 1 as the low moment and "
                "channel 2 as the high moment; that mapping does not hold for "
                "this instrument, and inverting anyway would apply the wrong "
                "gate factors and gate times." % (lm_moment, hm_moment))

        # Receiver orientation. Channels differing here means they are not two
        # moments of one measurement — most likely two components of one moment.
        lm_rx = _channel_field(self.gex, 1, "ReceiverPolarizationXYZ")
        hm_rx = _channel_field(self.gex, 2, "ReceiverPolarizationXYZ")
        if lm_rx is None or hm_rx is None:
            warnings.warn(
                "GEX does not declare ReceiverPolarizationXYZ for both channels; "
                "cannot confirm that they share a receiver orientation.")
        else:
            lm_rx, hm_rx = str(lm_rx).strip().lower(), str(hm_rx).strip().lower()
            assert lm_rx == hm_rx, (
                "Channel1 and Channel2 declare different receiver orientations "
                "(%r and %r). This class expects them to be the low and high "
                "moments of one measurement; differing orientations suggest they "
                "are two components of the same moment, in which case channel 1 "
                "and channel 2 are not the two moments." % (lm_rx, hm_rx))
            assert lm_rx == self.rx_orientation, (
                "GEX declares a %r receiver but the system is configured to model "
                "a %r one. Set rx_orientation to match the data, or correct the "
                "GEX." % (lm_rx, self.rx_orientation))

    @property
    def sounding_filter(self):
        if "dbdt_ch1gt" in self._xyz.layer_data and "dbdt_ch2gt" in self._xyz.layer_data:
            # Exclude soundings with no usable gates
            ch1 = np.isfinite(self._xyz.dbdt_ch1gt.values) & np.isfinite(self._xyz.dbdt_std_ch1gt.values) & self._xyz.dbdt_inuse_ch1gt
            ch2 = np.isfinite(self._xyz.dbdt_ch2gt.values) & np.isfinite(self._xyz.dbdt_std_ch2gt.values) & self._xyz.dbdt_inuse_ch2gt
            return ch1.sum(axis=1) + ch2.sum(axis=1) > 0
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
    
    @property
    def lm_data(self):
        dbdt = self.xyz.dbdt_ch1gt.values
        if "dbdt_inuse_ch1gt" in self.xyz.layer_data:
            dbdt = np.where(self.xyz.dbdt_inuse_ch1gt == 0, np.nan, dbdt)
        tiltcorrection = self.correct_tilt_pitch_for1Dinv
        tiltcorrection = np.tile(tiltcorrection, (dbdt.shape[1], 1)).T
        return - dbdt * self.xyz.model_info.get("scalefactor", 1) * self.gex.Channel1['GateFactor'] * tiltcorrection
    
    @property
    def hm_data(self):
        dbdt = self.xyz.dbdt_ch2gt.values
        if "dbdt_inuse_ch2gt" in self.xyz.layer_data:
            dbdt = np.where(self.xyz.dbdt_inuse_ch2gt == 0, np.nan, dbdt)
        tiltcorrection = self.correct_tilt_pitch_for1Dinv
        tiltcorrection = np.tile(tiltcorrection, (dbdt.shape[1], 1)).T
        return - dbdt * self.xyz.model_info.get("scalefactor", 1) * self.gex.Channel2['GateFactor'] * tiltcorrection

    # NOTE: dbdt_std is a fraction, not an actual standard deviation size!
    @property
    def lm_std(self):
        return self.xyz.dbdt_std_ch1gt.values
    
    @property
    def hm_std(self):
        return self.xyz.dbdt_std_ch2gt.values

    @property
    def data_array_nan(self):
        return np.hstack((self.lm_data, self.hm_data)).flatten()

    @property
    def data_uncert_array(self):
        return np.hstack((self.lm_std, self.hm_std)).flatten()

    @property
    def dipole_moments(self):
        return [self.gex.gex_dict['Channel1']['ApproxDipoleMoment'],
                self.gex.gex_dict['Channel2']['ApproxDipoleMoment']]
        
    @property
    def times_full(self):
        return (np.array(self.gex.gate_times('Channel1')[:,0]),
                np.array(self.gex.gate_times('Channel2')[:,0]))    

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

    
