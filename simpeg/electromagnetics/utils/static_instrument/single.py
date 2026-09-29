import simpeg.electromagnetics.time_domain as tdem
import numpy as np

from . import base

class SyntheticTEMXYZSystem(base.XYZSystem):
    """A very simple system description, suitable for working with
    synthetic data. It has a single transmitter with a perfect step
    function shut off, and a single receiver. It requires no setup and
    no GEX file — the instrument is described entirely by the class
    attributes below.

    Optionally, a custom waveform can be provided as a Pandas
    DataFrame with columns time and current in the waveform attribute.
    """

    area = 340
    i_max = 1
    rx_orientation = 'z'

    #turns = 1
    #rx_area = 1
    #alt = 30

    waveform = None

    @property
    def dipole_moments(self):
        return [self.area * self.i_max]

    def make_waveforms(self):
        if self.waveform is None:
            return [tdem.sources.StepOffWaveform()]

        return [tdem.sources.PiecewiseLinearWaveform(
            self.waveform.time.values,
            self.waveform.current.values)]

    def make_system(self, idx, location, times):
        waveforms = self.make_waveforms()
        return [tdem.sources.CircularLoop(
            location = location,
            receiver_list = [
                tdem.receivers.PointMagneticFluxTimeDerivative(
                    location,
                    times[0],
                    orientation = self.rx_orientation)],
            waveform = waveforms[0],
            radius = np.sqrt(self.area/np.pi),
            current = self.i_max,
            i_sounding = idx)]


# Backwards-compatible alias: this system used to be the "single moment" system.
SingleMomentTEMXYZSystem = SyntheticTEMXYZSystem
