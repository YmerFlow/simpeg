"""What each GEX receiver low-pass filter order means, as AarhusInv reads it.

An order of 0 is no filter, an integer n of 1 or more a Butterworth filter of
order n, and a value between 0 and 1 a Gaussian-RLC filter with that damping.
Each response is checked against its formula; order 1 is the first-order stage
simpeg applied before, unchanged.
"""

import numpy as np
import pytest

from simpeg.electromagnetics import time_domain as tdem
from simpeg.electromagnetics.time_domain.receivers import gex_low_pass_response

FC = 25.2e3  # cutoff [Hz]
F = np.array([1e2, 1e4, FC, 1e5, 1e6])  # [Hz]
S = 1j * F / FC  # complex frequency over the cutoff, dimensionless


def test_order_zero_is_no_filter():
    assert np.array_equal(gex_low_pass_response(F, 0, FC), np.ones(F.size))


def test_order_one_is_unchanged():
    """Butterworth order 1 is the first-order stage simpeg applied before."""
    h = gex_low_pass_response(F, 1, FC)
    np.testing.assert_allclose(h, 1 / (1 + S), rtol=1e-12)


@pytest.mark.parametrize(
    "order, denominator",
    [
        (2, lambda s: s**2 + np.sqrt(2) * s + 1),
        (3, lambda s: (s + 1) * (s**2 + s + 1)),
    ],
)
def test_integer_order_is_a_butterworth(order, denominator):
    h = gex_low_pass_response(F, order, FC)
    np.testing.assert_allclose(h, 1 / denominator(S), rtol=1e-12)
    power = 1 / (1 + (F / FC) ** (2 * order))
    np.testing.assert_allclose(np.abs(h) ** 2, power, rtol=1e-12)


@pytest.mark.parametrize("damping", [0.5, 0.99])
def test_order_between_zero_and_one_is_a_gaussian_rlc(damping):
    h = gex_low_pass_response(F, damping, FC)
    np.testing.assert_allclose(h, 1 / (S**2 + 2 * damping * S + 1), rtol=1e-12)
    # Gain at the cutoff is 1 / (2 * damping), lagging by a quarter turn.
    at_cutoff = h[F == FC][0]
    np.testing.assert_allclose(at_cutoff, -1j / (2 * damping), rtol=1e-12)


def test_critically_damped_rlc_is_two_first_order_stages():
    np.testing.assert_allclose(
        gex_low_pass_response(F, 1 - 1e-12, FC), 1 / (1 + S) ** 2, rtol=1e-9
    )


@pytest.mark.parametrize("order", [-1, 1.5])
def test_order_refused(order):
    with pytest.raises(ValueError):
        gex_low_pass_response(F, order, FC)


def test_order_two_forward_matches_the_receivers_butterworth():
    """A GEX order 2 forwards the same as the receiver's own ``bw_*`` filter."""
    times = np.logspace(-5, -3, 8)  # [s]
    waveform = np.array([[-1e-3, 1.0], [0.0, 0.0]])  # time [s], current [A]

    def forward(**filters):
        location = np.array([[0.0, 0.0, 30.0]])  # [m]
        rx = tdem.receivers.PointMagneticFluxTimeDerivative(
            location, times, "z", **filters
        )
        src = tdem.sources.CircularLoop(
            [rx],
            location=location[0],
            radius=10.0,
            waveform=tdem.sources.PiecewiseLinearWaveform(
                waveform[:, 0], waveform[:, 1]
            ),
        )
        sim = tdem.Simulation1DLayered(
            survey=tdem.Survey([src]),
            thicknesses=np.array([]),
            sigma=np.array([1 / 50.0]),
        )
        return sim.dpred(None)

    gex = forward(lp_power_1=2, lp_cutoff_frequency_1=FC)
    bw = forward(bw_power=2, bw_cutoff_frequency=FC)
    first_order = forward(lp_power_1=1, lp_cutoff_frequency_1=FC)
    np.testing.assert_allclose(gex, bw, rtol=1e-10)
    assert not np.allclose(gex, first_order, rtol=1e-3)
