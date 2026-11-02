"""Christiansen-Auken (2012) depth of investigation, simpeg.electromagnetics.utils.doi_utils.

The reference numbers are the AarhusInv 8.11 manual's emo example (Figure 15.1, a 3-layer DC
sounding, 20 data at 2 %): the depths, layer and cumulated sensitivities its DOI block lists,
and the absolute DOIs it reports for Workbench's thresholds (88.339 m at 1.5, 133.472 m at
0.75) and the relative ones (115.723 m at 5 %, 185.147 m at 2 %). The table has four
significant digits, so the DOIs are matched to 0.05 m, which is tight enough to rule out
interpolating in log depth (88.02 m) or in log sensitivity (87.89 m).
"""

import numpy as np
import pytest

import simpeg.electromagnetics.time_domain as tdem
from simpeg import maps
from simpeg.electromagnetics.utils.doi_utils import (
    cumulative_sensitivity,
    doi_christiansen_auken,
    doi_from_cumulative,
    layer_sensitivity,
    log_sensitivity,
)

# AarhusInv 8.11 manual, Figure 15.1, emo lines 110-128: depth [m], sensitivity, cumulated.
MANUAL_DEPTH = np.array([2.500, 7.803, 13.75, 20.42, 27.90, 36.28, 45.68, 56.22, 68.04, 81.30,
                         96.16, 112.8, 131.5, 152.5, 176.0, 202.3, 231.9, 265.0, 300.0])
MANUAL_SENS = np.array([8.206, 3.148, 1.833, 1.260, 0.9520, 0.7628, 0.6300, 0.5266, 0.4401,
                        0.3650, 0.2990, 0.2411, 0.1910, 0.1486, 0.1137, 0.08559, 0.06354,
                        0.04661, 0.1188])
MANUAL_CUM = np.array([19.43, 11.22, 8.077, 6.245, 4.984, 4.032, 3.270, 2.640, 2.113, 1.673,
                       1.308, 1.009, 0.7678, 0.5768, 0.4282, 0.3145, 0.2289, 0.1654, 0.1188])


# ── The manual's example ─────────────────────────────────────────────────────

def test_cumulated_from_the_bottom_reproduces_the_manual():
    np.testing.assert_allclose(cumulative_sensitivity(MANUAL_SENS), MANUAL_CUM, rtol=2e-3)


def test_absolute_doi_reproduces_the_manual():
    doi = [doi_from_cumulative(MANUAL_CUM, MANUAL_DEPTH, t)[0] for t in (1.5, 0.75)]
    np.testing.assert_allclose(doi, [88.339, 133.472], atol=0.05)


def test_relative_doi_reproduces_the_manual():
    normalized = MANUAL_CUM / MANUAL_CUM[0]
    doi = [doi_from_cumulative(normalized, MANUAL_DEPTH, t)[0] for t in (0.05, 0.02)]
    np.testing.assert_allclose(doi, [115.723, 185.147], atol=0.05)


# ── Edge cases of the threshold search ───────────────────────────────────────

DEPTH_TOP = np.array([0.0, 10.0, 30.0, 60.0])


def test_total_below_threshold_gives_zero():
    S = np.array([[1.0, 0.6, 0.3, 0.1]])
    assert doi_from_cumulative(S, DEPTH_TOP, 1.5)[0] == 0.0


def test_halfspace_above_threshold_gives_its_top():
    S = np.array([[9.0, 6.0, 4.0, 2.0]])
    assert doi_from_cumulative(S, DEPTH_TOP, 1.5)[0] == DEPTH_TOP[-1]


def test_threshold_between_tops_interpolates_linearly_in_depth():
    S = np.array([[4.0, 2.0, 1.0, 0.5]])
    assert doi_from_cumulative(S, DEPTH_TOP, 1.5)[0] == pytest.approx(20.0)
    assert doi_from_cumulative(S, DEPTH_TOP, 2.0)[0] == pytest.approx(10.0)   # exactly at a top


def test_rows_are_independent_and_nan_stays_nan():
    S = np.array([[4.0, 2.0, 1.0, 0.5], [np.nan] * 4, [1.0, 0.5, 0.2, 0.1]])
    np.testing.assert_allclose(doi_from_cumulative(S, DEPTH_TOP, 1.5), [20.0, np.nan, 0.0])


# ── Which data count, and how ────────────────────────────────────────────────

def test_unused_data_contribute_nothing_and_signs_do_not_cancel():
    G = np.array([[0.5, -0.2], [-0.5, 0.2], [100.0, 100.0]])
    rel = np.array([0.05, 0.05, np.inf])
    np.testing.assert_allclose(layer_sensitivity(G, rel), [1.0, 0.4])


def test_weighted_divides_by_the_relative_uncertainty():
    G = np.array([[0.5, 0.2], [0.5, 0.2]])
    rel = np.array([0.05, 0.10])
    np.testing.assert_allclose(layer_sensitivity(G, rel, weighted=True), [15.0, 6.0])


def test_non_finite_sensitivity_in_a_used_datum_raises():
    with np.errstate(divide="ignore"):
        G = log_sensitivity(np.ones((2, 2)), np.ones(2), np.array([1.0, 0.0]))
    with pytest.raises(ValueError):
        layer_sensitivity(G, np.array([0.05, 0.05]))
    layer_sensitivity(G, np.array([0.05, np.inf]))   # the same datum, unused, is fine


# ── On a real stitched simulation ────────────────────────────────────────────

THICKNESSES = 2.0 * 1.4 ** np.arange(11)        # 12 layers, the halfspace top at 197 m
TIMES = np.logspace(-5, -2.5, 12)               # [s]
HEIGHT = 30.0                                   # source height above ground [m]


def layered_sigma():
    """Conductivity [S/m] of a 100 ohm.m ground with a 10 ohm.m layer at 20-50 m."""
    top = np.r_[0.0, np.cumsum(THICKNESSES)]
    return np.where((top >= 20) & (top < 50), 0.1, 0.01)


def source(i_sounding=None, x=0.0):
    rx = tdem.receivers.PointMagneticFluxTimeDerivative(
        np.array([[x + 13.25, 0.0, HEIGHT + 2.0]]), TIMES, "z")
    kw = {} if i_sounding is None else {"i_sounding": i_sounding}
    return tdem.sources.MagDipole([rx], location=np.array([x, 0.0, HEIGHT]), orientation="z", **kw)


def stitched(n_sounding=2):
    survey = tdem.Survey([source(i, 20.0 * i) for i in range(n_sounding)])
    n_param = n_sounding * (len(THICKNESSES) + 1)
    return tdem.Simulation1DLayeredStitched(
        survey=survey, thicknesses=THICKNESSES, sigmaMap=maps.ExpMap(nP=n_param),
        parallel=False, n_cpu=1)


def test_log_sensitivity_is_d_ln_data_by_d_ln_sigma():
    """J["ds"] is with respect to sigma itself; a finite difference in ln(sigma) agrees."""
    sim = stitched(1)
    m = np.log(layered_sigma())
    d = sim.dpred(m)
    G = log_sensitivity(np.asarray(sim.getJ(m)[0]["ds"]), np.exp(m), d)
    h = 1e-3
    for j in (0, 4, len(m) - 1):
        e = np.zeros_like(m)
        e[j] = h
        fd = (np.log(np.abs(sim.dpred(m + e))) - np.log(np.abs(sim.dpred(m - e)))) / (2 * h)
        np.testing.assert_allclose(G[:, j], fd, rtol=1e-3, atol=1e-6)


def test_stitched_doi_per_sounding_and_given_dpred_changes_nothing():
    sim = stitched(2)
    m = np.log(np.r_[layered_sigma(), np.full(len(THICKNESSES) + 1, 0.01)])
    rel = np.full(sim.survey.nD, 0.03)
    doi, S, z = doi_christiansen_auken(sim, m, rel)
    assert doi.shape == (2, 2) and S.shape == (2, len(THICKNESSES) + 1)
    np.testing.assert_allclose(z, np.r_[0.0, np.cumsum(THICKNESSES)])
    assert (doi[:, 0] < doi[:, 1]).all()                      # 1.5 is shallower than 0.75
    assert ((doi[0] > 0) & (doi[0] < z[-1])).all()            # below the 10 ohm.m layer
    assert (doi[1] > doi[0]).all()                            # the 100 ohm.m halfspace: deeper
    assert doi[1, 1] == z[-1]                                 # ... to the model base, a lower bound
    doi_given, S_given, _ = doi_christiansen_auken(sim, m, rel, d_pred=sim.dpred(m))
    np.testing.assert_allclose(doi_given, doi)
    np.testing.assert_allclose(S_given, S)


def test_fixed_jacobian_is_refused():
    sim = stitched(1)
    sim.fix_Jmatrix = True
    with pytest.raises(ValueError, match="fix_Jmatrix"):
        doi_christiansen_auken(sim, np.log(layered_sigma()), np.full(sim.survey.nD, 0.03))


def test_single_sounding_method_matches_the_stitched_function():
    m = np.log(layered_sigma())
    rel = np.full(len(TIMES), 0.03)
    single = tdem.Simulation1DLayered(
        survey=tdem.Survey([source()]), thicknesses=THICKNESSES,
        sigmaMap=maps.ExpMap(nP=len(m)))
    doi, S = single.depth_of_investigation_christiansen_2012(m, rel)
    doi_st, S_st, _ = doi_christiansen_auken(stitched(1), m, rel)
    np.testing.assert_allclose(S, S_st[0], rtol=1e-6)
    np.testing.assert_allclose(doi, doi_st[0], rtol=1e-6)
