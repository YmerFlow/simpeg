"""Cluster IDs and the clustered start model land on the soundings they belong to.

``MeasuredTEMXYZSystem.sounding_filter`` drops every sounding without a usable
gate, so clustering runs on the kept soundings only. ``make_inversion_outputs`` then mapped the IDs back with
``self.xyz.flightlines.index`` - but ``FilteredXYZ`` resets that index to
0..n_kept-1, so the IDs went onto the first n_kept raw rows instead of the
kept ones, and the dropped soundings' NaNs piled up at the end of the line.

Neither K-means nor the inversion is run: the cluster IDs and medoids are set
by hand, and ``make_inversion_outputs`` gets a stand-in ``self.inv``, as in
``test_inversion_misfit_columns.py``.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd

import libaarhusxyz

from simpeg.electromagnetics.utils.static_instrument import base, dual


N_SOUNDINGS, N_GATES = 5, 4
DROPPED = 1                                  # the sounding with no usable gate
KEPT = [0, 2, 3, 4]                          # raw rows that survive the filter
THICKNESSES = np.array([10.0, 30.0])


class FakeGEX:
    def __init__(self):
        two_point = np.array([[-1e-3, 1.0], [0.0, 0.0]])
        self.General = {"TxLoopArea": 342.0, "RxCoilPosition": np.array([-13.25, 0.0, -2.0]),
                        "WaveformLMPoint": two_point, "WaveformHMPoint": two_point}
        self.gex_dict = {"General": self.General}
        for ch, (moment, dipole) in {1: ("LM", 2736.0), 2: ("HM", 451440.0)}.items():
            self.gex_dict["Channel%d" % ch] = {
                "TransmitterMoment": moment, "ReceiverPolarizationXYZ": "Z",
                "ApproxDipoleMoment": dipole, "GateFactor": 1.0, "NoGates": N_GATES}

    def gate_times(self, channel):
        return np.column_stack([np.logspace(-5, -3, N_GATES)] * 3)


def make_xyz():
    xyz = libaarhusxyz.XYZ()
    xyz.flightlines = pd.DataFrame({
        "x": np.arange(N_SOUNDINGS, dtype=float), "y": np.zeros(N_SOUNDINGS),
        "alt": np.full(N_SOUNDINGS, 30.0), "line_no": np.zeros(N_SOUNDINGS, dtype=int),
        "tilt_x": np.zeros(N_SOUNDINGS), "tilt_y": np.zeros(N_SOUNDINGS)})
    for ch in (1, 2):
        dbdt = np.full((N_SOUNDINGS, N_GATES), 1e-9)
        dbdt[DROPPED] = np.nan
        xyz.layer_data["dbdt_ch%dgt" % ch] = pd.DataFrame(dbdt)
        xyz.layer_data["dbdt_std_ch%dgt" % ch] = pd.DataFrame(np.full((N_SOUNDINGS, N_GATES), 0.05))
    return xyz


def make_system():
    class System(dual.DualMomentTEMXYZSystem):
        pass
    System.gex = FakeGEX()
    s = System(make_xyz(), validate=False)
    s.options.update(gate_filter__start_lm=0, gate_filter__end_lm=N_GATES,
                     gate_filter__start_hm=0, gate_filter__end_hm=N_GATES)
    return s


def fake_inversion(n_soundings):
    """A stand-in for ``s.inv`` after a run on ``n_soundings`` kept soundings."""
    n_data = n_soundings * 2 * N_GATES
    dobs = np.full(n_data, 1e-9)
    W = 1.0 / (0.05 * dobs)
    model = np.log(1 / 50.0) * np.ones(n_soundings * (len(THICKNESSES) + 1))
    dmisfit = SimpleNamespace(data=SimpleNamespace(dobs=dobs),
                              W=SimpleNamespace(diagonal=lambda: W),
                              simulation=SimpleNamespace(thicknesses=THICKNESSES))
    return SimpleNamespace(invProb=SimpleNamespace(dmisfit=dmisfit, model=model, dpred=dobs))


def test_the_filter_drops_the_sounding_without_gates():
    s = make_system()
    assert len(s.xyz.flightlines) == len(KEPT)
    assert np.flatnonzero(s.sounding_filter).tolist() == KEPT


# ── Cluster IDs go back onto the kept soundings ──────────────────────────────

def test_cluster_ids_land_on_the_kept_soundings():
    s = make_system()
    s._cluster_ids = np.array([7, 8, 9, 6])          # one per kept sounding, in order
    s.inv = fake_inversion(len(KEPT))
    s.make_inversion_outputs()

    expected = np.full(N_SOUNDINGS, np.nan)
    expected[KEPT] = [7, 8, 9, 6]
    for out in (s.l2, s.l2pred, s.corrected):
        np.testing.assert_array_equal(out.flightlines["cluster_id"].values, expected)


# ── The clustered start model reads each cluster's medoid from the right row ──

class HandClusterer(base.XYZClusterer):
    """Clusters and medoids set by hand, indexed like the real ones: into the kept soundings."""

    def fit(self):
        self.cluster_ids_ = np.array([0, 1, 0, 1])
        self.n_clusters_ = 2
        self.medoid_indices_ = np.array([2, 1])      # kept rows 2, 1 = raw rows 3, 2
        return self.cluster_ids_


def test_cluster_startmodel_uses_each_medoids_own_model(monkeypatch):
    seen = {}

    def invert_medoids(self):
        """Give each medoid sounding resistivity 100 + its raw row, so the reader is identifiable."""
        seen["sounding_filter"] = np.asarray(self.sounding_filter).tolist()
        raw_rows = np.asarray(self.sounding_filter, dtype=float)
        model = np.log(1 / np.repeat(100 + raw_rows, len(THICKNESSES) + 1))
        return None, self.inverted_model_to_xyz(model, THICKNESSES)

    s = make_system()
    s.options.update(clustering__enabled=True)
    monkeypatch.setattr(type(s), "_make_clusterer", lambda self: HandClusterer(self.xyz))
    monkeypatch.setattr(type(s), "invert", invert_medoids)

    startmodel = s.make_startmodel(THICKNESSES)

    assert seen["sounding_filter"] == [3, 2]          # raw rows of the two medoids
    res = 1 / np.exp(startmodel.reshape(len(KEPT), -1))
    medoid_res = {0: 103.0, 1: 102.0}
    for i, k in enumerate([0, 1, 0, 1]):
        np.testing.assert_allclose(res[i], medoid_res[k])
