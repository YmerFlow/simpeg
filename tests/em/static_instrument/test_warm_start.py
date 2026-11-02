"""Warm start: the inversion starts from earlier models (``startmodel__dataset``).

Each sounding takes its start from the model sounding with the same line and
sounding_id, else from the nearest model sounding within the fill distance, else
from the usual start (clustered or halfspace). Layers are resampled onto the
inversion's, or the run stops. With a dataset start the reference model defaults to
the halfspace, and beta starts from the recorded ``final_beta`` times
``warm_factor``. Every inversion writes ``final_beta``.

No inversion is run: start models are built in memory, and the outputs are checked
with a stand-in ``self.inv``, as in ``test_inversion_misfit_columns.py``.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import libaarhusxyz

from simpeg.electromagnetics.utils.static_instrument import base, dual
from simpeg.electromagnetics.utils.static_instrument.schedules import RecordBeta, StartingBeta


N_GATES = 4
LINES = [1, 2, 3]                     # 100 m apart
XS = [0.0, 25.0, 50.0, 75.0]          # soundings along each line
N = len(LINES) * len(XS)
THICKNESSES = np.array([10.0, 30.0])  # three layers, tops 0, 10, 40 m
TOPS = [0.0, 10.0, 40.0]


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


def soundings(lines=LINES, with_ids=True, dx=0.0):
    """Flightlines of the test block: line_no, sounding_id, and x, y [m]."""
    fl = pd.DataFrame([{"line_no": line, "sounding_id": i, "x": x + dx, "y": 100.0 * (line - 1)}
                       for line in lines for i, x in enumerate(XS)])
    return fl if with_ids else fl.drop(columns="sounding_id")


def model_res(fl):
    """A resistivity [ohm.m] that names its sounding: 100 * line + 10 * sounding_id + layer."""
    return np.array([[100.0 * line + 10.0 * sid + layer for layer in range(len(TOPS))]
                     for line, sid in zip(fl["line_no"], np.arange(len(fl)) % len(XS))])


def start_model(lines=LINES, with_ids=True, dx=0.0, tops=TOPS, res=None, final_beta=None):
    """An earlier inversion's output, in libaarhusxyz naming."""
    xyz = libaarhusxyz.XYZ()
    xyz.flightlines = soundings(lines, with_ids, dx)
    n = len(xyz.flightlines)
    res = model_res(xyz.flightlines) if res is None else np.tile(res, (n, 1))
    xyz.layer_data["resistivity"] = pd.DataFrame(res)
    xyz.layer_data["dep_top"] = pd.DataFrame(np.tile(tops, (n, 1)))
    xyz.layer_data["dep_bot"] = pd.DataFrame(np.tile(np.r_[tops[1:], np.inf], (n, 1)))
    if final_beta is not None:
        xyz.model_info["final_beta"] = final_beta
    return xyz


def system(with_ids=True, **options):
    xyz = libaarhusxyz.XYZ()
    xyz.flightlines = soundings(with_ids=with_ids).assign(
        alt=30.0, tilt_x=0.0, tilt_y=0.0)
    for ch in (1, 2):
        xyz.layer_data["dbdt_ch%dgt" % ch] = pd.DataFrame(np.full((N, N_GATES), 1e-9))
        xyz.layer_data["dbdt_std_ch%dgt" % ch] = pd.DataFrame(np.full((N, N_GATES), 0.05))

    class System(dual.DualMomentTEMXYZSystem):
        pass
    System.gex = FakeGEX()
    s = System(xyz, validate=False)
    s.options.update(gate_filter__start_lm=0, gate_filter__end_lm=N_GATES,
                     gate_filter__start_hm=0, gate_filter__end_hm=N_GATES, **options)
    return s


def start_res(s):
    """The start model as resistivity [ohm.m], one row per sounding."""
    return 1 / np.exp(s.make_startmodel(THICKNESSES).reshape(N, -1))


# ── Matching ──────────────────────────────────────────────────────────────────

def test_soundings_match_by_line_and_sounding_id():
    # The model's positions are shifted by most of a sounding spacing, so a match
    # by position would take the neighbor; the ids still find the same sounding.
    s = system(startmodel__dataset=start_model(dx=20.0))
    np.testing.assert_allclose(start_res(s), model_res(soundings()))
    assert s.startmodel_report["startmodel_by_id"] == N


def test_without_ids_soundings_match_by_position():
    s = system(with_ids=False, startmodel__dataset=start_model(with_ids=False))
    np.testing.assert_allclose(start_res(s), model_res(soundings()))
    assert s.startmodel_report["startmodel_by_position"] == N


def test_ids_are_used_only_when_the_data_carry_them_too():
    s = system(with_ids=False, startmodel__dataset=start_model(dx=20.0))
    # Shifted 20 m, the nearest model sounding is the one before (5 m away), except
    # at the start of each line.
    sid = np.arange(N) % len(XS)
    np.testing.assert_allclose(start_res(s), model_res(soundings())[np.arange(N) - (sid > 0)])


def test_a_line_without_a_model_is_filled_from_its_neighbor():
    # Default fill distance: 1.5 times the 100 m line spacing reaches line 2 from line 3.
    s = system(startmodel__dataset=start_model(lines=[1, 2]))
    res = start_res(s)
    on_3 = soundings()["line_no"].values == 3
    np.testing.assert_allclose(res[on_3], res[soundings()["line_no"].values == 2])
    report = s.startmodel_report
    assert (report["startmodel_by_id"], report["startmodel_by_position"], report["startmodel_by_fallback"]) == (8, 4, 0)
    assert report["startmodel_fill_distance"] == pytest.approx(150.0)


def test_beyond_the_fill_distance_the_halfspace_start_is_used():
    s = system(startmodel__dataset=start_model(lines=[1, 2]), startmodel__fill_distance=50.0,
               startmodel__res=33.0)
    res = start_res(s)
    np.testing.assert_allclose(res[soundings()["line_no"].values == 3], 33.0)
    assert s.startmodel_report["startmodel_by_fallback"] == 4
    assert s.startmodel_report["startmodel_fallback"] == "halfspace"


def test_beyond_the_fill_distance_the_clustered_start_is_used_when_clustering_is_on(monkeypatch):
    clustered = np.log(1 / np.full(N * len(TOPS), 7.0))
    monkeypatch.setattr(base.XYZSystem, "_clustered_startmodel", lambda self, thicknesses: clustered)
    s = system(startmodel__dataset=start_model(lines=[1, 2]), startmodel__fill_distance=50.0,
               clustering__enabled=True)
    np.testing.assert_allclose(start_res(s)[soundings()["line_no"].values == 3], 7.0)
    assert s.startmodel_report["startmodel_fallback"] == "clustered start"


def test_a_failed_model_sounding_is_not_used():
    model = start_model()
    model.layer_data["resistivity"].iloc[0] = np.nan
    s = system(startmodel__dataset=model)
    res = start_res(s)
    np.testing.assert_allclose(res[0], model_res(soundings())[1])   # its neighbor, by position
    assert s.startmodel_report["startmodel_by_id"] == N - 1


def test_an_empty_list_is_a_cold_start():
    """What a form sends for the field left empty."""
    s = system(startmodel__dataset=[], startmodel__res=33.0)
    np.testing.assert_allclose(start_res(s), 33.0)
    assert not hasattr(s, "startmodel_report")
    assert isinstance(s.make_directives()[0], base.directives.BetaEstimate_ByEig)


def test_several_start_models_are_combined():
    s = system(startmodel__dataset=[start_model(lines=[1]), start_model(lines=[2, 3])])
    np.testing.assert_allclose(start_res(s), model_res(soundings()))


def test_a_start_model_is_read_from_a_msgpack_path(tmp_path):
    import libaarhusxyz.export.msgpack
    model = start_model(final_beta=12.5)
    model.normalize(naming_standard="alc")   # as the platform writes its outputs
    path = tmp_path / "root.msgpack"
    libaarhusxyz.export.msgpack.dump(model, str(path))
    s = system(startmodel__dataset=str(path))
    np.testing.assert_allclose(start_res(s), model_res(soundings()))
    assert isinstance(s.make_directives()[0], StartingBeta)


# ── Layers ────────────────────────────────────────────────────────────────────

def test_other_layers_are_resampled_in_log10_resistivity():
    tops, res = [0.0, 5.0, 20.0, 60.0], [10.0, 100.0, 1000.0, 100.0]
    s = system(startmodel__dataset=start_model(tops=tops, res=res))
    # mid-depths: source 2.5, 12.5, 40, 80 m; target 5, 25, 55 m (a halfspace's is its
    # top plus half the layer above it)
    expected = 10 ** np.interp([5.0, 25.0, 55.0], [2.5, 12.5, 40.0, 80.0], np.log10(res))
    np.testing.assert_allclose(start_res(s), np.tile(expected, (N, 1)))
    layers = s.startmodel_report["startmodel_layers"]
    assert "3 layers" in layers and "4 layers" in layers


def test_other_layers_are_an_error_when_resampling_is_off():
    s = system(startmodel__dataset=start_model(tops=[0.0, 5.0, 20.0, 60.0], res=[10.0] * 4),
               startmodel__resample_layers=False)
    with pytest.raises(ValueError, match="4 layers.*3 layers"):
        s.make_startmodel(THICKNESSES)


# ── Reference model ───────────────────────────────────────────────────────────

def test_reference_defaults_to_the_halfspace_with_a_dataset_start():
    s = system(startmodel__dataset=start_model(), startmodel__res=33.0)
    np.testing.assert_allclose(1 / np.exp(s.make_mref(THICKNESSES)), 33.0)
    s.options["regularization__mref"] = "startmodel"
    np.testing.assert_allclose(s.make_mref(THICKNESSES), s.make_startmodel(THICKNESSES))


def test_reference_stays_the_clustered_start_without_a_dataset(monkeypatch):
    clustered = np.log(1 / np.full(N * len(TOPS), 7.0))
    monkeypatch.setattr(base.XYZSystem, "_clustered_startmodel", lambda self, thicknesses: clustered)
    s = system(clustering__enabled=True)
    np.testing.assert_allclose(s.make_mref(THICKNESSES), clustered)


# ── Starting beta ─────────────────────────────────────────────────────────────

def test_warm_beta_is_warm_factor_times_the_median_recorded_beta():
    models = [start_model(lines=[1], final_beta=10.0), start_model(lines=[2], final_beta=30.0),
              start_model(lines=[3], final_beta=20.0)]
    s = system(startmodel__dataset=models, directives__beta__warm_factor=3.0)
    first = s.make_directives()[0]
    assert isinstance(first, StartingBeta) and first.beta0 == pytest.approx(60.0)


def test_an_explicit_beta0_overrides_the_recorded_beta():
    s = system(startmodel__dataset=start_model(final_beta=10.0), directives__beta__beta0=5.0)
    assert s.make_directives()[0].beta0 == 5.0


def test_without_a_recorded_beta_beta_is_estimated_and_the_log_says_so(capsys):
    s = system(startmodel__dataset=start_model())
    first = s.make_directives()[0]
    assert isinstance(first, base.directives.BetaEstimate_ByEig)
    assert "no start model records final_beta" in capsys.readouterr().out


def test_a_cold_start_still_estimates_beta():
    assert isinstance(system().make_directives()[0], base.directives.BetaEstimate_ByEig)


def test_starting_beta_sets_beta_at_initialize():
    directive = StartingBeta(42.0)
    directive.inversion = SimpleNamespace(invProb=SimpleNamespace(beta=1.0))
    directive.initialize()
    assert directive.invProb.beta == 42.0


# ── final_beta ────────────────────────────────────────────────────────────────

def test_record_beta_keeps_the_beta_the_last_iteration_was_solved_with():
    prob = SimpleNamespace(beta=100.0)
    recorder = RecordBeta()
    recorder.inversion = SimpleNamespace(invProb=prob)
    recorder.initialize()
    for _ in range(2):          # each iteration: solve at prob.beta, then a schedule halves it
        prob.beta /= 2
        recorder.endIter()
    assert recorder.beta == 50.0 and prob.beta == 25.0


def fake_inversion(beta):
    n_data = N * 2 * N_GATES
    dobs = np.full(n_data, 1e-9)
    W = 1.0 / (0.05 * dobs)
    dmisfit = SimpleNamespace(data=SimpleNamespace(dobs=dobs), W=SimpleNamespace(diagonal=lambda: W),
                              simulation=SimpleNamespace(thicknesses=THICKNESSES))
    return SimpleNamespace(invProb=SimpleNamespace(
        dmisfit=dmisfit, model=np.log(1 / 50.0) * np.ones(N * len(TOPS)), dpred=dobs, beta=beta))


def test_outputs_carry_final_beta_and_the_start_report():
    s = system(startmodel__dataset=start_model(lines=[1, 2]))
    s.make_startmodel(THICKNESSES)
    s._beta_recorder = SimpleNamespace(beta=8.0)
    s.inv = fake_inversion(beta=4.0)        # cooled once more after the last iteration
    s.make_inversion_outputs()
    for out in (s.l2, s.l2pred, s.corrected):
        assert out.model_info["final_beta"] == 8.0
        assert out.model_info["startmodel_by_position"] == 4
    assert "final_beta" not in s._xyz.model_info


def test_a_warm_start_reads_back_the_final_beta_an_inversion_wrote():
    first = system()
    first._beta_recorder = SimpleNamespace(beta=8.0)
    first.inv = fake_inversion(beta=4.0)
    first.make_inversion_outputs()
    second = system(startmodel__dataset=first.l2, directives__beta__warm_factor=10.0)
    assert second.make_directives()[0].beta0 == pytest.approx(80.0)


# ── Lateral median ────────────────────────────────────────────────────────────

def test_lateral_median_takes_line_to_line_differences_out_of_the_start():
    model = start_model(res=[1.0, 1.0, 1.0])
    model.layer_data["resistivity"] = pd.DataFrame(
        np.repeat([[10.0] * 3, [100.0] * 3, [1000.0] * 3], len(XS), axis=0))
    s = system(startmodel__dataset=model, startmodel__lateral_median=130.0)
    res = start_res(s)
    line = soundings()["line_no"].values
    # Within 130 m every sounding reaches all four soundings of its own line and of each
    # neighboring line, but not the line beyond (200 m). On line 2 that is four of each
    # line (median 100 ohm.m); on lines 1 and 3, four of their own and four of line 2,
    # whose median is taken in log space (the geometric mean of the two middle values).
    np.testing.assert_allclose(res[line == 2], 100.0)
    np.testing.assert_allclose(res[line == 1], 10 ** 1.5)
    np.testing.assert_allclose(res[line == 3], 10 ** 2.5)
