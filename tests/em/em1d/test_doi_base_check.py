"""How often the DOI sits at the model base, per line and for the run (doi_base_check).

Layer tops 0, 10, 30, 60 m and a half-space top at 100 m: the bottom layer is 40 m thick, so a
DOI at or below 80 m counts as at the base, and an upper DOI above 10 m is in the first layer.
"""

import numpy as np
import pytest

from simpeg.electromagnetics.utils.doi_utils import doi_base_check, doi_check_messages

DEPTH_TOP = np.array([0.0, 10.0, 30.0, 60.0, 100.0])
NAN = np.nan

# line "1": four soundings with a DOI, one without
# line "2": two soundings, none at the base
# line "3": a single sounding without a DOI
LINE = np.array([1, 1, 1, 1, 1, 2, 2, 3])
LOWER = np.array([100.0, 85.0, 79.0, 50.0, NAN, 40.0, 60.0, NAN])
UPPER = np.array([80.0, 20.0, 5.0, 9.0, NAN, 20.0, 30.0, NAN])


def check(**kw):
    return doi_base_check(UPPER, LOWER, DEPTH_TOP, kw.pop("line", LINE), **kw)


def test_layering_figures():
    c = check()
    assert (c["base_depth"], c["bottom_layer"], c["first_layer"], c["threshold"]) == (100, 40, 10, 0.05)


def test_shares_per_line():
    one = check()["lines"]["1"]
    assert (one["n_soundings"], one["n_with_doi"]) == (5, 4)
    assert one["lower_at_base"] == 0.5                  # 100 and 85 m; 79 m is 1 m short
    assert one["upper_at_base"] == 0.25                 # 80 m, exactly half a layer up
    assert one["upper_in_first_layer"] == 0.5           # 5 and 9 m
    assert one["lower_p90_of_base"] == pytest.approx(np.percentile([100, 85, 79, 50], 90) / 100)
    two = check()["lines"]["2"]
    assert (two["lower_at_base"], two["upper_at_base"], two["upper_in_first_layer"]) == (0, 0, 0)
    assert two["lower_p90_of_base"] == pytest.approx(0.58)


def test_run_figures_pool_all_soundings_with_a_doi():
    run = check()["run"]
    assert (run["n_soundings"], run["n_with_doi"]) == (8, 6)
    assert run["lower_at_base"] == pytest.approx(2 / 6)
    assert run["upper_at_base"] == pytest.approx(1 / 6)
    assert run["upper_in_first_layer"] == pytest.approx(2 / 6)


def test_a_line_without_any_doi_is_counted_and_has_no_shares():
    three = check()["lines"]["3"]
    assert (three["n_soundings"], three["n_with_doi"]) == (1, 0)
    assert three["lower_at_base"] is None and three["lower_p90_of_base"] is None


def test_a_sounding_with_only_one_doi_is_left_out():
    upper = UPPER.copy()
    upper[0] = NAN                                      # the 100 m lower DOI loses its upper
    one = doi_base_check(upper, LOWER, DEPTH_TOP, LINE)["lines"]["1"]
    assert one["n_with_doi"] == 3
    assert one["lower_at_base"] == pytest.approx(1 / 3)


def test_lines_over_the_threshold():
    assert check()["lines_over_threshold"] == ["1"]
    assert check(threshold=0.5)["lines_over_threshold"] == []      # over, not at
    assert check(threshold=0.0)["lines_over_threshold"] == ["1"]   # line 2 is at 0


def test_line_keys_are_strings_in_order_of_appearance():
    c = check(line=np.array([7.0, 7.0, 7.0, 7.0, 7.0, 2.0, 2.0, 3.5]))
    assert list(c["lines"]) == ["7", "2", "3.5"]
    c = check(line=np.array(["L2", "L2", "L2", "L2", "L2", "L1", "L1", "L3"]))
    assert list(c["lines"]) == ["L2", "L1", "L3"]


def test_without_a_line_column_only_the_run():
    c = check(line=None)
    assert c["lines"] == {} and c["lines_over_threshold"] == []
    assert c["run"]["lower_at_base"] == pytest.approx(2 / 6)


def test_values_are_plain_python():
    def walk(v):
        if isinstance(v, dict):
            assert all(isinstance(k, str) for k in v)
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)
        else:
            assert v is None or type(v) in (int, float, str), (v, type(v))
    walk(check())


def test_bad_input():
    with pytest.raises(ValueError, match="at least one layer"):
        doi_base_check(UPPER, LOWER, [0.0])
    with pytest.raises(ValueError, match="differ in length"):
        doi_base_check(UPPER, LOWER[:-1], DEPTH_TOP)
    with pytest.raises(ValueError, match="differ in length"):
        doi_base_check(UPPER, LOWER, DEPTH_TOP, LINE[:-1])


# ── Log lines ─────────────────────────────────────────────────────────────────

def test_summary_and_a_warning_naming_exactly_the_lines_over():
    summary, warning = doi_check_messages(check(), "l2")
    assert summary.startswith("##DOI_CHECK## l2 model: lower DOI at the base on 33.3% of 6 soundings")
    assert "(upper 16.7%)" in summary and "of the 100 m base" in summary
    assert summary.endswith("; 1 of 3 lines over 5%")
    assert warning == ("WARNING: l2 model: the model base (100 m) cuts off the lower DOI on more "
                       "than 5% of soundings on these lines: 1; consider a deeper base")


def test_no_warning_when_every_line_is_under():
    assert len(doi_check_messages(check(threshold=0.5))) == 1


def test_without_lines_the_run_share_decides():
    messages = doi_check_messages(check(line=None))
    assert len(messages) == 2 and "lines" not in messages[1]
    assert len(doi_check_messages(check(line=None, threshold=0.5))) == 1


def test_no_doi_at_all():
    c = doi_base_check([NAN, NAN], [NAN, NAN], DEPTH_TOP, [1, 1])
    assert doi_check_messages(c) == ["##DOI_CHECK## no sounding has a DOI (2 soundings)"]
