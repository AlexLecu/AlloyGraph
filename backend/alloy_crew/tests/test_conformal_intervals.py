"""The platform's conformal intervals: what the user sees, and what the agents do not."""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT, "backend"))

from alloy_crew.conformal import OFFSETS_PATH, conformal_interval, load_offsets  # noqa: E402
from alloy_crew.tools.metallurgy_tools import compute_metallurgy_validation  # noqa: E402

WASPALOY = {"Ni": 58.0, "Cr": 19.5, "Co": 13.5, "Mo": 4.3, "Ti": 3.0, "Al": 1.4, "C": 0.08}
PROPS = {"Yield Strength": 800.0, "Tensile Strength": 1200.0,
         "Elongation": 25.0, "Elastic Modulus": 210.0}


def test_offsets_file_covers_the_four_properties():
    offsets = load_offsets()
    assert offsets is not None, f"missing {OFFSETS_PATH}"
    assert set(offsets["properties"]) == set(PROPS)
    assert offsets["nominal_coverage"] == 0.9


def test_interval_is_the_stored_offsets_around_the_value():
    entry = load_offsets()["properties"]["Yield Strength"]
    iv = conformal_interval("Yield Strength", 800.0)
    assert iv["lower"] == round(800.0 + entry["lower_offset"], 1)
    assert iv["upper"] == round(800.0 + entry["upper_offset"], 1)
    assert iv["method"] == "conformal_cv_plus_90"


def test_interval_is_clipped_to_the_physical_domain():
    """A 15 MPa yield strength near 1200 C must not get a negative lower bound."""
    assert conformal_interval("Yield Strength", 15.0)["lower"] == 0.0
    assert conformal_interval("Elongation", 95.0)["upper"] == 100.0


def test_uncalibrated_property_gets_no_conformal_interval():
    assert conformal_interval("Density", 8.2) is None


def test_final_validation_reports_conformal_intervals():
    res = compute_metallurgy_validation(PROPS, WASPALOY, 20, "wrought",
                                        conformal_intervals=True)
    for prop in PROPS:
        assert res["property_intervals"][prop]["method"] == "conformal_cv_plus_90"


def test_agent_facing_validation_is_unchanged():
    """The Reviewer's tool must keep the fixed-percentage band of the evaluated campaigns."""
    res = compute_metallurgy_validation(PROPS, WASPALOY, 20, "wrought")
    ys = res["property_intervals"]["Yield Strength"]
    assert "method" not in ys
    assert ys["lower"] == 720.0 and ys["upper"] == 880.0


def test_intervals_do_not_change_status_or_penalties():
    a = compute_metallurgy_validation(PROPS, WASPALOY, 20, "wrought")
    b = compute_metallurgy_validation(PROPS, WASPALOY, 20, "wrought", conformal_intervals=True)
    assert (a["status"], a["penalty_score"], a["tcp_risk"]) == \
           (b["status"], b["penalty_score"], b["tcp_risk"])
