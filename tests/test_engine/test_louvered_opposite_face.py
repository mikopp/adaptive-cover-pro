"""Opposite-face fallback at the louvered roof's travel limit (#1223).

A louvered roof whose drive turns partway past vertical (``max_slat_angle``
between 90° and 180°) cannot reach the facing-side blocking angle once it
exceeds the maximum. The travel clamp then pins the slats MORE open than the
blocking limit and the direct beam leaks. With ``opposite_face_fallback`` on,
the engine closes the slats from the other face instead.

The tests are driven by ``tests/fixtures/louvered_opposite_face_scenarios.json``:
an array of scenarios, each a roof plus a timeline of sun positions, so a new
case is a new entry in that file and no new code. Scenario schema::

    {
      "id": "unique-name",
      "description": "what this case is about",
      "roof": {
        "axis_azimuth": 212,              # direction the slats run (compass deg)
        "slat_depth_cm": 22, "slat_spacing_cm": 20,
        "tilt_mode": "mode2", "max_slat_angle": 125,   # 0 = the mode's maximum
        "opposite_face_fallback": true, "opposite_face_margin": 5.0
      },
      "samples": [                         # a timeline, in order
        {"time": "13:20", "sun_azimuth": 187.9, "sun_elevation": 65.8,
         "expect": {"branch": "opposite",  # facing | opposite | far | clamped
                    "angle": 24.55,        # commanded slat angle, degrees
                    "blocked": true}}      # is the direct sun blocked there
      ]
    }

Every scenario runs for BOTH ways the slats can turn. The drive opens toward the
side the roof faces; for slats along ``axis_azimuth`` that is either
``axis − 90°`` ("counterclockwise", the #1223 roof: axis 212°, facing 122°) or
``axis + 90°`` ("clockwise"). The listed sun positions are for the first; for
the second they are reflected across the slat axis, which is the same situation
seen in a mirror, so the same expectations apply. The checks per sample:

* **moved at the right time, to the right angle**: the branch and angle at each
  time match the file (the branch sequence is the timeline of when the slats
  switch face);
* **sun blocked**: the independent checker in ``tests/louvered_oracle.py`` says
  the commanded angle blocks the direct beam (or leaks, for the control
  scenario that documents the old clamp);
* **maximally open on either face**: the angle is the most open blocking angle on
  the face it chose, and a slightly more open angle leaks.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from custom_components.adaptive_cover_pro.const import (
    DEFAULT_LOUVERED_OPPOSITE_FACE_MARGIN,
)
from custom_components.adaptive_cover_pro.engine.covers.louvered_roof import (
    TRACE_KEY_OPPOSITE_FACE_FALLBACK,
)

from tests import louvered_oracle as oracle
from tests.cover_helpers import build_louvered_roof_cover

pytestmark = pytest.mark.unit

SCENARIO_FILE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "louvered_opposite_face_scenarios.json"
)
SCENARIOS = json.loads(SCENARIO_FILE.read_text())

# Offset from the slat axis to the direction the roof faces.
DIRECTIONS = {"counterclockwise": -90.0, "clockwise": +90.0}

ANGLE_TOL = 0.05  # engine vs checker: both are exact up to float/bisection error
FILE_ANGLE_TOL = 0.02  # golden values are stored to two decimals
OPEN_PROBE_DEG = 0.3  # "slightly more open" for the maximally-open check
BRANCHES = {"facing", "opposite", "far", "clamped"}

CASES = [
    pytest.param(sc, name, id=f"{sc['id']}-{name}")
    for sc in SCENARIOS
    for name in DIRECTIONS
]


def _facing(roof: dict, direction: str) -> float:
    return (roof["axis_azimuth"] + DIRECTIONS[direction]) % 360.0


def _sun_azimuth(roof: dict, sample: dict, direction: str) -> float:
    """Return the file's sun azimuth, reflected across the slat axis for 'clockwise'."""
    az = sample["sun_azimuth"]
    if direction == "counterclockwise":
        return az
    return (2.0 * roof["axis_azimuth"] - az) % 360.0


def _engine(roof: dict, sun_az: float, sun_el: float, direction: str):
    return build_louvered_roof_cover(
        sol_azi=sun_az,
        sol_elev=sun_el,
        roof_pitch=0.0,
        slat_distance=roof["slat_spacing_cm"] / 100.0,
        depth=roof["slat_depth_cm"] / 100.0,
        mode=roof["tilt_mode"],
        win_azi=_facing(roof, direction),
        max_slat_angle=roof["max_slat_angle"],
        opposite_face_fallback=roof["opposite_face_fallback"],
        opposite_face_margin=roof["opposite_face_margin"],
    )


def _limit(roof: dict) -> float:
    return oracle.travel(roof["tilt_mode"], roof["max_slat_angle"])


def _run(roof: dict, sample: dict, direction: str) -> tuple[str, float, object]:
    """Engine's ``(branch, angle, cover)`` for one sample."""
    cover = _engine(
        roof,
        _sun_azimuth(roof, sample, direction),
        sample["sun_elevation"],
        direction,
    )
    angle = cover.calculate_position()
    trace = cover._last_calc_details
    if trace["louvered_far_side_branch"]:
        branch = "far"
    elif trace[TRACE_KEY_OPPOSITE_FACE_FALLBACK]:
        branch = "opposite"
    elif angle == pytest.approx(_limit(roof)) and not oracle.blocks(
        angle,
        oracle.sun_tangent(
            _sun_azimuth(roof, sample, direction),
            sample["sun_elevation"],
            _facing(roof, direction),
        ),
        oracle.effective_depth(roof["slat_depth_cm"], roof["slat_spacing_cm"]),
        roof["slat_spacing_cm"],
    ):
        branch = "clamped"
    else:
        branch = "facing"
    return branch, angle, cover


def _checker_inputs(roof: dict, sample: dict, direction: str) -> dict:
    return {
        "sun_az": _sun_azimuth(roof, sample, direction),
        "sun_el": sample["sun_elevation"],
        "facing_az": _facing(roof, direction),
        "depth": roof["slat_depth_cm"],
        "spacing": roof["slat_spacing_cm"],
        "limit": _limit(roof),
        "fallback": roof["opposite_face_fallback"],
        "margin": roof["opposite_face_margin"],
    }


def _blocks_at(roof: dict, sample: dict, direction: str, angle: float) -> bool:
    tan_s = oracle.sun_tangent(
        _sun_azimuth(roof, sample, direction),
        sample["sun_elevation"],
        _facing(roof, direction),
    )
    return oracle.blocks(
        angle,
        tan_s,
        oracle.effective_depth(roof["slat_depth_cm"], roof["slat_spacing_cm"]),
        roof["slat_spacing_cm"],
    )


class TestScenarioFile:
    """The fixture itself: well-formed, and it exercises what it claims to."""

    def test_ids_are_unique(self) -> None:
        ids = [sc["id"] for sc in SCENARIOS]
        assert len(ids) == len(set(ids))

    @pytest.mark.parametrize("scenario", SCENARIOS, ids=[s["id"] for s in SCENARIOS])
    def test_entry_is_well_formed(self, scenario: dict) -> None:
        assert scenario["description"]
        assert {
            "axis_azimuth",
            "slat_depth_cm",
            "slat_spacing_cm",
            "tilt_mode",
            "max_slat_angle",
            "opposite_face_fallback",
            "opposite_face_margin",
        } <= set(scenario["roof"])
        assert scenario["samples"]
        for sample in scenario["samples"]:
            assert sample["expect"]["branch"] in BRANCHES
            assert 0.0 <= sample["expect"]["angle"] <= 180.0
            assert isinstance(sample["expect"]["blocked"], bool)

    def test_fallback_is_exercised_in_both_faces_and_the_control_leaks(self) -> None:
        branches = {s["expect"]["branch"] for sc in SCENARIOS for s in sc["samples"]}
        assert branches == BRANCHES
        control = next(
            sc for sc in SCENARIOS if not sc["roof"]["opposite_face_fallback"]
        )
        assert any(not s["expect"]["blocked"] for s in control["samples"])


@pytest.mark.parametrize(("scenario", "direction"), CASES)
class TestScenarios:
    """Every scenario in the file, for both ways the slats can turn."""

    def test_slats_move_to_the_right_angle_at_the_right_time(
        self, scenario: dict, direction: str
    ) -> None:
        roof = scenario["roof"]
        problems: list[str] = []
        prev_engine = prev_file = None
        for sample in scenario["samples"]:
            exp = sample["expect"]
            branch, angle, _ = _run(roof, sample, direction)
            if branch != exp["branch"]:
                problems.append(
                    f"{sample['time']}: took the {branch} branch, expected {exp['branch']}"
                )
            elif abs(angle - exp["angle"]) > max(FILE_ANGLE_TOL, ANGLE_TOL):
                problems.append(
                    f"{sample['time']}: commanded {angle:.2f}°, expected {exp['angle']:.2f}°"
                )
            if prev_engine is not None:
                moved = abs(angle - prev_engine) > 0.01
                should_move = abs(exp["angle"] - prev_file) > 0.01
                if moved != should_move:
                    problems.append(
                        f"{sample['time']}: slats {'moved' if moved else 'stayed'}, "
                        f"expected them to {'move' if should_move else 'stay'}"
                    )
            prev_engine, prev_file = angle, exp["angle"]
        assert not problems, "\n".join(problems)

    def test_direct_sun_is_blocked_where_expected(
        self, scenario: dict, direction: str
    ) -> None:
        roof = scenario["roof"]
        for sample in scenario["samples"]:
            _, angle, _ = _run(roof, sample, direction)
            blocked = _blocks_at(roof, sample, direction, angle)
            assert blocked == sample["expect"]["blocked"], (
                f"{sample['time']}: {angle:.2f}° "
                f"{'blocks' if blocked else 'lets sun through'}"
            )

    def test_slats_are_maximally_open_on_the_face_they_chose(
        self, scenario: dict, direction: str
    ) -> None:
        roof = scenario["roof"]
        d_eff = oracle.effective_depth(roof["slat_depth_cm"], roof["slat_spacing_cm"])
        spacing = roof["slat_spacing_cm"]
        for sample in scenario["samples"]:
            branch, angle, _ = _run(roof, sample, direction)
            tan_s = oracle.sun_tangent(
                _sun_azimuth(roof, sample, direction),
                sample["sun_elevation"],
                _facing(roof, direction),
            )
            where = f"{sample['time']} ({branch})"
            if branch == "clamped":
                # Out of reach with no fallback: pinned at the limit, which is
                # not a blocking pose (the leak this feature exists to fix).
                assert angle == pytest.approx(_limit(roof)), where
                assert (
                    oracle.max_open_angle("up", tan_s, d_eff, spacing, _limit(roof))
                    is None
                ), where
                continue
            face = "up" if branch == "facing" else "down"
            best = oracle.max_open_angle(face, tan_s, d_eff, spacing, _limit(roof))
            assert best is not None, where
            assert angle == pytest.approx(best, abs=ANGLE_TOL), where
            # A slightly more open pose on the same face would let sun through.
            if abs(angle - 90.0) > 1e-6:
                toward_open = angle + (
                    OPEN_PROBE_DEG if angle < 90.0 else -OPEN_PROBE_DEG
                )
                assert not oracle.blocks(toward_open, tan_s, d_eff, spacing), where

    def test_engine_agrees_with_the_independent_checker(
        self, scenario: dict, direction: str
    ) -> None:
        """Guards the file's golden values against drifting from the physics."""
        roof = scenario["roof"]
        for sample in scenario["samples"]:
            branch, angle, _ = _run(roof, sample, direction)
            want_branch, want_angle = oracle.expected_pose(
                **_checker_inputs(roof, sample, direction)
            )
            assert branch == want_branch, sample["time"]
            assert angle == pytest.approx(want_angle, abs=ANGLE_TOL), sample["time"]


class TestOutsideTheScenarioSchema:
    """Behaviour the scenario file does not describe."""

    @staticmethod
    def _cover(**kw):
        defaults = {
            "sol_azi": 187.9,
            "sol_elev": 65.8,
            "roof_pitch": 0.0,
            "slat_distance": 0.20,
            "depth": 0.22,
            "win_azi": 122.0,
            "max_slat_angle": 125.0,
        }
        return build_louvered_roof_cover(**{**defaults, **kw})

    def test_off_by_default(self) -> None:
        cover = self._cover()
        assert cover.roof_config.opposite_face_fallback is False
        assert cover.roof_config.opposite_face_margin == (
            DEFAULT_LOUVERED_OPPOSITE_FACE_MARGIN
        )
        assert cover.calculate_position() == pytest.approx(125.0)

    def test_vertical_pitch_keeps_the_venetian_anchor(self) -> None:
        off = self._cover(roof_pitch=90.0, sol_elev=30.0)
        on = self._cover(roof_pitch=90.0, sol_elev=30.0, opposite_face_fallback=True)
        assert on.calculate_position() == off.calculate_position()

    def test_safety_margin_closes_the_opposite_face_further(self) -> None:
        plain = self._cover(opposite_face_fallback=True)
        margined = self._cover(opposite_face_fallback=True, safety_margin=1.0)
        plain_angle = plain.calculate_position()
        margined_angle = margined.calculate_position()
        assert margined_angle < plain_angle
        # The raw trace records the angle before the safety margin.
        assert margined._last_calc_details["slat_angle_raw_deg"] == pytest.approx(
            plain_angle
        )
        tan_s = oracle.sun_tangent(187.9, 65.8, 122.0)
        assert oracle.blocks(margined_angle, tan_s, 24.0, 20.0)

    @pytest.mark.parametrize(
        ("margin", "engaged"),
        [(0.0, False), (1.0, False), (1.5, True), (5.0, True), (20.0, True)],
    )
    def test_margin_is_the_gap_below_the_limit(
        self, margin: float, engaged: bool
    ) -> None:
        """12:40 in #1223: the facing angle (123.9°) is 1.1° below the 125° limit."""
        cover = self._cover(
            sol_azi=165.7,
            sol_elev=65.4,
            opposite_face_fallback=True,
            opposite_face_margin=margin,
        )
        cover.calculate_position()
        assert cover._last_calc_details[TRACE_KEY_OPPOSITE_FACE_FALLBACK] is engaged
