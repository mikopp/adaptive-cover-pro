"""Opposite-face fallback at the louvered roof's travel limit (#1223).

A louvered roof whose drive turns partway past vertical (``max_slat_angle``
between 90° and 180°) cannot reach the facing-side blocking angle once it
exceeds the maximum. The travel clamp then pins the slats MORE open than the
blocking limit and the direct beam leaks. With ``opposite_face_fallback`` on,
the engine closes the slats from the other face instead.

The tests are driven by ``tests/fixtures/louvered_opposite_face_scenarios.json``,
an array of scenarios. Each is a roof at a place on a date; the test computes
the sun minute by minute over the day (NOAA equations, in
``tests/louvered_oracle.py``) and runs the engine at every minute. Schema::

    {
      "id": "unique-name",
      "description": "what this case is about",
      "location": {"latitude": 47.5, "longitude": 14.0, "utc_offset_hours": 2},
      "date": "2026-06-21",
      "day": {"start": "04:30", "end": "21:30"},   # local time, every minute
      "roof": {
        "axis_azimuth": 212,          # compass direction the slats run along
        "slat_depth_cm": 22, "slat_spacing_cm": 20,
        "tilt_mode": "mode2", "max_slat_angle": 125,   # 0 = the mode's maximum
        "opposite_face_fallback": true, "opposite_face_margin": 5.0
      },
      "expect": {                     # one entry per rotation direction
        "counterclockwise": {
          "flips": [{"time": "12:26", "from": "facing", "to": "opposite",
                     "angle_before": 120.0, "angle_after": 18.0}],
          "sun_gets_through": []      # [{"from": "12:44", "to": "14:06"}, ...]
        },
        "clockwise": {...}
      }
    }

**Rotation direction.** A roof can be built to turn either way. The drive
opens toward the side the roof faces, which for slats along ``axis_azimuth`` is
either ``axis − 90°`` ("counterclockwise"; the #1223 roof, facing 122°) or
``axis + 90°`` ("clockwise", facing 302°). Both run under the SAME sun, so the
flip happens at a different time of day for each, and each has its own
expectations in the file.

**A flip** is a minute where the commanded angle jumps by more than
``FLIP_JUMP_DEG``: the slats swing over to the other face. While they swing,
sun may get through; that is the only time it may. Per scenario and direction:

* the flips happen at the expected minutes, from and to the expected branch
  (``facing``, ``opposite``, ``far``, ``clamped``);
* in a window of minutes before, through and after each flip, the slats are on
  the right face, block the sun at every minute, and only jump at the flip;
* at every minute of the day the sun is blocked, except in the expected
  ``sun_gets_through`` windows (only the control scenario has any);
* at every minute the slats are as open as they can be on the face they chose,
  and the engine agrees with the independent checker.
"""

from __future__ import annotations

from functools import cache
import json
from pathlib import Path
from typing import NamedTuple

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
BY_ID = {sc["id"]: sc for sc in SCENARIOS}

# Offset from the slat axis to the direction the roof faces.
DIRECTIONS = {"counterclockwise": -90.0, "clockwise": +90.0}

MIN_ELEVATION_DEG = 5.0  # below this the sun is ignored (as in the generator)
FLIP_JUMP_DEG = 30.0  # a larger change in one minute is a flip
FLIP_WINDOW_MIN = 10  # minutes checked before and after each flip
STEADY_MOVE_DEG = 5.0  # outside a flip the slats move less than this a minute
MIN_MINUTES_BETWEEN_FLIPS = 30  # no back-and-forth
ANGLE_TOL = 0.05  # engine vs checker
FILE_ANGLE_TOL = 0.2  # file stores angles to one decimal
OPEN_PROBE_DEG = 0.3  # "slightly more open" for the maximally-open check
BRANCHES = {"facing", "opposite", "far", "clamped"}

CASES = [
    pytest.param(sc["id"], name, id=f"{sc['id']}-{name}")
    for sc in SCENARIOS
    for name in DIRECTIONS
]


class Minute(NamedTuple):
    """One minute of a scenario: sun, engine result and the checker's view."""

    time: str
    sun_az: float
    sun_el: float
    branch: str
    angle: float
    blocked: bool
    tan_s: float
    want_branch: str
    want_angle: float


def _facing(roof: dict, direction: str) -> float:
    return (roof["axis_azimuth"] + DIRECTIONS[direction]) % 360.0


def _limit(roof: dict) -> float:
    return oracle.travel(roof["tilt_mode"], roof["max_slat_angle"])


def _d_eff(roof: dict) -> float:
    return oracle.effective_depth(roof["slat_depth_cm"], roof["slat_spacing_cm"])


def _engine_branch(cover, angle: float, blocked: bool, limit: float) -> str:
    trace = cover._last_calc_details
    if trace["louvered_far_side_branch"]:
        return "far"
    if trace[TRACE_KEY_OPPOSITE_FACE_FALLBACK]:
        return "opposite"
    if abs(angle - limit) < 1e-6 and not blocked:
        return "clamped"
    return "facing"


@cache
def _day(scenario_id: str, direction: str) -> tuple[Minute, ...]:
    """Run the engine and the checker at every minute of the scenario's day."""
    sc = BY_ID[scenario_id]
    roof, loc = sc["roof"], sc["location"]
    facing = _facing(roof, direction)
    limit = _limit(roof)
    out = []
    for label, utc in oracle.local_minutes(
        sc["date"], sc["day"]["start"], sc["day"]["end"], loc["utc_offset_hours"]
    ):
        az, el = oracle.sun_position(loc["latitude"], loc["longitude"], utc)
        if el < MIN_ELEVATION_DEG:
            continue
        cover = build_louvered_roof_cover(
            sol_azi=az,
            sol_elev=el,
            roof_pitch=0.0,
            slat_distance=roof["slat_spacing_cm"] / 100.0,
            depth=roof["slat_depth_cm"] / 100.0,
            mode=roof["tilt_mode"],
            win_azi=facing,
            max_slat_angle=roof["max_slat_angle"],
            opposite_face_fallback=roof["opposite_face_fallback"],
            opposite_face_margin=roof["opposite_face_margin"],
        )
        angle = cover.calculate_position()
        tan_s = oracle.sun_tangent(az, el, facing)
        blocked = oracle.blocks(angle, tan_s, _d_eff(roof), roof["slat_spacing_cm"])
        want_branch, want_angle = oracle.expected_pose(
            sun_az=az,
            sun_el=el,
            facing_az=facing,
            depth=roof["slat_depth_cm"],
            spacing=roof["slat_spacing_cm"],
            limit=limit,
            fallback=roof["opposite_face_fallback"],
            margin=roof["opposite_face_margin"],
        )
        out.append(
            Minute(
                label,
                az,
                el,
                _engine_branch(cover, angle, blocked, limit),
                angle,
                blocked,
                tan_s,
                want_branch,
                want_angle,
            )
        )
    return tuple(out)


def _flip_indices(day: tuple[Minute, ...]) -> list[int]:
    return [
        i
        for i in range(1, len(day))
        if abs(day[i].angle - day[i - 1].angle) > FLIP_JUMP_DEG
    ]


def _in_windows(time: str, windows: list[dict]) -> bool:
    return any(w["from"] <= time <= w["to"] for w in windows)


def _minutes(time: str) -> int:
    h, m = time.split(":")
    return int(h) * 60 + int(m)


class TestScenarioFile:
    """The fixture itself: well-formed, and it exercises what it claims to."""

    def test_ids_are_unique(self) -> None:
        assert len(BY_ID) == len(SCENARIOS)

    @pytest.mark.parametrize("scenario", SCENARIOS, ids=list(BY_ID))
    def test_entry_is_well_formed(self, scenario: dict) -> None:
        assert scenario["description"]
        assert {"latitude", "longitude", "utc_offset_hours"} <= set(
            scenario["location"]
        )
        assert {
            "axis_azimuth",
            "slat_depth_cm",
            "slat_spacing_cm",
            "tilt_mode",
            "max_slat_angle",
            "opposite_face_fallback",
            "opposite_face_margin",
        } <= set(scenario["roof"])
        assert set(scenario["expect"]) == set(DIRECTIONS)
        for exp in scenario["expect"].values():
            for flip in exp["flips"]:
                assert {flip["from"], flip["to"]} <= BRANCHES

    def test_the_control_leaks_and_every_fallback_roof_stays_shaded(self) -> None:
        for sc in SCENARIOS:
            leaks = [w for e in sc["expect"].values() for w in e["sun_gets_through"]]
            if sc["roof"]["opposite_face_fallback"]:
                assert not leaks, sc["id"]
            else:
                assert leaks, sc["id"]

    def test_the_two_rotation_directions_flip_at_different_times(self) -> None:
        """The point of running both: the flip-over moves to another time of day."""
        differing = 0
        for sc in SCENARIOS:
            # Only fallback flips move; the far/facing flip where the sun crosses
            # the slat axis happens at the same minute whichever way the roof turns.
            ccw, cw = (
                [
                    f["time"]
                    for f in sc["expect"][d]["flips"]
                    if "opposite" in (f["from"], f["to"])
                ]
                for d in ("counterclockwise", "clockwise")
            )
            if sc["roof"]["opposite_face_fallback"] and (ccw or cw):
                assert ccw != cw, sc["id"]
                differing += 1
        assert differing >= 10


@pytest.mark.parametrize(("scenario_id", "direction"), CASES)
class TestScenarios:
    """Every scenario in the file, for both ways the slats can turn."""

    def test_slats_flip_at_the_right_time(
        self, scenario_id: str, direction: str
    ) -> None:
        day = _day(scenario_id, direction)
        got = [
            {
                "time": day[i].time,
                "from": day[i - 1].branch,
                "to": day[i].branch,
                "angle_before": day[i - 1].angle,
                "angle_after": day[i].angle,
            }
            for i in _flip_indices(day)
        ]
        want = BY_ID[scenario_id]["expect"][direction]["flips"]
        assert [(f["time"], f["from"], f["to"]) for f in got] == [
            (f["time"], f["from"], f["to"]) for f in want
        ]
        for g, w in zip(got, want, strict=True):
            assert g["angle_before"] == pytest.approx(
                w["angle_before"], abs=FILE_ANGLE_TOL
            ), w["time"]
            assert g["angle_after"] == pytest.approx(
                w["angle_after"], abs=FILE_ANGLE_TOL
            ), w["time"]

    def test_shade_before_through_and_after_each_flip(
        self, scenario_id: str, direction: str
    ) -> None:
        day = _day(scenario_id, direction)
        leaks = BY_ID[scenario_id]["expect"][direction]["sun_gets_through"]
        flips = _flip_indices(day)
        for i in flips:
            lo, hi = max(0, i - FLIP_WINDOW_MIN), min(len(day), i + FLIP_WINDOW_MIN + 1)
            for j in range(lo, hi):
                m = day[j]
                if not _in_windows(m.time, leaks):
                    assert (
                        m.blocked
                    ), f"flip at {day[i].time}: sun gets through at {m.time}"
                if j > lo and j != i:
                    step = abs(m.angle - day[j - 1].angle)
                    assert (
                        step < STEADY_MOVE_DEG
                    ), f"flip at {day[i].time}: slats jumped {step:.1f}° at {m.time}"
            # Before the flip the slats sit on one face, after it on the other.
            assert (day[i - 1].angle > 90.0) != (day[i].angle > 90.0) or (
                day[i - 1].branch != day[i].branch
            ), day[i].time
        for a, b in zip(flips, flips[1:]):
            gap = _minutes(day[b].time) - _minutes(day[a].time)
            assert (
                gap >= MIN_MINUTES_BETWEEN_FLIPS
            ), f"flipped back after {gap} min at {day[b].time}"

    def test_sun_is_blocked_all_day_except_where_expected(
        self, scenario_id: str, direction: str
    ) -> None:
        day = _day(scenario_id, direction)
        leaks = BY_ID[scenario_id]["expect"][direction]["sun_gets_through"]
        wrong = [
            f"{m.time}: {'blocked' if m.blocked else 'sun gets through'} at {m.angle:.1f}°"
            for m in day
            if m.blocked == _in_windows(m.time, leaks)
        ]
        assert not wrong, "\n".join(wrong[:20])

    def test_slats_are_maximally_open_every_minute(
        self, scenario_id: str, direction: str
    ) -> None:
        roof = BY_ID[scenario_id]["roof"]
        d_eff, spacing, limit = _d_eff(roof), roof["slat_spacing_cm"], _limit(roof)
        for m in _day(scenario_id, direction):
            where = f"{m.time} ({m.branch}, {m.angle:.2f}°)"
            if m.branch == "clamped":
                # Out of reach and no fallback: pinned at the limit, which leaks.
                assert m.angle == pytest.approx(limit), where
                assert (
                    oracle.max_open_angle("up", m.tan_s, d_eff, spacing, limit) is None
                ), where
                continue
            face = "up" if m.branch == "facing" else "down"
            best = oracle.max_open_angle(face, m.tan_s, d_eff, spacing, limit)
            assert best is not None, where
            assert m.angle == pytest.approx(best, abs=ANGLE_TOL), where
            if abs(m.angle - 90.0) > 1e-6:
                probe = m.angle + (
                    OPEN_PROBE_DEG if m.angle < 90.0 else -OPEN_PROBE_DEG
                )
                assert not oracle.blocks(probe, m.tan_s, d_eff, spacing), where

    def test_engine_agrees_with_the_independent_checker_every_minute(
        self, scenario_id: str, direction: str
    ) -> None:
        wrong = [
            f"{m.time}: engine {m.branch} {m.angle:.2f}°, "
            f"checker {m.want_branch} {m.want_angle:.2f}°"
            for m in _day(scenario_id, direction)
            if m.branch != m.want_branch or abs(m.angle - m.want_angle) > ANGLE_TOL
        ]
        assert not wrong, "\n".join(wrong[:20])


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
