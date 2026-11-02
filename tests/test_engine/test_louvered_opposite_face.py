"""Opposite-face fallback at the louvered roof's travel limit (#1223).

A louvered roof whose drive turns partway past vertical (``max_slat_angle``
between 90° and 180°) cannot reach the facing-side blocking angle ``90° + i``
once it exceeds the maximum. The travel clamp then pins the slats MORE open
than the blocking limit and the direct beam leaks. With
``opposite_face_fallback`` on, the engine closes the slats from the other face
instead (the most open angle below 90° that still blocks, ``cutoff − 2β``).

The blocking check here is an independent 2-D slat-shadow model, not the
engine's cut-off formula: a slat of effective depth ``D`` at angle ``θ`` (from
horizontal, 90° = vertical) over pivot spacing ``s`` intercepts a facing-side
ray at profile angle ``β`` (from the roof normal) iff the slat's projection
perpendicular to the ray covers the gap's, ``D·|cos(θ + β)| ≥ s·cos β``.

The reporter's geometry (#1223): flat roof, slat axis 212° → ``win_azi`` 122°,
slat depth 22 cm, pivot spacing 20 cm, MODE2, ``max_slat_angle`` 125°.
"""

from __future__ import annotations

import math

import pytest

from custom_components.adaptive_cover_pro.const import (
    LOUVERED_OPPOSITE_FACE_SWITCH_MARGIN_DEG,
    TILT_HORIZONTAL_DEG,
)
from custom_components.adaptive_cover_pro.engine.covers.louvered_roof import (
    TRACE_KEY_OPPOSITE_FACE_FALLBACK,
)

from tests.cover_helpers import build_louvered_roof_cover

pytestmark = pytest.mark.unit

WIN_AZI = 122.0
DEPTH = 0.22
SPACING = 0.20
MAX_SLAT = 125.0
# Effective blocking chord for interlocking lamellae (2d − s, #830).
D_EFF = 2 * DEPTH - SPACING

# Summer-solstice sun positions from the #1223 report, ≈47.5° N, where the
# facing-side solve asks for 129–145°, beyond the 125° travel.
LEAK_WINDOW = [
    (176.7, 65.9),
    (187.9, 65.8),
    (198.8, 65.0),
    (208.9, 63.6),
]


def _roof(
    sol_azi: float,
    sol_elev: float,
    *,
    fallback: bool,
    max_slat_angle: float = MAX_SLAT,
    roof_pitch: float = 0.0,
    safety_margin: float = 0.0,
    mode: str = "mode2",
):
    return build_louvered_roof_cover(
        sol_azi=sol_azi,
        sol_elev=sol_elev,
        roof_pitch=roof_pitch,
        slat_distance=SPACING,
        depth=DEPTH,
        mode=mode,
        win_azi=WIN_AZI,
        fov_left=90.0,
        fov_right=90.0,
        max_slat_angle=max_slat_angle,
        safety_margin=safety_margin,
        opposite_face_fallback=fallback,
    )


def _blocks(theta_deg: float, beta_deg: float, *, tol: float = 1e-9) -> bool:
    """Independent 2-D check: does a slat at ``theta`` block a facing-side ray."""
    lhs = D_EFF * abs(math.cos(math.radians(theta_deg + beta_deg)))
    rhs = SPACING * math.cos(math.radians(beta_deg))
    return lhs + tol >= rhs


def _beta_deg(cover) -> float:
    return math.degrees(cover.beta)


class TestReportedLeak:
    """The #1223 window: clamp leaks, fallback blocks."""

    @pytest.mark.parametrize(("azi", "elev"), LEAK_WINDOW)
    def test_default_clamps_and_leaks(self, azi: float, elev: float) -> None:
        cover = _roof(azi, elev, fallback=False)
        result = cover.calculate_position()
        assert result == pytest.approx(MAX_SLAT)
        assert cover._last_calc_details["slat_angle_raw_deg"] > MAX_SLAT
        assert not _blocks(result, _beta_deg(cover))
        assert cover._last_calc_details[TRACE_KEY_OPPOSITE_FACE_FALLBACK] is False

    @pytest.mark.parametrize(("azi", "elev"), LEAK_WINDOW)
    def test_fallback_blocks_from_the_other_face(self, azi: float, elev: float) -> None:
        cover = _roof(azi, elev, fallback=True)
        result = cover.calculate_position()
        beta = _beta_deg(cover)
        assert 0.0 <= result < TILT_HORIZONTAL_DEG
        assert _blocks(result, beta)
        assert cover._last_calc_details[TRACE_KEY_OPPOSITE_FACE_FALLBACK] is True
        assert cover._last_calc_details["louvered_far_side_branch"] is False

    @pytest.mark.parametrize(("azi", "elev"), LEAK_WINDOW)
    def test_fallback_is_the_most_open_blocking_angle(
        self, azi: float, elev: float
    ) -> None:
        """One degree more open on the opposite face already leaks."""
        cover = _roof(azi, elev, fallback=True)
        result = cover.calculate_position()
        assert not _blocks(result + 1.0, _beta_deg(cover))

    def test_matches_reported_angle(self) -> None:
        """13:20 CEST: the report's 2-D model gives ≈24.5° on the far face."""
        cover = _roof(187.9, 65.8, fallback=True)
        assert cover.calculate_position() == pytest.approx(24.5, abs=0.5)

    def test_not_a_mirror_about_horizontal(self) -> None:
        """Reflecting the facing angle about 90° would leak on this face."""
        off = _roof(187.9, 65.8, fallback=False)
        off.calculate_position()
        mirrored = 180.0 - off._last_calc_details["slat_angle_raw_deg"]
        assert not _blocks(mirrored, _beta_deg(off))


class TestUnchangedWhereNotNeeded:
    """Everywhere outside the leak the fallback is a byte-for-byte no-op."""

    @pytest.mark.parametrize("azi", [90.0, 120.0, 150.0, 230.0, 260.0, 300.0])
    @pytest.mark.parametrize("elev", [15.0, 35.0, 55.0])
    def test_reachable_or_far_side_positions_are_identical(
        self, azi: float, elev: float
    ) -> None:
        off = _roof(azi, elev, fallback=False)
        on = _roof(azi, elev, fallback=True)
        expected = off.calculate_position()
        if expected <= MAX_SLAT - LOUVERED_OPPOSITE_FACE_SWITCH_MARGIN_DEG:
            assert on.calculate_position() == expected
            assert on._last_calc_details[TRACE_KEY_OPPOSITE_FACE_FALLBACK] is False

    def test_far_side_never_engages(self) -> None:
        """Afternoon sun past the slat axis already closes from the far face."""
        off = _roof(230.0, 55.0, fallback=False)
        on = _roof(230.0, 55.0, fallback=True)
        assert on._is_far_side()
        assert on.calculate_position() == off.calculate_position()
        assert on._last_calc_details[TRACE_KEY_OPPOSITE_FACE_FALLBACK] is False

    def test_vertical_pitch_keeps_venetian_anchor(self) -> None:
        off = _roof(187.9, 30.0, fallback=False, roof_pitch=90.0)
        on = _roof(187.9, 30.0, fallback=True, roof_pitch=90.0)
        assert on.calculate_position() == off.calculate_position()

    def test_default_config_is_off(self) -> None:
        cover = build_louvered_roof_cover(
            sol_azi=187.9, sol_elev=65.8, roof_pitch=0.0, max_slat_angle=MAX_SLAT
        )
        assert cover.roof_config.opposite_face_fallback is False


class TestSwitchMargin:
    """The switch fires ``LOUVERED_OPPOSITE_FACE_SWITCH_MARGIN_DEG`` before the rail."""

    @staticmethod
    def _facing_raw(azi: float, elev: float) -> float:
        cover = _roof(azi, elev, fallback=False, max_slat_angle=0.0)
        cover.calculate_position()
        return cover._last_calc_details["slat_angle_raw_deg"]

    def test_switches_inside_the_margin(self) -> None:
        # 12:40 CEST from the report: raw 123.9°, reachable but within 5° of 125°.
        raw = self._facing_raw(165.7, 65.4)
        assert MAX_SLAT - LOUVERED_OPPOSITE_FACE_SWITCH_MARGIN_DEG < raw < MAX_SLAT
        cover = _roof(165.7, 65.4, fallback=True)
        assert cover.calculate_position() < TILT_HORIZONTAL_DEG
        assert cover._last_calc_details[TRACE_KEY_OPPOSITE_FACE_FALLBACK] is True

    def test_keeps_facing_side_below_the_margin(self) -> None:
        raw = self._facing_raw(150.0, 60.0)
        assert (
            TILT_HORIZONTAL_DEG
            < raw
            < (MAX_SLAT - LOUVERED_OPPOSITE_FACE_SWITCH_MARGIN_DEG)
        )
        cover = _roof(150.0, 60.0, fallback=True)
        assert cover.calculate_position() == pytest.approx(raw)


class TestInteractions:
    """Safety margin, MODE1 travel, and the self-blocking horizontal."""

    def test_safety_margin_closes_the_opposite_face_further(self) -> None:
        plain = _roof(187.9, 65.8, fallback=True)
        margined = _roof(187.9, 65.8, fallback=True, safety_margin=1.0)
        plain_angle = plain.calculate_position()
        margined_angle = margined.calculate_position()
        assert margined_angle < plain_angle
        assert _blocks(margined_angle, _beta_deg(margined))
        # The raw trace records the unmargined opposite-face angle.
        assert margined._last_calc_details["slat_angle_raw_deg"] == pytest.approx(
            plain_angle
        )

    def test_mode1_roof_closes_instead_of_opening_fully(self) -> None:
        """A 0–90° louvered drive clamps facing-side sun to 90° (fully open)."""
        off = _roof(187.9, 65.8, fallback=False, max_slat_angle=0.0, mode="mode1")
        on = _roof(187.9, 65.8, fallback=True, max_slat_angle=0.0, mode="mode1")
        assert off.calculate_position() == pytest.approx(90.0)
        assert not _blocks(90.0, _beta_deg(off))
        result = on.calculate_position()
        assert result < TILT_HORIZONTAL_DEG
        assert _blocks(result, _beta_deg(on))

    def test_self_blocking_horizontal_is_kept(self) -> None:
        """When 90° already blocks (``i = 0``) the slats stay fully open."""
        on = _roof(150.0, 20.0, fallback=True, max_slat_angle=0.0, mode="mode1")
        off = _roof(150.0, 20.0, fallback=False, max_slat_angle=0.0, mode="mode1")
        assert off.calculate_position() == pytest.approx(90.0)
        assert on.calculate_position() == pytest.approx(90.0)
        assert on._last_calc_details[TRACE_KEY_OPPOSITE_FACE_FALLBACK] is False
