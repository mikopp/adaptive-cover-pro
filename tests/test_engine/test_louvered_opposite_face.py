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
    DEFAULT_LOUVERED_OPPOSITE_FACE_MARGIN,
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
    margin: float = DEFAULT_LOUVERED_OPPOSITE_FACE_MARGIN,
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
        opposite_face_margin=margin,
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


class TestOrientationsAndRotationDirection:
    """The fix holds for any roof orientation and either slat rotation direction.

    On a flat roof the solve depends only on ``gamma`` (sun azimuth minus the
    facing azimuth), so a roof rotating the other way is the same roof with
    ``win_azi`` turned 180°. The #1223 roof is one of these; the grid below
    covers the cardinal and diagonal facings in both directions plus drives
    with a shorter and a longer travel than 125°, sweeping the whole sky.
    """

    FACINGS = [0.0, 45.0, 90.0, 122.0, 180.0, 270.0, 302.0, 350.0]
    SKY = [(azi, elev) for azi in range(0, 360, 15) for elev in range(5, 90, 10)]

    @staticmethod
    def _leaks(cover) -> bool:
        beta = _beta_deg(cover)
        theta = cover.calculate_position()
        if cover._is_far_side():
            theta = 180.0 - theta  # far-side sun is realized on the flipped face
        return not _blocks(theta, beta)

    @staticmethod
    def _build(facing: float, azi: float, elev: float, *, max_slat: float, fb: bool):
        cover = _roof(azi, elev, fallback=fb, max_slat_angle=max_slat)
        cover.config.win_azi = facing
        return cover

    @pytest.mark.parametrize("max_slat", [100.0, 125.0, 150.0])
    @pytest.mark.parametrize("facing", FACINGS)
    def test_fallback_never_leaks(self, facing: float, max_slat: float) -> None:
        for azi, elev in self.SKY:
            cover = self._build(facing, azi, elev, max_slat=max_slat, fb=True)
            if cover.cos_aoi <= 0:
                continue
            assert not self._leaks(cover), (facing, max_slat, azi, elev)

    @pytest.mark.parametrize("facing", FACINGS)
    def test_the_clamp_does_leak_without_it(self, facing: float) -> None:
        """Guards the test above against passing vacuously."""
        leaking = 0
        for azi, elev in self.SKY:
            cover = self._build(facing, azi, elev, max_slat=125.0, fb=False)
            if cover.cos_aoi > 0 and self._leaks(cover):
                leaking += 1
        assert leaking > 0

    @pytest.mark.parametrize("facing", [122.0, 0.0, 270.0])
    def test_opposite_rotation_direction_is_a_relabelling(self, facing: float) -> None:
        """Turn the roof 180° and the sun with it: identical commanded angle."""
        for azi, elev in self.SKY:
            a = self._build(facing, azi, elev, max_slat=125.0, fb=True)
            b = self._build(
                (facing + 180.0) % 360.0,
                (azi + 180.0) % 360.0,
                elev,
                max_slat=125.0,
                fb=True,
            )
            assert a.calculate_position() == pytest.approx(b.calculate_position())


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
        if expected <= MAX_SLAT - DEFAULT_LOUVERED_OPPOSITE_FACE_MARGIN:
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
    """The switch fires ``DEFAULT_LOUVERED_OPPOSITE_FACE_MARGIN`` before the rail."""

    @staticmethod
    def _facing_raw(azi: float, elev: float) -> float:
        cover = _roof(azi, elev, fallback=False, max_slat_angle=0.0)
        cover.calculate_position()
        return cover._last_calc_details["slat_angle_raw_deg"]

    def test_switches_inside_the_margin(self) -> None:
        # 12:40 CEST from the report: raw 123.9°, reachable but within 5° of 125°.
        raw = self._facing_raw(165.7, 65.4)
        assert MAX_SLAT - DEFAULT_LOUVERED_OPPOSITE_FACE_MARGIN < raw < MAX_SLAT
        cover = _roof(165.7, 65.4, fallback=True)
        assert cover.calculate_position() < TILT_HORIZONTAL_DEG
        assert cover._last_calc_details[TRACE_KEY_OPPOSITE_FACE_FALLBACK] is True

    def test_keeps_facing_side_below_the_margin(self) -> None:
        raw = self._facing_raw(150.0, 60.0)
        assert (
            TILT_HORIZONTAL_DEG
            < raw
            < (MAX_SLAT - DEFAULT_LOUVERED_OPPOSITE_FACE_MARGIN)
        )
        cover = _roof(150.0, 60.0, fallback=True)
        assert cover.calculate_position() == pytest.approx(raw)


class TestConfigurableMargin:
    """The switch margin is a parameter (default 5°)."""

    # 12:40 CEST from the report: facing-side raw angle 123.9°, 1.1° below 125°.
    AZI, ELEV = 165.7, 65.4

    def test_default_is_five_degrees(self) -> None:
        assert DEFAULT_LOUVERED_OPPOSITE_FACE_MARGIN == 5.0

    def test_margin_below_the_gap_keeps_the_facing_side(self) -> None:
        cover = _roof(self.AZI, self.ELEV, fallback=True, margin=1.0)
        assert cover.calculate_position() == pytest.approx(123.9, abs=0.1)
        assert cover._last_calc_details[TRACE_KEY_OPPOSITE_FACE_FALLBACK] is False

    def test_margin_above_the_gap_switches(self) -> None:
        cover = _roof(self.AZI, self.ELEV, fallback=True, margin=1.5)
        assert cover.calculate_position() < TILT_HORIZONTAL_DEG
        assert cover._last_calc_details[TRACE_KEY_OPPOSITE_FACE_FALLBACK] is True

    def test_zero_margin_still_switches_at_the_limit(self) -> None:
        """With margin 0 the raw angle past the max (the #1223 leak) still switches."""
        cover = _roof(187.9, 65.8, fallback=True, margin=0.0)
        result = cover.calculate_position()
        assert _blocks(result, _beta_deg(cover))
        assert cover._last_calc_details[TRACE_KEY_OPPOSITE_FACE_FALLBACK] is True

    def test_zero_margin_keeps_the_exactly_reachable_angle(self) -> None:
        """An angle at or below the max is left alone when the margin is 0."""
        cover = _roof(self.AZI, self.ELEV, fallback=True, margin=0.0)
        assert cover.calculate_position() == pytest.approx(123.9, abs=0.1)

    @pytest.mark.parametrize("margin", [0.0, 2.0, 10.0, 20.0])
    def test_never_leaks_for_any_margin(self, margin: float) -> None:
        for azi, elev in LEAK_WINDOW:
            cover = _roof(azi, elev, fallback=True, margin=margin)
            assert _blocks(cover.calculate_position(), _beta_deg(cover))


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
