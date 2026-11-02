"""Independent physics checker for the louvered-roof opposite-face tests (#1223).

Nothing here imports the engine. It answers, from first principles and for a
FLAT roof, the three questions the scenario tests ask of the engine:

* is the direct sun blocked at a given slat angle?
* what is the most open blocking angle on each face of the slats?
* which pose should the roof command, and on which branch?

Geometry. Take the cross-section perpendicular to the slat axis, with ``y`` the
horizontal direction the roof FACES (its ``win_azi``) and ``z`` up. A slat
pivoting at the roof plane has direction ``u = (cos θ, sin θ)`` for slat angle
``θ`` (0° and 180° are horizontal/closed, 90° is vertical/open). The sun
direction projected into that plane is ``(y_s, z_s)``, from the 3-D sun vector
alone: ``y_s = cos(el)·cos(az − facing)``, ``z_s = sin(el)``. Sliding a slat's
two end points along the sun direction onto a plane high above the roof gives
its shadow width ``L = D·|cos θ − sin θ·(y_s/z_s)|``. Slats repeat every
``spacing``, so the direct beam is blocked exactly when one shadow is at least
one spacing wide.

``D`` is the effective blocking chord of an interlocking louvered roof,
``d + max(0, d − s)`` (#830, measured on a real pergola). It is taken from that
issue, not derived here, so a wrong chord would be wrong in the engine and in
this checker alike.
"""

from __future__ import annotations

import math

SCAN_STEP_DEG = 0.05
BLOCK_TOL = 1e-9


def effective_depth(depth: float, spacing: float) -> float:
    """Interlock chord ``2d − s`` (nominal ``d`` when the slats do not overlap)."""
    return depth + max(0.0, depth - spacing)


def travel(tilt_mode: str, max_slat_angle: float) -> float:
    """Reachable maximum slat angle: the configured one, else the mode's."""
    if max_slat_angle:
        return float(max_slat_angle)
    return {"mode1": 90.0, "mode2": 180.0}[tilt_mode]


def sun_tangent(sun_az: float, sun_el: float, facing_az: float) -> float:
    """``y_s / z_s``: signed sideways slope of the sun seen from the slat axis."""
    y = math.cos(math.radians(sun_el)) * math.cos(math.radians(sun_az - facing_az))
    z = math.sin(math.radians(sun_el))
    return y / z


def is_near_side(sun_az: float, facing_az: float) -> bool:
    """Whether the sun is on the side the roof faces (before the slat axis)."""
    return math.cos(math.radians(sun_az - facing_az)) >= 0.0


def shadow_width(theta: float, tan_s: float, depth_eff: float) -> float:
    t = math.radians(theta)
    return depth_eff * abs(math.cos(t) - math.sin(t) * tan_s)


def blocks(theta: float, tan_s: float, depth_eff: float, spacing: float) -> bool:
    return shadow_width(theta, tan_s, depth_eff) >= spacing - BLOCK_TOL


def max_open_angle(
    face: str, tan_s: float, depth_eff: float, spacing: float, limit: float
) -> float | None:
    """Most open (closest to 90°) blocking slat angle on one face of horizontal.

    ``face`` is ``"up"`` (angles in [90°, limit]) or ``"down"`` (angles in
    [0°, 90°]). Returns ``None`` when no reachable angle on that face blocks.
    Scans away from 90° and bisects the first transition, so the answer is the
    edge of the blocking band, not a grid point.
    """
    sign = 1.0 if face == "up" else -1.0
    reach = (limit - 90.0) if face == "up" else 90.0
    if reach < 0:
        return None
    if blocks(90.0, tan_s, depth_eff, spacing):
        return 90.0
    prev = 90.0
    step = SCAN_STEP_DEG
    off = step
    while off <= reach + 1e-12:
        cur = 90.0 + sign * min(off, reach)
        if blocks(cur, tan_s, depth_eff, spacing):
            lo, hi = prev, cur  # lo leaks, hi blocks
            for _ in range(60):
                mid = (lo + hi) / 2.0
                if blocks(mid, tan_s, depth_eff, spacing):
                    hi = mid
                else:
                    lo = mid
            return hi
        prev = cur
        off += step
    return None


def expected_pose(
    *,
    sun_az: float,
    sun_el: float,
    facing_az: float,
    depth: float,
    spacing: float,
    limit: float,
    fallback: bool,
    margin: float,
) -> tuple[str, float]:
    """Return the pose a correct roof commands, as ``(branch, slat_angle)``.

    Branches: ``far`` (sun past the slat axis, closed from the far face),
    ``facing`` (most open blocking angle past vertical), ``opposite`` (the
    facing angle is out of reach, so the other face), ``clamped`` (out of reach
    and no fallback: pinned at the travel limit, which leaks).
    """
    d_eff = effective_depth(depth, spacing)
    tan_s = sun_tangent(sun_az, sun_el, facing_az)
    if not is_near_side(sun_az, facing_az):
        down = max_open_angle("down", tan_s, d_eff, spacing, limit)
        assert down is not None, "no blocking angle on the far face"
        return "far", down
    up = max_open_angle("up", tan_s, d_eff, spacing, limit)
    # Already blocking when fully open (up == 90): nothing to give up, so the
    # roof never swaps that for a more closed pose, whatever the margin says.
    self_blocking = up is not None and up <= 90.0 + 1e-9
    if fallback and not self_blocking and (up is None or up > limit - margin):
        down = max_open_angle("down", tan_s, d_eff, spacing, limit)
        assert down is not None, "no blocking angle on the opposite face"
        return "opposite", down
    if up is None:
        return "clamped", limit
    return "facing", up
