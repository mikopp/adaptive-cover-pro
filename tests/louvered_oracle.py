"""Independent physics checker for the louvered-roof opposite-face tests (#1223).

Nothing here imports the engine. It answers, for a FLAT roof, the questions
the scenario tests ask of the engine:

* where is the sun at a given place and time (NOAA solar-position equations)?
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

from datetime import datetime, timedelta, timezone, UTC
import math

BLOCK_TOL = 1e-9


# ---------------------------------------------------------------------------
# Sun position (NOAA General Solar Position equations, no refraction)
# ---------------------------------------------------------------------------


def sun_position(
    latitude: float, longitude: float, when_utc: datetime
) -> tuple[float, float]:
    """Return ``(azimuth, elevation)`` in degrees; azimuth clockwise from north."""
    jd = when_utc.timestamp() / 86400.0 + 2440587.5
    jc = (jd - 2451545.0) / 36525.0
    l0 = (280.46646 + jc * (36000.76983 + jc * 0.0003032)) % 360.0
    m = 357.52911 + jc * (35999.05029 - 0.0001537 * jc)
    e = 0.016708634 - jc * (0.000042037 + 0.0000001267 * jc)
    mr = math.radians(m)
    c = (
        math.sin(mr) * (1.914602 - jc * (0.004817 + 0.000014 * jc))
        + math.sin(2 * mr) * (0.019993 - 0.000101 * jc)
        + math.sin(3 * mr) * 0.000289
    )
    true_long = l0 + c
    omega = 125.04 - 1934.136 * jc
    app_long = true_long - 0.00569 - 0.00478 * math.sin(math.radians(omega))
    obliq0 = (
        23.0
        + (26.0 + (21.448 - jc * (46.815 + jc * (0.00059 - jc * 0.001813))) / 60.0)
        / 60.0
    )
    obliq = obliq0 + 0.00256 * math.cos(math.radians(omega))
    decl = math.asin(math.sin(math.radians(obliq)) * math.sin(math.radians(app_long)))
    y = math.tan(math.radians(obliq / 2.0)) ** 2
    l0r = math.radians(l0)
    eq_time = 4.0 * math.degrees(
        y * math.sin(2 * l0r)
        - 2 * e * math.sin(mr)
        + 4 * e * y * math.sin(mr) * math.cos(2 * l0r)
        - 0.5 * y * y * math.sin(4 * l0r)
        - 1.25 * e * e * math.sin(2 * mr)
    )
    minutes = when_utc.hour * 60 + when_utc.minute + when_utc.second / 60.0
    true_solar = (minutes + eq_time + 4.0 * longitude) % 1440.0
    hour_angle = true_solar / 4.0 - 180.0
    lat = math.radians(latitude)
    ha = math.radians(hour_angle)
    cos_zen = math.sin(lat) * math.sin(decl) + math.cos(lat) * math.cos(
        decl
    ) * math.cos(ha)
    zen = math.acos(max(-1.0, min(1.0, cos_zen)))
    az = math.degrees(
        math.atan2(
            math.sin(ha),
            math.cos(ha) * math.sin(lat) - math.tan(decl) * math.cos(lat),
        )
    )
    return (az + 180.0) % 360.0, 90.0 - math.degrees(zen)


def local_minutes(date: str, start: str, end: str, utc_offset_hours: float):
    """Yield ``(label, utc_datetime)`` for every local minute in ``[start, end]``."""
    tz = timezone(timedelta(hours=utc_offset_hours))
    t = datetime.fromisoformat(f"{date}T{start}:00").replace(tzinfo=tz)
    stop = datetime.fromisoformat(f"{date}T{end}:00").replace(tzinfo=tz)
    while t <= stop:
        yield t.strftime("%H:%M"), t.astimezone(UTC)
        t += timedelta(minutes=1)


# ---------------------------------------------------------------------------
# Slat shadow
# ---------------------------------------------------------------------------


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
    """Width of one slat's shadow, projected along the sun onto a level plane."""
    t = math.radians(theta)
    return depth_eff * abs(math.cos(t) - math.sin(t) * tan_s)


def blocks(theta: float, tan_s: float, depth_eff: float, spacing: float) -> bool:
    """Whether the direct beam is blocked at slat angle ``theta``."""
    return shadow_width(theta, tan_s, depth_eff) >= spacing - BLOCK_TOL


def _band_edges(tan_s: float, depth_eff: float, spacing: float) -> list[float]:
    """Slat angles in [0°, 180°] where the shadow is exactly one spacing wide.

    ``cos θ − sin θ·tan φ = cos(θ + φ)/cos φ`` with ``φ = atan(tan_s)``, so the
    edges solve ``|cos(θ + φ)| = (spacing/D)·cos φ``.
    """
    phi = math.atan(tan_s)
    c = spacing * math.cos(phi) / depth_eff
    if c > 1.0:
        return []
    a = math.degrees(math.acos(c))
    p = math.degrees(phi)
    edges = []
    for base in (a, -a, 180.0 - a, 180.0 + a):
        for k in (-2, -1, 0, 1, 2):
            theta = base - p + 180.0 * k
            if -1e-9 <= theta <= 180.0 + 1e-9:
                edges.append(min(180.0, max(0.0, theta)))
    return edges


def max_open_angle(
    face: str, tan_s: float, depth_eff: float, spacing: float, limit: float
) -> float | None:
    """Most open (closest to 90°) blocking slat angle on one face of vertical.

    ``face`` is ``"up"`` (angles in [90°, limit]) or ``"down"`` (angles in
    [0°, 90°]). Returns ``None`` when no reachable angle on that face blocks.
    The blocking set is a union of closed intervals whose ends are band edges
    or the ends of travel, so the answer is one of those, checked by ``blocks``.
    """
    lo, hi = (90.0, limit) if face == "up" else (0.0, 90.0)
    if hi < lo:
        return None
    candidates = [90.0, lo, hi] + [
        e for e in _band_edges(tan_s, depth_eff, spacing) if lo <= e <= hi
    ]
    blocking = [
        th
        for th in candidates
        if blocks(th, tan_s, depth_eff, spacing)
        or shadow_width(th, tan_s, depth_eff) >= spacing * (1 - 1e-9)
    ]
    if not blocking:
        return None
    return min(blocking, key=lambda th: abs(th - 90.0))


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
