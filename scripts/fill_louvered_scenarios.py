#!/usr/bin/env python3
"""Fill in the ``expect`` block of louvered-roof scenarios from the physics checker.

Usage::

    python scripts/fill_louvered_scenarios.py            # only entries without "expect"
    python scripts/fill_louvered_scenarios.py --all      # recompute every entry

Reads and rewrites ``tests/fixtures/louvered_opposite_face_scenarios.json``. To
add a scenario, append an entry with ``id``, ``description``, ``location``,
``date``, ``day`` and ``roof`` (see ``tests/test_engine/test_louvered_opposite_face.py``),
run this, and review the flips and ``sun_gets_through`` windows it writes before
committing: they come from ``tests/louvered_oracle.py``, not from the engine, and
are what the tests then hold the engine to.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests import louvered_oracle as oracle  # noqa: E402

FILE = ROOT / "tests" / "fixtures" / "louvered_opposite_face_scenarios.json"
DIRECTIONS = {"counterclockwise": -90.0, "clockwise": 90.0}
MIN_ELEVATION_DEG = 5.0  # keep in step with the test module
FLIP_JUMP_DEG = 30.0


def _timeline(sc: dict, direction: str) -> list[tuple[str, str, float, bool]]:
    roof, loc = sc["roof"], sc["location"]
    facing = (roof["axis_azimuth"] + DIRECTIONS[direction]) % 360.0
    d_eff = oracle.effective_depth(roof["slat_depth_cm"], roof["slat_spacing_cm"])
    out = []
    for label, utc in oracle.local_minutes(
        sc["date"], sc["day"]["start"], sc["day"]["end"], loc["utc_offset_hours"]
    ):
        az, el = oracle.sun_position(loc["latitude"], loc["longitude"], utc)
        if el < MIN_ELEVATION_DEG:
            continue
        branch, angle = oracle.expected_pose(
            sun_az=az,
            sun_el=el,
            facing_az=facing,
            depth=roof["slat_depth_cm"],
            spacing=roof["slat_spacing_cm"],
            limit=oracle.travel(roof["tilt_mode"], roof["max_slat_angle"]),
            fallback=roof["opposite_face_fallback"],
            margin=roof["opposite_face_margin"],
        )
        blocked = oracle.blocks(
            angle,
            oracle.sun_tangent(az, el, facing),
            d_eff,
            roof["slat_spacing_cm"],
        )
        out.append((label, branch, angle, blocked))
    return out


def expectations(sc: dict, direction: str) -> dict:
    """Return the flips and sun-gets-through windows for one rotation direction."""
    tl = _timeline(sc, direction)
    flips, leaks, open_leak = [], [], None
    for i, (label, branch, angle, blocked) in enumerate(tl):
        if i and abs(angle - tl[i - 1][2]) > FLIP_JUMP_DEG:
            flips.append(
                {
                    "time": label,
                    "from": tl[i - 1][1],
                    "to": branch,
                    "angle_before": round(tl[i - 1][2], 1),
                    "angle_after": round(angle, 1),
                }
            )
        if not blocked and open_leak is None:
            open_leak = {"from": label}
        if blocked and open_leak is not None:
            open_leak["to"] = tl[i - 1][0]
            leaks.append(open_leak)
            open_leak = None
    if open_leak is not None:
        open_leak["to"] = tl[-1][0]
        leaks.append(open_leak)
    return {"flips": flips, "sun_gets_through": leaks}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--all", action="store_true", help="recompute every entry")
    args = parser.parse_args()
    scenarios = json.loads(FILE.read_text())
    for sc in scenarios:
        if args.all or "expect" not in sc:
            sc["expect"] = {d: expectations(sc, d) for d in DIRECTIONS}
            for d, exp in sc["expect"].items():
                flips = ", ".join(
                    f"{f['time']} {f['from']}->{f['to']}" for f in exp["flips"]
                )
                print(
                    f"{sc['id']} [{d}]: flips: {flips or 'none'}; "
                    f"sun gets through: {exp['sun_gets_through'] or 'never'}"
                )
    FILE.write_text(json.dumps(scenarios, indent=2) + "\n")


if __name__ == "__main__":
    main()
