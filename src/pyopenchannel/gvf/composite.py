"""
Profiles with a hydraulic jump - PyOpenChannel

Author: Alexius Academia
Date: 2026-10-24

A reach with a supercritical control upstream (for example the jet below a
sluice gate) and a subcritical control downstream (for example a weir or
tailwater) has a supercritical profile computed downstream from the first
and a subcritical profile computed upstream from the second. The flow jumps
from one to the other where their momentum functions are equal:

    M = A * y_bar + Q^2 / (g A)

so the supercritical depth and the subcritical depth there are sequent
depths. The jump is drawn as a vertical step at its toe; its length is
reported separately.
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional

from ..geometry import ChannelGeometry
from ..hydraulics import CriticalDepth, NormalDepth
from ..units import get_gravity
from .solver import BoundaryType, GVFResult, GVFSolver, ProfilePoint


@dataclass
class JumpLocation:
    """Where the jump forms and its main properties."""
    x: float                 # toe of the jump
    upstream_depth: float    # supercritical depth at the toe
    sequent_depth: float     # subcritical depth after the jump
    froude_number: float     # upstream Froude number
    energy_loss: float       # specific energy lost in the jump
    length: float            # jump length (6.1 times the sequent depth, USBR)


@dataclass
class CompositeProfile:
    """
    A water surface profile made of a supercritical part, an optional jump
    and a subcritical part. Points run in the flow direction; at a jump
    there are two points at the same x (before and after the jump).
    """
    profile_points: List[ProfilePoint]
    discharge: float
    slope: float
    manning_n: float
    critical_depth: float
    normal_depth: Optional[float]
    x_start: float
    x_end: float
    jump: Optional[JumpLocation]
    supercritical: Optional[GVFResult]
    subcritical: Optional[GVFResult]
    success: bool
    message: str
    notes: List[str] = field(default_factory=list)

    @property
    def x_values(self) -> List[float]:
        return [p.x for p in self.profile_points]

    @property
    def depths(self) -> List[float]:
        return [p.depth for p in self.profile_points]


def solve_profile_with_jump(
    channel: ChannelGeometry,
    discharge: float,
    slope: float,
    manning_n: float,
    x_start: float,
    x_end: float,
    upstream_depth: Optional[float] = None,
    downstream_depth: Optional[float] = None,
    solver: Optional[GVFSolver] = None
) -> CompositeProfile:
    """
    Water surface profile between an upstream and a downstream control,
    including the hydraulic jump between them if one forms.

    Args:
        channel: Channel geometry
        discharge: Discharge
        slope: Bed slope
        manning_n: Manning's n
        x_start, x_end: Ends of the reach (x increases downstream)
        upstream_depth: Depth at x_start. It controls the flow only if it is
            supercritical (below critical depth), e.g. the jet below a gate.
        downstream_depth: Depth at x_end. It controls the flow only if it
            is subcritical (above critical depth), e.g. a weir or tailwater.
        solver: GVF solver to use (default GVFSolver())

    Returns:
        CompositeProfile. `jump` is set when a jump forms inside the reach.
    """
    solver = solver or GVFSolver()
    critical_depth = CriticalDepth.calculate(channel, discharge)
    normal_depth = GVFSolver._normal_depth_or_none(channel, discharge, slope, manning_n)
    notes: List[str] = []

    def result(points, jump, sup, sub, message, success=True):
        return CompositeProfile(
            profile_points=sorted(points, key=lambda p: p.x) if jump is None else points,
            discharge=discharge, slope=slope, manning_n=manning_n,
            critical_depth=critical_depth, normal_depth=normal_depth,
            x_start=x_start, x_end=x_end, jump=jump,
            supercritical=sup, subcritical=sub, success=success, message=message, notes=notes)

    sup = sub = None
    if upstream_depth is not None:
        if upstream_depth < critical_depth:
            sup = solver.solve_profile(channel, discharge, slope, manning_n, x_start, x_end,
                                       upstream_depth, BoundaryType.UPSTREAM_DEPTH)
        else:
            notes.append("The upstream depth is subcritical, so it does not control the flow.")
    if downstream_depth is not None:
        if downstream_depth > critical_depth:
            sub = solver.solve_profile(channel, discharge, slope, manning_n, x_start, x_end,
                                       downstream_depth, BoundaryType.DOWNSTREAM_DEPTH)
        else:
            notes.append("The downstream depth is supercritical, so it does not control the flow.")

    for name, part in (("supercritical", sup), ("subcritical", sub)):
        if part is not None and not part.success:
            return result([], None, sup, sub, f"The {name} profile failed: {part.message}", success=False)

    if sup is None and sub is None:
        return result([], None, None, None, "Neither depth controls the flow; give a supercritical "
                      "upstream depth or a subcritical downstream depth.", success=False)
    if sub is None:
        return result(list(sup.profile_points), None, sup, None, _single_message(sup, "supercritical"))
    if sup is None:
        return result(list(sub.profile_points), None, None, sub, _single_message(sub, "subcritical"))

    # Both controls: find where the momentum functions are equal.
    sup_points = sorted(sup.profile_points, key=lambda p: p.x)
    sub_points = sorted(sub.profile_points, key=lambda p: p.x)
    lo = max(sup_points[0].x, sub_points[0].x)
    hi = min(sup_points[-1].x, sub_points[-1].x)

    def momentum(depth):
        return channel.first_moment(depth) + discharge ** 2 / (get_gravity() * channel.area(depth))

    def depth_at(points, x):
        return _interpolate(points, x)

    def excess(x):
        return momentum(depth_at(sup_points, x)) - momentum(depth_at(sub_points, x))

    if lo >= hi:
        # The profiles do not overlap; one of them ended at critical depth first.
        if sup.ended_at_critical_depth and sub.ended_at_critical_depth:
            return result([], None, sup, sub, "The supercritical and subcritical profiles both reach "
                          "critical depth before meeting; no steady profile joins them.", success=False)
        if sup_points[-1].x < sub_points[0].x:
            return result([], None, sup, sub, "The supercritical profile reaches critical depth "
                          "before the subcritical profile begins; no steady profile joins them.", success=False)

    if excess(lo) <= 0:
        notes.append("The tailwater drowns the jump at the upstream control.")
        return result(list(sub.profile_points), None, sup, sub,
                      "Subcritical throughout: the jump is drowned at the upstream end")
    if excess(hi) >= 0:
        notes.append("The supercritical flow sweeps the jump out of the reach.")
        return result(list(sup.profile_points), None, sup, sub,
                      "Supercritical throughout: the jump is swept out downstream")

    # Bisection on the interpolated profiles, then refine both depths at the
    # toe by integrating each part from its nearest computed point.
    a, b = lo, hi
    for _ in range(100):
        mid = 0.5 * (a + b)
        if excess(mid) > 0:
            a = mid
        else:
            b = mid
    x_jump = 0.5 * (a + b)
    for _ in range(3):
        y1 = _refined_depth(solver, channel, discharge, slope, manning_n, sup_points, x_jump, upstream=True)
        y2 = _refined_depth(solver, channel, discharge, slope, manning_n, sub_points, x_jump, upstream=False)
        # Move the toe by the momentum mismatch over the local momentum gradient
        h = max((hi - lo) * 1e-4, 1e-6)
        gradient = (excess(min(x_jump + h, hi)) - excess(max(x_jump - h, lo))) / (2 * h)
        mismatch = momentum(y1.depth) - momentum(y2.depth)
        if gradient == 0 or abs(mismatch) < 1e-10 * momentum(y2.depth):
            break
        x_jump = min(max(x_jump - mismatch / gradient, lo), hi)
    y1 = _refined_depth(solver, channel, discharge, slope, manning_n, sup_points, x_jump, upstream=True)
    y2 = _refined_depth(solver, channel, discharge, slope, manning_n, sub_points, x_jump, upstream=False)

    g = get_gravity()
    energy_loss = (y1.depth + y1.velocity ** 2 / (2 * g)) - (y2.depth + y2.velocity ** 2 / (2 * g))
    jump = JumpLocation(
        x=x_jump,
        upstream_depth=y1.depth,
        sequent_depth=y2.depth,
        froude_number=y1.froude_number,
        energy_loss=energy_loss,
        length=6.1 * y2.depth,
    )
    points = ([p for p in sup_points if p.x < x_jump] + [y1, y2]
              + [p for p in sub_points if p.x > x_jump])
    return result(points, jump, sup, sub,
                  f"Hydraulic jump at x = {x_jump:.2f}: {y1.depth:.3f} to {y2.depth:.3f}")


def _single_message(part: GVFResult, regime: str) -> str:
    if part.ended_at_critical_depth:
        return (f"Only the {regime} control acts; its profile reaches critical depth "
                f"at x = {part.critical_depth_location:.2f}")
    return f"Only the {regime} control acts; no jump in the reach"


def _interpolate(points: List[ProfilePoint], x: float) -> float:
    """Linear interpolation of depth in points sorted by x."""
    if x <= points[0].x:
        return points[0].depth
    if x >= points[-1].x:
        return points[-1].depth
    lo, hi = 0, len(points) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if points[mid].x <= x:
            lo = mid
        else:
            hi = mid
    p, q = points[lo], points[hi]
    t = (x - p.x) / (q.x - p.x)
    return p.depth + t * (q.depth - p.depth)


def _refined_depth(solver, channel, discharge, slope, manning_n, points, x, upstream: bool) -> ProfilePoint:
    """
    The profile point at x, integrated from the nearest computed point on the
    control side (upstream for supercritical, downstream for subcritical).
    """
    if upstream:
        start = max((p for p in points if p.x <= x), key=lambda p: p.x)
        if abs(start.x - x) < 1e-9:
            return start
        part = solver.solve_profile(channel, discharge, slope, manning_n, start.x, x,
                                    start.depth, BoundaryType.UPSTREAM_DEPTH)
    else:
        start = min((p for p in points if p.x >= x), key=lambda p: p.x)
        if abs(start.x - x) < 1e-9:
            return start
        part = solver.solve_profile(channel, discharge, slope, manning_n, x, start.x,
                                    start.depth, BoundaryType.DOWNSTREAM_DEPTH)
    return min(part.profile_points, key=lambda p: abs(p.x - x))
