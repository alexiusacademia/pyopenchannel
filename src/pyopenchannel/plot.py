"""
Plots as SVG - PyOpenChannel

Author: Alexius Academia
Date: 2026-10-24

Charts for open channel flow written as SVG, with no plotting library:
water surface profiles (with jumps and critical depth), cross-sections,
specific energy diagrams and rating curves. Each function returns a
Figure; save it with figure.save("name.svg"), view it with figure.show(),
or let Jupyter display it inline.

    import pyopenchannel as poc
    from pyopenchannel import plot

    result = poc.GVFSolver().solve_profile(...)
    plot.profile(result).save("profile.svg")
"""

import math
import os
import tempfile
import webbrowser
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple
from xml.sax.saxutils import escape

from .units import get_gravity, get_unit_system

# Colours (light theme; SVG files are viewed on white)
TEXT = "#1f2937"
MUTED = "#6b7280"
GRID = "#e5e7eb"
AXIS = "#9ca3af"
WATER_FILL = "#bfdbfe"
WATER_LINE = "#2563eb"
BED_FILL = "#e7e5e4"
BED_LINE = "#57534e"
NORMAL = "#0d9488"
CRITICAL = "#dc2626"
ENERGY = "#7c3aed"
JUMP = "#d97706"
FONT = "system-ui, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

@dataclass
class _Series:
    kind: str                      # "line", "area", "polygon", "marker", "text", "vline"
    xs: Sequence[float]
    ys: Sequence[float]
    color: str = TEXT
    width: float = 2.0
    dash: Optional[str] = None
    fill: Optional[str] = None
    opacity: float = 1.0
    label: Optional[str] = None    # legend entry
    text: Optional[str] = None
    anchor: str = "start"
    base: Optional[Sequence[float]] = None  # lower edge for "area"


@dataclass
class Figure:
    """An SVG chart with linear axes."""
    title: str = ""
    subtitle: str = ""
    x_label: str = ""
    y_label: str = ""
    width: int = 860
    height: int = 460
    equal_aspect: bool = False
    series: List[_Series] = field(default_factory=list)

    # -- adding content -----------------------------------------------------
    def line(self, xs, ys, color=TEXT, width=2.0, dash=None, label=None):
        self.series.append(_Series("line", list(xs), list(ys), color, width, dash, label=label))
        return self

    def area(self, xs, top, bottom, fill, opacity=1.0, label=None, color=None):
        self.series.append(_Series("area", list(xs), list(top), color or fill, 0, None, fill,
                                   opacity, label=label, base=list(bottom)))
        return self

    def polygon(self, xs, ys, fill, color=None, width=1.5, opacity=1.0, label=None):
        self.series.append(_Series("polygon", list(xs), list(ys), color or fill, width, None,
                                   fill, opacity, label=label))
        return self

    def marker(self, x, y, color=TEXT, label=None):
        self.series.append(_Series("marker", [x], [y], color, label=label))
        return self

    def text(self, x, y, text, color=MUTED, anchor="start"):
        self.series.append(_Series("text", [x], [y], color, text=text, anchor=anchor))
        return self

    # -- output ---------------------------------------------------------------
    def to_svg(self) -> str:
        return _render(self)

    def save(self, path: str) -> str:
        """Write the SVG file and return its path."""
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(self.to_svg())
        return path

    def show(self) -> str:
        """Write the SVG to a temporary file and open it in the browser."""
        handle, path = tempfile.mkstemp(suffix=".svg", prefix="pyopenchannel-")
        os.close(handle)
        self.save(path)
        webbrowser.open("file://" + path)
        return path

    def _repr_svg_(self) -> str:  # Jupyter
        return self.to_svg()


def _nice_step(span: float, target: int = 6) -> float:
    if span <= 0:
        return 1.0
    raw = span / target
    power = 10 ** math.floor(math.log10(raw))
    for factor in (1, 2, 2.5, 5, 10):
        if raw <= factor * power:
            return factor * power
    return 10 * power


def _ticks(lo: float, hi: float, target: int = 6) -> Tuple[float, float, List[float]]:
    """Rounded axis limits and tick values covering lo..hi."""
    if hi - lo < 1e-12:
        pad = abs(hi) * 0.1 or 1.0
        lo, hi = lo - pad, hi + pad
    step = _nice_step(hi - lo, target)
    start = math.floor(lo / step + 1e-9) * step
    end = math.ceil(hi / step - 1e-9) * step
    count = int(round((end - start) / step))
    return start, end, [start + i * step for i in range(count + 1)]


def _fmt(value: float, step: float) -> str:
    decimals = max(0, -int(math.floor(math.log10(step))) if step < 1 else 0)
    if step < 1 and abs(step * 10 ** decimals - round(step * 10 ** decimals)) > 1e-9:
        decimals += 1
    text = f"{value:.{decimals}f}"
    return "0" if text in ("-0", "-0.0", "-0.00", "-0.000") else text


def _render(fig: Figure) -> str:
    xs_all = [x for s in fig.series for x in s.xs]
    ys_all = [y for s in fig.series for y in s.ys]
    ys_all += [y for s in fig.series if s.base for y in s.base]
    if not xs_all:
        xs_all, ys_all = [0.0, 1.0], [0.0, 1.0]

    left, right, bottom = 72, 24, 56
    legend = [s for s in fig.series if s.label]
    legend_rows: List[List[Tuple[_Series, float]]] = []
    if legend:
        row: List[Tuple[_Series, float]] = []
        x = left
        for s in legend:
            item_width = 24 + 7 * len(s.label) + 22
            if row and x + item_width > fig.width - right:
                legend_rows.append(row)
                row, x = [], left
            row.append((s, x))
            x += item_width
        legend_rows.append(row)
    top = 30
    if fig.title:
        top += 34
    if fig.subtitle:
        top += 20
    top += 24 * len(legend_rows)
    plot_w = fig.width - left - right
    plot_h = fig.height - top - bottom

    x0, x1, xticks = _ticks(min(xs_all), max(xs_all))
    y0, y1, yticks = _ticks(min(ys_all), max(ys_all))
    if fig.equal_aspect:
        # Same scale on both axes (cross-sections), centred across, resting on the bed
        data_x0, data_x1 = min(xs_all), max(xs_all)
        data_y0, data_y1 = min(ys_all), max(ys_all)
        pad = 0.08 * max(data_x1 - data_x0, data_y1 - data_y0)
        scale = min(plot_w / (data_x1 - data_x0 + 2 * pad), plot_h / (data_y1 - data_y0 + 2 * pad))
        xc = (data_x0 + data_x1) / 2
        x0, x1 = xc - plot_w / scale / 2, xc + plot_w / scale / 2
        y0 = data_y0 - pad
        y1 = y0 + plot_h / scale
        step = _nice_step(y1 - y0)
        xticks = [t for t in _frange(math.ceil(x0 / step) * step, x1, step)]
        yticks = [t for t in _frange(math.ceil(y0 / step) * step, y1, step)]

    def px(x):
        return left + (x - x0) / (x1 - x0) * plot_w

    def py(y):
        return top + (y1 - y) / (y1 - y0) * plot_h

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{fig.width}" height="{fig.height}" '
        f'viewBox="0 0 {fig.width} {fig.height}" font-family="{FONT}">',
        f'<rect width="{fig.width}" height="{fig.height}" fill="#ffffff"/>',
    ]
    if fig.title:
        out.append(f'<text x="{left}" y="34" font-size="18" font-weight="600" fill="{TEXT}">{escape(fig.title)}</text>')
    if fig.subtitle:
        out.append(f'<text x="{left}" y="{56 if fig.title else 34}" font-size="13" fill="{MUTED}">{escape(fig.subtitle)}</text>')

    # Grid and tick labels
    xstep = xticks[1] - xticks[0] if len(xticks) > 1 else 1.0
    ystep = yticks[1] - yticks[0] if len(yticks) > 1 else 1.0
    for t in xticks:
        X = px(t)
        out.append(f'<line x1="{X:.1f}" y1="{top}" x2="{X:.1f}" y2="{top + plot_h}" stroke="{GRID}" stroke-width="1"/>')
        out.append(f'<text x="{X:.1f}" y="{top + plot_h + 18}" font-size="12" fill="{MUTED}" text-anchor="middle">{_fmt(t, xstep)}</text>')
    for t in yticks:
        Y = py(t)
        out.append(f'<line x1="{left}" y1="{Y:.1f}" x2="{left + plot_w}" y2="{Y:.1f}" stroke="{GRID}" stroke-width="1"/>')
        out.append(f'<text x="{left - 8}" y="{Y + 4:.1f}" font-size="12" fill="{MUTED}" text-anchor="end">{_fmt(t, ystep)}</text>')
    out.append(f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="{AXIS}" stroke-width="1"/>')
    if fig.x_label:
        out.append(f'<text x="{left + plot_w / 2:.1f}" y="{fig.height - 14}" font-size="13" fill="{TEXT}" text-anchor="middle">{escape(fig.x_label)}</text>')
    if fig.y_label:
        cy = top + plot_h / 2
        out.append(f'<text x="18" y="{cy:.1f}" font-size="13" fill="{TEXT}" text-anchor="middle" transform="rotate(-90 18 {cy:.1f})">{escape(fig.y_label)}</text>')

    # Data, clipped to the plot area
    out.append(f'<clipPath id="plot-area"><rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}"/></clipPath>')
    out.append('<g clip-path="url(#plot-area)">')
    for s in fig.series:
        if s.kind == "area":
            upper = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in zip(s.xs, s.ys))
            lower = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in zip(reversed(s.xs), reversed(s.base)))
            out.append(f'<polygon points="{upper} {lower}" fill="{s.fill}" fill-opacity="{s.opacity}" stroke="none"/>')
        elif s.kind == "polygon":
            pts = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in zip(s.xs, s.ys))
            out.append(f'<polygon points="{pts}" fill="{s.fill}" fill-opacity="{s.opacity}" stroke="{s.color}" stroke-width="{s.width}" stroke-linejoin="round"/>')
    for s in fig.series:
        if s.kind == "line":
            pts = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in zip(s.xs, s.ys))
            dash = f' stroke-dasharray="{s.dash}"' if s.dash else ""
            out.append(f'<polyline points="{pts}" fill="none" stroke="{s.color}" stroke-width="{s.width}"{dash} stroke-linejoin="round" stroke-linecap="round"/>')
    for s in fig.series:
        if s.kind == "marker":
            out.append(f'<circle cx="{px(s.xs[0]):.1f}" cy="{py(s.ys[0]):.1f}" r="4.5" fill="{s.color}" stroke="#ffffff" stroke-width="1.5"/>')
    out.append('</g>')
    for s in fig.series:
        if s.kind == "text":
            X, Y = px(s.xs[0]), py(s.ys[0])
            X = min(max(X, left + 4), left + plot_w - 4)
            out.append(f'<text x="{X:.1f}" y="{Y:.1f}" font-size="12" fill="{s.color}" text-anchor="{s.anchor}" '
                       f'paint-order="stroke" stroke="#ffffff" stroke-width="3">{escape(s.text)}</text>')

    # Legend
    for row_index, row in enumerate(legend_rows):
        Y = top - 16 - 24 * (len(legend_rows) - 1 - row_index)
        for s, X in row:
            if s.kind in ("area", "polygon"):
                out.append(f'<rect x="{X}" y="{Y - 9}" width="18" height="11" fill="{s.fill}" stroke="{s.color if s.kind == "polygon" else "none"}"/>')
            elif s.kind == "marker":
                out.append(f'<circle cx="{X + 9}" cy="{Y - 4}" r="4.5" fill="{s.color}"/>')
            else:
                dash = f' stroke-dasharray="{s.dash}"' if s.dash else ""
                out.append(f'<line x1="{X}" y1="{Y - 4}" x2="{X + 18}" y2="{Y - 4}" stroke="{s.color}" stroke-width="{max(s.width, 2)}"{dash}/>')
            out.append(f'<text x="{X + 24}" y="{Y}" font-size="12" fill="{TEXT}">{escape(s.label)}</text>')
    out.append("</svg>")
    return "\n".join(out)


def _frange(start, stop, step):
    values = []
    value = start
    while value <= stop + step * 1e-9:
        values.append(value)
        value += step
    return values


def _length_unit() -> str:
    return get_unit_system().length_unit


def _discharge_unit() -> str:
    return get_unit_system().discharge_unit


# ---------------------------------------------------------------------------
# Water surface profiles
# ---------------------------------------------------------------------------

def profile(
    result: Any,
    title: Optional[str] = None,
    show_normal_depth: bool = True,
    show_critical_depth: bool = True,
    show_energy_line: bool = False,
    datum: float = 0.0,
    width: int = 860,
    height: int = 460
) -> Figure:
    """
    Longitudinal profile: bed, water surface, normal and critical depth
    lines, a hydraulic jump and where a profile reached critical depth.

    Accepts a GVFResult, a CompositeProfile (from solve_profile_with_jump)
    or an AnalysisResult from the dam, bridge, chute or transition analyses.
    The bed is drawn at `datum` at the downstream end of the reach.
    """
    if hasattr(result, "gvf_result") and result.gvf_result is not None:   # AnalysisResult
        result = result.gvf_result
    points = sorted(result.profile_points, key=lambda p: p.x) if not hasattr(result, "jump") \
        else list(result.profile_points)
    if not points:
        raise ValueError("The result has no profile points to plot")

    if hasattr(result, "jump"):                                            # CompositeProfile
        slope, yn, yc = result.slope, result.normal_depth, result.critical_depth
        jump = result.jump
        critical_end = None
    else:                                                                  # GVFResult
        props = result.channel_properties
        slope, yn, yc = props.get("slope", 0.0), props.get("normal_depth"), props.get("critical_depth")
        jump = None
        critical_end = result.critical_depth_location if result.ended_at_critical_depth else None

    unit = _length_unit()
    xs = [p.x for p in points]
    x_max = max(xs)

    def bed(x):
        return datum + slope * (x_max - x)

    fig = Figure(
        title=title or "Water surface profile",
        x_label=f"Distance ({unit}), flow left to right",
        y_label=f"Elevation ({unit})",
        width=width, height=height,
    )
    bed_xs = [min(xs), x_max]
    surface = [bed(p.x) + p.depth for p in points]
    beds = [bed(x) for x in xs]
    depth_span = max(p.depth for p in points)
    low = min(beds) - 0.25 * depth_span

    fig.area(bed_xs, [bed(x) for x in bed_xs], [low, low], BED_FILL)
    fig.area(xs, surface, beds, WATER_FILL, opacity=0.85, label="Water")
    fig.line(bed_xs, [bed(x) for x in bed_xs], BED_LINE, 2.0, label="Bed")
    if show_normal_depth and yn:
        fig.line(bed_xs, [bed(x) + yn for x in bed_xs], NORMAL, 1.5, "7 5", label="Normal depth")
    if show_critical_depth and yc:
        fig.line(bed_xs, [bed(x) + yc for x in bed_xs], CRITICAL, 1.5, "3 4", label="Critical depth")
    if show_energy_line:
        g = get_gravity()
        fig.line(xs, [bed(p.x) + p.depth + p.velocity ** 2 / (2 * g) for p in points], ENERGY, 1.5,
                 "10 4 2 4", label="Energy line")
    fig.line(xs, surface, WATER_LINE, 2.4, label="Water surface")

    if jump is not None:
        top_y = bed(jump.x) + jump.sequent_depth
        fig.line([jump.x, jump.x], [bed(jump.x) + jump.upstream_depth, top_y], JUMP, 3.0)
        fig.marker(jump.x, top_y, JUMP, label="Hydraulic jump")
        fig.text(jump.x, top_y + 0.12 * depth_span,
                 f"Jump: {jump.upstream_depth:.2f} → {jump.sequent_depth:.2f} {unit}", JUMP, "middle")
    if critical_end is not None:
        fig.marker(critical_end, bed(critical_end) + yc, CRITICAL, label="Reaches critical depth")
    return fig


# ---------------------------------------------------------------------------
# Cross-section
# ---------------------------------------------------------------------------

def _section_height(channel, depth: float) -> float:
    if hasattr(channel, "diameter"):
        return channel.diameter
    return max(depth * 1.35, depth + 0.3)


def cross_section(
    channel: Any,
    depth: float,
    title: Optional[str] = None,
    width: int = 640,
    height: int = 460
) -> Figure:
    """Channel cross-section with the water at `depth`, drawn to scale."""
    unit = _length_unit()
    full = _section_height(channel, depth)

    def outline(level, samples=120):
        heights = [level * i / samples for i in range(samples + 1)]
        half = [channel.top_width(max(h, 1e-9)) / 2 if h > 0 else channel.top_width(1e-9) / 2
                for h in heights]
        if hasattr(channel, "diameter"):
            half[0] = 0.0
            if level >= channel.diameter - 1e-9:
                half[-1] = 0.0
        left = [(-w, h) for w, h in zip(half, heights)]
        right = [(w, h) for w, h in zip(half, heights)]
        return list(reversed(left)) + right

    shape = outline(full)
    water = outline(depth)
    top_width = channel.top_width(depth)
    area = channel.area(depth)
    radius = channel.hydraulic_radius(depth)

    fig = Figure(
        title=title or f"Cross-section at depth {depth:.3f} {unit}",
        subtitle=f"Area {area:.3f} {unit}², wetted perimeter {channel.wetted_perimeter(depth):.3f} {unit}, "
                 f"hydraulic radius {radius:.3f} {unit}, top width {top_width:.3f} {unit}",
        x_label=f"Width ({unit})", y_label=f"Height above bed ({unit})",
        width=width, height=height, equal_aspect=True,
    )
    fig.polygon([x for x, _ in water], [y for _, y in water], WATER_FILL, WATER_LINE, 0, 0.9, label="Water")
    fig.line([x for x, _ in shape], [y for _, y in shape], BED_LINE, 2.5, label="Section")
    fig.line([-top_width / 2, top_width / 2], [depth, depth], WATER_LINE, 2.0)
    fig.text(0, depth + 0.04 * full, f"T = {top_width:.3f} {unit}", WATER_LINE, "middle")
    fig.text(0, depth / 2, f"y = {depth:.3f} {unit}", TEXT, "middle")
    return fig


# ---------------------------------------------------------------------------
# Specific energy and rating curves
# ---------------------------------------------------------------------------

def specific_energy(
    channel: Any,
    discharge: float,
    depth: Optional[float] = None,
    max_depth: Optional[float] = None,
    title: Optional[str] = None,
    width: int = 640,
    height: int = 520
) -> Figure:
    """
    Specific energy diagram E = y + Q²/(2gA²) for one discharge, with
    critical depth and, if `depth` is given, that depth and its alternate.
    """
    from .hydraulics import CriticalDepth
    g = get_gravity()
    unit = _length_unit()
    yc = CriticalDepth.calculate(channel, discharge)

    def energy(y):
        return y + discharge ** 2 / (2 * g * channel.area(y) ** 2)

    top = max_depth or max(3 * yc, (depth or 0) * 1.3)
    if hasattr(channel, "diameter"):
        top = min(top, channel.diameter * 0.98)
    # Start where E is about 3 times its minimum, so the supercritical arm fits
    e_min = energy(yc)
    low = yc
    while low > yc * 1e-3 and energy(low) < 3 * e_min:
        low *= 0.9
    ys = [low + (top - low) * i / 300 for i in range(301)]
    es = [energy(y) for y in ys]

    fig = Figure(
        title=title or f"Specific energy, Q = {discharge:g} {_discharge_unit()}",
        x_label=f"Specific energy E ({unit})", y_label=f"Depth y ({unit})",
        width=width, height=height,
    )
    span = max(max(es), top)
    fig.line([0, span], [0, span], AXIS, 1.2, "6 5", label="E = y")
    fig.line(es, ys, WATER_LINE, 2.4, label="E(y)")
    fig.marker(e_min, yc, CRITICAL, label="Critical depth")
    fig.text(e_min + 0.03 * span, yc, f"yc = {yc:.3f}, Emin = {e_min:.3f}", CRITICAL)
    if depth is not None:
        from .flow_analysis import EnergyEquation
        e = energy(depth)
        sub, sup = EnergyEquation.alternate_depths(channel, discharge, e)
        other = sup if abs(depth - sub) < abs(depth - sup) else sub
        fig.line([e, e], [min(depth, other), max(depth, other)], JUMP, 1.2, "4 4")
        fig.marker(e, depth, JUMP, label="Given depth")
        fig.marker(e, other, MUTED, label="Alternate depth")
        fig.text(e + 0.03 * span, depth, f"y = {depth:.3f}", JUMP)
        fig.text(e + 0.03 * span, other, f"alternate {other:.3f}", MUTED)
    return fig


def rating_curve(
    channel: Any,
    slope: float,
    manning_n: float,
    max_depth: float,
    title: Optional[str] = None,
    width: int = 640,
    height: int = 460
) -> Figure:
    """Uniform-flow rating curve: normal depth against discharge (Manning)."""
    from .units import get_manning_factor
    unit = _length_unit()
    k = get_manning_factor()
    ys = [max_depth * i / 200 for i in range(1, 201)]
    qs = [k / manning_n * channel.area(y) * channel.hydraulic_radius(y) ** (2 / 3) * math.sqrt(slope)
          for y in ys]
    fig = Figure(
        title=title or "Rating curve (uniform flow)",
        x_label=f"Discharge ({_discharge_unit()})", y_label=f"Depth ({unit})",
        width=width, height=height,
    )
    fig.line([0.0] + qs, [0.0] + ys, WATER_LINE, 2.4, label=f"S = {slope:g}, n = {manning_n:g}")
    return fig
