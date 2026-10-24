"""
Profiles with a hydraulic jump, momentum of non-rectangular sections, dense
profile output and SVG plots.

Author: Alexius Academia
Date: 2026-10-24
"""

import math
import os
import sys
from xml.etree import ElementTree

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pyopenchannel as poc
from pyopenchannel import plot
from pyopenchannel.geometry import ChannelGeometry

G = 9.81
SVG = "{http://www.w3.org/2000/svg}"


@pytest.fixture(autouse=True)
def si_units():
    poc.set_unit_system(poc.UnitSystem.SI)


# --- independent reference ---------------------------------------------------

B, Q, S, N = 5.0, 20.0, 0.0004, 0.015


def _slope(y):
    area = B * y
    radius = area / (B + 2 * y)
    velocity = Q / area
    friction = (N * velocity) ** 2 / radius ** (4 / 3)
    return (S - friction) / (1 - velocity ** 2 / (G * y))


def _rk4(y, x, x_end, h):
    depths = {round(x, 2): y}
    while (x_end - x) * h > 1e-12:
        k1 = _slope(y)
        k2 = _slope(y + h * k1 / 2)
        k3 = _slope(y + h * k2 / 2)
        k4 = _slope(y + h * k3)
        y += h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        x += h
        depths[round(x, 2)] = y
        if abs(1 - (Q / (B * y)) / math.sqrt(G * y)) < 0.01:
            break
    return depths


def _momentum(y):
    return B * y * y / 2 + Q * Q / (G * B * y)


def _independent_jump(tailwater, length=500.0):
    supercritical = _rk4(0.35, 0.0, length, 0.01)
    subcritical = _rk4(tailwater, length, 0.0, -0.01)
    for x in sorted(supercritical):
        if x in subcritical and _momentum(supercritical[x]) <= _momentum(subcritical[x]):
            return x, supercritical[x], subcritical[x]
    return None


# --- jumps -------------------------------------------------------------------

class TestProfileWithJump:

    @pytest.mark.parametrize("tailwater", [2.75, 2.5])
    def test_jump_matches_independent_calculation(self, tailwater):
        result = poc.solve_profile_with_jump(
            poc.RectangularChannel(B), Q, S, N, 0.0, 500.0,
            upstream_depth=0.35, downstream_depth=tailwater)
        assert result.success
        jump = result.jump
        x, y1, y2 = _independent_jump(tailwater)
        assert jump.x == pytest.approx(x, abs=0.05)
        assert jump.upstream_depth == pytest.approx(y1, abs=1e-3)
        assert jump.sequent_depth == pytest.approx(y2, abs=1e-3)
        # Sequent depths: equal momentum, and Belanger for a rectangle
        assert _momentum(jump.upstream_depth) == pytest.approx(_momentum(jump.sequent_depth), rel=1e-4)
        froude = jump.froude_number
        belanger = jump.upstream_depth / 2 * (math.sqrt(1 + 8 * froude ** 2) - 1)
        assert jump.sequent_depth == pytest.approx(belanger, rel=1e-3)
        assert jump.energy_loss == pytest.approx(
            (jump.sequent_depth - jump.upstream_depth) ** 3 / (4 * jump.upstream_depth * jump.sequent_depth),
            rel=1e-3)

    def test_profile_points_run_through_the_jump(self):
        result = poc.solve_profile_with_jump(
            poc.RectangularChannel(B), Q, S, N, 0.0, 500.0, upstream_depth=0.35, downstream_depth=2.5)
        points = result.profile_points
        assert len(points) > 300
        assert all(a.x <= b.x for a, b in zip(points, points[1:]))
        assert points[0].x == pytest.approx(0.0) and points[0].depth == pytest.approx(0.35)
        assert points[-1].x == pytest.approx(500.0) and points[-1].depth == pytest.approx(2.5)
        before = [p for p in points if p.x < result.jump.x]
        assert all(p.froude_number > 1 for p in before)

    def test_high_tailwater_drowns_the_jump(self):
        result = poc.solve_profile_with_jump(
            poc.RectangularChannel(B), Q, S, N, 0.0, 500.0, upstream_depth=0.35, downstream_depth=4.5)
        assert result.success
        assert result.jump is None
        assert "drowned" in result.message
        assert all(p.depth > result.critical_depth for p in result.profile_points)

    def test_short_reach_sweeps_the_jump_out(self):
        result = poc.solve_profile_with_jump(
            poc.RectangularChannel(B), Q, S, N, 0.0, 15.0, upstream_depth=0.35, downstream_depth=2.5)
        assert result.success
        assert result.jump is None
        assert "swept out" in result.message
        assert all(p.depth < result.critical_depth for p in result.profile_points)

    def test_one_control_only(self):
        result = poc.solve_profile_with_jump(
            poc.RectangularChannel(B), Q, S, N, 0.0, 500.0, downstream_depth=3.0)
        assert result.success and result.jump is None and result.supercritical is None
        no_control = poc.solve_profile_with_jump(
            poc.RectangularChannel(B), Q, S, N, 0.0, 500.0, upstream_depth=2.0)
        assert not no_control.success


# --- momentum of non-rectangular sections --------------------------------------

class TestMomentum:

    @pytest.mark.parametrize("channel, depth", [
        (poc.TrapezoidalChannel(3.0, 1.5), 1.2),
        (poc.TriangularChannel(2.0), 1.2),
        (poc.CircularChannel(2.0), 0.7),
        (poc.ParabolicChannel(0.5), 1.2),
    ])
    def test_first_moment_matches_integration(self, channel, depth):
        steps = 20000
        h = depth / steps
        reference = sum((depth - (i + 0.5) * h) * channel.top_width((i + 0.5) * h) for i in range(steps)) * h
        assert channel.first_moment(depth) == pytest.approx(reference, rel=1e-3)
        assert ChannelGeometry.first_moment(channel, depth) == pytest.approx(reference, rel=1e-3)

    def test_trapezoidal_sequent_depth(self):
        channel = poc.TrapezoidalChannel(3.0, 1.5)
        discharge, y1 = 20.0, 0.4

        def momentum(y):
            return 3.0 * y * y / 2 + 1.5 * y ** 3 / 3 + discharge ** 2 / (G * (3.0 + 1.5 * y) * y)

        lo, hi = 1.0, 5.0
        for _ in range(200):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if momentum(mid) < momentum(y1) else (lo, mid)
        assert poc.MomentumEquation.conjugate_depths(channel, discharge, y1) == pytest.approx(lo, rel=1e-4)
        jump = poc.RVFSolver().analyze_hydraulic_jump(channel, discharge, y1)
        assert jump.downstream_depth == pytest.approx(lo, rel=1e-4)


# --- dense output ---------------------------------------------------------------

def test_profiles_have_dense_points():
    result = poc.GVFSolver().solve_profile(
        poc.RectangularChannel(B), Q, S, N, 0.0, 500.0, 3.0, poc.BoundaryType.DOWNSTREAM_DEPTH)
    points = sorted(result.profile_points, key=lambda p: p.x)
    assert len(points) >= 201
    assert max(b.x - a.x for a, b in zip(points, points[1:])) <= 500.0 / 200 + 1e-9
    assert result.channel_properties["normal_depth"] == pytest.approx(
        poc.NormalDepth.calculate(poc.RectangularChannel(B), Q, S, N))


# --- plots ----------------------------------------------------------------------

def _svg(figure):
    root = ElementTree.fromstring(figure.to_svg())
    assert root.tag == SVG + "svg"
    texts = " ".join(t.text or "" for t in root.iter(SVG + "text"))
    return root, texts


class TestPlots:

    def test_profile_with_jump(self):
        result = poc.solve_profile_with_jump(
            poc.RectangularChannel(B), Q, S, N, 0.0, 500.0, upstream_depth=0.35, downstream_depth=2.5)
        root, texts = _svg(plot.profile(result, title="Gate and jump"))
        assert "Gate and jump" in texts
        for label in ("Water surface", "Normal depth", "Critical depth", "Hydraulic jump", "Jump:"):
            assert label in texts
        assert root.find(f".//{SVG}polyline") is not None

    def test_profile_from_gvf_result_marks_critical_depth(self):
        result = poc.GVFSolver().solve_profile(
            poc.RectangularChannel(3.0), 10.0, 0.02, 0.015, -200.0, 0.0, 1.5,
            poc.BoundaryType.DOWNSTREAM_DEPTH)
        _, texts = _svg(plot.profile(result, show_energy_line=True))
        assert "Reaches critical depth" in texts
        assert "Energy line" in texts

    def test_profile_from_dam_analysis(self):
        analysis = poc.DamAnalysis().analyze_backwater(
            poc.RectangularChannel(10.0), 30.0, 0.0008, 0.025, 2.0, analysis_distance=8000.0)
        _, texts = _svg(plot.profile(analysis))
        assert "Water surface profile" in texts

    def test_cross_sections_and_curves(self):
        for figure in (
            plot.cross_section(poc.TrapezoidalChannel(3.0, 1.5), 1.783),
            plot.cross_section(poc.CircularChannel(1.2), 0.57),
            plot.specific_energy(poc.RectangularChannel(5.0), 20.0, depth=2.5),
            plot.rating_curve(poc.TrapezoidalChannel(3.0, 1.5), 0.0008, 0.025, 3.0),
        ):
            _svg(figure)
        _, texts = _svg(plot.cross_section(poc.TrapezoidalChannel(3.0, 1.5), 1.783))
        assert "Area 10.118" in texts and "top width 8.349" in texts

    def test_us_units_label_axes_in_feet(self):
        poc.set_unit_system(poc.UnitSystem.US_CUSTOMARY)
        _, texts = _svg(plot.rating_curve(poc.RectangularChannel(10.0), 0.001, 0.015, 4.0))
        assert "Depth (ft)" in texts and "ft³/s" in texts

    def test_save(self, tmp_path):
        path = plot.rating_curve(poc.RectangularChannel(10.0), 0.001, 0.015, 4.0).save(str(tmp_path / "r.svg"))
        assert open(path, encoding="utf-8").read().startswith("<svg")

    def test_nice_ticks(self):
        lo, hi, ticks = plot._ticks(0.13, 4.87)
        assert lo == 0 and hi == 5
        assert ticks == pytest.approx([0, 1, 2, 3, 4, 5])
        _, _, ticks = plot._ticks(-8000, 0)
        assert ticks[1] - ticks[0] in (1000, 2000)
