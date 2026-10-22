"""
Physics checks for GVF profiles and the GVF applications.

Author: Alexius Academia
Date: 2026-10-22

These tests compare results with independent calculations (RK4 integration,
energy balances, critical depth formulas) rather than only checking that
fields exist.
"""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pyopenchannel as poc
from pyopenchannel.geometry import RectangularChannel
from pyopenchannel.gvf import BoundaryType, BridgeAnalysis, DamAnalysis, GVFSolver
from pyopenchannel.hydraulics import NormalDepth

G = 9.81


def rk4_upstream(width, discharge, slope, n, start_depth, length, step=1.0):
    """Independent RK4 integration of dy/dx upstream from x = 0 (rectangular, SI)."""
    def dydx(y):
        area = width * y
        radius = area / (width + 2 * y)
        velocity = discharge / area
        friction = (n * velocity) ** 2 / radius ** (4 / 3)
        froude2 = velocity ** 2 / (G * y)
        return (slope - friction) / (1 - froude2)

    depths = {0: start_depth}
    y, x, h = start_depth, 0.0, -step
    while -x < length:
        k1 = dydx(y)
        k2 = dydx(y + h * k1 / 2)
        k3 = dydx(y + h * k2 / 2)
        k4 = dydx(y + h * k3)
        y += h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        x += h
        depths[round(-x)] = y
    return depths


class TestProfileDirection:
    """Subcritical profiles are computed upstream from their control."""

    def setup_method(self):
        poc.set_unit_system(poc.UnitSystem.SI)
        self.solver = GVFSolver()
        self.channel = RectangularChannel(width=8.0)
        self.discharge = 40.0
        self.slope = 0.001
        self.n = 0.030
        self.yn = NormalDepth.calculate(self.channel, self.discharge, self.slope, self.n)

    def test_m1_from_downstream_control_approaches_normal_depth(self):
        control = 1.3 * self.yn
        result = self.solver.solve_profile(
            self.channel, self.discharge, self.slope, self.n,
            x_start=0.0, x_end=3000.0,
            boundary_depth=control, boundary_type=BoundaryType.DOWNSTREAM_DEPTH)
        assert result.success
        points = result.profile_points
        assert points[0].x == pytest.approx(3000.0)
        assert points[0].depth == pytest.approx(control)
        depths = [p.depth for p in points]
        # Going upstream the M1 curve falls toward normal depth and stays above it
        assert all(a >= b - 1e-9 for a, b in zip(depths, depths[1:]))
        assert depths[-1] > self.yn
        assert depths[-1] - self.yn < 0.25 * (control - self.yn)

    def test_normal_depth_boundary_is_uniform(self):
        result = self.solver.solve_profile(
            self.channel, self.discharge, self.slope, self.n,
            x_start=0.0, x_end=500.0,
            boundary_depth=0.0, boundary_type=BoundaryType.NORMAL_DEPTH)
        assert result.success
        for point in result.profile_points:
            assert point.depth == pytest.approx(self.yn, rel=1e-6)

    @pytest.mark.parametrize("bad", [
        dict(discharge=-10.0),
        dict(manning_n=-0.03),
        dict(boundary_depth=-1.0),
    ])
    def test_invalid_inputs_raise(self, bad):
        args = dict(channel=self.channel, discharge=self.discharge, slope=self.slope,
                    manning_n=self.n, x_start=0.0, x_end=100.0, boundary_depth=4.0,
                    boundary_type=BoundaryType.DOWNSTREAM_DEPTH)
        args.update(bad)
        with pytest.raises(Exception):
            self.solver.solve_profile(**args)


class TestDamBackwater:

    def setup_method(self):
        poc.set_unit_system(poc.UnitSystem.SI)
        self.width, self.discharge, self.slope, self.n = 10.0, 30.0, 0.0008, 0.025
        self.result = DamAnalysis().analyze_backwater(
            RectangularChannel(self.width), self.discharge, self.slope, self.n,
            dam_height=2.0, analysis_distance=8000.0)

    def test_profile_runs_upstream_from_the_dam(self):
        assert self.result.success
        points = self.result.gvf_result.profile_points
        params = self.result.design_parameters
        assert points[0].x == pytest.approx(0.0)
        assert points[0].depth == pytest.approx(params['design_dam_depth'])
        assert all(p.x <= 1e-9 for p in points)
        depths = [p.depth for p in points]
        assert all(a >= b - 1e-9 for a, b in zip(depths, depths[1:]))
        assert depths[-1] == pytest.approx(params['normal_depth'], rel=0.02)
        assert self.result.profile.profile_type.value == "M1"
        assert 0 < params['backwater_extent'] < 8000.0

    def test_matches_independent_integration(self):
        points = self.result.gvf_result.profile_points
        reference = rk4_upstream(self.width, self.discharge, self.slope, self.n,
                                 points[0].depth, 7000.0)
        compared = 0
        for point in points:
            distance = round(-point.x)
            if 0 < distance <= 7000:
                assert point.depth == pytest.approx(reference[distance], abs=2e-3)
                compared += 1
        assert compared >= 5


class TestBridgeEnergyMethod:

    def setup_method(self):
        poc.set_unit_system(poc.UnitSystem.SI)
        self.approach_width, self.discharge, self.slope, self.n = 8.0, 40.0, 0.001, 0.030

    def analyze(self, opening_width):
        return BridgeAnalysis().analyze_bridge_hydraulics(
            RectangularChannel(self.approach_width), RectangularChannel(opening_width),
            self.discharge, self.slope, self.n)

    @pytest.mark.parametrize("opening_width", [7.0, 5.0, 4.0])
    def test_energy_balances(self, opening_width):
        result = self.analyze(opening_width)
        assert result.success
        p = result.design_parameters
        assert not p['bridge_choked']
        y3 = p['approach_normal_depth']
        y2 = p['bridge_depth']
        y1 = p['upstream_depth']
        v3 = self.discharge / (self.approach_width * y3)
        v2 = self.discharge / (opening_width * y2)
        v1 = self.discharge / (self.approach_width * y1)
        e3 = y3 + v3 ** 2 / (2 * G)
        e2 = y2 + v2 ** 2 / (2 * G)
        e1 = y1 + v1 ** 2 / (2 * G)
        assert e2 == pytest.approx(e3 + 0.5 * (v2 ** 2 - v3 ** 2) / (2 * G), abs=1e-6)
        assert e1 == pytest.approx(e2 + 0.3 * (v2 ** 2 - v1 ** 2) / (2 * G), abs=1e-6)
        assert p['backwater_rise'] == pytest.approx(y1 - y3)
        assert p['backwater_rise'] > 0

    def test_backwater_grows_as_the_opening_narrows(self):
        rises = [self.analyze(w).design_parameters['backwater_rise'] for w in (7.0, 6.0, 5.0, 4.0)]
        assert all(a < b for a, b in zip(rises, rises[1:]))

    def test_narrow_opening_chokes_at_critical_depth(self):
        width = 3.0
        p = self.analyze(width).design_parameters
        critical = ((self.discharge / width) ** 2 / G) ** (1 / 3)
        assert p['bridge_choked']
        assert p['bridge_depth'] == pytest.approx(critical, rel=1e-3)
        assert p['froude_number_bridge'] == pytest.approx(1.0, rel=1e-3)
