"""
Weir and gate flow checks against standard equations.

Author: Alexius Academia
Date: 2026-10-24
"""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pyopenchannel as poc
from pyopenchannel.geometry import RectangularChannel
from pyopenchannel.rvf.core import RVFSolver
from pyopenchannel.rvf.gates import FlowCondition, GateFlowSolver, GateGeometry, GateType
from pyopenchannel.rvf.weirs import WeirCondition, WeirFlowSolver, WeirGeometry, WeirType

G = 9.81
ROOT_2G = math.sqrt(2 * G)


@pytest.fixture(autouse=True)
def si_units():
    poc.set_unit_system(poc.UnitSystem.SI)


class TestWeirs:

    def setup_method(self):
        self.solver = WeirFlowSolver()

    def test_rectangular_full_width_matches_rehbock(self):
        weir = WeirGeometry(WeirType.SHARP_CRESTED_RECTANGULAR, weir_height=1.0, weir_width=2.0)
        result = self.solver.analyze_weir_flow(RectangularChannel(2.0), weir, approach_depth=1.3)
        assert result.success, result.message
        expected = (2 / 3) * (0.602 + 0.083 * 0.3) * ROOT_2G * 2.0 * 0.3 ** 1.5
        assert result.discharge == pytest.approx(expected, rel=1e-9)
        assert result.effective_length == pytest.approx(2.0)
        assert result.weir_condition == WeirCondition.FREE_FLOW

    def test_contracted_rectangular_uses_francis_length(self):
        weir = WeirGeometry(WeirType.SHARP_CRESTED_RECTANGULAR, weir_height=1.0, weir_width=1.5)
        result = self.solver.analyze_weir_flow(RectangularChannel(3.0), weir, approach_depth=1.3)
        assert result.effective_length == pytest.approx(1.5 - 0.2 * 0.3)

    def test_v_notch(self):
        weir = WeirGeometry(WeirType.SHARP_CRESTED_TRIANGULAR, weir_height=0.5, vertex_angle=90)
        result = self.solver.analyze_weir_flow(RectangularChannel(1.0), weir, approach_depth=0.75)
        # Cd for a 90 degree V-notch is about 0.58-0.60
        low = 0.58 * (8 / 15) * ROOT_2G * 0.25 ** 2.5
        high = 0.60 * (8 / 15) * ROOT_2G * 0.25 ** 2.5
        assert low <= result.discharge <= high

    def test_trapezoidal_counts_the_side_triangles_once(self):
        weir = WeirGeometry(WeirType.SHARP_CRESTED_TRAPEZOIDAL, weir_height=1.0,
                            weir_width=2.0, side_slope=0.25)
        result = self.solver.analyze_weir_flow(RectangularChannel(4.0), weir, approach_depth=1.4)
        cd = 0.602 + 0.083 * 0.4
        expected = ((2 / 3) * cd * ROOT_2G * 2.0 * 0.4 ** 1.5
                    + (8 / 15) * 0.58 * ROOT_2G * 0.25 * 0.4 ** 2.5)
        assert result.discharge == pytest.approx(expected, rel=1e-9)

    def test_ogee_matches_usbr_coefficient_with_approach_head(self):
        weir = WeirGeometry(WeirType.OGEE_SPILLWAY, weir_height=5.0, crest_length=10.0)
        result = self.solver.analyze_weir_flow(RectangularChannel(10.0), weir, approach_depth=6.0)
        total_head = 1.0 + (result.discharge / 60.0) ** 2 / (2 * G)
        # Q = C L He^1.5 with C = (2/3)(0.738)sqrt(2g) = 2.18 (SI)
        c = (2 / 3) * 0.738 * ROOT_2G
        assert c == pytest.approx(2.18, abs=0.005)
        assert result.discharge == pytest.approx(c * 10.0 * total_head ** 1.5, rel=1e-6)
        assert result.velocity_coefficient > 1.0

    def test_villemonte_submergence(self):
        weir = WeirGeometry(WeirType.SHARP_CRESTED_RECTANGULAR, weir_height=1.0, weir_width=2.0)
        channel = RectangularChannel(2.0)
        free = self.solver.analyze_weir_flow(channel, weir, approach_depth=1.3)
        submerged = self.solver.analyze_weir_flow(channel, weir, approach_depth=1.3, downstream_depth=1.15)
        factor = (1 - 0.5 ** 1.5) ** 0.385
        assert submerged.submergence_ratio == pytest.approx(0.5)
        assert submerged.submergence_factor == pytest.approx(factor)
        assert submerged.discharge == pytest.approx(free.discharge * factor)
        assert submerged.weir_condition == WeirCondition.SUBMERGED

    def test_broad_crested_free_below_modular_limit(self):
        weir = WeirGeometry(WeirType.BROAD_CRESTED, weir_height=1.0, crest_length=3.0, crest_thickness=2.0)
        channel = RectangularChannel(3.0)
        free = self.solver.analyze_weir_flow(channel, weir, approach_depth=1.5)
        partly = self.solver.analyze_weir_flow(channel, weir, approach_depth=1.5, downstream_depth=1.35)
        assert partly.submergence_ratio == pytest.approx(0.7)
        assert partly.submergence_factor == 1.0
        assert partly.discharge == pytest.approx(free.discharge)

    @pytest.mark.parametrize("weir_type, key", [
        (WeirType.SHARP_CRESTED_RECTANGULAR, "weir_length"),
        (WeirType.OGEE_SPILLWAY, "spillway_length"),
    ])
    def test_design_length_reproduces_the_discharge(self, weir_type, key):
        design = self.solver.design_weir_dimensions(weir_type, 5.0, 0.5, RectangularChannel(10.0))
        assert design["success"]
        cd = design["discharge_coefficient"]
        assert (2 / 3) * cd * ROOT_2G * design[key] * 0.5 ** 1.5 == pytest.approx(5.0)

    def test_trapezoidal_design_exists(self):
        design = self.solver.design_weir_dimensions(
            WeirType.SHARP_CRESTED_TRAPEZOIDAL, 5.0, 0.5, RectangularChannel(10.0))
        assert design["success"]
        assert design["weir_width"] > 0

    def test_invalid_method_raises(self):
        weir = WeirGeometry(WeirType.SHARP_CRESTED_RECTANGULAR, weir_height=1.0, weir_width=2.0)
        with pytest.raises(ValueError):
            self.solver.analyze_weir_flow(RectangularChannel(2.0), weir, 1.3, method="invalid")


class TestGates:

    def setup_method(self):
        self.solver = GateFlowSolver()
        self.channel = RectangularChannel(3.0)
        self.gate = GateGeometry(GateType.SLUICE, gate_width=3.0, gate_opening=0.5)
        self.upstream = 3.0
        self.jet = 0.61 * 0.5

    def test_free_flow_matches_energy_equation(self):
        result = self.solver.analyze_gate_flow(self.channel, self.gate, self.upstream)
        assert result.success, result.message
        y2 = self.jet
        expected = 3.0 * y2 * math.sqrt(2 * G * (3.0 - y2) / (1 - (y2 / 3.0) ** 2))
        assert result.flow_condition == FlowCondition.FREE_FLOW
        assert result.discharge == pytest.approx(expected)
        # Henry's chart gives Cd of about 0.58 for upstream depth = 6 x opening
        assert result.discharge_coefficient == pytest.approx(0.58, abs=0.01)

    def test_low_tailwater_is_free_flow_with_a_jump(self):
        result = self.solver.analyze_gate_flow(self.channel, self.gate, self.upstream, downstream_depth=1.5)
        assert result.flow_condition == FlowCondition.FREE_FLOW
        assert result.hydraulic_jump_required
        assert result.jump_analysis is not None

    @pytest.mark.parametrize("tailwater", [2.2, 2.6])
    def test_submerged_flow_satisfies_energy_and_momentum(self, tailwater):
        free = self.solver.analyze_gate_flow(self.channel, self.gate, self.upstream)
        result = self.solver.analyze_gate_flow(self.channel, self.gate, self.upstream,
                                               downstream_depth=tailwater)
        assert tailwater > result.properties["sequent_depth"]
        assert result.flow_condition in (FlowCondition.SUBMERGED, FlowCondition.DROWNED)
        assert 0 < result.discharge < free.discharge
        assert not result.hydraulic_jump_required

        q = result.discharge / 3.0
        y = result.properties["depth_behind_gate"]
        upstream_energy = 3.0 + (result.discharge / 9.0) ** 2 / (2 * G)
        assert upstream_energy == pytest.approx(y + q ** 2 / (2 * G * self.jet ** 2))
        assert y ** 2 / 2 + q ** 2 / (G * self.jet) == pytest.approx(
            tailwater ** 2 / 2 + q ** 2 / (G * tailwater))

    def test_more_tailwater_means_less_flow(self):
        flows = [self.solver.analyze_gate_flow(self.channel, self.gate, self.upstream,
                                               downstream_depth=t).discharge
                 for t in (2.0, 2.3, 2.6, 2.9)]
        assert all(a > b for a, b in zip(flows, flows[1:]))

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            self.solver.analyze_gate_flow(self.channel, self.gate, self.upstream, method="invalid")


def test_hydraulic_jump_matches_belanger():
    result = RVFSolver().analyze_hydraulic_jump(RectangularChannel(4.0), 12.0, 0.4)
    froude = 3.0 / 0.4 / math.sqrt(G * 0.4)
    sequent = 0.4 / 2 * (math.sqrt(1 + 8 * froude ** 2) - 1)
    assert result.downstream_depth == pytest.approx(sequent)
    assert result.energy_loss == pytest.approx((sequent - 0.4) ** 3 / (4 * 0.4 * sequent))
