"""
Gate Flow Analysis - PyOpenChannel

Author: Alexius Academia
Email: alexius.sayco.academia@gmail.com

This module provides comprehensive analysis of flow under gates including:
- Sluice gates (vertical lift gates)
- Radial gates (Tainter gates)
- Free flow and submerged flow conditions
- Discharge coefficient calculations
- Vena contracta effects
- Downstream hydraulic jump analysis
- Cavitation risk assessment

The module handles both design (given flow, find gate opening) and 
analysis (given gate opening, find flow) problems.
"""

import math
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum

from ..geometry import ChannelGeometry, RectangularChannel, TrapezoidalChannel
from ..hydraulics import ManningEquation
from ..flow_analysis import CriticalFlow, EnergyEquation
from ..validators import validate_discharge, validate_depth, validate_positive
from ..exceptions import ConvergenceError, InvalidFlowConditionError
from ..units import get_gravity
from .core import RVFSolver, RVFResult, TransitionType


class GateType(Enum):
    """Types of gates for flow analysis."""
    SLUICE = "sluice"              # Vertical lift gate
    RADIAL = "radial"              # Tainter gate (curved)
    ROLLER = "roller"              # Roller gate
    FLAP = "flap"                  # Flap gate
    STOP_LOG = "stop_log"          # Stop log gate


class FlowCondition(Enum):
    """Flow conditions under gates."""
    FREE_FLOW = "free_flow"        # No downstream submergence
    SUBMERGED = "submerged"        # Downstream submergence affects flow
    DROWNED = "drowned"            # Heavily submerged, reduced discharge


class CavitationRisk(Enum):
    """Cavitation risk levels."""
    LOW = "low"                    # Cavitation index > 0.25
    MODERATE = "moderate"          # Cavitation index 0.1 - 0.25
    HIGH = "high"                  # Cavitation index < 0.1


@dataclass
class GateGeometry:
    """Gate geometry parameters."""
    gate_type: GateType
    gate_width: float              # Gate width (m or ft)
    gate_opening: float            # Gate opening height (m or ft)
    gate_height: Optional[float] = None    # Total gate height
    radius: Optional[float] = None         # Radius for radial gates
    angle: Optional[float] = None          # Gate angle for radial gates
    sill_elevation: float = 0.0            # Sill elevation above channel bottom


@dataclass
class GateFlowResult:
    """
    Result of gate flow analysis.
    
    Contains all computed hydraulic parameters for flow under gates.
    """
    # Gate properties
    gate_type: GateType
    gate_opening: float
    gate_width: float
    
    # Flow conditions
    discharge: float
    upstream_depth: float
    downstream_depth: float
    contracted_depth: float        # Depth at vena contracta
    
    # Hydraulic parameters
    upstream_velocity: float
    gate_velocity: float           # Velocity through gate opening
    downstream_velocity: float
    contraction_coefficient: float # Cc
    discharge_coefficient: float   # Cd
    velocity_coefficient: float    # Cv
    
    # Energy analysis
    upstream_energy: float
    gate_energy: float
    downstream_energy: float
    energy_loss: float
    
    # Flow classification
    flow_condition: FlowCondition
    froude_upstream: float
    froude_gate: float
    froude_downstream: float
    
    # Downstream conditions
    hydraulic_jump_required: bool
    cavitation_risk: CavitationRisk
    cavitation_index: float
    
    # Analysis metadata
    success: bool
    message: str
    properties: Dict[str, Any]
    
    # Optional fields (must come last)
    jump_analysis: Optional[RVFResult] = None


class GateFlowSolver:
    """
    Comprehensive gate flow analysis solver.
    
    This solver handles flow under various gate types using:
    - Energy equations for free flow
    - Momentum equations for submerged flow
    - Empirical coefficients for discharge calculations
    - Vena contracta analysis
    - Downstream hydraulic jump prediction
    """
    
    def __init__(
        self,
        convergence_tolerance: float = 1e-6,
        max_iterations: int = 100
    ):
        """
        Initialize gate flow solver.
        
        Args:
            convergence_tolerance: Convergence tolerance for iterative solutions
            max_iterations: Maximum iterations for convergence
        """
        self.tolerance = convergence_tolerance
        self.max_iterations = max_iterations
        self.gravity = get_gravity()
        self.rvf_solver = RVFSolver()
    
    def analyze_gate_flow(
        self,
        channel: ChannelGeometry,
        gate: GateGeometry,
        upstream_depth: float,
        downstream_depth: Optional[float] = None,
        discharge: Optional[float] = None
    ) -> GateFlowResult:
        """
        Analyze flow under a gate.
        
        Args:
            channel: Channel geometry
            gate: Gate geometry and properties
            upstream_depth: Upstream water depth
            downstream_depth: Downstream water depth (optional)
            discharge: Known discharge (optional, will be calculated if not provided)
            
        Returns:
            GateFlowResult with complete analysis
        """
        try:
            # Validate inputs
            upstream_depth = validate_depth(upstream_depth)
            if downstream_depth is not None:
                downstream_depth = validate_depth(downstream_depth)
            if discharge is not None:
                discharge = validate_discharge(discharge)
            
            # Calculate discharge if not provided
            if discharge is None:
                discharge = self._calculate_gate_discharge(
                    channel, gate, upstream_depth, downstream_depth
                )
            
            # Determine flow condition
            flow_condition = self._classify_flow_condition(
                gate, upstream_depth, downstream_depth
            )
            
            # Calculate hydraulic parameters
            hydraulic_params = self._calculate_hydraulic_parameters(
                channel, gate, discharge, upstream_depth, downstream_depth, flow_condition
            )
            
            # Analyze downstream conditions
            downstream_analysis = self._analyze_downstream_conditions(
                channel, discharge, hydraulic_params["contracted_depth"],
                hydraulic_params["gate_velocity"], downstream_depth
            )
            
            # Assess cavitation risk
            cavitation_analysis = self._assess_cavitation_risk(
                upstream_depth, hydraulic_params["contracted_depth"],
                hydraulic_params["gate_velocity"]
            )
            
            # Calculate energy parameters
            energy_analysis = self._calculate_energy_analysis(
                channel, discharge, upstream_depth, 
                hydraulic_params["contracted_depth"], downstream_depth
            )
            
            return GateFlowResult(
                gate_type=gate.gate_type,
                gate_opening=gate.gate_opening,
                gate_width=gate.gate_width,
                discharge=discharge,
                upstream_depth=upstream_depth,
                downstream_depth=downstream_depth or downstream_analysis["depth"],
                contracted_depth=hydraulic_params["contracted_depth"],
                upstream_velocity=hydraulic_params["upstream_velocity"],
                gate_velocity=hydraulic_params["gate_velocity"],
                downstream_velocity=downstream_analysis["velocity"],
                contraction_coefficient=hydraulic_params["contraction_coefficient"],
                discharge_coefficient=hydraulic_params["discharge_coefficient"],
                velocity_coefficient=hydraulic_params["velocity_coefficient"],
                upstream_energy=energy_analysis["upstream_energy"],
                gate_energy=energy_analysis["gate_energy"],
                downstream_energy=energy_analysis["downstream_energy"],
                energy_loss=energy_analysis["energy_loss"],
                flow_condition=flow_condition,
                froude_upstream=hydraulic_params["froude_upstream"],
                froude_gate=hydraulic_params["froude_gate"],
                froude_downstream=downstream_analysis["froude_number"],
                hydraulic_jump_required=downstream_analysis["jump_required"],
                jump_analysis=downstream_analysis.get("jump_analysis"),
                cavitation_risk=cavitation_analysis["risk_level"],
                cavitation_index=cavitation_analysis["cavitation_index"],
                success=True,
                message=f"Gate flow analysis complete. Flow condition: {flow_condition.value}",
                properties={
                    "submergence_ratio": downstream_analysis.get("submergence_ratio", 0),
                    "energy_efficiency": energy_analysis["efficiency"],
                    "gate_coefficient_source": hydraulic_params.get("coefficient_source", "empirical")
                }
            )
            
        except Exception as e:
            return GateFlowResult(
                gate_type=gate.gate_type,
                gate_opening=gate.gate_opening,
                gate_width=gate.gate_width,
                discharge=discharge or 0,
                upstream_depth=upstream_depth,
                downstream_depth=downstream_depth or 0,
                contracted_depth=0,
                upstream_velocity=0,
                gate_velocity=0,
                downstream_velocity=0,
                contraction_coefficient=0,
                discharge_coefficient=0,
                velocity_coefficient=0,
                upstream_energy=0,
                gate_energy=0,
                downstream_energy=0,
                energy_loss=0,
                flow_condition=FlowCondition.FREE_FLOW,
                froude_upstream=0,
                froude_gate=0,
                froude_downstream=0,
                hydraulic_jump_required=False,
                cavitation_risk=CavitationRisk.LOW,
                cavitation_index=0,
                success=False,
                message=f"Gate flow analysis failed: {str(e)}",
                properties={"error": str(e)}
            )
    
    def design_gate_opening(
        self,
        channel: ChannelGeometry,
        gate_type: GateType,
        gate_width: float,
        discharge: float,
        upstream_depth: float,
        target_downstream_depth: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Design gate opening for specified discharge and conditions.
        
        Args:
            channel: Channel geometry
            gate_type: Type of gate
            gate_width: Gate width
            discharge: Required discharge
            upstream_depth: Upstream water depth
            target_downstream_depth: Target downstream depth (optional)
            
        Returns:
            Dictionary with design results
        """
        try:
            # Initial guess for gate opening
            # Start with critical depth as rough estimate
            critical_depth = CriticalFlow(channel).calculate_critical_depth(discharge)
            gate_opening = critical_depth * 0.8  # Start with 80% of critical depth
            
            for iteration in range(self.max_iterations):
                # Create gate geometry
                gate = GateGeometry(
                    gate_type=gate_type,
                    gate_width=gate_width,
                    gate_opening=gate_opening
                )
                
                # Analyze flow with current gate opening
                result = self.analyze_gate_flow(
                    channel, gate, upstream_depth, target_downstream_depth
                )
                
                if not result.success:
                    continue
                
                # Check discharge convergence
                discharge_error = abs(result.discharge - discharge) / discharge
                
                if discharge_error < self.tolerance:
                    return {
                        "success": True,
                        "gate_opening": gate_opening,
                        "discharge_achieved": result.discharge,
                        "discharge_error": discharge_error,
                        "flow_analysis": result,
                        "iterations": iteration + 1
                    }
                
                # Update gate opening using Newton-Raphson approach
                # Approximate derivative
                delta_opening = gate_opening * 0.01
                gate_test = GateGeometry(
                    gate_type=gate_type,
                    gate_width=gate_width,
                    gate_opening=gate_opening + delta_opening
                )
                
                result_test = self.analyze_gate_flow(
                    channel, gate_test, upstream_depth, target_downstream_depth
                )
                
                if result_test.success and result_test.discharge != result.discharge:
                    # Calculate derivative dQ/da (discharge vs gate opening)
                    dQ_da = (result_test.discharge - result.discharge) / delta_opening
                    
                    # Newton-Raphson update
                    gate_opening_new = gate_opening + (discharge - result.discharge) / dQ_da
                    
                    # Ensure positive opening
                    if gate_opening_new > 0:
                        gate_opening = gate_opening_new
                    else:
                        gate_opening *= 1.1  # Increase by 10%
                else:
                    # Simple adjustment if derivative calculation fails
                    if result.discharge < discharge:
                        gate_opening *= 1.1  # Increase opening
                    else:
                        gate_opening *= 0.9  # Decrease opening
            
            return {
                "success": False,
                "message": "Gate opening design did not converge",
                "gate_opening": gate_opening,
                "discharge_achieved": result.discharge if 'result' in locals() else 0,
                "iterations": self.max_iterations
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Gate design failed: {str(e)}",
                "error": str(e)
            }
    
    def _calculate_gate_discharge(
        self,
        channel: ChannelGeometry,
        gate: GateGeometry,
        upstream_depth: float,
        downstream_depth: Optional[float]
    ) -> float:
        """Calculate discharge through gate opening."""
        
        # Determine discharge coefficient based on gate type and conditions
        Cd = self._get_discharge_coefficient(gate, upstream_depth, downstream_depth)
        
        # Effective gate area
        effective_area = gate.gate_width * gate.gate_opening
        
        # Calculate driving head
        if gate.gate_type == GateType.SLUICE:
            # For sluice gates, use upstream depth minus gate opening
            driving_head = upstream_depth - gate.gate_opening / 2
        else:
            # For other gates, use full upstream depth
            driving_head = upstream_depth
        
        # Ensure positive driving head
        driving_head = max(driving_head, 0.1)
        
        # Calculate discharge using orifice equation
        discharge = Cd * effective_area * math.sqrt(2 * self.gravity * driving_head)
        
        # Apply submergence correction if applicable
        if downstream_depth is not None and downstream_depth > gate.gate_opening:
            submergence_ratio = downstream_depth / upstream_depth
            if submergence_ratio > 0.7:  # Significant submergence
                submergence_factor = 1 - 0.5 * (submergence_ratio - 0.7) / 0.3
                discharge *= submergence_factor
        
        return discharge
    
    def _get_discharge_coefficient(
        self,
        gate: GateGeometry,
        upstream_depth: float,
        downstream_depth: Optional[float]
    ) -> float:
        """Get discharge coefficient based on gate type and conditions."""
        
        if gate.gate_type == GateType.SLUICE:
            # Sluice gate coefficients
            opening_ratio = gate.gate_opening / upstream_depth
            
            if opening_ratio < 0.2:
                return 0.61  # Sharp-edged orifice
            elif opening_ratio < 0.5:
                return 0.65  # Partially contracted
            else:
                return 0.70  # Fully contracted
                
        elif gate.gate_type == GateType.RADIAL:
            # Radial gate coefficients (typically higher)
            return 0.75
            
        elif gate.gate_type == GateType.ROLLER:
            # Roller gate coefficients
            return 0.68
            
        else:
            # Default coefficient for other gate types
            return 0.62
    
    def _classify_flow_condition(
        self,
        gate: GateGeometry,
        upstream_depth: float,
        downstream_depth: Optional[float]
    ) -> FlowCondition:
        """Classify flow condition under gate."""
        
        if downstream_depth is None:
            return FlowCondition.FREE_FLOW
        
        # Calculate submergence ratio
        submergence_ratio = downstream_depth / gate.gate_opening
        
        if submergence_ratio < 0.7:
            return FlowCondition.FREE_FLOW
        elif submergence_ratio < 1.5:
            return FlowCondition.SUBMERGED
        else:
            return FlowCondition.DROWNED
    
    def _calculate_hydraulic_parameters(
        self,
        channel: ChannelGeometry,
        gate: GateGeometry,
        discharge: float,
        upstream_depth: float,
        downstream_depth: Optional[float],
        flow_condition: FlowCondition
    ) -> Dict[str, Any]:
        """Calculate hydraulic parameters for gate flow."""
        
        # Upstream conditions
        upstream_area = channel.area(upstream_depth)
        upstream_velocity = discharge / upstream_area
        froude_upstream = upstream_velocity / math.sqrt(self.gravity * upstream_depth)
        
        # Gate conditions
        gate_area = gate.gate_width * gate.gate_opening
        gate_velocity = discharge / gate_area
        
        # Contraction coefficient (Cc)
        Cc = self._get_contraction_coefficient(gate, upstream_depth)
        
        # Contracted depth (vena contracta)
        contracted_depth = Cc * gate.gate_opening
        
        # Froude number at gate
        froude_gate = gate_velocity / math.sqrt(self.gravity * contracted_depth)
        
        # Discharge coefficient
        Cd = self._get_discharge_coefficient(gate, upstream_depth, downstream_depth)
        
        # Velocity coefficient
        Cv = Cd / Cc if Cc > 0 else 1.0
        
        return {
            "upstream_velocity": upstream_velocity,
            "gate_velocity": gate_velocity,
            "contracted_depth": contracted_depth,
            "contraction_coefficient": Cc,
            "discharge_coefficient": Cd,
            "velocity_coefficient": Cv,
            "froude_upstream": froude_upstream,
            "froude_gate": froude_gate,
            "coefficient_source": "empirical"
        }
    
    def _get_contraction_coefficient(
        self,
        gate: GateGeometry,
        upstream_depth: float
    ) -> float:
        """Get contraction coefficient for vena contracta calculation."""
        
        if gate.gate_type == GateType.SLUICE:
            # Contraction coefficient for sluice gates
            opening_ratio = gate.gate_opening / upstream_depth
            
            if opening_ratio < 0.2:
                return 0.61  # Sharp contraction
            elif opening_ratio < 0.5:
                return 0.65  # Moderate contraction
            else:
                return 0.70  # Mild contraction
                
        elif gate.gate_type == GateType.RADIAL:
            # Radial gates have less contraction
            return 0.75
            
        else:
            # Default contraction coefficient
            return 0.62
    
    def _analyze_downstream_conditions(
        self,
        channel: ChannelGeometry,
        discharge: float,
        contracted_depth: float,
        gate_velocity: float,
        downstream_depth: Optional[float]
    ) -> Dict[str, Any]:
        """Analyze downstream flow conditions and hydraulic jump potential."""
        
        # If downstream depth is specified, use it
        if downstream_depth is not None:
            downstream_area = channel.area(downstream_depth)
            downstream_velocity = discharge / downstream_area
            froude_downstream = downstream_velocity / math.sqrt(self.gravity * downstream_depth)
            
            # Check for hydraulic jump
            froude_contracted = gate_velocity / math.sqrt(self.gravity * contracted_depth)
            jump_required = froude_contracted > 1.0 and froude_downstream < 1.0
            
            jump_analysis = None
            if jump_required:
                # Analyze hydraulic jump from contracted section to downstream
                jump_analysis = self.rvf_solver.analyze_hydraulic_jump(
                    channel, discharge, contracted_depth, downstream_depth
                )
            
            return {
                "depth": downstream_depth,
                "velocity": downstream_velocity,
                "froude_number": froude_downstream,
                "jump_required": jump_required,
                "jump_analysis": jump_analysis,
                "submergence_ratio": downstream_depth / contracted_depth
            }
        
        else:
            # Calculate downstream depth assuming no hydraulic jump
            # Use contracted depth as starting point
            downstream_depth_calc = contracted_depth
            downstream_area = channel.area(downstream_depth_calc)
            downstream_velocity = discharge / downstream_area
            froude_downstream = downstream_velocity / math.sqrt(self.gravity * downstream_depth_calc)
            
            return {
                "depth": downstream_depth_calc,
                "velocity": downstream_velocity,
                "froude_number": froude_downstream,
                "jump_required": froude_downstream > 1.0,
                "submergence_ratio": 1.0
            }
    
    def _assess_cavitation_risk(
        self,
        upstream_depth: float,
        contracted_depth: float,
        gate_velocity: float
    ) -> Dict[str, Any]:
        """Assess cavitation risk at gate opening."""
        
        # Calculate cavitation index
        # σ = (P_atm + ρgh - P_vapor) / (0.5 * ρ * V²)
        # Simplified: σ ≈ (h_upstream - h_contracted) / (V² / 2g)
        
        pressure_head_diff = upstream_depth - contracted_depth
        velocity_head = gate_velocity**2 / (2 * self.gravity)
        
        if velocity_head > 0:
            cavitation_index = pressure_head_diff / velocity_head
        else:
            cavitation_index = 10.0  # Very high (safe)
        
        # Classify risk level
        if cavitation_index > 0.25:
            risk_level = CavitationRisk.LOW
        elif cavitation_index > 0.1:
            risk_level = CavitationRisk.MODERATE
        else:
            risk_level = CavitationRisk.HIGH
        
        return {
            "cavitation_index": cavitation_index,
            "risk_level": risk_level,
            "pressure_head_diff": pressure_head_diff,
            "velocity_head": velocity_head
        }
    
    def _calculate_energy_analysis(
        self,
        channel: ChannelGeometry,
        discharge: float,
        upstream_depth: float,
        contracted_depth: float,
        downstream_depth: Optional[float]
    ) -> Dict[str, Any]:
        """Calculate energy analysis for gate flow."""
        
        # Upstream energy
        upstream_area = channel.area(upstream_depth)
        upstream_velocity = discharge / upstream_area
        upstream_energy = upstream_depth + upstream_velocity**2 / (2 * self.gravity)
        
        # Gate energy (at vena contracta)
        gate_area = channel.area(contracted_depth)
        gate_velocity = discharge / gate_area
        gate_energy = contracted_depth + gate_velocity**2 / (2 * self.gravity)
        
        # Downstream energy
        if downstream_depth is not None:
            downstream_area = channel.area(downstream_depth)
            downstream_velocity = discharge / downstream_area
            downstream_energy = downstream_depth + downstream_velocity**2 / (2 * self.gravity)
        else:
            downstream_energy = gate_energy
        
        # Energy loss
        energy_loss = upstream_energy - downstream_energy
        
        # Energy efficiency
        efficiency = downstream_energy / upstream_energy if upstream_energy > 0 else 0
        
        return {
            "upstream_energy": upstream_energy,
            "gate_energy": gate_energy,
            "downstream_energy": downstream_energy,
            "energy_loss": energy_loss,
            "efficiency": efficiency
        }


class GateFlowAnalyzer:
    """
    High-level gate flow analysis tools.
    
    Provides convenient methods for common gate analysis tasks
    and professional design recommendations.
    """
    
    def __init__(self):
        self.solver = GateFlowSolver()
    
    def recommend_gate_design(self, result: GateFlowResult) -> List[str]:
        """Generate design recommendations for gate flow."""
        recommendations = []
        
        if not result.success:
            recommendations.append("❌ Gate analysis failed - check input conditions")
            return recommendations
        
        # Flow condition recommendations
        if result.flow_condition == FlowCondition.FREE_FLOW:
            recommendations.extend([
                "✅ Free flow conditions - optimal discharge efficiency",
                "📏 Gate opening provides good flow control",
                "🔧 Consider downstream energy dissipation if needed"
            ])
        elif result.flow_condition == FlowCondition.SUBMERGED:
            recommendations.extend([
                "⚠️ Submerged flow - reduced discharge efficiency",
                "📊 Consider increasing gate opening for better performance",
                "🌊 Monitor downstream water levels"
            ])
        else:  # DROWNED
            recommendations.extend([
                "❌ Drowned flow conditions - significantly reduced efficiency",
                "🚨 Increase gate opening or reduce downstream levels",
                "⚡ Poor flow control under current conditions"
            ])
        
        # Cavitation risk recommendations
        if result.cavitation_risk == CavitationRisk.HIGH:
            recommendations.extend([
                "🚨 HIGH CAVITATION RISK - immediate attention required",
                "🔧 Consider gate design modifications or operational changes",
                "⚠️ Potential for gate damage and vibration"
            ])
        elif result.cavitation_risk == CavitationRisk.MODERATE:
            recommendations.extend([
                "⚠️ Moderate cavitation risk - monitor closely",
                "🔍 Consider cavitation-resistant materials"
            ])
        else:
            recommendations.append("✅ Low cavitation risk - acceptable operation")
        
        # Hydraulic jump recommendations
        if result.hydraulic_jump_required:
            recommendations.extend([
                "🌊 Hydraulic jump expected downstream",
                "🛡️ Provide adequate stilling basin or energy dissipation",
                "📏 Consider jump length in downstream design"
            ])
        
        # Energy efficiency recommendations
        if result.properties.get("energy_efficiency", 0) < 0.5:
            recommendations.append("⚡ High energy dissipation - good for energy dissipation structures")
        elif result.properties.get("energy_efficiency", 0) > 0.8:
            recommendations.append("💡 High energy efficiency - minimal energy loss")
        
        # Gate-specific recommendations
        if result.gate_type == GateType.SLUICE:
            recommendations.append("📐 Sluice gate: Consider trash rack upstream for debris protection")
        elif result.gate_type == GateType.RADIAL:
            recommendations.append("🔄 Radial gate: Excellent for precise flow control")
        
        return recommendations
    
    def compare_gate_alternatives(
        self,
        channel: ChannelGeometry,
        gate_types: List[GateType],
        gate_width: float,
        discharge: float,
        upstream_depth: float
    ) -> Dict[str, Any]:
        """Compare different gate types for given conditions."""
        
        results = {}
        
        for gate_type in gate_types:
            # Design gate opening for each type
            design_result = self.solver.design_gate_opening(
                channel, gate_type, gate_width, discharge, upstream_depth
            )
            
            if design_result["success"]:
                results[gate_type.value] = {
                    "gate_opening": design_result["gate_opening"],
                    "flow_analysis": design_result["flow_analysis"],
                    "design_iterations": design_result["iterations"]
                }
        
        # Find best options
        if results:
            # Best efficiency
            best_efficiency = max(
                results.items(),
                key=lambda x: x[1]["flow_analysis"].properties.get("energy_efficiency", 0)
            )
            
            # Lowest cavitation risk
            cavitation_scores = {
                CavitationRisk.LOW: 3,
                CavitationRisk.MODERATE: 2,
                CavitationRisk.HIGH: 1
            }
            
            best_cavitation = max(
                results.items(),
                key=lambda x: cavitation_scores.get(x[1]["flow_analysis"].cavitation_risk, 0)
            )
            
            return {
                "success": True,
                "gate_comparisons": results,
                "best_efficiency": {
                    "gate_type": best_efficiency[0],
                    "efficiency": best_efficiency[1]["flow_analysis"].properties.get("energy_efficiency", 0),
                    "gate_opening": best_efficiency[1]["gate_opening"]
                },
                "best_cavitation": {
                    "gate_type": best_cavitation[0],
                    "cavitation_risk": best_cavitation[1]["flow_analysis"].cavitation_risk.value,
                    "cavitation_index": best_cavitation[1]["flow_analysis"].cavitation_index
                }
            }
        
        return {"success": False, "message": "No successful gate designs found"}
