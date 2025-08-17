"""
Core RVF Engine - PyOpenChannel

Author: Alexius Academia
Email: alexius.sayco.academia@gmail.com

This module provides the core Rapidly Varied Flow (RVF) analysis engine.
RVF occurs when flow depth changes rapidly over short distances, such as:
- Hydraulic jumps
- Flow over weirs
- Flow under gates
- Shock waves and surges
- Choking conditions

The RVF solver uses energy-momentum balance methods instead of differential
equations, making it suitable for discontinuous flow transitions.
"""

import math
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum

from ..geometry import ChannelGeometry, RectangularChannel, TrapezoidalChannel
from ..hydraulics import ManningEquation
from ..flow_analysis import CriticalFlow, EnergyEquation, MomentumEquation
from ..validators import validate_discharge, validate_depth, validate_slope, validate_manning_n
from ..exceptions import ConvergenceError, InvalidFlowConditionError
from ..units import get_gravity


class RVFRegime(Enum):
    """Flow regime classification for RVF analysis."""
    SUBCRITICAL = "subcritical"
    CRITICAL = "critical"
    SUPERCRITICAL = "supercritical"
    HYDRAULIC_JUMP = "hydraulic_jump"
    SHOCK_WAVE = "shock_wave"
    CHOKING = "choking"


class JumpType(Enum):
    """Hydraulic jump classification based on upstream Froude number."""
    UNDULAR = "undular"          # Fr1 = 1.0 - 1.7
    WEAK = "weak"                # Fr1 = 1.7 - 2.5
    OSCILLATING = "oscillating"  # Fr1 = 2.5 - 4.5
    STEADY = "steady"            # Fr1 = 4.5 - 9.0
    STRONG = "strong"            # Fr1 = 9.0+


class TransitionType(Enum):
    """Type of rapid flow transition."""
    HYDRAULIC_JUMP = "hydraulic_jump"
    POSITIVE_SURGE = "positive_surge"
    NEGATIVE_SURGE = "negative_surge"
    BORE = "bore"
    CHOKING = "choking"
    CONTROL_STRUCTURE = "control_structure"


@dataclass
class RVFResult:
    """
    Result of RVF analysis.
    
    Contains all computed hydraulic parameters and analysis results
    for rapidly varied flow conditions.
    """
    # Flow conditions
    upstream_depth: float
    downstream_depth: float
    upstream_velocity: float
    downstream_velocity: float
    upstream_froude: float
    downstream_froude: float
    
    # Energy and momentum
    upstream_energy: float
    downstream_energy: float
    energy_loss: float
    energy_efficiency: float
    momentum_change: float
    
    # Jump characteristics (if applicable)
    jump_type: Optional[JumpType]
    jump_length: Optional[float]
    jump_height: Optional[float]
    sequent_depth_ratio: Optional[float]
    
    # Analysis metadata
    transition_type: TransitionType
    regime_upstream: RVFRegime
    regime_downstream: RVFRegime
    success: bool
    message: str
    
    # Additional properties
    properties: Dict[str, Any]


@dataclass
class ShockProperties:
    """Properties of shock wave or surge."""
    wave_celerity: float
    wave_height: float
    wave_type: str
    propagation_direction: str
    reflection_coefficient: Optional[float] = None
    transmission_coefficient: Optional[float] = None


class RVFSolver:
    """
    Core Rapidly Varied Flow solver.
    
    This solver handles flow situations where depth changes rapidly over
    short distances, using energy-momentum balance methods instead of
    differential equations.
    
    Features:
    - Hydraulic jump analysis (all types)
    - Shock wave and surge analysis
    - Automatic regime detection
    - GVF-RVF transition detection
    - Energy dissipation calculations
    - Professional design recommendations
    """
    
    def __init__(
        self,
        transition_threshold: float = 0.1,
        froude_gradient_threshold: float = 0.5,
        convergence_tolerance: float = 1e-6,
        max_iterations: int = 100
    ):
        """
        Initialize RVF solver.
        
        Args:
            transition_threshold: Threshold for detecting rapid transitions
            froude_gradient_threshold: Froude number gradient threshold for RVF detection
            convergence_tolerance: Convergence tolerance for iterative solutions
            max_iterations: Maximum iterations for convergence
        """
        self.transition_threshold = transition_threshold
        self.froude_gradient_threshold = froude_gradient_threshold
        self.tolerance = convergence_tolerance
        self.max_iterations = max_iterations
        self.gravity = get_gravity()
    
    def analyze_hydraulic_jump(
        self,
        channel: ChannelGeometry,
        discharge: float,
        upstream_depth: float,
        tailwater_depth: Optional[float] = None
    ) -> RVFResult:
        """
        Analyze hydraulic jump characteristics.
        
        Args:
            channel: Channel geometry
            discharge: Discharge (m³/s or ft³/s)
            upstream_depth: Upstream depth (supercritical)
            tailwater_depth: Tailwater depth (optional)
            
        Returns:
            RVFResult with jump analysis
        """
        try:
            # Validate inputs
            discharge = validate_discharge(discharge)
            upstream_depth = validate_depth(upstream_depth)
            
            # Calculate upstream conditions
            upstream_area = channel.area(upstream_depth)
            upstream_velocity = discharge / upstream_area
            upstream_froude = upstream_velocity / math.sqrt(self.gravity * upstream_depth)
            
            # Check if flow is supercritical (required for hydraulic jump)
            if upstream_froude <= 1.0:
                return RVFResult(
                    upstream_depth=upstream_depth,
                    downstream_depth=upstream_depth,
                    upstream_velocity=upstream_velocity,
                    downstream_velocity=upstream_velocity,
                    upstream_froude=upstream_froude,
                    downstream_froude=upstream_froude,
                    upstream_energy=0,
                    downstream_energy=0,
                    energy_loss=0,
                    energy_efficiency=1.0,
                    momentum_change=0,
                    jump_type=None,
                    jump_length=None,
                    jump_height=None,
                    sequent_depth_ratio=None,
                    transition_type=TransitionType.HYDRAULIC_JUMP,
                    regime_upstream=RVFRegime.SUBCRITICAL,
                    regime_downstream=RVFRegime.SUBCRITICAL,
                    success=False,
                    message="Upstream flow is subcritical - no hydraulic jump possible",
                    properties={}
                )
            
            # Calculate sequent depth using momentum equation
            sequent_depth = self._calculate_sequent_depth(
                channel, discharge, upstream_depth, upstream_velocity
            )
            
            # Calculate downstream conditions
            downstream_area = channel.area(sequent_depth)
            downstream_velocity = discharge / downstream_area
            downstream_froude = downstream_velocity / math.sqrt(self.gravity * sequent_depth)
            
            # Calculate energy conditions
            upstream_energy = upstream_depth + upstream_velocity**2 / (2 * self.gravity)
            downstream_energy = sequent_depth + downstream_velocity**2 / (2 * self.gravity)
            energy_loss = upstream_energy - downstream_energy
            energy_efficiency = downstream_energy / upstream_energy
            
            # Calculate momentum change
            momentum_change = self._calculate_momentum_change(
                channel, discharge, upstream_depth, sequent_depth
            )
            
            # Classify jump type
            jump_type = self._classify_jump_type(upstream_froude)
            
            # Calculate jump geometry
            jump_length = self._calculate_jump_length(
                upstream_depth, sequent_depth, upstream_froude, channel
            )
            jump_height = sequent_depth - upstream_depth
            sequent_depth_ratio = sequent_depth / upstream_depth
            
            # Check tailwater adequacy
            tailwater_adequate = True
            tailwater_message = ""
            if tailwater_depth is not None:
                if tailwater_depth < sequent_depth * 0.9:
                    tailwater_adequate = False
                    tailwater_message = f"Tailwater depth ({tailwater_depth:.3f}) insufficient for stable jump"
                elif tailwater_depth > sequent_depth * 1.1:
                    tailwater_message = f"Tailwater depth ({tailwater_depth:.3f}) higher than required - submerged jump"
            
            # Determine regimes
            regime_upstream = RVFRegime.SUPERCRITICAL
            regime_downstream = RVFRegime.SUBCRITICAL if downstream_froude < 1.0 else RVFRegime.CRITICAL
            
            # Success message
            message = f"Hydraulic jump analysis complete. Jump type: {jump_type.value}"
            if tailwater_message:
                message += f". {tailwater_message}"
            
            return RVFResult(
                upstream_depth=upstream_depth,
                downstream_depth=sequent_depth,
                upstream_velocity=upstream_velocity,
                downstream_velocity=downstream_velocity,
                upstream_froude=upstream_froude,
                downstream_froude=downstream_froude,
                upstream_energy=upstream_energy,
                downstream_energy=downstream_energy,
                energy_loss=energy_loss,
                energy_efficiency=energy_efficiency,
                momentum_change=momentum_change,
                jump_type=jump_type,
                jump_length=jump_length,
                jump_height=jump_height,
                sequent_depth_ratio=sequent_depth_ratio,
                transition_type=TransitionType.HYDRAULIC_JUMP,
                regime_upstream=regime_upstream,
                regime_downstream=regime_downstream,
                success=True,
                message=message,
                properties={
                    "tailwater_adequate": tailwater_adequate,
                    "tailwater_depth": tailwater_depth,
                    "energy_dissipation_ratio": energy_loss / upstream_energy,
                    "momentum_function_upstream": self._momentum_function(channel, discharge, upstream_depth),
                    "momentum_function_downstream": self._momentum_function(channel, discharge, sequent_depth)
                }
            )
            
        except Exception as e:
            return RVFResult(
                upstream_depth=upstream_depth,
                downstream_depth=upstream_depth,
                upstream_velocity=0,
                downstream_velocity=0,
                upstream_froude=0,
                downstream_froude=0,
                upstream_energy=0,
                downstream_energy=0,
                energy_loss=0,
                energy_efficiency=0,
                momentum_change=0,
                jump_type=None,
                jump_length=None,
                jump_height=None,
                sequent_depth_ratio=None,
                transition_type=TransitionType.HYDRAULIC_JUMP,
                regime_upstream=RVFRegime.SUBCRITICAL,
                regime_downstream=RVFRegime.SUBCRITICAL,
                success=False,
                message=f"Hydraulic jump analysis failed: {str(e)}",
                properties={"error": str(e)}
            )
    
    def detect_rvf_transition(
        self,
        froude_numbers: List[float],
        depths: List[float],
        distances: List[float]
    ) -> Dict[str, Any]:
        """
        Detect when flow transitions from GVF to RVF conditions.
        
        Args:
            froude_numbers: List of Froude numbers along the profile
            depths: List of depths along the profile
            distances: List of distances along the profile
            
        Returns:
            Dictionary with transition detection results
        """
        transitions = []
        
        if len(froude_numbers) < 2:
            return {"transitions": transitions, "rvf_required": False}
        
        for i in range(1, len(froude_numbers)):
            # Calculate gradients
            dx = distances[i] - distances[i-1]
            if dx <= 0:
                continue
                
            froude_gradient = abs(froude_numbers[i] - froude_numbers[i-1]) / dx
            depth_gradient = abs(depths[i] - depths[i-1]) / dx
            depth_change_rate = abs(depths[i] - depths[i-1]) / depths[i-1]
            
            # Check for rapid transitions
            if froude_gradient > self.froude_gradient_threshold:
                transitions.append({
                    "location": distances[i],
                    "type": "froude_gradient",
                    "upstream_froude": froude_numbers[i-1],
                    "downstream_froude": froude_numbers[i],
                    "gradient": froude_gradient
                })
            
            if depth_change_rate > self.transition_threshold:
                transitions.append({
                    "location": distances[i],
                    "type": "rapid_depth_change",
                    "upstream_depth": depths[i-1],
                    "downstream_depth": depths[i],
                    "change_rate": depth_change_rate
                })
            
            # Check for critical flow transitions
            if (froude_numbers[i-1] < 1.0 < froude_numbers[i]) or \
               (froude_numbers[i-1] > 1.0 > froude_numbers[i]):
                transitions.append({
                    "location": distances[i],
                    "type": "critical_transition",
                    "upstream_froude": froude_numbers[i-1],
                    "downstream_froude": froude_numbers[i]
                })
        
        return {
            "transitions": transitions,
            "rvf_required": len(transitions) > 0,
            "transition_count": len(transitions)
        }
    
    def analyze_shock_wave(
        self,
        channel: ChannelGeometry,
        discharge: float,
        initial_depth: float,
        final_depth: float,
        wave_type: str = "positive_surge"
    ) -> Dict[str, Any]:
        """
        Analyze shock wave or surge characteristics.
        
        Args:
            channel: Channel geometry
            discharge: Discharge
            initial_depth: Initial flow depth
            final_depth: Final flow depth
            wave_type: Type of wave ("positive_surge", "negative_surge", "bore")
            
        Returns:
            Dictionary with shock wave analysis
        """
        try:
            # Calculate wave celerity using shallow water wave theory
            if wave_type == "positive_surge":
                # Positive surge (depth increases)
                celerity = math.sqrt(self.gravity * final_depth * 
                                   (1 + final_depth/initial_depth) / 2)
            elif wave_type == "negative_surge":
                # Negative surge (depth decreases)
                celerity = math.sqrt(self.gravity * initial_depth)
            else:  # bore
                # Bore (moving hydraulic jump)
                celerity = math.sqrt(self.gravity * final_depth * 
                                   (final_depth + initial_depth) / (2 * initial_depth))
            
            # Calculate wave height
            wave_height = abs(final_depth - initial_depth)
            
            # Calculate velocities
            initial_area = channel.area(initial_depth)
            final_area = channel.area(final_depth)
            initial_velocity = discharge / initial_area
            final_velocity = discharge / final_area
            
            # Calculate Froude numbers
            initial_froude = initial_velocity / math.sqrt(self.gravity * initial_depth)
            final_froude = final_velocity / math.sqrt(self.gravity * final_depth)
            
            return {
                "wave_celerity": celerity,
                "wave_height": wave_height,
                "wave_type": wave_type,
                "initial_depth": initial_depth,
                "final_depth": final_depth,
                "initial_velocity": initial_velocity,
                "final_velocity": final_velocity,
                "initial_froude": initial_froude,
                "final_froude": final_froude,
                "energy_change": self._calculate_energy_change(
                    initial_depth, final_depth, initial_velocity, final_velocity
                ),
                "success": True
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "wave_celerity": 0,
                "wave_height": 0
            }
    
    def _calculate_sequent_depth(
        self,
        channel: ChannelGeometry,
        discharge: float,
        upstream_depth: float,
        upstream_velocity: float
    ) -> float:
        """Calculate sequent depth for hydraulic jump using momentum equation."""
        
        if isinstance(channel, RectangularChannel):
            # Analytical solution for rectangular channel
            froude1 = upstream_velocity / math.sqrt(self.gravity * upstream_depth)
            y2_over_y1 = 0.5 * (-1 + math.sqrt(1 + 8 * froude1**2))
            return upstream_depth * y2_over_y1
        
        else:
            # Iterative solution for general channel geometry
            # Use momentum equation: M1 = M2
            momentum1 = self._momentum_function(channel, discharge, upstream_depth)
            
            # Initial guess
            y2 = upstream_depth * 2.0
            
            for iteration in range(self.max_iterations):
                momentum2 = self._momentum_function(channel, discharge, y2)
                residual = momentum2 - momentum1
                
                if abs(residual) < self.tolerance:
                    return y2
                
                # Newton-Raphson iteration
                # dM/dy = A + Q²/gA² * dA/dy - Q²/gA³ * T
                area2 = channel.area(y2)
                top_width2 = channel.top_width(y2)
                
                dM_dy = area2 + discharge**2 / (self.gravity * area2**2) - \
                        discharge**2 * top_width2 / (self.gravity * area2**3)
                
                y2_new = y2 - residual / dM_dy
                
                # Ensure positive depth
                if y2_new <= 0:
                    y2_new = y2 * 0.5
                
                y2 = y2_new
            
            raise ConvergenceError("Sequent depth calculation did not converge")
    
    def _momentum_function(self, channel: ChannelGeometry, discharge: float, depth: float) -> float:
        """Calculate momentum function M = yc*A + Q²/(g*A)"""
        area = channel.area(depth)
        centroid = self._calculate_centroidal_depth(channel, depth)
        return centroid * area + discharge**2 / (self.gravity * area)
    
    def _calculate_centroidal_depth(self, channel: ChannelGeometry, depth: float) -> float:
        """Calculate centroidal depth for different channel geometries."""
        if isinstance(channel, RectangularChannel):
            return depth / 2.0
        elif isinstance(channel, TrapezoidalChannel):
            # For trapezoidal channel: yc = (b*y² + m*y³) / (2*(b*y + m*y²))
            b = channel.bottom_width
            m = channel.side_slope
            return (b * depth**2 + m * depth**3) / (2 * (b * depth + m * depth**2))
        else:
            # General approximation for other geometries
            return depth / 2.0
    
    def _classify_jump_type(self, upstream_froude: float) -> JumpType:
        """Classify hydraulic jump type based on upstream Froude number."""
        if upstream_froude <= 1.7:
            return JumpType.UNDULAR
        elif upstream_froude <= 2.5:
            return JumpType.WEAK
        elif upstream_froude <= 4.5:
            return JumpType.OSCILLATING
        elif upstream_froude <= 9.0:
            return JumpType.STEADY
        else:
            return JumpType.STRONG
    
    def _calculate_jump_length(
        self,
        upstream_depth: float,
        downstream_depth: float,
        upstream_froude: float,
        channel: ChannelGeometry
    ) -> float:
        """Calculate hydraulic jump length using empirical relations."""
        
        if isinstance(channel, RectangularChannel):
            # Empirical relation for rectangular channels
            # L = 6 * (y2 - y1) for Fr1 < 4.5
            # L = 5 * y2 for Fr1 >= 4.5
            if upstream_froude < 4.5:
                return 6.0 * (downstream_depth - upstream_depth)
            else:
                return 5.0 * downstream_depth
        else:
            # General approximation
            return 5.0 * (downstream_depth - upstream_depth)
    
    def _calculate_momentum_change(
        self,
        channel: ChannelGeometry,
        discharge: float,
        upstream_depth: float,
        downstream_depth: float
    ) -> float:
        """Calculate momentum change across hydraulic jump."""
        momentum_upstream = self._momentum_function(channel, discharge, upstream_depth)
        momentum_downstream = self._momentum_function(channel, discharge, downstream_depth)
        return momentum_downstream - momentum_upstream
    
    def _calculate_energy_change(
        self,
        depth1: float,
        depth2: float,
        velocity1: float,
        velocity2: float
    ) -> float:
        """Calculate energy change between two flow states."""
        energy1 = depth1 + velocity1**2 / (2 * self.gravity)
        energy2 = depth2 + velocity2**2 / (2 * self.gravity)
        return energy2 - energy1


class RVFAnalyzer:
    """
    High-level RVF analysis tools.
    
    Provides convenient methods for common RVF analysis tasks
    and integration with GVF analysis.
    """
    
    def __init__(self):
        self.solver = RVFSolver()
    
    def recommend_jump_design(self, jump_result: RVFResult) -> List[str]:
        """Generate design recommendations for hydraulic jump."""
        recommendations = []
        
        if not jump_result.success:
            recommendations.append("❌ Jump analysis failed - check input conditions")
            return recommendations
        
        # Jump type specific recommendations
        if jump_result.jump_type == JumpType.UNDULAR:
            recommendations.extend([
                "✅ Undular jump - smooth transition with minimal energy loss",
                "📏 Consider shorter stilling basin length",
                "⚠️ Monitor for potential oscillations in tailwater"
            ])
        elif jump_result.jump_type == JumpType.WEAK:
            recommendations.extend([
                "✅ Weak jump - stable with moderate energy dissipation",
                "📏 Standard stilling basin design applicable",
                "🔧 Consider baffle blocks for enhanced mixing"
            ])
        elif jump_result.jump_type == JumpType.OSCILLATING:
            recommendations.extend([
                "⚠️ Oscillating jump - unstable, avoid if possible",
                "🔧 Consider forced jump with sill or baffle blocks",
                "📊 Monitor tailwater levels carefully"
            ])
        elif jump_result.jump_type == JumpType.STEADY:
            recommendations.extend([
                "✅ Steady jump - excellent energy dissipation",
                "📏 Use standard jump length calculations",
                "🛡️ Provide adequate erosion protection"
            ])
        elif jump_result.jump_type == JumpType.STRONG:
            recommendations.extend([
                "⚠️ Strong jump - high energy dissipation but rough",
                "🛡️ Heavy erosion protection required",
                "📏 Extended stilling basin recommended"
            ])
        
        # Energy efficiency recommendations
        if jump_result.energy_efficiency < 0.3:
            recommendations.append("⚡ High energy dissipation - excellent for energy dissipation structures")
        elif jump_result.energy_efficiency > 0.7:
            recommendations.append("💡 Low energy loss - consider if more dissipation needed")
        
        # Tailwater adequacy
        if jump_result.properties.get("tailwater_adequate", True):
            recommendations.append("✅ Tailwater depth adequate for stable jump")
        else:
            recommendations.append("❌ Insufficient tailwater - jump may be swept out")
        
        # Jump length recommendations
        if jump_result.jump_length:
            recommendations.append(f"📏 Minimum stilling basin length: {jump_result.jump_length:.1f} units")
        
        return recommendations
    
    def compare_jump_alternatives(
        self,
        channel: ChannelGeometry,
        discharge: float,
        upstream_depths: List[float]
    ) -> Dict[str, Any]:
        """Compare hydraulic jump characteristics for different upstream conditions."""
        
        results = []
        for depth in upstream_depths:
            result = self.solver.analyze_hydraulic_jump(channel, discharge, depth)
            results.append(result)
        
        # Find optimal conditions
        successful_results = [r for r in results if r.success]
        
        if not successful_results:
            return {"success": False, "message": "No successful jump analyses"}
        
        # Find best energy dissipation
        best_dissipation = max(successful_results, key=lambda r: r.energy_loss)
        
        # Find most stable jump (avoid oscillating)
        stable_results = [r for r in successful_results 
                         if r.jump_type != JumpType.OSCILLATING]
        most_stable = min(stable_results, key=lambda r: abs(r.upstream_froude - 6.0)) \
                     if stable_results else None
        
        return {
            "success": True,
            "total_analyses": len(results),
            "successful_analyses": len(successful_results),
            "best_energy_dissipation": {
                "upstream_depth": best_dissipation.upstream_depth,
                "energy_loss": best_dissipation.energy_loss,
                "efficiency": best_dissipation.energy_efficiency,
                "jump_type": best_dissipation.jump_type.value
            },
            "most_stable": {
                "upstream_depth": most_stable.upstream_depth,
                "jump_type": most_stable.jump_type.value,
                "upstream_froude": most_stable.upstream_froude
            } if most_stable else None,
            "all_results": results
        }
