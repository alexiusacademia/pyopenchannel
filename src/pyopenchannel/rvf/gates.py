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
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
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
    
    # FVM integration fields
    method_used: str = "analytical"
    computation_time: float = 0.0
    fvm_profile: Optional['GateFlowProfile'] = None
    
    @property
    def has_detailed_profile(self) -> bool:
        """Check if detailed FVM profile is available."""
        return self.method_used == "fvm" and self.fvm_profile is not None
    
    @property
    def profile_resolution(self) -> str:
        """Get profile resolution description."""
        if self.method_used == "analytical":
            return "3-point (analytical)"
        elif self.has_detailed_profile:
            return f"{len(self.fvm_profile.x_coordinates)}-point (FVM)"
        else:
            return "3-point (analytical fallback)"


@dataclass
class GateFlowProfile:
    """
    Detailed FVM profile data for gate flow analysis.
    
    Contains high-resolution flow field data from FVM simulation,
    providing detailed insight into gate flow structure including
    velocity profiles, pressure distribution, and cavitation analysis.
    """
    # Spatial coordinates
    x_coordinates: np.ndarray
    
    # Primary flow variables
    depths: np.ndarray
    velocities: np.ndarray
    froude_numbers: np.ndarray
    
    # Energy and pressure
    specific_energies: np.ndarray
    pressure_heads: np.ndarray
    
    # Gate-specific properties
    gate_location: float
    gate_length: float
    
    # Computational metadata
    grid_points: int = 0
    scheme_used: str = ""
    convergence_iterations: int = 0
    
    @property
    def domain_length(self) -> float:
        """Get total domain length."""
        return self.x_coordinates[-1] - self.x_coordinates[0]
    
    @property
    def resolution(self) -> float:
        """Get average grid resolution."""
        return self.domain_length / len(self.x_coordinates)
    
    def get_profile_at_x(self, x: float) -> Dict[str, float]:
        """Get flow properties at specific x location using interpolation."""
        if x < self.x_coordinates[0] or x > self.x_coordinates[-1]:
            raise ValueError(f"x = {x} is outside profile domain")
        
        # Linear interpolation
        depth = np.interp(x, self.x_coordinates, self.depths)
        velocity = np.interp(x, self.x_coordinates, self.velocities)
        froude = np.interp(x, self.x_coordinates, self.froude_numbers)
        energy = np.interp(x, self.x_coordinates, self.specific_energies)
        pressure = np.interp(x, self.x_coordinates, self.pressure_heads)
        
        return {
            'depth': depth,
            'velocity': velocity,
            'froude': froude,
            'specific_energy': energy,
            'pressure_head': pressure
        }
    
    def find_vena_contracta(self) -> Optional[float]:
        """Find vena contracta location based on minimum depth."""
        if len(self.depths) < 3:
            return None
        
        # Find location of minimum depth (vena contracta)
        min_depth_idx = np.argmin(self.depths)
        return self.x_coordinates[min_depth_idx]
    
    def find_maximum_velocity(self) -> Tuple[Optional[float], Optional[float]]:
        """Find location and value of maximum velocity."""
        if len(self.velocities) < 3:
            return None, None
        
        max_velocity_idx = np.argmax(self.velocities)
        return self.x_coordinates[max_velocity_idx], self.velocities[max_velocity_idx]
    
    def analyze_gate_flow_details(self) -> Dict[str, float]:
        """Calculate detailed gate flow characteristics from FVM profile."""
        vena_contracta_location = self.find_vena_contracta()
        max_velocity_location, max_velocity = self.find_maximum_velocity()
        
        if vena_contracta_location is None:
            return {}
        
        # Find gate region indices
        gate_start_idx = np.argmin(np.abs(self.x_coordinates - self.gate_location))
        gate_end_idx = np.argmin(np.abs(self.x_coordinates - (self.gate_location + self.gate_length)))
        
        # Upstream region (before gate)
        upstream_region = slice(0, gate_start_idx)
        upstream_depth = np.mean(self.depths[upstream_region])
        upstream_velocity = np.mean(self.velocities[upstream_region])
        upstream_pressure = np.mean(self.pressure_heads[upstream_region])
        
        # Gate region
        gate_region = slice(gate_start_idx, gate_end_idx)
        min_depth = np.min(self.depths[gate_region])
        max_velocity_gate = np.max(self.velocities[gate_region])
        min_pressure = np.min(self.pressure_heads[gate_region])
        
        # Downstream region (after gate)
        downstream_region = slice(gate_end_idx, len(self.depths))
        downstream_depth = np.mean(self.depths[downstream_region])
        downstream_velocity = np.mean(self.velocities[downstream_region])
        downstream_pressure = np.mean(self.pressure_heads[downstream_region])
        
        return {
            'vena_contracta_location': vena_contracta_location,
            'max_velocity_location': max_velocity_location or 0,
            'upstream_depth': upstream_depth,
            'downstream_depth': downstream_depth,
            'min_depth': min_depth,
            'max_velocity': max_velocity_gate,
            'upstream_velocity': upstream_velocity,
            'downstream_velocity': downstream_velocity,
            'upstream_pressure': upstream_pressure,
            'downstream_pressure': downstream_pressure,
            'min_pressure': min_pressure,
            'pressure_drop': upstream_pressure - min_pressure,
            'velocity_increase': max_velocity_gate - upstream_velocity
        }
    
    def assess_cavitation_risk(self) -> Dict[str, Any]:
        """Assess cavitation risk based on pressure distribution."""
        min_pressure = np.min(self.pressure_heads)
        min_pressure_location = self.x_coordinates[np.argmin(self.pressure_heads)]
        
        # Cavitation risk assessment (simplified)
        # In practice, this would use vapor pressure and atmospheric pressure
        cavitation_threshold = 2.0  # meters of water (approximate)
        
        if min_pressure < cavitation_threshold:
            if min_pressure < 0.5:
                risk = CavitationRisk.SEVERE
            elif min_pressure < 1.0:
                risk = CavitationRisk.HIGH
            else:
                risk = CavitationRisk.MODERATE
        else:
            risk = CavitationRisk.LOW
        
        return {
            'cavitation_risk': risk,
            'min_pressure': min_pressure,
            'min_pressure_location': min_pressure_location,
            'cavitation_threshold': cavitation_threshold,
            'pressure_margin': min_pressure - cavitation_threshold
        }


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
        method: str = "analytical",
        convergence_tolerance: float = 1e-6,
        max_iterations: int = 100
    ):
        """
        Initialize gate flow solver.
        
        Args:
            method: Solution method ("analytical" or "fvm")
            convergence_tolerance: Convergence tolerance for iterative solutions
            max_iterations: Maximum iterations for convergence
        """
        # Validate method
        if method not in ["analytical", "fvm"]:
            raise ValueError(f"Invalid method: {method}. Must be 'analytical' or 'fvm'")
        
        self.method = method
        self.tolerance = convergence_tolerance
        self.max_iterations = max_iterations
        self.gravity = get_gravity()
        
        # Initialize FVM solver if needed
        self.fvm_solver = None
        if method == "fvm":
            self._initialize_fvm_solver()
    
    def _initialize_fvm_solver(self):
        """Initialize FVM solver for detailed gate flow analysis."""
        try:
            from ..numerical.fvm import (
                ShallowWaterSolver, ConvergenceCriteria, 
                TimeIntegrationMethod, InletBC, OutletBC, BoundaryData
            )
            
            # Create FVM solver optimized for gate flow
            criteria = ConvergenceCriteria(
                max_iterations=1000,
                residual_tolerance=1e-8,
                steady_state_tolerance=1e-10,
                cfl_number=0.2  # Conservative for complex geometry
            )
            
            self.fvm_solver = ShallowWaterSolver(
                scheme_name="hllc",  # Best for complex flow patterns
                time_integration=TimeIntegrationMethod.RK4,
                convergence_criteria=criteria
            )
            
        except ImportError:
            raise ImportError(
                "FVM solver not available. Install FVM components or use method='analytical'"
            )
        self.rvf_solver = RVFSolver()
    
    def analyze_gate_flow(
        self,
        channel: ChannelGeometry,
        gate: GateGeometry,
        upstream_depth: float,
        downstream_depth: Optional[float] = None,
        discharge: Optional[float] = None,
        method: Optional[str] = None
    ) -> GateFlowResult:
        """
        Analyze flow under a gate.
        
        Args:
            channel: Channel geometry
            gate: Gate geometry and properties
            upstream_depth: Upstream water depth
            downstream_depth: Downstream water depth (optional)
            discharge: Known discharge (optional, will be calculated if not provided)
            method: Override method ("analytical" or "fvm", None uses default)
            
        Returns:
            GateFlowResult with complete analysis
        """
        try:
            # Determine method to use
            analysis_method = method if method is not None else self.method
            
            # Validate method
            if analysis_method not in ["analytical", "fvm"]:
                raise ValueError(f"Invalid method: {analysis_method}")
            
            # Validate inputs
            upstream_depth = validate_depth(upstream_depth)
            if downstream_depth is not None:
                downstream_depth = validate_depth(downstream_depth)
            if discharge is not None:
                discharge = validate_discharge(discharge)
            
            # Route to appropriate analysis method
            if analysis_method == "fvm":
                return self._analyze_gate_flow_fvm(
                    channel, gate, upstream_depth, downstream_depth, discharge
                )
            else:
                return self._analyze_gate_flow_analytical(
                    channel, gate, upstream_depth, downstream_depth, discharge
                )
            
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
    
    def _analyze_gate_flow_analytical(
        self,
        channel: ChannelGeometry,
        gate: GateGeometry,
        upstream_depth: float,
        downstream_depth: Optional[float] = None,
        discharge: Optional[float] = None
    ) -> GateFlowResult:
        """Analyze gate flow using analytical methods (original implementation)."""
        import time
        start_time = time.time()
        
        # Calculate discharge if not provided
        if discharge is None:
            discharge = self._calculate_gate_discharge(
                channel, gate, upstream_depth, downstream_depth
            )
        
        # Determine flow condition
        flow_condition = self._classify_flow_condition(
            gate, upstream_depth, downstream_depth
        )
        
        # Analyze based on flow condition
        if flow_condition == FlowCondition.FREE_FLOW:
            result = self._analyze_free_flow(
                channel, gate, upstream_depth, discharge
            )
        elif flow_condition == FlowCondition.SUBMERGED:
            result = self._analyze_submerged_flow(
                channel, gate, upstream_depth, downstream_depth, discharge
            )
        else:  # DROWNED
            result = self._analyze_drowned_flow(
                channel, gate, upstream_depth, downstream_depth, discharge
            )
        
        # Add method information
        result.method_used = "analytical"
        result.computation_time = time.time() - start_time
        
        return result
    
    def _analyze_gate_flow_fvm(
        self,
        channel: ChannelGeometry,
        gate: GateGeometry,
        upstream_depth: float,
        downstream_depth: Optional[float] = None,
        discharge: Optional[float] = None
    ) -> GateFlowResult:
        """Analyze gate flow using FVM for detailed velocity and pressure profiles."""
        import time
        start_time = time.time()
        
        if self.fvm_solver is None:
            raise RuntimeError("FVM solver not initialized. Use method='analytical' or initialize with method='fvm'")
        
        try:
            from ..numerical.fvm import (
                UniformGrid, ConservativeVariables, 
                InletBC, OutletBC, BoundaryData
            )
            
            # First get analytical solution for comparison and setup
            analytical_result = self._analyze_gate_flow_analytical(
                channel, gate, upstream_depth, downstream_depth, discharge
            )
            
            if not analytical_result.success:
                # Return analytical result with FVM method flag
                analytical_result.method_used = "fvm"
                analytical_result.computation_time = time.time() - start_time
                return analytical_result
            
            # Create FVM grid for detailed gate flow analysis
            # Domain: upstream approach + gate region + downstream recovery
            gate_length = gate.gate_width * 2  # Assume gate influence zone
            upstream_length = max(20.0, upstream_depth * 10)  # 10x upstream depth
            downstream_length = max(30.0, analytical_result.downstream_depth * 15)  # 15x downstream depth
            domain_length = upstream_length + gate_length + downstream_length
            
            # High resolution near gate for accurate velocity profiles
            grid_points = min(300, max(150, int(domain_length * 3)))  # 3 points per meter
            
            grid = UniformGrid(
                x_min=0.0,
                x_max=domain_length,
                num_cells=grid_points
            )
            
            # Initialize flow field with gate flow conditions
            gate_location = upstream_length  # Gate at transition point
            
            for cell in grid.cells:
                x = cell.x_center
                if x < gate_location:
                    # Upstream approach flow
                    cell.U = ConservativeVariables(
                        h=upstream_depth,
                        hu=upstream_depth * analytical_result.upstream_velocity
                    )
                elif x < gate_location + gate_length:
                    # Gate region - contracted flow
                    contracted_depth = analytical_result.contracted_depth
                    gate_velocity = analytical_result.gate_velocity
                    cell.U = ConservativeVariables(
                        h=contracted_depth,
                        hu=contracted_depth * gate_velocity
                    )
                else:
                    # Downstream recovery
                    downstream_depth = analytical_result.downstream_depth
                    downstream_velocity = analytical_result.downstream_velocity
                    cell.U = ConservativeVariables(
                        h=downstream_depth,
                        hu=downstream_depth * downstream_velocity
                    )
            
            # Set boundary conditions
            inlet_data = BoundaryData(
                depth=upstream_depth, 
                velocity=analytical_result.upstream_velocity
            )
            outlet_data = BoundaryData(depth=analytical_result.downstream_depth)
            
            self.fvm_solver.set_boundary_conditions(
                InletBC("left", inlet_data),
                OutletBC("right", outlet_data)
            )
            
            # Solve FVM system
            fvm_result = self.fvm_solver.solve(grid, steady_state=True)
            
            if not fvm_result.converged:
                # Fallback to analytical with warning
                analytical_result.method_used = "fvm"
                analytical_result.computation_time = time.time() - start_time
                analytical_result.message += " (FVM failed to converge - using analytical fallback)"
                return analytical_result
            
            # Create detailed FVM profile for gate flow
            g = self.gravity
            specific_energies = fvm_result.depths + fvm_result.velocities**2 / (2 * g)
            
            # Calculate pressure distribution (critical for cavitation analysis)
            pressure_heads = []
            for i, (depth, velocity) in enumerate(zip(fvm_result.depths, fvm_result.velocities)):
                # Bernoulli equation: P/γ = (E - z - V²/2g) where z = channel bottom
                pressure_head = specific_energies[i] - velocity**2 / (2 * g)  # Hydrostatic + dynamic
                pressure_heads.append(max(0, pressure_head))  # Prevent negative pressures
            
            pressure_heads = np.array(pressure_heads)
            
            # Create gate flow profile
            gate_profile = GateFlowProfile(
                x_coordinates=fvm_result.x_coordinates,
                depths=fvm_result.depths,
                velocities=fvm_result.velocities,
                froude_numbers=fvm_result.froude_numbers,
                specific_energies=specific_energies,
                pressure_heads=pressure_heads,
                gate_location=gate_location,
                gate_length=gate_length,
                grid_points=len(fvm_result.x_coordinates),
                scheme_used=self.fvm_solver.scheme.name,
                convergence_iterations=fvm_result.iterations
            )
            
            # Analyze detailed gate flow characteristics
            gate_characteristics = gate_profile.analyze_gate_flow_details()
            
            # Assess cavitation risk from FVM pressure field
            cavitation_analysis = gate_profile.assess_cavitation_risk()
            
            # Create enhanced gate flow result with FVM data
            result = GateFlowResult(
                gate_type=gate.gate_type,
                gate_opening=gate.gate_opening,
                gate_width=gate.gate_width,
                discharge=analytical_result.discharge,
                upstream_depth=analytical_result.upstream_depth,
                downstream_depth=analytical_result.downstream_depth,
                contracted_depth=gate_characteristics.get('min_depth', analytical_result.contracted_depth),
                upstream_velocity=analytical_result.upstream_velocity,
                gate_velocity=gate_characteristics.get('max_velocity', analytical_result.gate_velocity),
                downstream_velocity=analytical_result.downstream_velocity,
                upstream_energy=analytical_result.upstream_energy,
                gate_energy=analytical_result.gate_energy,
                downstream_energy=analytical_result.downstream_energy,
                energy_loss=analytical_result.energy_loss,
                discharge_coefficient=analytical_result.discharge_coefficient,
                contraction_coefficient=analytical_result.contraction_coefficient,
                velocity_coefficient=analytical_result.velocity_coefficient,
                flow_condition=analytical_result.flow_condition,
                froude_upstream=analytical_result.froude_upstream,
                froude_gate=analytical_result.froude_gate,
                froude_downstream=analytical_result.froude_downstream,
                hydraulic_jump_required=analytical_result.hydraulic_jump_required,
                cavitation_risk=cavitation_analysis['cavitation_risk'],
                cavitation_index=cavitation_analysis['pressure_margin'],
                success=True,
                message=f"FVM gate flow analysis complete. {grid_points} grid points, {fvm_result.iterations} iterations",
                method_used="fvm",
                computation_time=time.time() - start_time,
                fvm_profile=gate_profile,
                properties={
                    **analytical_result.properties,
                    "fvm_converged": fvm_result.converged,
                    "fvm_iterations": fvm_result.iterations,
                    "fvm_final_residual": fvm_result.final_residual,
                    "fvm_grid_points": grid_points,
                    "fvm_domain_length": domain_length,
                    "fvm_scheme": self.fvm_solver.scheme.name,
                    "mass_conservation_error": fvm_result.calculate_mass_conservation_error(),
                    "vena_contracta_location": gate_characteristics.get('vena_contracta_location', 0),
                    "pressure_drop": gate_characteristics.get('pressure_drop', 0),
                    "velocity_increase": gate_characteristics.get('velocity_increase', 0)
                }
            )
            
            return result
            
        except Exception as e:
            # Fallback to analytical method with error message
            analytical_result = self._analyze_gate_flow_analytical(
                channel, gate, upstream_depth, downstream_depth, discharge
            )
            analytical_result.method_used = "fvm"
            analytical_result.computation_time = time.time() - start_time
            analytical_result.message += f" (FVM analysis failed: {str(e)} - using analytical fallback)"
            return analytical_result
    
    def _assess_cavitation_risk_fvm(self, gate_profile: GateFlowProfile) -> CavitationRisk:
        """Assess cavitation risk from FVM pressure field."""
        cavitation_analysis = gate_profile.assess_cavitation_risk()
        return cavitation_analysis['cavitation_risk']
    
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
