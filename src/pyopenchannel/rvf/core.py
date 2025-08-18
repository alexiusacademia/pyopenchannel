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
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
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
    for rapidly varied flow conditions. Supports both analytical and
    FVM methods with detailed profile data.
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
    
    # Method information
    method_used: str = "analytical"  # "analytical" or "fvm"
    computation_time: float = 0.0
    
    # FVM-specific detailed results (only populated if method_used == "fvm")
    fvm_profile: Optional['FVMProfile'] = None
    
    # Additional properties
    properties: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def has_detailed_profile(self) -> bool:
        """Check if detailed FVM profile is available."""
        return self.method_used == "fvm" and self.fvm_profile is not None
    
    @property
    def profile_resolution(self) -> str:
        """Get profile resolution description."""
        if self.method_used == "analytical":
            return "2-point (analytical)"
        elif self.has_detailed_profile:
            return f"{len(self.fvm_profile.x_coordinates)}-point (FVM)"
        else:
            return "2-point (analytical fallback)"
    
    @property
    def num_profile_points(self) -> int:
        """Get number of profile points."""
        if self.has_detailed_profile:
            return len(self.fvm_profile.x_coordinates)
        else:
            return 2


@dataclass
class FVMProfile:
    """
    Detailed FVM profile data for RVF analysis.
    
    Contains high-resolution flow field data from FVM simulation,
    providing detailed insight into hydraulic jump structure.
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
    
    # Advanced properties (optional)
    turbulence_intensity: Optional[np.ndarray] = None
    air_entrainment_rate: Optional[np.ndarray] = None
    
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
    
    def find_jump_location(self) -> Optional[float]:
        """Find hydraulic jump location based on maximum depth gradient."""
        if len(self.depths) < 3:
            return None
        
        # Calculate depth gradients
        depth_gradients = np.gradient(self.depths)
        
        # Find location of maximum positive gradient
        max_gradient_idx = np.argmax(depth_gradients)
        
        return self.x_coordinates[max_gradient_idx]
    
    def calculate_jump_characteristics(self) -> Dict[str, float]:
        """Calculate detailed jump characteristics from FVM profile."""
        jump_location = self.find_jump_location()
        if jump_location is None:
            return {}
        
        # Find upstream and downstream regions
        jump_idx = np.argmin(np.abs(self.x_coordinates - jump_location))
        
        # Upstream (average over 10% of domain before jump)
        upstream_region = max(0, jump_idx - len(self.x_coordinates) // 10)
        upstream_depth = np.mean(self.depths[upstream_region:jump_idx])
        upstream_velocity = np.mean(self.velocities[upstream_region:jump_idx])
        upstream_froude = np.mean(self.froude_numbers[upstream_region:jump_idx])
        
        # Downstream (average over 10% of domain after jump)
        downstream_region = min(len(self.x_coordinates), jump_idx + len(self.x_coordinates) // 10)
        downstream_depth = np.mean(self.depths[jump_idx:downstream_region])
        downstream_velocity = np.mean(self.velocities[jump_idx:downstream_region])
        downstream_froude = np.mean(self.froude_numbers[jump_idx:downstream_region])
        
        return {
            'jump_location': jump_location,
            'upstream_depth': upstream_depth,
            'downstream_depth': downstream_depth,
            'upstream_velocity': upstream_velocity,
            'downstream_velocity': downstream_velocity,
            'upstream_froude': upstream_froude,
            'downstream_froude': downstream_froude,
            'jump_height': downstream_depth - upstream_depth,
            'depth_ratio': downstream_depth / upstream_depth if upstream_depth > 0 else 0
        }


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
        method: str = "analytical",
        transition_threshold: float = 0.1,
        froude_gradient_threshold: float = 0.5,
        convergence_tolerance: float = 1e-6,
        max_iterations: int = 100
    ):
        """
        Initialize RVF solver.
        
        Args:
            method: Solution method ("analytical" or "fvm")
            transition_threshold: Threshold for detecting rapid transitions
            froude_gradient_threshold: Froude number gradient threshold for RVF detection
            convergence_tolerance: Convergence tolerance for iterative solutions
            max_iterations: Maximum iterations for convergence
        """
        # Validate method
        if method not in ["analytical", "fvm"]:
            raise ValueError(f"Invalid method: {method}. Must be 'analytical' or 'fvm'")
        
        self.method = method
        self.transition_threshold = transition_threshold
        self.froude_gradient_threshold = froude_gradient_threshold
        self.tolerance = convergence_tolerance
        self.max_iterations = max_iterations
        self.gravity = get_gravity()
        
        # Initialize FVM solver if needed
        self.fvm_solver = None
        if method == "fvm":
            self._initialize_fvm_solver()
    
    def _initialize_fvm_solver(self):
        """Initialize FVM solver for detailed analysis."""
        try:
            from ..numerical.fvm import (
                ShallowWaterSolver, ConvergenceCriteria, 
                TimeIntegrationMethod, InletBC, OutletBC, BoundaryData
            )
            
            # Create FVM solver with hydraulic-optimized settings
            criteria = ConvergenceCriteria(
                max_iterations=1000,
                residual_tolerance=1e-8,
                steady_state_tolerance=1e-10,
                cfl_number=0.3  # Conservative for stability
            )
            
            self.fvm_solver = ShallowWaterSolver(
                scheme_name="hllc",  # Best for shocks
                time_integration=TimeIntegrationMethod.RK4,
                convergence_criteria=criteria
            )
            
        except ImportError:
            raise ImportError(
                "FVM solver not available. Install FVM components or use method='analytical'"
            )
    
    def analyze_hydraulic_jump(
        self,
        channel: ChannelGeometry,
        discharge: float,
        upstream_depth: float,
        tailwater_depth: Optional[float] = None,
        method: Optional[str] = None
    ) -> RVFResult:
        """
        Analyze hydraulic jump characteristics.
        
        Args:
            channel: Channel geometry
            discharge: Discharge (m³/s or ft³/s)
            upstream_depth: Upstream depth (supercritical)
            tailwater_depth: Tailwater depth (optional)
            method: Override method ("analytical" or "fvm", None uses default)
            
        Returns:
            RVFResult with jump analysis
        """
        try:
            # Determine method to use
            analysis_method = method if method is not None else self.method
            
            # Validate method
            if analysis_method not in ["analytical", "fvm"]:
                raise ValueError(f"Invalid method: {analysis_method}")
            
            # Validate inputs
            discharge = validate_discharge(discharge)
            upstream_depth = validate_depth(upstream_depth)
            
            # Route to appropriate analysis method
            if analysis_method == "fvm":
                return self._analyze_hydraulic_jump_fvm(
                    channel, discharge, upstream_depth, tailwater_depth
                )
            else:
                return self._analyze_hydraulic_jump_analytical(
                    channel, discharge, upstream_depth, tailwater_depth
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
    
    def _analyze_hydraulic_jump_analytical(
        self,
        channel: ChannelGeometry,
        discharge: float,
        upstream_depth: float,
        tailwater_depth: Optional[float] = None
    ) -> RVFResult:
        """Analyze hydraulic jump using analytical methods (original implementation)."""
        import time
        start_time = time.time()
        
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
                method_used="analytical",
                computation_time=time.time() - start_time,
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
            method_used="analytical",
            computation_time=time.time() - start_time,
            properties={
                "tailwater_adequate": tailwater_adequate,
                "tailwater_depth": tailwater_depth,
                "energy_dissipation_ratio": energy_loss / upstream_energy,
                "momentum_function_upstream": self._momentum_function(channel, discharge, upstream_depth),
                "momentum_function_downstream": self._momentum_function(channel, discharge, sequent_depth)
            }
        )
    
    def _analyze_hydraulic_jump_fvm(
        self,
        channel: ChannelGeometry,
        discharge: float,
        upstream_depth: float,
        tailwater_depth: Optional[float] = None
    ) -> RVFResult:
        """Analyze hydraulic jump using FVM for detailed profile."""
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
            analytical_result = self._analyze_hydraulic_jump_analytical(
                channel, discharge, upstream_depth, tailwater_depth
            )
            
            if not analytical_result.success:
                # Return analytical result with FVM method flag
                analytical_result.method_used = "fvm"
                analytical_result.computation_time = time.time() - start_time
                return analytical_result
            
            # Create FVM grid for detailed analysis
            # Domain: upstream region + jump region + downstream region
            jump_length_est = analytical_result.jump_length or 10.0
            domain_length = max(50.0, jump_length_est * 5)  # 5x jump length or 50m minimum
            grid_points = min(200, max(100, int(domain_length * 2)))  # 2 points per meter, capped
            
            grid = UniformGrid(
                x_min=0.0,
                x_max=domain_length,
                num_cells=grid_points
            )
            
            # Initialize with hydraulic jump conditions
            jump_location = domain_length * 0.3  # Place jump at 30% of domain
            
            for cell in grid.cells:
                x = cell.x_center
                if x < jump_location:
                    # Upstream supercritical
                    cell.U = ConservativeVariables(
                        h=upstream_depth,
                        hu=upstream_depth * (discharge / channel.area(upstream_depth))
                    )
                else:
                    # Downstream subcritical
                    downstream_depth = analytical_result.downstream_depth
                    cell.U = ConservativeVariables(
                        h=downstream_depth,
                        hu=downstream_depth * (discharge / channel.area(downstream_depth))
                    )
            
            # Set boundary conditions
            upstream_velocity = discharge / channel.area(upstream_depth)
            downstream_depth = tailwater_depth or analytical_result.downstream_depth
            
            inlet_data = BoundaryData(depth=upstream_depth, velocity=upstream_velocity)
            outlet_data = BoundaryData(depth=downstream_depth)
            
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
            
            # Create detailed FVM profile
            g = self.gravity
            specific_energies = fvm_result.depths + fvm_result.velocities**2 / (2 * g)
            pressure_heads = fvm_result.depths  # Hydrostatic assumption
            
            fvm_profile = FVMProfile(
                x_coordinates=fvm_result.x_coordinates,
                depths=fvm_result.depths,
                velocities=fvm_result.velocities,
                froude_numbers=fvm_result.froude_numbers,
                specific_energies=specific_energies,
                pressure_heads=pressure_heads,
                grid_points=len(fvm_result.x_coordinates),
                scheme_used=self.fvm_solver.scheme.name,
                convergence_iterations=fvm_result.iterations
            )
            
            # Analyze FVM results for jump characteristics
            jump_characteristics = fvm_profile.calculate_jump_characteristics()
            
            # Create enhanced RVF result with FVM data
            result = RVFResult(
                upstream_depth=jump_characteristics.get('upstream_depth', analytical_result.upstream_depth),
                downstream_depth=jump_characteristics.get('downstream_depth', analytical_result.downstream_depth),
                upstream_velocity=jump_characteristics.get('upstream_velocity', analytical_result.upstream_velocity),
                downstream_velocity=jump_characteristics.get('downstream_velocity', analytical_result.downstream_velocity),
                upstream_froude=jump_characteristics.get('upstream_froude', analytical_result.upstream_froude),
                downstream_froude=jump_characteristics.get('downstream_froude', analytical_result.downstream_froude),
                upstream_energy=analytical_result.upstream_energy,
                downstream_energy=analytical_result.downstream_energy,
                energy_loss=analytical_result.energy_loss,
                energy_efficiency=analytical_result.energy_efficiency,
                momentum_change=analytical_result.momentum_change,
                jump_type=analytical_result.jump_type,
                jump_length=jump_characteristics.get('jump_location', analytical_result.jump_length),
                jump_height=jump_characteristics.get('jump_height', analytical_result.jump_height),
                sequent_depth_ratio=jump_characteristics.get('depth_ratio', analytical_result.sequent_depth_ratio),
                transition_type=TransitionType.HYDRAULIC_JUMP,
                regime_upstream=RVFRegime.SUPERCRITICAL,
                regime_downstream=RVFRegime.SUBCRITICAL,
                success=True,
                message=f"FVM hydraulic jump analysis complete. {grid_points} grid points, {fvm_result.iterations} iterations",
                method_used="fvm",
                computation_time=time.time() - start_time,
                fvm_profile=fvm_profile,
                properties={
                    **analytical_result.properties,
                    "fvm_converged": fvm_result.converged,
                    "fvm_iterations": fvm_result.iterations,
                    "fvm_final_residual": fvm_result.final_residual,
                    "fvm_grid_points": grid_points,
                    "fvm_domain_length": domain_length,
                    "fvm_scheme": self.fvm_solver.scheme.name,
                    "mass_conservation_error": fvm_result.calculate_mass_conservation_error()
                }
            )
            
            return result
            
        except Exception as e:
            # Fallback to analytical method with error message
            analytical_result = self._analyze_hydraulic_jump_analytical(
                channel, discharge, upstream_depth, tailwater_depth
            )
            analytical_result.method_used = "fvm"
            analytical_result.computation_time = time.time() - start_time
            analytical_result.message += f" (FVM analysis failed: {str(e)} - using analytical fallback)"
            return analytical_result
    
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
