"""
Comprehensive Weir Analysis - PyOpenChannel

Author: Alexius Academia
Email: alexius.sayco.academia@gmail.com

This module provides comprehensive analysis of flow over weirs including:

WEIR TYPES:
- Sharp-crested weirs (rectangular, triangular, trapezoidal)
- Broad-crested weirs and spillways
- Ogee spillway design with standard shapes

ANALYSIS FEATURES:
- Discharge coefficient calculations with approach velocity effects
- Submergence effects and modular limits
- Aeration requirements and cavitation analysis
- Energy dissipation downstream of weirs
- Approach channel design and velocity distribution

PROFESSIONAL CAPABILITIES:
- Standard weir equations (Rehbock, Kindsvater-Carter, etc.)
- Approach velocity corrections
- Submergence ratio calculations
- Cavitation index analysis
- Downstream energy dissipation
- Spillway shape optimization
- Aeration system design
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
from .gates import CavitationRisk


class WeirType(Enum):
    """Types of weirs for flow analysis."""
    SHARP_CRESTED_RECTANGULAR = "sharp_crested_rectangular"
    SHARP_CRESTED_TRIANGULAR = "sharp_crested_triangular"
    SHARP_CRESTED_TRAPEZOIDAL = "sharp_crested_trapezoidal"
    BROAD_CRESTED = "broad_crested"
    OGEE_SPILLWAY = "ogee_spillway"
    LABYRINTH = "labyrinth"
    PIANO_KEY = "piano_key"


class WeirCondition(Enum):
    """Flow conditions over weirs."""
    FREE_FLOW = "free_flow"           # Modular flow, no downstream influence
    SUBMERGED = "submerged"           # Non-modular flow, downstream influence
    TRANSITIONAL = "transitional"     # Between free and submerged
    AERATED = "aerated"               # With air entrainment
    NON_AERATED = "non_aerated"       # Without air entrainment


class AerationLevel(Enum):
    """Aeration requirement levels."""
    NOT_REQUIRED = "not_required"     # Low head, no aeration needed
    RECOMMENDED = "recommended"       # Moderate head, aeration beneficial
    REQUIRED = "required"             # High head, aeration essential
    CRITICAL = "critical"             # Very high head, multiple aeration points


class CavitationRisk(Enum):
    """Cavitation risk levels for weirs."""
    NEGLIGIBLE = "negligible"         # σ > 0.5
    LOW = "low"                       # 0.3 < σ ≤ 0.5
    MODERATE = "moderate"             # 0.1 < σ ≤ 0.3
    HIGH = "high"                     # 0.05 < σ ≤ 0.1
    CRITICAL = "critical"             # σ ≤ 0.05


@dataclass
class WeirGeometry:
    """Weir geometry parameters."""
    weir_type: WeirType
    weir_height: float                # Height of weir crest above channel bottom
    weir_width: Optional[float] = None        # Width for rectangular weirs
    crest_length: Optional[float] = None      # Effective crest length
    side_slope: Optional[float] = None        # Side slope for trapezoidal weirs
    vertex_angle: Optional[float] = None      # Vertex angle for triangular weirs (degrees)
    crest_thickness: Optional[float] = None   # Crest thickness for broad-crested weirs
    upstream_slope: Optional[float] = None    # Upstream face slope
    downstream_slope: Optional[float] = None  # Downstream face slope
    approach_length: float = 10.0             # Approach channel length
    spillway_shape: Optional[str] = None      # Standard spillway shape (WES, etc.)


@dataclass
class WeirFlowResult:
    """
    Result of weir flow analysis.
    
    Contains all computed hydraulic parameters for flow over weirs.
    """
    # Weir properties
    weir_type: WeirType
    weir_height: float
    effective_length: float
    
    # Flow conditions
    discharge: float
    head_over_weir: float             # H - head over weir crest
    approach_depth: float
    approach_velocity: float
    downstream_depth: Optional[float]
    
    # Hydraulic parameters
    discharge_coefficient: float      # Cd
    velocity_coefficient: float       # Cv
    approach_velocity_factor: float   # Kv
    submergence_factor: float         # Ks
    
    # Energy analysis
    approach_energy: float
    crest_energy: float
    downstream_energy: Optional[float]
    energy_dissipated: float
    energy_efficiency: float
    
    # Flow classification
    weir_condition: WeirCondition
    froude_approach: float
    froude_downstream: Optional[float]
    
    # Professional analysis
    modular_limit: float              # Maximum submergence for free flow
    submergence_ratio: float          # Downstream/upstream head ratio
    aeration_requirement: AerationLevel
    cavitation_risk: CavitationRisk
    cavitation_index: float
    
    # Downstream conditions
    nappe_trajectory: Optional[Dict[str, float]]
    energy_dissipation_length: float
    scour_potential: str
    
    # Analysis metadata
    success: bool
    message: str
    properties: Dict[str, Any]
    
    # Optional analysis results
    spillway_profile: Optional[List[Tuple[float, float]]] = None
    aeration_analysis: Optional[Dict[str, Any]] = None
    approach_flow_analysis: Optional[Dict[str, Any]] = None
    
    # FVM integration fields
    method_used: str = "analytical"
    computation_time: float = 0.0
    fvm_profile: Optional['WeirFlowProfile'] = None
    
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
class WeirFlowProfile:
    """
    Detailed FVM profile data for weir flow analysis.
    
    Contains high-resolution flow field data from FVM simulation,
    providing detailed insight into weir flow structure including
    pressure distribution over weir crest, velocity profiles,
    aeration analysis, and cavitation assessment.
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
    
    # Weir-specific properties
    weir_crest_location: float
    weir_length: float
    weir_height: float
    
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
    
    def find_weir_crest_conditions(self) -> Dict[str, float]:
        """Find flow conditions at weir crest."""
        # Find index closest to weir crest
        crest_idx = np.argmin(np.abs(self.x_coordinates - self.weir_crest_location))
        
        return {
            'crest_depth': self.depths[crest_idx],
            'crest_velocity': self.velocities[crest_idx],
            'crest_froude': self.froude_numbers[crest_idx],
            'crest_energy': self.specific_energies[crest_idx],
            'crest_pressure': self.pressure_heads[crest_idx]
        }
    
    def analyze_pressure_distribution(self) -> Dict[str, Any]:
        """Analyze pressure distribution over weir for cavitation assessment."""
        # Find weir region
        weir_start = self.weir_crest_location
        weir_end = self.weir_crest_location + self.weir_length
        
        weir_mask = (self.x_coordinates >= weir_start) & (self.x_coordinates <= weir_end)
        weir_pressures = self.pressure_heads[weir_mask]
        weir_x = self.x_coordinates[weir_mask]
        
        if len(weir_pressures) == 0:
            return {}
        
        min_pressure = np.min(weir_pressures)
        min_pressure_location = weir_x[np.argmin(weir_pressures)]
        max_pressure = np.max(weir_pressures)
        avg_pressure = np.mean(weir_pressures)
        
        # Cavitation risk assessment
        cavitation_threshold = 2.0  # meters of water (approximate)
        
        if min_pressure < cavitation_threshold:
            if min_pressure < 0.5:
                cavitation_risk = "SEVERE"
            elif min_pressure < 1.0:
                cavitation_risk = "HIGH"
            else:
                cavitation_risk = "MODERATE"
        else:
            cavitation_risk = "LOW"
        
        return {
            'min_pressure': min_pressure,
            'min_pressure_location': min_pressure_location,
            'max_pressure': max_pressure,
            'avg_pressure': avg_pressure,
            'pressure_range': max_pressure - min_pressure,
            'cavitation_risk': cavitation_risk,
            'cavitation_threshold': cavitation_threshold,
            'pressure_margin': min_pressure - cavitation_threshold
        }
    
    def analyze_velocity_distribution(self) -> Dict[str, Any]:
        """Analyze velocity distribution for aeration requirements."""
        max_velocity = np.max(self.velocities)
        max_velocity_location = self.x_coordinates[np.argmax(self.velocities)]
        avg_velocity = np.mean(self.velocities)
        
        # Find velocity at weir crest
        crest_conditions = self.find_weir_crest_conditions()
        crest_velocity = crest_conditions['crest_velocity']
        
        # Aeration requirement assessment (simplified)
        if max_velocity > 15.0:
            aeration_level = "HIGH"
        elif max_velocity > 10.0:
            aeration_level = "MODERATE"
        elif max_velocity > 6.0:
            aeration_level = "LOW"
        else:
            aeration_level = "NONE"
        
        return {
            'max_velocity': max_velocity,
            'max_velocity_location': max_velocity_location,
            'avg_velocity': avg_velocity,
            'crest_velocity': crest_velocity,
            'velocity_increase': max_velocity - avg_velocity,
            'aeration_requirement': aeration_level
        }
    
    def analyze_energy_dissipation(self) -> Dict[str, Any]:
        """Analyze energy dissipation downstream of weir."""
        # Upstream energy (before weir)
        upstream_region = self.x_coordinates < self.weir_crest_location
        if np.any(upstream_region):
            upstream_energy = np.mean(self.specific_energies[upstream_region])
        else:
            upstream_energy = self.specific_energies[0]
        
        # Downstream energy (after weir)
        downstream_region = self.x_coordinates > (self.weir_crest_location + self.weir_length)
        if np.any(downstream_region):
            downstream_energy = np.mean(self.specific_energies[downstream_region])
        else:
            downstream_energy = self.specific_energies[-1]
        
        energy_loss = upstream_energy - downstream_energy
        energy_efficiency = downstream_energy / upstream_energy if upstream_energy > 0 else 0
        
        return {
            'upstream_energy': upstream_energy,
            'downstream_energy': downstream_energy,
            'energy_loss': energy_loss,
            'energy_efficiency': energy_efficiency,
            'energy_dissipation_ratio': energy_loss / upstream_energy if upstream_energy > 0 else 0
        }
    
    def calculate_spillway_characteristics(self) -> Dict[str, Any]:
        """Calculate detailed spillway flow characteristics."""
        crest_conditions = self.find_weir_crest_conditions()
        pressure_analysis = self.analyze_pressure_distribution()
        velocity_analysis = self.analyze_velocity_distribution()
        energy_analysis = self.analyze_energy_dissipation()
        
        return {
            **crest_conditions,
            **pressure_analysis,
            **velocity_analysis,
            **energy_analysis,
            'weir_efficiency': energy_analysis['energy_efficiency'],
            'flow_stability': 'stable' if pressure_analysis.get('cavitation_risk', 'HIGH') == 'LOW' else 'unstable'
        }


class WeirFlowSolver:
    """
    Comprehensive weir flow analysis solver.
    
    This solver handles all types of weirs using:
    - Standard weir equations (Rehbock, Kindsvater-Carter, etc.)
    - Approach velocity corrections
    - Submergence effects analysis
    - Cavitation and aeration assessment
    - Energy dissipation calculations
    - Professional design recommendations
    """
    
    def __init__(
        self,
        method: str = "analytical",
        convergence_tolerance: float = 1e-6,
        max_iterations: int = 100
    ):
        """
        Initialize weir flow solver.
        
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
        """Initialize FVM solver for detailed weir flow analysis."""
        try:
            from ..numerical.fvm import (
                ShallowWaterSolver, ConvergenceCriteria, 
                TimeIntegrationMethod, InletBC, OutletBC, BoundaryData
            )
            
            # Create FVM solver optimized for weir flow
            criteria = ConvergenceCriteria(
                max_iterations=1000,
                residual_tolerance=1e-8,
                steady_state_tolerance=1e-10,
                cfl_number=0.15  # Very conservative for complex weir geometry
            )
            
            self.fvm_solver = ShallowWaterSolver(
                scheme_name="hllc",  # Best for complex flow patterns and shocks
                time_integration=TimeIntegrationMethod.RK4,
                convergence_criteria=criteria
            )
            
        except ImportError:
            raise ImportError(
                "FVM solver not available. Install FVM components or use method='analytical'"
            )
        self.rvf_solver = RVFSolver()
    
    def analyze_weir_flow(
        self,
        channel: ChannelGeometry,
        weir: WeirGeometry,
        approach_depth: float,
        discharge: Optional[float] = None,
        downstream_depth: Optional[float] = None,
        method: Optional[str] = None
    ) -> WeirFlowResult:
        """
        Analyze flow over a weir.
        
        Args:
            channel: Channel geometry
            weir: Weir geometry and properties
            approach_depth: Approach flow depth
            discharge: Known discharge (optional, will be calculated if not provided)
            downstream_depth: Downstream water depth (optional)
            method: Override method ("analytical" or "fvm", None uses default)
            
        Returns:
            WeirFlowResult with complete analysis
        """
        try:
            # Determine method to use
            analysis_method = method if method is not None else self.method
            
            # Validate method
            if analysis_method not in ["analytical", "fvm"]:
                raise ValueError(f"Invalid method: {analysis_method}")
            
            # Validate inputs
            approach_depth = validate_depth(approach_depth)
            if discharge is not None:
                discharge = validate_discharge(discharge)
            if downstream_depth is not None:
                downstream_depth = validate_depth(downstream_depth)
            
            # Route to appropriate analysis method
            if analysis_method == "fvm":
                return self._analyze_weir_flow_fvm(
                    channel, weir, approach_depth, discharge, downstream_depth
                )
            else:
                return self._analyze_weir_flow_analytical(
                    channel, weir, approach_depth, discharge, downstream_depth
                )
            
            # Calculate head over weir
            head_over_weir = approach_depth - weir.weir_height
            
            if head_over_weir <= 0:
                return self._create_failed_result(
                    weir, "Approach depth is below weir crest - no flow over weir"
                )
            
            # Calculate discharge if not provided
            if discharge is None:
                discharge = self._calculate_weir_discharge(
                    channel, weir, approach_depth, head_over_weir, downstream_depth
                )
            
            # Calculate approach conditions
            approach_analysis = self._analyze_approach_conditions(
                channel, discharge, approach_depth, weir
            )
            
            # Determine flow condition and submergence effects
            flow_condition, submergence_analysis = self._analyze_submergence_effects(
                weir, head_over_weir, downstream_depth, discharge
            )
            
            # Calculate discharge coefficients
            coefficients = self._calculate_discharge_coefficients(
                weir, head_over_weir, approach_analysis, submergence_analysis
            )
            
            # Energy analysis
            energy_analysis = self._calculate_energy_analysis(
                channel, weir, discharge, approach_depth, head_over_weir, downstream_depth
            )
            
            # Aeration and cavitation analysis
            aeration_analysis = self._analyze_aeration_requirements(
                weir, head_over_weir, discharge
            )
            cavitation_analysis = self._analyze_cavitation_risk(
                weir, head_over_weir, approach_analysis["velocity"]
            )
            
            # Downstream conditions analysis
            downstream_analysis = self._analyze_downstream_conditions(
                channel, weir, discharge, head_over_weir, downstream_depth
            )
            
            # Calculate effective length
            effective_length = self._calculate_effective_length(weir, head_over_weir)
            
            return WeirFlowResult(
                weir_type=weir.weir_type,
                weir_height=weir.weir_height,
                effective_length=effective_length,
                discharge=discharge,
                head_over_weir=head_over_weir,
                approach_depth=approach_depth,
                approach_velocity=approach_analysis["velocity"],
                downstream_depth=downstream_depth,
                discharge_coefficient=coefficients["discharge_coefficient"],
                velocity_coefficient=coefficients["velocity_coefficient"],
                approach_velocity_factor=coefficients["approach_velocity_factor"],
                submergence_factor=submergence_analysis["submergence_factor"],
                approach_energy=energy_analysis["approach_energy"],
                crest_energy=energy_analysis["crest_energy"],
                downstream_energy=energy_analysis.get("downstream_energy"),
                energy_dissipated=energy_analysis["energy_dissipated"],
                energy_efficiency=energy_analysis["energy_efficiency"],
                weir_condition=flow_condition,
                froude_approach=approach_analysis["froude_number"],
                froude_downstream=downstream_analysis.get("froude_number"),
                modular_limit=submergence_analysis["modular_limit"],
                submergence_ratio=submergence_analysis["submergence_ratio"],
                aeration_requirement=aeration_analysis["requirement_level"],
                cavitation_risk=cavitation_analysis["risk_level"],
                cavitation_index=cavitation_analysis["cavitation_index"],
                nappe_trajectory=downstream_analysis.get("nappe_trajectory"),
                energy_dissipation_length=downstream_analysis["energy_dissipation_length"],
                scour_potential=downstream_analysis["scour_potential"],
                success=True,
                message=f"Weir flow analysis complete. Condition: {flow_condition.value}",
                properties={
                    "weir_efficiency": coefficients["discharge_coefficient"],
                    "approach_channel_adequate": approach_analysis["channel_adequate"],
                    "aeration_details": aeration_analysis,
                    "cavitation_details": cavitation_analysis
                },
                spillway_profile=self._generate_spillway_profile(weir) if weir.weir_type == WeirType.OGEE_SPILLWAY else None,
                aeration_analysis=aeration_analysis,
                approach_flow_analysis=approach_analysis
            )
            
        except Exception as e:
            return self._create_failed_result(weir, f"Weir flow analysis failed: {str(e)}")
    
    def _analyze_weir_flow_analytical(
        self,
        channel: ChannelGeometry,
        weir: WeirGeometry,
        approach_depth: float,
        discharge: Optional[float] = None,
        downstream_depth: Optional[float] = None
    ) -> WeirFlowResult:
        """Analyze weir flow using analytical methods (original implementation)."""
        import time
        start_time = time.time()
        
        # Calculate head over weir
        head_over_weir = approach_depth - weir.weir_height
        
        if head_over_weir <= 0:
            result = self._create_failed_result(
                weir, "Approach depth is below weir crest - no flow over weir"
            )
            result.method_used = "analytical"
            result.computation_time = time.time() - start_time
            return result
        
        # Calculate discharge if not provided
        if discharge is None:
            discharge = self._calculate_weir_discharge(
                channel, weir, approach_depth, head_over_weir
            )
        
        # Determine weir condition (free flow vs submerged)
        weir_condition = self._classify_weir_condition(
            weir, approach_depth, downstream_depth, head_over_weir
        )
        
        # Calculate hydraulic parameters
        result = self._calculate_weir_hydraulics(
            channel, weir, approach_depth, discharge, head_over_weir, 
            downstream_depth, weir_condition
        )
        
        # Add method information
        result.method_used = "analytical"
        result.computation_time = time.time() - start_time
        
        return result
    
    def _analyze_weir_flow_fvm(
        self,
        channel: ChannelGeometry,
        weir: WeirGeometry,
        approach_depth: float,
        discharge: Optional[float] = None,
        downstream_depth: Optional[float] = None
    ) -> WeirFlowResult:
        """Analyze weir flow using FVM for detailed pressure and velocity profiles."""
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
            analytical_result = self._analyze_weir_flow_analytical(
                channel, weir, approach_depth, discharge, downstream_depth
            )
            
            if not analytical_result.success:
                # Return analytical result with FVM method flag
                analytical_result.method_used = "fvm"
                analytical_result.computation_time = time.time() - start_time
                return analytical_result
            
            # Create FVM grid for detailed weir flow analysis
            # Domain: upstream approach + weir region + downstream recovery
            weir_length = max(weir.effective_length, 5.0)  # Minimum 5m weir length
            upstream_length = max(30.0, approach_depth * 15)  # 15x approach depth
            downstream_length = max(40.0, approach_depth * 20)  # 20x approach depth for energy dissipation
            domain_length = upstream_length + weir_length + downstream_length
            
            # Very high resolution for weir flow (pressure gradients are steep)
            grid_points = min(400, max(200, int(domain_length * 4)))  # 4 points per meter
            
            grid = UniformGrid(
                x_min=0.0,
                x_max=domain_length,
                num_cells=grid_points
            )
            
            # Initialize flow field with weir flow conditions
            weir_crest_location = upstream_length
            
            for cell in grid.cells:
                x = cell.x_center
                if x < weir_crest_location:
                    # Upstream approach flow
                    approach_velocity = analytical_result.approach_velocity
                    cell.U = ConservativeVariables(
                        h=approach_depth,
                        hu=approach_depth * approach_velocity
                    )
                elif x < weir_crest_location + weir_length:
                    # Weir region - critical/supercritical flow
                    # Use critical depth as approximation
                    critical_depth = analytical_result.head_over_weir * 0.67  # Approximate
                    critical_velocity = analytical_result.discharge / (channel.area(critical_depth))
                    cell.U = ConservativeVariables(
                        h=critical_depth,
                        hu=critical_depth * critical_velocity
                    )
                else:
                    # Downstream recovery
                    if downstream_depth is not None:
                        downstream_velocity = analytical_result.discharge / channel.area(downstream_depth)
                        cell.U = ConservativeVariables(
                            h=downstream_depth,
                            hu=downstream_depth * downstream_velocity
                        )
                    else:
                        # Use approach conditions as fallback
                        cell.U = ConservativeVariables(
                            h=approach_depth,
                            hu=approach_depth * analytical_result.approach_velocity
                        )
            
            # Set boundary conditions
            inlet_data = BoundaryData(
                depth=approach_depth, 
                velocity=analytical_result.approach_velocity
            )
            
            if downstream_depth is not None:
                outlet_data = BoundaryData(depth=downstream_depth)
            else:
                # Use critical depth boundary
                outlet_data = BoundaryData(depth=approach_depth * 0.8)  # Approximate
            
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
            
            # Create detailed FVM profile for weir flow
            g = self.gravity
            specific_energies = fvm_result.depths + fvm_result.velocities**2 / (2 * g)
            
            # Calculate pressure distribution (critical for cavitation analysis)
            pressure_heads = []
            for i, (depth, velocity) in enumerate(zip(fvm_result.depths, fvm_result.velocities)):
                # Pressure head considering elevation changes over weir
                x = fvm_result.x_coordinates[i]
                
                # Approximate weir profile elevation
                if weir_crest_location <= x <= weir_crest_location + weir_length:
                    # On weir crest
                    weir_elevation = weir.weir_height
                else:
                    # Channel bottom
                    weir_elevation = 0.0
                
                # Pressure head = total head - velocity head - elevation
                pressure_head = specific_energies[i] - velocity**2 / (2 * g) - weir_elevation
                pressure_heads.append(max(0, pressure_head))  # Prevent negative pressures
            
            pressure_heads = np.array(pressure_heads)
            
            # Create weir flow profile
            weir_profile = WeirFlowProfile(
                x_coordinates=fvm_result.x_coordinates,
                depths=fvm_result.depths,
                velocities=fvm_result.velocities,
                froude_numbers=fvm_result.froude_numbers,
                specific_energies=specific_energies,
                pressure_heads=pressure_heads,
                weir_crest_location=weir_crest_location,
                weir_length=weir_length,
                weir_height=weir.weir_height,
                grid_points=len(fvm_result.x_coordinates),
                scheme_used=self.fvm_solver.scheme.name,
                convergence_iterations=fvm_result.iterations
            )
            
            # Analyze detailed weir flow characteristics
            spillway_characteristics = weir_profile.calculate_spillway_characteristics()
            
            # Create enhanced weir flow result with FVM data
            result = WeirFlowResult(
                weir_type=weir.weir_type,
                weir_height=weir.weir_height,
                effective_length=weir.effective_length,
                discharge=analytical_result.discharge,
                head_over_weir=analytical_result.head_over_weir,
                approach_depth=analytical_result.approach_depth,
                approach_velocity=analytical_result.approach_velocity,
                downstream_depth=analytical_result.downstream_depth,
                discharge_coefficient=analytical_result.discharge_coefficient,
                velocity_coefficient=analytical_result.velocity_coefficient,
                approach_velocity_factor=analytical_result.approach_velocity_factor,
                submergence_factor=analytical_result.submergence_factor,
                approach_energy=analytical_result.approach_energy,
                crest_energy=spillway_characteristics.get('crest_energy', analytical_result.crest_energy),
                downstream_energy=analytical_result.downstream_energy,
                energy_dissipated=spillway_characteristics.get('energy_loss', analytical_result.energy_dissipated),
                energy_efficiency=spillway_characteristics.get('energy_efficiency', analytical_result.energy_efficiency),
                weir_condition=analytical_result.weir_condition,
                froude_approach=analytical_result.froude_approach,
                froude_downstream=analytical_result.froude_downstream,
                modular_limit=analytical_result.modular_limit,
                submergence_ratio=analytical_result.submergence_ratio,
                aeration_requirement=self._assess_aeration_from_fvm(spillway_characteristics),
                cavitation_risk=self._assess_cavitation_from_fvm(spillway_characteristics),
                cavitation_index=spillway_characteristics.get('pressure_margin', analytical_result.cavitation_index),
                nappe_trajectory=analytical_result.nappe_trajectory,
                energy_dissipation_length=analytical_result.energy_dissipation_length,
                scour_potential=analytical_result.scour_potential,
                success=True,
                message=f"FVM weir flow analysis complete. {grid_points} grid points, {fvm_result.iterations} iterations",
                method_used="fvm",
                computation_time=time.time() - start_time,
                fvm_profile=weir_profile,
                properties={
                    **analytical_result.properties,
                    "fvm_converged": fvm_result.converged,
                    "fvm_iterations": fvm_result.iterations,
                    "fvm_final_residual": fvm_result.final_residual,
                    "fvm_grid_points": grid_points,
                    "fvm_domain_length": domain_length,
                    "fvm_scheme": self.fvm_solver.scheme.name,
                    "mass_conservation_error": fvm_result.calculate_mass_conservation_error(),
                    **spillway_characteristics
                }
            )
            
            return result
            
        except Exception as e:
            # Fallback to analytical method with error message
            analytical_result = self._analyze_weir_flow_analytical(
                channel, weir, approach_depth, discharge, downstream_depth
            )
            analytical_result.method_used = "fvm"
            analytical_result.computation_time = time.time() - start_time
            analytical_result.message += f" (FVM analysis failed: {str(e)} - using analytical fallback)"
            return analytical_result
    
    def _assess_aeration_from_fvm(self, spillway_characteristics: Dict[str, Any]) -> 'AerationLevel':
        """Assess aeration requirements from FVM velocity analysis."""
        aeration_req = spillway_characteristics.get('aeration_requirement', 'LOW')
        
        # Convert string to enum
        if aeration_req == 'HIGH':
            return AerationLevel.HIGH
        elif aeration_req == 'MODERATE':
            return AerationLevel.MODERATE
        elif aeration_req == 'LOW':
            return AerationLevel.LOW
        else:
            return AerationLevel.NONE
    
    def _assess_cavitation_from_fvm(self, spillway_characteristics: Dict[str, Any]) -> 'CavitationRisk':
        """Assess cavitation risk from FVM pressure analysis."""
        cavitation_risk = spillway_characteristics.get('cavitation_risk', 'LOW')
        
        # Convert string to enum (assuming CavitationRisk is imported from gates)
        if cavitation_risk == 'SEVERE':
            return CavitationRisk.SEVERE
        elif cavitation_risk == 'HIGH':
            return CavitationRisk.HIGH
        elif cavitation_risk == 'MODERATE':
            return CavitationRisk.MODERATE
        else:
            return CavitationRisk.LOW
    
    def design_weir_dimensions(
        self,
        weir_type: WeirType,
        target_discharge: float,
        available_head: float,
        channel: ChannelGeometry,
        design_criteria: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Design weir dimensions for target discharge and available head.
        
        Args:
            weir_type: Type of weir to design
            target_discharge: Required discharge
            available_head: Available head over weir
            channel: Channel geometry
            design_criteria: Optional design criteria and constraints
            
        Returns:
            Dictionary with design results
        """
        try:
            criteria = design_criteria or {}
            
            if weir_type == WeirType.SHARP_CRESTED_RECTANGULAR:
                return self._design_rectangular_weir(
                    target_discharge, available_head, channel, criteria
                )
            elif weir_type == WeirType.SHARP_CRESTED_TRIANGULAR:
                return self._design_triangular_weir(
                    target_discharge, available_head, channel, criteria
                )
            elif weir_type == WeirType.SHARP_CRESTED_TRAPEZOIDAL:
                return self._design_trapezoidal_weir(
                    target_discharge, available_head, channel, criteria
                )
            elif weir_type == WeirType.BROAD_CRESTED:
                return self._design_broad_crested_weir(
                    target_discharge, available_head, channel, criteria
                )
            elif weir_type == WeirType.OGEE_SPILLWAY:
                return self._design_ogee_spillway(
                    target_discharge, available_head, channel, criteria
                )
            else:
                return {
                    "success": False,
                    "message": f"Design not implemented for weir type: {weir_type.value}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "message": f"Weir design failed: {str(e)}",
                "error": str(e)
            }
    
    def _calculate_weir_discharge(
        self,
        channel: ChannelGeometry,
        weir: WeirGeometry,
        approach_depth: float,
        head_over_weir: float,
        downstream_depth: Optional[float]
    ) -> float:
        """Calculate discharge over weir using appropriate equations."""
        
        if weir.weir_type == WeirType.SHARP_CRESTED_RECTANGULAR:
            return self._calculate_rectangular_weir_discharge(weir, head_over_weir, approach_depth)
        elif weir.weir_type == WeirType.SHARP_CRESTED_TRIANGULAR:
            return self._calculate_triangular_weir_discharge(weir, head_over_weir)
        elif weir.weir_type == WeirType.SHARP_CRESTED_TRAPEZOIDAL:
            return self._calculate_trapezoidal_weir_discharge(weir, head_over_weir)
        elif weir.weir_type == WeirType.BROAD_CRESTED:
            return self._calculate_broad_crested_weir_discharge(weir, head_over_weir, approach_depth)
        elif weir.weir_type == WeirType.OGEE_SPILLWAY:
            return self._calculate_ogee_spillway_discharge(weir, head_over_weir, approach_depth)
        else:
            # Default to rectangular weir equation
            return self._calculate_rectangular_weir_discharge(weir, head_over_weir, approach_depth)
    
    def _calculate_rectangular_weir_discharge(
        self, weir: WeirGeometry, head: float, approach_depth: float
    ) -> float:
        """Calculate discharge for rectangular sharp-crested weir using Rehbock equation."""
        
        # Rehbock equation with approach velocity correction
        # Q = Cd * L * H^(3/2) * sqrt(2g)
        
        # Base discharge coefficient (Rehbock)
        Cd = 0.602 + 0.083 * (head / weir.weir_height)
        
        # Approach velocity correction
        if approach_depth > 0:
            approach_velocity_head = head * 0.1  # Approximate correction
            effective_head = head + approach_velocity_head
        else:
            effective_head = head
        
        # Effective length (considering end contractions)
        if weir.weir_width:
            effective_length = weir.weir_width - 0.1 * head  # End contraction correction
        else:
            effective_length = weir.crest_length or 1.0
        
        discharge = Cd * effective_length * effective_head**(3/2) * math.sqrt(2 * self.gravity)
        
        return discharge
    
    def _calculate_triangular_weir_discharge(
        self, weir: WeirGeometry, head: float
    ) -> float:
        """Calculate discharge for triangular sharp-crested weir."""
        
        # Standard triangular weir equation
        # Q = Cd * (8/15) * sqrt(2g) * tan(θ/2) * H^(5/2)
        
        vertex_angle = weir.vertex_angle or 90.0  # Default to 90-degree V-notch
        theta_rad = math.radians(vertex_angle)
        
        # Discharge coefficient for triangular weir
        Cd = 0.58 + 0.05 * (head / (weir.weir_height + head))
        
        discharge = Cd * (8/15) * math.sqrt(2 * self.gravity) * math.tan(theta_rad/2) * head**(5/2)
        
        return discharge
    
    def _calculate_trapezoidal_weir_discharge(
        self, weir: WeirGeometry, head: float
    ) -> float:
        """Calculate discharge for trapezoidal sharp-crested weir."""
        
        # Trapezoidal weir = rectangular + triangular components
        
        # Rectangular component
        weir_width = weir.weir_width or weir.crest_length or 1.0
        Cd_rect = 0.602 + 0.083 * (head / weir.weir_height)
        Q_rect = Cd_rect * weir_width * head**(3/2) * math.sqrt(2 * self.gravity)
        
        # Triangular components (both sides)
        side_slope = weir.side_slope or 1.0  # Default 1:1 side slopes
        Cd_tri = 0.58
        Q_tri = 2 * Cd_tri * (8/15) * math.sqrt(2 * self.gravity) * side_slope * head**(5/2)
        
        return Q_rect + Q_tri
    
    def _calculate_broad_crested_weir_discharge(
        self, weir: WeirGeometry, head: float, approach_depth: float
    ) -> float:
        """Calculate discharge for broad-crested weir."""
        
        # Broad-crested weir equation
        # Q = Cd * L * H * sqrt(2g * H)
        
        # Discharge coefficient depends on head/crest_thickness ratio
        thickness = weir.crest_thickness or 0.5
        head_thickness_ratio = head / thickness
        
        if head_thickness_ratio < 0.1:
            Cd = 0.32  # Thick weir
        elif head_thickness_ratio > 2.0:
            Cd = 0.385  # Thin weir (approaching sharp-crested)
        else:
            # Interpolate
            Cd = 0.32 + (0.385 - 0.32) * (head_thickness_ratio - 0.1) / 1.9
        
        effective_length = weir.crest_length or weir.weir_width or 1.0
        
        discharge = Cd * effective_length * head * math.sqrt(2 * self.gravity * head)
        
        return discharge
    
    def _calculate_ogee_spillway_discharge(
        self, weir: WeirGeometry, head: float, approach_depth: float
    ) -> float:
        """Calculate discharge for ogee spillway."""
        
        # Standard ogee spillway equation
        # Q = Cd * L * H^(3/2) * sqrt(2g)
        
        # Discharge coefficient for ogee spillway (typically higher than sharp-crested)
        Cd = 0.75  # Standard value for well-designed ogee spillway
        
        # Approach velocity correction
        approach_velocity_correction = 1 + 0.1 * (head / approach_depth)
        
        effective_length = weir.crest_length or weir.weir_width or 1.0
        
        discharge = Cd * effective_length * head**(3/2) * math.sqrt(2 * self.gravity) * approach_velocity_correction
        
        return discharge
    
    def _analyze_approach_conditions(
        self,
        channel: ChannelGeometry,
        discharge: float,
        approach_depth: float,
        weir: WeirGeometry
    ) -> Dict[str, Any]:
        """Analyze approach channel flow conditions."""
        
        approach_area = channel.area(approach_depth)
        approach_velocity = discharge / approach_area
        froude_number = approach_velocity / math.sqrt(self.gravity * approach_depth)
        
        # Check if approach channel is adequate
        # Velocity should be reasonable (< 3 m/s for good measurement accuracy)
        channel_adequate = approach_velocity < 3.0 and froude_number < 0.5
        
        # Velocity distribution factor
        velocity_distribution_factor = 1.0  # Assume uniform distribution
        
        return {
            "velocity": approach_velocity,
            "froude_number": froude_number,
            "channel_adequate": channel_adequate,
            "velocity_distribution_factor": velocity_distribution_factor,
            "approach_length_adequate": weir.approach_length > 10 * approach_depth
        }
    
    def _analyze_submergence_effects(
        self,
        weir: WeirGeometry,
        head_over_weir: float,
        downstream_depth: Optional[float],
        discharge: float
    ) -> Tuple[WeirCondition, Dict[str, Any]]:
        """Analyze submergence effects and determine flow condition."""
        
        if downstream_depth is None:
            return WeirCondition.FREE_FLOW, {
                "submergence_ratio": 0.0,
                "submergence_factor": 1.0,
                "modular_limit": 0.7  # Typical modular limit
            }
        
        # Calculate submergence ratio
        downstream_head = downstream_depth - weir.weir_height
        
        if downstream_head <= 0:
            submergence_ratio = 0.0
        else:
            submergence_ratio = downstream_head / head_over_weir
        
        # Determine modular limit based on weir type
        if weir.weir_type in [WeirType.SHARP_CRESTED_RECTANGULAR, WeirType.SHARP_CRESTED_TRAPEZOIDAL]:
            modular_limit = 0.7
        elif weir.weir_type == WeirType.SHARP_CRESTED_TRIANGULAR:
            modular_limit = 0.6
        elif weir.weir_type == WeirType.BROAD_CRESTED:
            modular_limit = 0.8
        elif weir.weir_type == WeirType.OGEE_SPILLWAY:
            modular_limit = 0.75
        else:
            modular_limit = 0.7
        
        # Determine flow condition
        if submergence_ratio < modular_limit:
            flow_condition = WeirCondition.FREE_FLOW
            submergence_factor = 1.0
        elif submergence_ratio < 0.9:
            flow_condition = WeirCondition.TRANSITIONAL
            # Villemonte equation for submergence factor
            submergence_factor = (1 - (submergence_ratio / modular_limit)**1.5)**0.385
        else:
            flow_condition = WeirCondition.SUBMERGED
            submergence_factor = (1 - (submergence_ratio / modular_limit)**1.5)**0.385
        
        return flow_condition, {
            "submergence_ratio": submergence_ratio,
            "submergence_factor": submergence_factor,
            "modular_limit": modular_limit
        }
    
    def _calculate_discharge_coefficients(
        self,
        weir: WeirGeometry,
        head: float,
        approach_analysis: Dict[str, Any],
        submergence_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate various discharge coefficients."""
        
        # Base discharge coefficient
        if weir.weir_type == WeirType.SHARP_CRESTED_RECTANGULAR:
            Cd_base = 0.602 + 0.083 * (head / weir.weir_height)
        elif weir.weir_type == WeirType.SHARP_CRESTED_TRIANGULAR:
            Cd_base = 0.58 + 0.05 * (head / (weir.weir_height + head))
        elif weir.weir_type == WeirType.BROAD_CRESTED:
            Cd_base = 0.35
        elif weir.weir_type == WeirType.OGEE_SPILLWAY:
            Cd_base = 0.75
        else:
            Cd_base = 0.6
        
        # Approach velocity factor
        approach_velocity = approach_analysis["velocity"]
        approach_velocity_factor = 1 + approach_velocity**2 / (2 * self.gravity * head)
        
        # Final discharge coefficient
        discharge_coefficient = Cd_base * approach_velocity_factor * submergence_analysis["submergence_factor"]
        
        # Velocity coefficient
        velocity_coefficient = discharge_coefficient / 0.6  # Normalized to theoretical
        
        return {
            "discharge_coefficient": discharge_coefficient,
            "velocity_coefficient": velocity_coefficient,
            "approach_velocity_factor": approach_velocity_factor,
            "base_coefficient": Cd_base
        }
    
    def _calculate_energy_analysis(
        self,
        channel: ChannelGeometry,
        weir: WeirGeometry,
        discharge: float,
        approach_depth: float,
        head: float,
        downstream_depth: Optional[float]
    ) -> Dict[str, Any]:
        """Calculate energy analysis for weir flow."""
        
        # Approach energy
        approach_area = channel.area(approach_depth)
        approach_velocity = discharge / approach_area
        approach_energy = approach_depth + approach_velocity**2 / (2 * self.gravity)
        
        # Energy at weir crest (critical flow assumption)
        critical_depth = (discharge**2 / (self.gravity * (weir.crest_length or 1.0)**2))**(1/3)
        critical_velocity = discharge / (weir.crest_length or 1.0) / critical_depth
        crest_energy = weir.weir_height + critical_depth + critical_velocity**2 / (2 * self.gravity)
        
        # Downstream energy
        if downstream_depth is not None:
            downstream_area = channel.area(downstream_depth)
            downstream_velocity = discharge / downstream_area
            downstream_energy = downstream_depth + downstream_velocity**2 / (2 * self.gravity)
        else:
            downstream_energy = None
        
        # Energy dissipated
        if downstream_energy is not None:
            energy_dissipated = approach_energy - downstream_energy
            energy_efficiency = downstream_energy / approach_energy
        else:
            energy_dissipated = approach_energy - crest_energy
            energy_efficiency = crest_energy / approach_energy
        
        return {
            "approach_energy": approach_energy,
            "crest_energy": crest_energy,
            "downstream_energy": downstream_energy,
            "energy_dissipated": energy_dissipated,
            "energy_efficiency": energy_efficiency
        }
    
    def _analyze_aeration_requirements(
        self,
        weir: WeirGeometry,
        head: float,
        discharge: float
    ) -> Dict[str, Any]:
        """Analyze aeration requirements for weir flow."""
        
        # Aeration requirements based on head and discharge
        unit_discharge = discharge / (weir.crest_length or 1.0)
        
        # Criteria for aeration requirements
        if head < 1.0:
            requirement_level = AerationLevel.NOT_REQUIRED
            aeration_capacity = 0.0
        elif head < 3.0:
            requirement_level = AerationLevel.RECOMMENDED
            aeration_capacity = 0.02 * unit_discharge  # 2% of discharge
        elif head < 10.0:
            requirement_level = AerationLevel.REQUIRED
            aeration_capacity = 0.05 * unit_discharge  # 5% of discharge
        else:
            requirement_level = AerationLevel.CRITICAL
            aeration_capacity = 0.08 * unit_discharge  # 8% of discharge
        
        # Aeration system design
        if requirement_level in [AerationLevel.REQUIRED, AerationLevel.CRITICAL]:
            aeration_slots = max(2, int(head / 2))  # One slot per 2m of head
            slot_width = 0.3  # 30cm wide slots
            slot_area = aeration_slots * slot_width * 1.0  # 1m deep slots
        else:
            aeration_slots = 0
            slot_width = 0.0
            slot_area = 0.0
        
        return {
            "requirement_level": requirement_level,
            "aeration_capacity": aeration_capacity,
            "unit_discharge": unit_discharge,
            "aeration_slots": aeration_slots,
            "slot_width": slot_width,
            "slot_area": slot_area,
            "aeration_ratio": aeration_capacity / discharge if discharge > 0 else 0
        }
    
    def _analyze_cavitation_risk(
        self,
        weir: WeirGeometry,
        head: float,
        approach_velocity: float
    ) -> Dict[str, Any]:
        """Analyze cavitation risk for weir flow."""
        
        # Cavitation index calculation
        # σ = (P_atm + ρgh - P_vapor) / (0.5 * ρ * V²)
        
        # Simplified cavitation index
        velocity_head = approach_velocity**2 / (2 * self.gravity)
        pressure_head = head  # Simplified
        
        if velocity_head > 0:
            cavitation_index = pressure_head / velocity_head
        else:
            cavitation_index = 10.0  # Very high (safe)
        
        # Risk classification
        if cavitation_index > 0.5:
            risk_level = CavitationRisk.NEGLIGIBLE
        elif cavitation_index > 0.3:
            risk_level = CavitationRisk.LOW
        elif cavitation_index > 0.1:
            risk_level = CavitationRisk.MODERATE
        elif cavitation_index > 0.05:
            risk_level = CavitationRisk.HIGH
        else:
            risk_level = CavitationRisk.CRITICAL
        
        return {
            "cavitation_index": cavitation_index,
            "risk_level": risk_level,
            "velocity_head": velocity_head,
            "pressure_head": pressure_head
        }
    
    def _analyze_downstream_conditions(
        self,
        channel: ChannelGeometry,
        weir: WeirGeometry,
        discharge: float,
        head: float,
        downstream_depth: Optional[float]
    ) -> Dict[str, Any]:
        """Analyze downstream flow conditions and energy dissipation."""
        
        # Nappe trajectory calculation (for free-falling nappe)
        if downstream_depth is None or weir.weir_type in [WeirType.SHARP_CRESTED_RECTANGULAR, WeirType.SHARP_CRESTED_TRIANGULAR]:
            # Free nappe trajectory
            nappe_trajectory = self._calculate_nappe_trajectory(weir, head)
        else:
            nappe_trajectory = None
        
        # Energy dissipation length
        if downstream_depth is not None:
            # Estimate energy dissipation length based on head difference
            head_difference = head
            energy_dissipation_length = 4 * head_difference  # Empirical relation
        else:
            energy_dissipation_length = 6 * head  # For free nappe
        
        # Scour potential assessment
        unit_discharge = discharge / (weir.crest_length or 1.0)
        if unit_discharge < 2.0:
            scour_potential = "Low"
        elif unit_discharge < 5.0:
            scour_potential = "Moderate"
        elif unit_discharge < 10.0:
            scour_potential = "High"
        else:
            scour_potential = "Very High"
        
        # Downstream Froude number
        if downstream_depth is not None:
            downstream_area = channel.area(downstream_depth)
            downstream_velocity = discharge / downstream_area
            froude_downstream = downstream_velocity / math.sqrt(self.gravity * downstream_depth)
        else:
            froude_downstream = None
        
        return {
            "nappe_trajectory": nappe_trajectory,
            "energy_dissipation_length": energy_dissipation_length,
            "scour_potential": scour_potential,
            "froude_number": froude_downstream,
            "unit_discharge": unit_discharge
        }
    
    def _calculate_nappe_trajectory(
        self, weir: WeirGeometry, head: float
    ) -> Dict[str, float]:
        """Calculate free nappe trajectory."""
        
        # Initial conditions at weir crest
        initial_velocity = math.sqrt(2 * self.gravity * head)
        
        # Trajectory parameters
        trajectory_points = []
        for x in [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]:  # Horizontal distances
            # y = x * tan(θ) - (g * x²) / (2 * V₀² * cos²(θ))
            # For horizontal discharge: θ = 0
            y = (self.gravity * x**2) / (2 * initial_velocity**2)
            trajectory_points.append((x, y))
        
        return {
            "initial_velocity": initial_velocity,
            "trajectory_points": trajectory_points,
            "maximum_range": initial_velocity**2 / self.gravity
        }
    
    def _calculate_effective_length(
        self, weir: WeirGeometry, head: float
    ) -> float:
        """Calculate effective crest length considering end contractions."""
        
        if weir.weir_type == WeirType.SHARP_CRESTED_RECTANGULAR:
            # Francis formula for end contractions
            if weir.weir_width:
                return weir.weir_width - 0.1 * head
            else:
                return weir.crest_length or 1.0
        else:
            return weir.crest_length or weir.weir_width or 1.0
    
    def _generate_spillway_profile(
        self, weir: WeirGeometry
    ) -> List[Tuple[float, float]]:
        """Generate standard spillway profile coordinates."""
        
        if weir.weir_type != WeirType.OGEE_SPILLWAY:
            return []
        
        # Standard WES (Waterways Experiment Station) spillway profile
        # Coordinates for design head
        profile_points = [
            (0.0, 0.0),      # Crest
            (0.1, -0.007),
            (0.2, -0.027),
            (0.3, -0.060),
            (0.4, -0.105),
            (0.5, -0.162),
            (0.6, -0.232),
            (0.7, -0.315),
            (0.8, -0.410),
            (0.9, -0.518),
            (1.0, -0.638)
        ]
        
        return profile_points
    
    def _design_rectangular_weir(
        self, discharge: float, head: float, channel: ChannelGeometry, criteria: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Design rectangular sharp-crested weir."""
        
        # Rehbock equation: Q = Cd * L * H^(3/2) * sqrt(2g)
        Cd = criteria.get("discharge_coefficient", 0.62)
        
        required_length = discharge / (Cd * head**(3/2) * math.sqrt(2 * self.gravity))
        
        # Apply practical constraints
        min_length = criteria.get("min_length", 0.5)
        max_length = criteria.get("max_length", 50.0)
        
        design_length = max(min_length, min(required_length, max_length))
        
        # Calculate actual discharge with designed length
        actual_discharge = Cd * design_length * head**(3/2) * math.sqrt(2 * self.gravity)
        
        return {
            "success": True,
            "weir_type": "rectangular",
            "weir_length": design_length,
            "required_length": required_length,
            "actual_discharge": actual_discharge,
            "discharge_error": abs(actual_discharge - discharge) / discharge,
            "design_head": head,
            "discharge_coefficient": Cd
        }
    
    def _design_triangular_weir(
        self, discharge: float, head: float, channel: ChannelGeometry, criteria: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Design triangular sharp-crested weir."""
        
        # Standard 90-degree V-notch
        vertex_angle = criteria.get("vertex_angle", 90.0)
        theta_rad = math.radians(vertex_angle)
        
        Cd = criteria.get("discharge_coefficient", 0.58)
        
        # Q = Cd * (8/15) * sqrt(2g) * tan(θ/2) * H^(5/2)
        theoretical_discharge = Cd * (8/15) * math.sqrt(2 * self.gravity) * math.tan(theta_rad/2) * head**(5/2)
        
        # Scale factor to match required discharge
        scale_factor = discharge / theoretical_discharge
        
        return {
            "success": True,
            "weir_type": "triangular",
            "vertex_angle": vertex_angle,
            "scale_factor": scale_factor,
            "theoretical_discharge": theoretical_discharge,
            "actual_discharge": discharge,
            "design_head": head,
            "discharge_coefficient": Cd
        }
    
    def _design_broad_crested_weir(
        self, discharge: float, head: float, channel: ChannelGeometry, criteria: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Design broad-crested weir."""
        
        Cd = criteria.get("discharge_coefficient", 0.35)
        crest_thickness = criteria.get("crest_thickness", head * 2)  # Default to 2H
        
        # Q = Cd * L * H * sqrt(2g * H)
        required_length = discharge / (Cd * head * math.sqrt(2 * self.gravity * head))
        
        return {
            "success": True,
            "weir_type": "broad_crested",
            "weir_length": required_length,
            "crest_thickness": crest_thickness,
            "actual_discharge": discharge,
            "design_head": head,
            "discharge_coefficient": Cd
        }
    
    def _design_ogee_spillway(
        self, discharge: float, head: float, channel: ChannelGeometry, criteria: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Design ogee spillway."""
        
        Cd = criteria.get("discharge_coefficient", 0.75)
        
        # Q = Cd * L * H^(3/2) * sqrt(2g)
        required_length = discharge / (Cd * head**(3/2) * math.sqrt(2 * self.gravity))
        
        # Generate spillway profile
        profile = self._generate_spillway_profile(
            WeirGeometry(weir_type=WeirType.OGEE_SPILLWAY, weir_height=0, crest_length=required_length)
        )
        
        return {
            "success": True,
            "weir_type": "ogee_spillway",
            "spillway_length": required_length,
            "actual_discharge": discharge,
            "design_head": head,
            "discharge_coefficient": Cd,
            "spillway_profile": profile
        }
    
    def _create_failed_result(self, weir: WeirGeometry, message: str) -> WeirFlowResult:
        """Create a failed analysis result."""
        return WeirFlowResult(
            weir_type=weir.weir_type,
            weir_height=weir.weir_height,
            effective_length=0,
            discharge=0,
            head_over_weir=0,
            approach_depth=0,
            approach_velocity=0,
            downstream_depth=None,
            discharge_coefficient=0,
            velocity_coefficient=0,
            approach_velocity_factor=0,
            submergence_factor=0,
            approach_energy=0,
            crest_energy=0,
            downstream_energy=None,
            energy_dissipated=0,
            energy_efficiency=0,
            weir_condition=WeirCondition.FREE_FLOW,
            froude_approach=0,
            froude_downstream=None,
            modular_limit=0,
            submergence_ratio=0,
            aeration_requirement=AerationLevel.NOT_REQUIRED,
            cavitation_risk=CavitationRisk.NEGLIGIBLE,
            cavitation_index=0,
            nappe_trajectory=None,
            energy_dissipation_length=0,
            scour_potential="Unknown",
            success=False,
            message=message,
            properties={}
        )


class WeirFlowAnalyzer:
    """
    High-level weir flow analysis tools.
    
    Provides convenient methods for common weir analysis tasks
    and professional design recommendations.
    """
    
    def __init__(self):
        self.solver = WeirFlowSolver()
    
    def recommend_weir_design(self, result: WeirFlowResult) -> List[str]:
        """Generate design recommendations for weir flow."""
        recommendations = []
        
        if not result.success:
            recommendations.append("❌ Weir analysis failed - check input conditions")
            return recommendations
        
        # Flow condition recommendations
        if result.weir_condition == WeirCondition.FREE_FLOW:
            recommendations.extend([
                "✅ Free flow conditions - optimal discharge accuracy",
                "📏 Weir operating within modular limits",
                "🔧 Good measurement accuracy expected"
            ])
        elif result.weir_condition == WeirCondition.SUBMERGED:
            recommendations.extend([
                "⚠️ Submerged flow - reduced discharge accuracy",
                "📊 Consider increasing weir height or reducing downstream levels",
                "🌊 Apply submergence corrections to discharge calculations"
            ])
        
        # Aeration recommendations
        if result.aeration_requirement == AerationLevel.REQUIRED:
            recommendations.extend([
                "🌬️ Aeration required - install aeration slots",
                f"📏 Recommended aeration capacity: {result.aeration_analysis['aeration_capacity']:.2f} m³/s",
                "⚠️ Risk of cavitation damage without proper aeration"
            ])
        elif result.aeration_requirement == AerationLevel.CRITICAL:
            recommendations.extend([
                "🚨 CRITICAL aeration requirements - multiple aeration points needed",
                "🔧 Consider stepped spillway or other energy dissipation methods",
                "⚡ High risk of structural damage without adequate aeration"
            ])
        
        # Cavitation risk recommendations
        if result.cavitation_risk in [CavitationRisk.HIGH, CavitationRisk.CRITICAL]:
            recommendations.extend([
                f"🚨 {result.cavitation_risk.value.upper()} cavitation risk",
                "🛡️ Use cavitation-resistant materials",
                "🔧 Consider design modifications to reduce velocities"
            ])
        
        # Energy dissipation recommendations
        if result.energy_dissipated > 5.0:
            recommendations.extend([
                "⚡ High energy dissipation - provide adequate downstream protection",
                f"📏 Energy dissipation length: {result.energy_dissipation_length:.1f} m",
                "🛡️ Heavy scour protection required"
            ])
        
        # Approach channel recommendations
        if not result.approach_flow_analysis.get("channel_adequate", True):
            recommendations.extend([
                "⚠️ Approach channel conditions not optimal",
                "📏 Consider lengthening approach channel",
                "🌊 High approach velocities may affect accuracy"
            ])
        
        return recommendations
    
    def compare_weir_types(
        self,
        channel: ChannelGeometry,
        target_discharge: float,
        available_head: float,
        approach_depth: float
    ) -> Dict[str, Any]:
        """Compare different weir types for given conditions."""
        
        weir_types = [
            WeirType.SHARP_CRESTED_RECTANGULAR,
            WeirType.SHARP_CRESTED_TRIANGULAR,
            WeirType.BROAD_CRESTED,
            WeirType.OGEE_SPILLWAY
        ]
        
        results = {}
        
        for weir_type in weir_types:
            # Design weir for target conditions
            design_result = self.solver.design_weir_dimensions(
                weir_type, target_discharge, available_head, channel
            )
            
            if design_result["success"]:
                # Create weir geometry
                if weir_type == WeirType.SHARP_CRESTED_RECTANGULAR:
                    weir = WeirGeometry(
                        weir_type=weir_type,
                        weir_height=approach_depth - available_head,
                        weir_width=design_result["weir_length"]
                    )
                elif weir_type == WeirType.SHARP_CRESTED_TRIANGULAR:
                    weir = WeirGeometry(
                        weir_type=weir_type,
                        weir_height=approach_depth - available_head,
                        vertex_angle=design_result["vertex_angle"]
                    )
                else:
                    weir = WeirGeometry(
                        weir_type=weir_type,
                        weir_height=approach_depth - available_head,
                        crest_length=design_result.get("weir_length", design_result.get("spillway_length", 1.0))
                    )
                
                # Analyze the designed weir
                analysis_result = self.solver.analyze_weir_flow(
                    channel, weir, approach_depth, target_discharge
                )
                
                results[weir_type.value] = {
                    "design": design_result,
                    "analysis": analysis_result,
                    "efficiency": analysis_result.discharge_coefficient if analysis_result.success else 0
                }
        
        # Find best options
        if results:
            best_efficiency = max(
                results.items(),
                key=lambda x: x[1]["efficiency"]
            )
            
            return {
                "success": True,
                "weir_comparisons": results,
                "best_efficiency": {
                    "weir_type": best_efficiency[0],
                    "efficiency": best_efficiency[1]["efficiency"],
                    "design_parameters": best_efficiency[1]["design"]
                },
                "recommendations": self._generate_comparison_recommendations(results)
            }
        
        return {"success": False, "message": "No successful weir designs found"}
    
    def _generate_comparison_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations from weir comparison."""
        recommendations = []
        
        if "sharp_crested_rectangular" in results:
            recommendations.append("📏 Rectangular weir: Best for wide flow ranges and high accuracy")
        
        if "sharp_crested_triangular" in results:
            recommendations.append("📐 Triangular weir: Excellent for low flows and precise measurement")
        
        if "broad_crested" in results:
            recommendations.append("🏗️ Broad-crested weir: Good for high flows and structural stability")
        
        if "ogee_spillway" in results:
            recommendations.append("🌊 Ogee spillway: Optimal for dam spillways and high discharge capacity")
        
        return recommendations
