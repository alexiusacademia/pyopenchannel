"""
File: gvf/solver.py
Author: Alexius Academia
Date: 2025-08-17

Core GVF (Gradually Varied Flow) solver engine.

This module implements the fundamental GVF differential equation solver:
dy/dx = (S₀ - Sf) / (1 - Fr²)

Features:
- High-accuracy numerical integration (Dormand-Prince, RKF45, RK4)
- Automatic event detection (critical transitions, hydraulic jumps)
- Analytical validation and cross-checking
- Adaptive step size control
- Multiple boundary condition types
- Unit-aware calculations

The solver handles all flow regimes:
- Subcritical flow (Fr < 1)
- Supercritical flow (Fr > 1) 
- Critical transitions (Fr ≈ 1)
- Mixed flow conditions

Applications:
- Water surface profile computation
- Backwater analysis
- Dam and bridge hydraulics
- Channel design verification
"""

import math
from typing import Callable, List, Optional, Tuple, Dict, Any, Union
from dataclasses import dataclass, field
from enum import Enum

from ..geometry import ChannelGeometry
from ..hydraulics import ManningEquation, CriticalDepth
from ..numerical import (
    AdaptiveIntegrator, 
    IntegrationResult, 
    EventDetector, 
    AnalyticalValidator,
    EventType
)
from ..exceptions import (
    ConvergenceError, 
    InvalidFlowConditionError,
    SupercriticalFlowError,
    SubcriticalFlowError
)
from ..validators import (
    validate_positive,
    validate_discharge,
    validate_manning_n,
    validate_slope
)


class BoundaryType(Enum):
    """Types of boundary conditions for GVF analysis."""
    UPSTREAM_DEPTH = "upstream_depth"
    DOWNSTREAM_DEPTH = "downstream_depth"
    CRITICAL_DEPTH = "critical_depth"
    NORMAL_DEPTH = "normal_depth"
    CONTROL_STRUCTURE = "control_structure"


class FlowRegime(Enum):
    """Flow regime classification."""
    SUBCRITICAL = "subcritical"
    SUPERCRITICAL = "supercritical"
    CRITICAL = "critical"
    MIXED = "mixed"


@dataclass
class ProfilePoint:
    """
    Represents a point on the water surface profile.
    
    Attributes:
        x: Distance along channel (m)
        depth: Flow depth (m)
        velocity: Flow velocity (m/s)
        discharge: Discharge (m³/s)
        area: Cross-sectional area (m²)
        top_width: Top width (m)
        hydraulic_radius: Hydraulic radius (m)
        froude_number: Froude number
        specific_energy: Specific energy (m)
        slope_friction: Friction slope
        slope_energy: Energy slope
        regime: Flow regime at this point
        properties: Additional computed properties
    """
    x: float
    depth: float
    velocity: float
    discharge: float
    area: float
    top_width: float
    hydraulic_radius: float
    froude_number: float
    specific_energy: float
    slope_friction: float
    slope_energy: float
    regime: FlowRegime
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GVFResult:
    """
    Results from GVF analysis.
    
    Attributes:
        profile_points: Array of computed profile points
        integration_result: Raw integration results
        boundary_conditions: Applied boundary conditions
        channel_properties: Channel geometry and flow properties
        computation_summary: Summary of computation process
        events_detected: List of detected hydraulic events
        validation_results: Analytical validation results
        success: Whether computation completed successfully
        message: Status message or error description
    """
    profile_points: List[ProfilePoint]
    integration_result: IntegrationResult
    boundary_conditions: Dict[str, Any]
    channel_properties: Dict[str, Any]
    computation_summary: Dict[str, Any]
    events_detected: List[Any] = field(default_factory=list)
    validation_results: List[Dict[str, Any]] = field(default_factory=list)
    success: bool = True
    message: str = "GVF computation completed successfully"
    
    @property
    def length(self) -> float:
        """Total length of computed profile."""
        if len(self.profile_points) < 2:
            return 0.0
        return abs(self.profile_points[-1].x - self.profile_points[0].x)
    
    @property
    def max_depth(self) -> float:
        """Maximum depth in profile."""
        if not self.profile_points:
            return 0.0
        return max(point.depth for point in self.profile_points)
    
    @property
    def min_depth(self) -> float:
        """Minimum depth in profile."""
        if not self.profile_points:
            return 0.0
        return min(point.depth for point in self.profile_points)
    
    @property
    def critical_points(self) -> List[ProfilePoint]:
        """Points where flow is critical (Fr ≈ 1)."""
        return [point for point in self.profile_points 
                if abs(point.froude_number - 1.0) < 0.05]


class GVFEquation:
    """
    Encapsulates the GVF differential equation and related computations.
    
    The fundamental GVF equation is:
    dy/dx = (S₀ - Sf) / (1 - Fr²)
    
    This class handles:
    - Equation evaluation
    - Singularity detection (Fr ≈ 1)
    - Unit-aware calculations
    - Flow property computation
    """
    
    def __init__(
        self,
        channel: ChannelGeometry,
        discharge: float,
        slope: float,
        manning_n: float,
        critical_tolerance: float = 0.01
    ):
        """
        Initialize GVF equation.
        
        Args:
            channel: Channel geometry
            discharge: Discharge (m³/s or ft³/s)
            slope: Channel slope (dimensionless)
            manning_n: Manning's roughness coefficient
            critical_tolerance: Tolerance for critical flow detection
        """
        self.channel = channel
        self.discharge = validate_discharge(discharge)
        self.slope = validate_slope(slope)
        self.manning_n = validate_manning_n(manning_n)
        self.critical_tolerance = critical_tolerance
        
        # Pre-compute critical depth for efficiency
        self.critical_depth = CriticalDepth.calculate(channel, discharge)
        
        # Import unit-aware constants
        from ..units import get_gravity, get_manning_factor
        self.gravity = get_gravity()
        self.manning_factor = get_manning_factor()
    
    def evaluate(self, x: float, depth: float) -> float:
        """
        Evaluate the GVF equation: dy/dx = f(x, y).
        
        Args:
            x: Distance along channel (m)
            depth: Flow depth (m)
            
        Returns:
            Derivative dy/dx
            
        Raises:
            InvalidFlowConditionError: If flow conditions are invalid
        """
        if depth <= 0:
            raise InvalidFlowConditionError(f"Depth must be positive, got {depth}")
        
        # Compute geometric properties
        area = self.channel.area(depth)
        hydraulic_radius = self.channel.hydraulic_radius(depth)
        top_width = self.channel.top_width(depth)
        
        if area <= 0 or hydraulic_radius <= 0:
            raise InvalidFlowConditionError("Invalid channel geometry")
        
        # Compute flow properties
        velocity = self.discharge / area
        hydraulic_depth = area / top_width
        froude_number = velocity / math.sqrt(self.gravity * hydraulic_depth)
        
        # Check for critical flow singularity
        if abs(froude_number - 1.0) < self.critical_tolerance:
            # Near critical flow - use special handling
            return self._handle_critical_flow(depth, froude_number)
        
        # Compute friction slope using Manning's equation
        friction_slope = self._compute_friction_slope(
            velocity, hydraulic_radius, self.manning_n
        )
        
        # GVF equation: dy/dx = (S₀ - Sf) / (1 - Fr²)
        numerator = self.slope - friction_slope
        denominator = 1.0 - froude_number**2
        
        return numerator / denominator
    
    def _compute_friction_slope(
        self, 
        velocity: float, 
        hydraulic_radius: float, 
        manning_n: float
    ) -> float:
        """Compute friction slope using Manning's equation."""
        # Sf = (n * V)² / (C * R^(4/3))
        # Where C is the Manning factor (1.0 for SI, 1.486 for US)
        return (manning_n * velocity)**2 / (self.manning_factor**2 * hydraulic_radius**(4/3))
    
    def _handle_critical_flow(self, depth: float, froude_number: float) -> float:
        """
        Handle critical flow conditions with special numerical treatment.
        
        Near critical flow (Fr ≈ 1), the denominator approaches zero,
        requiring special handling to avoid numerical instability.
        """
        # Use L'Hôpital's rule or asymptotic expansion
        # For now, use a small perturbation method
        epsilon = 1e-6
        
        if froude_number > 1.0:
            # Slightly supercritical
            perturbed_depth = depth * (1 - epsilon)
        else:
            # Slightly subcritical  
            perturbed_depth = depth * (1 + epsilon)
        
        try:
            return self.evaluate(0, perturbed_depth)  # x doesn't matter for uniform channels
        except RecursionError:
            # Fallback: assume horizontal water surface near critical
            return 0.0
    
    def compute_flow_properties(self, depth: float) -> Dict[str, float]:
        """
        Compute all flow properties at given depth.
        
        Args:
            depth: Flow depth
            
        Returns:
            Dictionary of flow properties
        """
        # Geometric properties
        area = self.channel.area(depth)
        perimeter = self.channel.wetted_perimeter(depth)
        hydraulic_radius = self.channel.hydraulic_radius(depth)
        top_width = self.channel.top_width(depth)
        
        # Flow properties
        velocity = self.discharge / area
        hydraulic_depth = area / top_width
        froude_number = velocity / math.sqrt(self.gravity * hydraulic_depth)
        
        # Energy properties
        specific_energy = depth + velocity**2 / (2 * self.gravity)
        
        # Slopes
        friction_slope = self._compute_friction_slope(velocity, hydraulic_radius, self.manning_n)
        energy_slope = friction_slope  # For uniform channels
        
        # Flow regime
        if abs(froude_number - 1.0) < self.critical_tolerance:
            regime = FlowRegime.CRITICAL
        elif froude_number < 1.0:
            regime = FlowRegime.SUBCRITICAL
        else:
            regime = FlowRegime.SUPERCRITICAL
        
        return {
            'depth': depth,
            'area': area,
            'perimeter': perimeter,
            'hydraulic_radius': hydraulic_radius,
            'top_width': top_width,
            'velocity': velocity,
            'hydraulic_depth': hydraulic_depth,
            'froude_number': froude_number,
            'specific_energy': specific_energy,
            'friction_slope': friction_slope,
            'energy_slope': energy_slope,
            'regime': regime
        }


class GVFSolver:
    """
    High-level GVF solver with advanced features.
    
    Combines numerical integration, event detection, and analytical validation
    to provide comprehensive gradually varied flow analysis.
    
    Features:
    - Multiple integration methods (RK4, RKF45, Dormand-Prince)
    - Automatic event detection
    - Analytical cross-validation
    - Adaptive step size control
    - Comprehensive error handling
    """
    
    def __init__(
        self,
        integration_method: str = "dormand_prince",
        rtol: float = 1e-6,
        atol: float = 1e-9,
        enable_event_detection: bool = True,
        enable_validation: bool = True,
        max_steps: int = 50000
    ):
        """
        Initialize GVF solver.
        
        Args:
            integration_method: "rk4", "rkf45", or "dormand_prince"
            rtol: Relative tolerance for integration
            atol: Absolute tolerance for integration
            enable_event_detection: Enable hydraulic event detection
            enable_validation: Enable analytical validation
            max_steps: Maximum integration steps
        """
        self.integration_method = integration_method
        self.rtol = rtol
        self.atol = atol
        self.enable_event_detection = enable_event_detection
        self.enable_validation = enable_validation
        self.max_steps = max_steps
        
        # Initialize numerical components
        self.integrator = AdaptiveIntegrator(
            method=integration_method,
            rtol=rtol,
            atol=atol
        )
        
        if enable_event_detection:
            self.event_detector = EventDetector(
                froude_tolerance=0.05,
                shock_threshold=0.1,
                enable_critical_detection=True,
                enable_shock_detection=True
            )
        
        if enable_validation:
            self.validator = AnalyticalValidator(tolerance=1e-3)
    
    def solve_profile(
        self,
        channel: ChannelGeometry,
        discharge: float,
        slope: float,
        manning_n: float,
        x_start: float,
        x_end: float,
        boundary_depth: float,
        boundary_type: BoundaryType = BoundaryType.UPSTREAM_DEPTH,
        initial_step: Optional[float] = None
    ) -> GVFResult:
        """
        Solve water surface profile using GVF equation.
        
        Args:
            channel: Channel geometry
            discharge: Discharge (m³/s or ft³/s)
            slope: Channel slope (dimensionless)
            manning_n: Manning's roughness coefficient
            x_start: Starting x-coordinate (m or ft)
            x_end: Ending x-coordinate (m or ft)
            boundary_depth: Boundary condition depth (m or ft)
            boundary_type: Type of boundary condition
            initial_step: Initial integration step size
            
        Returns:
            GVFResult with computed water surface profile
        """
        try:
            # Create GVF equation
            gvf_eq = GVFEquation(channel, discharge, slope, manning_n)
            
            # Set up boundary conditions
            if boundary_type == BoundaryType.UPSTREAM_DEPTH:
                x0, y0 = x_start, boundary_depth
                x_final = x_end
            elif boundary_type == BoundaryType.DOWNSTREAM_DEPTH:
                x0, y0 = x_end, boundary_depth
                x_final = x_start
            elif boundary_type == BoundaryType.CRITICAL_DEPTH:
                x0, y0 = x_start, gvf_eq.critical_depth
                x_final = x_end
            else:
                raise ValueError(f"Unsupported boundary type: {boundary_type}")
            
            # Auto-estimate initial step if not provided
            if initial_step is None:
                initial_step = abs(x_final - x0) / 1000
            
            # Reset event detector if enabled
            if self.enable_event_detection:
                self.event_detector.reset()
            
            # Integrate the GVF equation
            integration_result = self.integrator.integrate(
                func=gvf_eq.evaluate,
                x0=x0,
                y0=y0,
                x_end=x_final,
                initial_step=initial_step,
                max_steps=self.max_steps
            )
            
            if not integration_result.success:
                return GVFResult(
                    profile_points=[],
                    integration_result=integration_result,
                    boundary_conditions={
                        "type": boundary_type.value,
                        "depth": boundary_depth,
                        "x_start": x_start,
                        "x_end": x_end
                    },
                    channel_properties={
                        "discharge": discharge,
                        "slope": slope,
                        "manning_n": manning_n,
                        "critical_depth": gvf_eq.critical_depth
                    },
                    computation_summary={
                        "integration_method": self.integration_method,
                        "total_steps": integration_result.total_steps
                    },
                    success=False,
                    message=f"Integration failed: {integration_result.message}"
                )
            
            # Convert integration results to profile points
            profile_points = self._create_profile_points(
                integration_result, gvf_eq
            )
            
            # Perform event detection if enabled
            events_detected = []
            if self.enable_event_detection:
                events_detected = self._detect_events(profile_points)
            
            # Perform analytical validation if enabled
            validation_results = []
            if self.enable_validation:
                validation_results = self._validate_solution(
                    profile_points, gvf_eq
                )
            
            return GVFResult(
                profile_points=profile_points,
                integration_result=integration_result,
                boundary_conditions={
                    "type": boundary_type.value,
                    "depth": boundary_depth,
                    "x_start": x_start,
                    "x_end": x_end
                },
                channel_properties={
                    "discharge": discharge,
                    "slope": slope,
                    "manning_n": manning_n,
                    "critical_depth": gvf_eq.critical_depth
                },
                computation_summary={
                    "integration_method": self.integration_method,
                    "total_steps": integration_result.total_steps,
                    "profile_length": abs(x_final - x0),
                    "max_depth": max(point.depth for point in profile_points),
                    "min_depth": min(point.depth for point in profile_points)
                },
                events_detected=events_detected,
                validation_results=validation_results,
                success=True,
                message="GVF profile computed successfully"
            )
            
        except Exception as e:
            return GVFResult(
                profile_points=[],
                integration_result=IntegrationResult([], [], [], success=False),
                boundary_conditions={},
                channel_properties={},
                computation_summary={},
                success=False,
                message=f"GVF solver error: {e}"
            )
    
    def _create_profile_points(
        self, 
        integration_result: IntegrationResult, 
        gvf_eq: GVFEquation
    ) -> List[ProfilePoint]:
        """Convert integration results to profile points with full hydraulic properties."""
        profile_points = []
        
        for x, depth in zip(integration_result.x_values, integration_result.y_values):
            # Compute all flow properties at this point
            props = gvf_eq.compute_flow_properties(depth)
            
            profile_point = ProfilePoint(
                x=x,
                depth=depth,
                velocity=props['velocity'],
                discharge=gvf_eq.discharge,
                area=props['area'],
                top_width=props['top_width'],
                hydraulic_radius=props['hydraulic_radius'],
                froude_number=props['froude_number'],
                specific_energy=props['specific_energy'],
                slope_friction=props['friction_slope'],
                slope_energy=props['energy_slope'],
                regime=props['regime'],
                properties={
                    'hydraulic_depth': props['hydraulic_depth'],
                    'perimeter': props['perimeter']
                }
            )
            
            profile_points.append(profile_point)
        
        return profile_points
    
    def _detect_events(self, profile_points: List[ProfilePoint]) -> List[Any]:
        """Detect hydraulic events in the computed profile."""
        events = []
        
        for point in profile_points:
            detected = self.event_detector.check_events(
                x=point.x,
                y=point.depth,
                froude_number=point.froude_number,
                velocity=point.velocity
            )
            events.extend(detected)
        
        return events
    
    def _validate_solution(
        self, 
        profile_points: List[ProfilePoint], 
        gvf_eq: GVFEquation
    ) -> List[Dict[str, Any]]:
        """Validate solution against analytical solutions where possible."""
        validation_results = []
        
        # Sample a few points for validation
        sample_indices = [0, len(profile_points)//4, len(profile_points)//2, 
                         3*len(profile_points)//4, -1]
        
        for i in sample_indices:
            if i >= len(profile_points):
                continue
                
            point = profile_points[i]
            
            # Validate critical flow condition if near critical
            if abs(point.froude_number - 1.0) < 0.1:
                validation = self.validator.validate_critical_flow(
                    depth=point.depth,
                    discharge=point.discharge,
                    channel_area=point.area,
                    top_width=point.top_width
                )
                validation['point_index'] = i
                validation['x_location'] = point.x
                validation_results.append(validation)
        
        return validation_results
