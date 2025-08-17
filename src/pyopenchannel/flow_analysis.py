"""
File: flow_analysis.py
Author: Alexius Academia
Date: 2025-08-17

Flow analysis for open channel hydraulics.

This module provides classes for analyzing different types of flow:
- Uniform flow
- Critical flow
- Gradually varied flow
- Energy and momentum equations
"""

import math
from typing import Union, Optional, Tuple, List
from dataclasses import dataclass

from .constants import DEFAULT_TOLERANCE, MAX_ITERATIONS
from .geometry import ChannelGeometry
from .hydraulics import ManningEquation, CriticalDepth, NormalDepth
from .validators import (
    validate_positive,
    validate_depth,
    validate_discharge,
    validate_manning_n,
    validate_slope,
)
from .exceptions import (
    ConvergenceError,
    InvalidFlowConditionError,
    SupercriticalFlowError,
    SubcriticalFlowError,
)


@dataclass
class FlowState:
    """
    Represents the hydraulic state at a cross-section.
    
    Attributes:
        depth: Flow depth (m)
        velocity: Average velocity (m/s)
        discharge: Discharge (m³/s)
        area: Cross-sectional area (m²)
        top_width: Top width (m)
        hydraulic_radius: Hydraulic radius (m)
        froude_number: Froude number
        specific_energy: Specific energy (m)
        momentum: Momentum function (m³)
    """
    depth: float
    velocity: float
    discharge: float
    area: float
    top_width: float
    hydraulic_radius: float
    froude_number: float
    specific_energy: float
    momentum: float
    
    @property
    def is_critical(self, tolerance: float = 0.01) -> bool:
        """Check if flow is critical (Fr ≈ 1.0)."""
        return abs(self.froude_number - 1.0) < tolerance
    
    @property
    def is_subcritical(self) -> bool:
        """Check if flow is subcritical (Fr < 1.0)."""
        return self.froude_number < 1.0
    
    @property
    def is_supercritical(self) -> bool:
        """Check if flow is supercritical (Fr > 1.0)."""
        return self.froude_number > 1.0


class UniformFlow:
    """
    Analysis of uniform flow in open channels.
    
    Uniform flow occurs when depth, velocity, and discharge are constant
    along the channel length.
    """
    
    def __init__(
        self,
        channel: ChannelGeometry,
        slope: float,
        manning_n: float
    ):
        """
        Initialize uniform flow analysis.
        
        Args:
            channel: Channel geometry object
            slope: Channel slope (dimensionless)
            manning_n: Manning's roughness coefficient
        """
        self.channel = channel
        self.slope = validate_slope(slope)
        self.manning_n = validate_manning_n(manning_n)
    
    def calculate_flow_state(self, discharge: float) -> FlowState:
        """
        Calculate flow state for given discharge.
        
        Args:
            discharge: Discharge (m³/s)
            
        Returns:
            FlowState object with all hydraulic properties
        """
        discharge = validate_discharge(discharge)
        
        # Calculate normal depth
        normal_depth = NormalDepth.calculate(
            self.channel, discharge, self.slope, self.manning_n
        )
        
        return self._create_flow_state(normal_depth, discharge)
    
    def calculate_discharge(self, depth: float) -> float:
        """
        Calculate discharge for given depth.
        
        Args:
            depth: Flow depth (m)
            
        Returns:
            Discharge (m³/s)
        """
        depth = validate_depth(depth)
        
        area = self.channel.area(depth)
        hydraulic_radius = self.channel.hydraulic_radius(depth)
        
        return ManningEquation.discharge(area, hydraulic_radius, self.slope, self.manning_n)
    
    def _create_flow_state(self, depth: float, discharge: float) -> FlowState:
        """Create FlowState object from depth and discharge."""
        # Import here to avoid circular imports
        from .units import get_gravity
        
        area = self.channel.area(depth)
        top_width = self.channel.top_width(depth)
        hydraulic_radius = self.channel.hydraulic_radius(depth)
        hydraulic_depth = area / top_width if top_width > 0 else 0
        gravity = get_gravity()
        
        velocity = discharge / area if area > 0 else 0
        froude_number = velocity / math.sqrt(gravity * hydraulic_depth) if hydraulic_depth > 0 else 0
        specific_energy = depth + velocity**2 / (2 * gravity)
        momentum = discharge**2 / (gravity * area) + area * (depth / 2) if area > 0 else 0
        
        return FlowState(
            depth=depth,
            velocity=velocity,
            discharge=discharge,
            area=area,
            top_width=top_width,
            hydraulic_radius=hydraulic_radius,
            froude_number=froude_number,
            specific_energy=specific_energy,
            momentum=momentum
        )


class CriticalFlow:
    """
    Analysis of critical flow conditions.
    
    Critical flow occurs when the Froude number equals 1.0.
    """
    
    def __init__(self, channel: ChannelGeometry):
        """
        Initialize critical flow analysis.
        
        Args:
            channel: Channel geometry object
        """
        self.channel = channel
    
    def calculate_critical_depth(self, discharge: float) -> float:
        """
        Calculate critical depth for given discharge.
        
        Args:
            discharge: Discharge (m³/s)
            
        Returns:
            Critical depth (m)
        """
        return CriticalDepth.calculate(self.channel, discharge)
    
    def calculate_critical_discharge(self, depth: float) -> float:
        """
        Calculate critical discharge for given depth.
        
        Args:
            depth: Flow depth (current unit system)
            
        Returns:
            Critical discharge (current unit system)
        """
        # Import here to avoid circular imports
        from .units import get_gravity
        
        depth = validate_depth(depth)
        gravity = get_gravity()
        
        area = self.channel.area(depth)
        top_width = self.channel.top_width(depth)
        
        if top_width == 0:
            raise InvalidFlowConditionError("Top width is zero - no free surface")
        
        # Q_critical = sqrt(g * A³ / T)
        return math.sqrt(gravity * (area**3) / top_width)
    
    def calculate_critical_velocity(self, depth: float) -> float:
        """
        Calculate critical velocity for given depth.
        
        Args:
            depth: Flow depth (current unit system)
            
        Returns:
            Critical velocity (current unit system)
        """
        # Import here to avoid circular imports
        from .units import get_gravity
        
        depth = validate_depth(depth)
        gravity = get_gravity()
        
        area = self.channel.area(depth)
        hydraulic_depth = area / self.channel.top_width(depth)
        
        # V_critical = sqrt(g * D)
        return math.sqrt(gravity * hydraulic_depth)
    
    def calculate_critical_slope(
        self,
        discharge: float,
        manning_n: float
    ) -> float:
        """
        Calculate critical slope for given discharge.
        
        Args:
            discharge: Discharge (m³/s)
            manning_n: Manning's roughness coefficient
            
        Returns:
            Critical slope (dimensionless)
        """
        discharge = validate_discharge(discharge)
        manning_n = validate_manning_n(manning_n)
        
        critical_depth = self.calculate_critical_depth(discharge)
        area = self.channel.area(critical_depth)
        hydraulic_radius = self.channel.hydraulic_radius(critical_depth)
        
        # At critical conditions, normal depth equals critical depth
        return ManningEquation.required_slope(discharge, area, hydraulic_radius, manning_n)


class EnergyEquation:
    """
    Energy equation for open channel flow.
    
    E = y + V²/(2g) = y + Q²/(2gA²)
    """
    
    @staticmethod
    def specific_energy(depth: float, velocity: float) -> float:
        """
        Calculate specific energy.
        
        Args:
            depth: Flow depth (current unit system)
            velocity: Average velocity (current unit system)
            
        Returns:
            Specific energy (current unit system)
        """
        # Import here to avoid circular imports
        from .units import get_gravity
        
        depth = validate_depth(depth)
        velocity = validate_positive(velocity, "velocity")
        gravity = get_gravity()
        
        return depth + velocity**2 / (2 * gravity)
    
    @staticmethod
    def specific_energy_from_discharge(
        channel: ChannelGeometry,
        depth: float,
        discharge: float
    ) -> float:
        """
        Calculate specific energy from discharge.
        
        Args:
            channel: Channel geometry object
            depth: Flow depth (m)
            discharge: Discharge (m³/s)
            
        Returns:
            Specific energy (m)
        """
        depth = validate_depth(depth)
        discharge = validate_discharge(discharge)
        
        area = channel.area(depth)
        if area == 0:
            raise InvalidFlowConditionError("Cross-sectional area is zero")
        
        velocity = discharge / area
        return EnergyEquation.specific_energy(depth, velocity)
    
    @staticmethod
    def minimum_specific_energy(channel: ChannelGeometry, discharge: float) -> float:
        """
        Calculate minimum specific energy (occurs at critical depth).
        
        Args:
            channel: Channel geometry object
            discharge: Discharge (m³/s)
            
        Returns:
            Minimum specific energy (m)
        """
        critical_depth = CriticalDepth.calculate(channel, discharge)
        return EnergyEquation.specific_energy_from_discharge(channel, critical_depth, discharge)
    
    @staticmethod
    def alternate_depths(
        channel: ChannelGeometry,
        discharge: float,
        specific_energy: float
    ) -> Tuple[float, float]:
        """
        Calculate alternate depths for given specific energy.
        
        Args:
            channel: Channel geometry object
            discharge: Discharge (m³/s)
            specific_energy: Specific energy (m)
            
        Returns:
            Tuple of (subcritical_depth, supercritical_depth)
        """
        discharge = validate_discharge(discharge)
        specific_energy = validate_positive(specific_energy, "specific_energy")
        
        # Check if specific energy is sufficient
        min_energy = EnergyEquation.minimum_specific_energy(channel, discharge)
        if specific_energy < min_energy:
            raise InvalidFlowConditionError(
                f"Specific energy ({specific_energy:.3f} m) is less than minimum "
                f"({min_energy:.3f} m)"
            )
        
        # Find depths using Newton-Raphson
        critical_depth = CriticalDepth.calculate(channel, discharge)
        
        # Subcritical depth (greater than critical)
        depth_sub = EnergyEquation._solve_depth_for_energy(
            channel, discharge, specific_energy, critical_depth * 1.5
        )
        
        # Supercritical depth (less than critical)
        depth_super = EnergyEquation._solve_depth_for_energy(
            channel, discharge, specific_energy, critical_depth * 0.5
        )
        
        return depth_sub, depth_super
    
    @staticmethod
    def _solve_depth_for_energy(
        channel: ChannelGeometry,
        discharge: float,
        target_energy: float,
        initial_guess: float,
        tolerance: float = DEFAULT_TOLERANCE,
        max_iterations: int = MAX_ITERATIONS
    ) -> float:
        """Solve for depth given specific energy using Newton-Raphson."""
        # Import here to avoid circular imports
        from .units import get_gravity
        
        depth = initial_guess
        gravity = get_gravity()  # Cache the gravity value
        
        for _ in range(max_iterations):
            area = channel.area(depth)
            top_width = channel.top_width(depth)
            
            if area == 0:
                raise InvalidFlowConditionError("Cross-sectional area is zero")
            
            velocity = discharge / area
            current_energy = depth + velocity**2 / (2 * gravity)
            
            f = current_energy - target_energy
            
            # Derivative: dE/dy = 1 - Q²T/(gA³)
            df_dy = 1 - (discharge**2 * top_width) / (gravity * area**3)
            
            if abs(df_dy) < 1e-12:
                raise ConvergenceError("Derivative too small")
            
            depth_new = depth - f / df_dy
            
            if depth_new <= 0:
                depth_new = depth / 2
            
            if abs(depth_new - depth) < tolerance:
                return depth_new
            
            depth = depth_new
        
        raise ConvergenceError("Depth solution did not converge")


class MomentumEquation:
    """
    Momentum equation for open channel flow.
    
    M = Q²/(gA) + A*yc
    
    Where yc is the depth to centroid of the cross-section.
    """
    
    @staticmethod
    def momentum_function(
        channel: ChannelGeometry,
        depth: float,
        discharge: float
    ) -> float:
        """
        Calculate momentum function.
        
        Args:
            channel: Channel geometry object
            depth: Flow depth (current unit system)
            discharge: Discharge (current unit system)
            
        Returns:
            Momentum function (current unit system³)
        """
        # Import here to avoid circular imports
        from .units import get_gravity
        
        depth = validate_depth(depth)
        discharge = validate_discharge(discharge)
        gravity = get_gravity()
        
        area = channel.area(depth)
        if area == 0:
            raise InvalidFlowConditionError("Cross-sectional area is zero")
        
        # For most channel shapes, centroid depth ≈ depth/2
        # This is exact for rectangular and trapezoidal channels
        centroid_depth = depth / 2
        
        return discharge**2 / (gravity * area) + area * centroid_depth
    
    @staticmethod
    def conjugate_depths(
        channel: ChannelGeometry,
        discharge: float,
        upstream_depth: float
    ) -> float:
        """
        Calculate conjugate depth for hydraulic jump.
        
        Args:
            channel: Channel geometry object
            discharge: Discharge (m³/s)
            upstream_depth: Upstream depth (m)
            
        Returns:
            Downstream conjugate depth (m)
        """
        discharge = validate_discharge(discharge)
        upstream_depth = validate_depth(upstream_depth)
        
        # Check if upstream flow is supercritical
        from .units import get_gravity
        gravity = get_gravity()
        
        upstream_area = channel.area(upstream_depth)
        upstream_velocity = discharge / upstream_area
        upstream_hydraulic_depth = upstream_area / channel.top_width(upstream_depth)
        upstream_froude = upstream_velocity / math.sqrt(gravity * upstream_hydraulic_depth)
        
        if upstream_froude <= 1.0:
            raise SupercriticalFlowError(
                f"Upstream flow must be supercritical (Fr > 1), got Fr = {upstream_froude:.3f}"
            )
        
        # Calculate upstream momentum
        upstream_momentum = MomentumEquation.momentum_function(
            channel, upstream_depth, discharge
        )
        
        # Solve for downstream depth using Newton-Raphson
        # Initial guess: use rectangular channel approximation
        y1 = upstream_depth
        q = discharge / channel.top_width(upstream_depth)  # Unit discharge approximation
        y2_guess = y1 * (-1 + math.sqrt(1 + 8 * upstream_froude**2)) / 2
        
        return MomentumEquation._solve_conjugate_depth(
            channel, discharge, upstream_momentum, y2_guess
        )
    
    @staticmethod
    def _solve_conjugate_depth(
        channel: ChannelGeometry,
        discharge: float,
        target_momentum: float,
        initial_guess: float,
        tolerance: float = DEFAULT_TOLERANCE,
        max_iterations: int = MAX_ITERATIONS
    ) -> float:
        """Solve for conjugate depth using Newton-Raphson."""
        depth = initial_guess
        
        for _ in range(max_iterations):
            current_momentum = MomentumEquation.momentum_function(channel, depth, discharge)
            f = current_momentum - target_momentum
            
            # Calculate derivative using finite differences
            delta = depth * 1e-6
            if delta < 1e-9:
                delta = 1e-9
            
            momentum_plus = MomentumEquation.momentum_function(channel, depth + delta, discharge)
            df_dy = (momentum_plus - current_momentum) / delta
            
            if abs(df_dy) < 1e-12:
                raise ConvergenceError("Derivative too small")
            
            depth_new = depth - f / df_dy
            
            if depth_new <= 0:
                depth_new = depth / 2
            
            if abs(depth_new - depth) < tolerance:
                return depth_new
            
            depth = depth_new
        
        raise ConvergenceError("Conjugate depth solution did not converge")


class GraduallyVariedFlow:
    """
    Analysis of gradually varied flow using the standard step method.
    
    This class handles non-uniform flow where depth varies gradually
    along the channel length.
    """
    
    def __init__(
        self,
        channel: ChannelGeometry,
        slope: float,
        manning_n: float
    ):
        """
        Initialize gradually varied flow analysis.
        
        Args:
            channel: Channel geometry object
            slope: Channel slope (dimensionless)
            manning_n: Manning's roughness coefficient
        """
        self.channel = channel
        self.slope = validate_slope(slope, allow_zero=True)
        self.manning_n = validate_manning_n(manning_n)
    
    def classify_flow_profile(
        self,
        discharge: float,
        depth: float
    ) -> str:
        """
        Classify the flow profile type.
        
        Args:
            discharge: Discharge (m³/s)
            depth: Flow depth (m)
            
        Returns:
            Flow profile classification (e.g., 'M1', 'S2', etc.)
        """
        discharge = validate_discharge(discharge)
        depth = validate_depth(depth)
        
        # Calculate critical and normal depths
        critical_depth = CriticalDepth.calculate(self.channel, discharge)
        
        try:
            normal_depth = NormalDepth.calculate(
                self.channel, discharge, self.slope, self.manning_n
            )
        except:
            # Steep slope - no normal depth exists
            normal_depth = float('inf')
        
        # Classify slope
        if self.slope == 0:
            slope_type = 'H'  # Horizontal
        elif normal_depth > critical_depth:
            slope_type = 'M'  # Mild
        elif normal_depth < critical_depth:
            slope_type = 'S'  # Steep
        else:
            slope_type = 'C'  # Critical
        
        # Classify depth zone
        if slope_type == 'H':
            if depth > critical_depth:
                zone = '2'
            else:
                zone = '3'
        elif slope_type in ['M', 'S']:
            if depth > max(normal_depth, critical_depth):
                zone = '1'
            elif depth > min(normal_depth, critical_depth):
                zone = '2'
            else:
                zone = '3'
        else:  # Critical slope
            if depth > critical_depth:
                zone = '1'
            else:
                zone = '3'
        
        return f"{slope_type}{zone}"
    
    def water_surface_profile(
        self,
        discharge: float,
        initial_depth: float,
        distance: float,
        num_steps: int = 100
    ) -> List[Tuple[float, float]]:
        """
        Calculate water surface profile using standard step method.
        
        Args:
            discharge: Discharge (m³/s)
            initial_depth: Initial depth (m)
            distance: Total distance (m)
            num_steps: Number of calculation steps
            
        Returns:
            List of (distance, depth) tuples
        """
        discharge = validate_discharge(discharge)
        initial_depth = validate_depth(initial_depth)
        distance = validate_positive(distance, "distance")
        
        step_length = distance / num_steps
        profile = [(0.0, initial_depth)]
        
        current_depth = initial_depth
        current_distance = 0.0
        
        for _ in range(num_steps):
            try:
                # Calculate next depth using standard step method
                next_depth = self._standard_step(
                    discharge, current_depth, step_length
                )
                
                current_distance += step_length
                current_depth = next_depth
                
                profile.append((current_distance, current_depth))
                
            except Exception:
                # Stop if calculation fails (e.g., critical point reached)
                break
        
        return profile
    
    def _standard_step(
        self,
        discharge: float,
        depth1: float,
        step_length: float
    ) -> float:
        """
        Calculate depth at next section using standard step method.
        
        This is a simplified implementation. A full implementation would
        use iterative methods to solve the energy equation with friction losses.
        """
        # This is a placeholder for the full standard step implementation
        # In practice, this would solve:
        # E1 + S0*Δx - Sf*Δx = E2
        
        # For now, return a simple approximation
        area1 = self.channel.area(depth1)
        velocity1 = discharge / area1
        
        # Friction slope
        hydraulic_radius1 = self.channel.hydraulic_radius(depth1)
        sf = (self.manning_n * velocity1)**2 / (hydraulic_radius1**(4/3))
        
        # Energy gradient
        energy_gradient = self.slope - sf
        
        # Approximate depth change (this is simplified)
        depth_change = energy_gradient * step_length * 0.1  # Damping factor
        
        return depth1 + depth_change
