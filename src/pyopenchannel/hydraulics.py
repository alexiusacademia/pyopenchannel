"""
hydraulics.py
Author: Alexius Academia
Date: 2025-08-17

Hydraulic calculations for open channel flow.

This module provides classes and functions for fundamental hydraulic calculations:
- Manning's equation
- Chezy equation  
- Critical depth calculations
- Normal depth calculations
- Hydraulic radius calculations
"""

import math
from typing import Union, Optional, Tuple

from .constants import DEFAULT_TOLERANCE, MAX_ITERATIONS
from .geometry import ChannelGeometry
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
    InvalidSlopeError,
)


class ManningEquation:
    """
    Manning's equation for uniform flow in open channels.
    
    SI Units: Q = (1/n) * A * R^(2/3) * S^(1/2)
    US Units: Q = (1.49/n) * A * R^(2/3) * S^(1/2)
    
    Where:
    - Q = discharge (m³/s or ft³/s)
    - n = Manning's roughness coefficient
    - A = cross-sectional area (m² or ft²)
    - R = hydraulic radius (m or ft)
    - S = channel slope (dimensionless)
    """
    
    @staticmethod
    def discharge(
        area: float,
        hydraulic_radius: float,
        slope: float,
        manning_n: float
    ) -> float:
        """
        Calculate discharge using Manning's equation.
        
        Automatically uses the correct unit factor based on the current unit system.
        
        Args:
            area: Cross-sectional area (current unit system)
            hydraulic_radius: Hydraulic radius (current unit system)
            slope: Channel slope (dimensionless)
            manning_n: Manning's roughness coefficient
            
        Returns:
            Discharge (current unit system)
        """
        # Import here to avoid circular imports
        from .units import get_manning_factor
        
        area = validate_positive(area, "area")
        hydraulic_radius = validate_positive(hydraulic_radius, "hydraulic_radius")
        slope = validate_slope(slope)
        manning_n = validate_manning_n(manning_n)
        
        # Get the appropriate Manning factor for current unit system
        manning_factor = get_manning_factor()
        
        return (manning_factor / manning_n) * area * (hydraulic_radius ** (2/3)) * (slope ** 0.5)
    
    @staticmethod
    def velocity(
        hydraulic_radius: float,
        slope: float,
        manning_n: float
    ) -> float:
        """
        Calculate average velocity using Manning's equation.
        
        Automatically uses the correct unit factor based on the current unit system.
        
        Args:
            hydraulic_radius: Hydraulic radius (current unit system)
            slope: Channel slope (dimensionless)
            manning_n: Manning's roughness coefficient
            
        Returns:
            Average velocity (current unit system)
        """
        # Import here to avoid circular imports
        from .units import get_manning_factor
        
        hydraulic_radius = validate_positive(hydraulic_radius, "hydraulic_radius")
        slope = validate_slope(slope)
        manning_n = validate_manning_n(manning_n)
        
        # Get the appropriate Manning factor for current unit system
        manning_factor = get_manning_factor()
        
        return (manning_factor / manning_n) * (hydraulic_radius ** (2/3)) * (slope ** 0.5)
    
    @staticmethod
    def required_slope(
        discharge: float,
        area: float,
        hydraulic_radius: float,
        manning_n: float
    ) -> float:
        """
        Calculate required slope for given discharge.
        
        Automatically uses the correct unit factor based on the current unit system.
        
        Args:
            discharge: Discharge (current unit system)
            area: Cross-sectional area (current unit system)
            hydraulic_radius: Hydraulic radius (current unit system)
            manning_n: Manning's roughness coefficient
            
        Returns:
            Required slope (dimensionless)
        """
        # Import here to avoid circular imports
        from .units import get_manning_factor
        
        discharge = validate_discharge(discharge)
        area = validate_positive(area, "area")
        hydraulic_radius = validate_positive(hydraulic_radius, "hydraulic_radius")
        manning_n = validate_manning_n(manning_n)
        
        # Get the appropriate Manning factor for current unit system
        manning_factor = get_manning_factor()
        
        # S = (Q * n / (manning_factor * A * R^(2/3)))^2
        term = (discharge * manning_n) / (manning_factor * area * (hydraulic_radius ** (2/3)))
        return term ** 2


class ChezyEquation:
    """
    Chezy equation for uniform flow in open channels.
    
    Q = C * A * sqrt(R * S)
    
    Where:
    - Q = discharge (m³/s)
    - C = Chezy coefficient
    - A = cross-sectional area (m²)
    - R = hydraulic radius (m)
    - S = channel slope (dimensionless)
    """
    
    @staticmethod
    def discharge(
        area: float,
        hydraulic_radius: float,
        slope: float,
        chezy_c: float
    ) -> float:
        """
        Calculate discharge using Chezy equation.
        
        Args:
            area: Cross-sectional area (m²)
            hydraulic_radius: Hydraulic radius (m)
            slope: Channel slope (dimensionless)
            chezy_c: Chezy coefficient
            
        Returns:
            Discharge (m³/s)
        """
        area = validate_positive(area, "area")
        hydraulic_radius = validate_positive(hydraulic_radius, "hydraulic_radius")
        slope = validate_slope(slope)
        chezy_c = validate_positive(chezy_c, "chezy_c")
        
        return chezy_c * area * math.sqrt(hydraulic_radius * slope)
    
    @staticmethod
    def chezy_from_manning(manning_n: float, hydraulic_radius: float) -> float:
        """
        Convert Manning's n to Chezy coefficient.
        
        C = R^(1/6) / n
        
        Args:
            manning_n: Manning's roughness coefficient
            hydraulic_radius: Hydraulic radius (m)
            
        Returns:
            Chezy coefficient
        """
        manning_n = validate_manning_n(manning_n)
        hydraulic_radius = validate_positive(hydraulic_radius, "hydraulic_radius")
        
        return (hydraulic_radius ** (1/6)) / manning_n


class HydraulicRadius:
    """Utility class for hydraulic radius calculations."""
    
    @staticmethod
    def from_geometry(channel: ChannelGeometry, depth: float) -> float:
        """
        Calculate hydraulic radius from channel geometry.
        
        Args:
            channel: Channel geometry object
            depth: Flow depth (m)
            
        Returns:
            Hydraulic radius (m)
        """
        return channel.hydraulic_radius(depth)
    
    @staticmethod
    def optimal_rectangular(width: float) -> float:
        """
        Calculate hydraulic radius for optimal rectangular section.
        
        For rectangular channels, optimal hydraulic efficiency occurs when
        depth = width/2, giving R = width/4.
        
        Args:
            width: Channel width (m)
            
        Returns:
            Optimal hydraulic radius (m)
        """
        width = validate_positive(width, "width")
        return width / 4
    
    @staticmethod
    def optimal_trapezoidal(bottom_width: float, side_slope: float) -> float:
        """
        Calculate hydraulic radius for optimal trapezoidal section.
        
        Args:
            bottom_width: Bottom width (m)
            side_slope: Side slope ratio (horizontal:vertical)
            
        Returns:
            Optimal hydraulic radius (m)
        """
        bottom_width = validate_positive(bottom_width, "bottom_width")
        side_slope = validate_positive(side_slope, "side_slope")
        
        # For optimal trapezoidal section: depth = bottom_width / (2 * sqrt(3))
        optimal_depth = bottom_width / (2 * math.sqrt(3))
        
        # Calculate area and perimeter for optimal depth
        area = optimal_depth * (bottom_width + side_slope * optimal_depth)
        side_length = optimal_depth * math.sqrt(1 + side_slope**2)
        perimeter = bottom_width + 2 * side_length
        
        return area / perimeter


class CriticalDepth:
    """
    Critical depth calculations for open channels.
    
    Critical flow occurs when the Froude number equals 1.0.
    """
    
    @staticmethod
    def calculate(
        channel: ChannelGeometry,
        discharge: float,
        tolerance: float = DEFAULT_TOLERANCE,
        max_iterations: int = MAX_ITERATIONS
    ) -> float:
        """
        Calculate critical depth using Newton-Raphson method.
        
        Args:
            channel: Channel geometry object
            discharge: Discharge (current unit system)
            tolerance: Convergence tolerance
            max_iterations: Maximum iterations
            
        Returns:
            Critical depth (current unit system)
            
        Raises:
            ConvergenceError: If solution doesn't converge
        """
        # Import here to avoid circular imports
        from .units import get_gravity
        
        discharge = validate_discharge(discharge)
        gravity = get_gravity()
        
        # Initial guess - use hydraulic depth approximation
        depth = (discharge**2 / gravity)**(1/3)
        
        for iteration in range(max_iterations):
            try:
                area = channel.area(depth)
                top_width = channel.top_width(depth)
                
                if top_width == 0:
                    raise InvalidFlowConditionError("Top width is zero - no free surface")
                
                # Critical flow condition: Q² = g * A³ / T
                f = discharge**2 - gravity * (area**3) / top_width
                
                # Derivative for Newton-Raphson
                dA_dy = top_width  # dA/dy = T for most geometries
                dT_dy = CriticalDepth._top_width_derivative(channel, depth)
                
                df_dy = -gravity * (3 * area**2 * dA_dy * top_width - area**3 * dT_dy) / (top_width**2)
                
                if abs(df_dy) < 1e-12:
                    raise ConvergenceError("Derivative too small - cannot continue iteration")
                
                depth_new = depth - f / df_dy
                
                if depth_new <= 0:
                    depth_new = depth / 2  # Prevent negative depth
                
                if abs(depth_new - depth) < tolerance:
                    return depth_new
                
                depth = depth_new
                
            except Exception as e:
                if iteration == 0:
                    raise ConvergenceError(f"Critical depth calculation failed: {e}")
                # Try with different initial guess
                depth = depth * 1.5
        
        raise ConvergenceError(f"Critical depth did not converge after {max_iterations} iterations")
    
    @staticmethod
    def _top_width_derivative(channel: ChannelGeometry, depth: float) -> float:
        """
        Calculate derivative of top width with respect to depth.
        
        This is approximated using finite differences.
        """
        delta = depth * 1e-6
        if delta < 1e-9:
            delta = 1e-9
        
        try:
            t1 = channel.top_width(depth + delta)
            t2 = channel.top_width(depth - delta)
            return (t1 - t2) / (2 * delta)
        except:
            # Fallback to forward difference
            t1 = channel.top_width(depth + delta)
            t0 = channel.top_width(depth)
            return (t1 - t0) / delta
    
    @staticmethod
    def froude_number(velocity: float, hydraulic_depth: float) -> float:
        """
        Calculate Froude number.
        
        Fr = V / sqrt(g * D)
        
        Args:
            velocity: Average velocity (current unit system)
            hydraulic_depth: Hydraulic depth (current unit system)
            
        Returns:
            Froude number (dimensionless)
        """
        # Import here to avoid circular imports
        from .units import get_gravity
        
        velocity = validate_positive(velocity, "velocity")
        hydraulic_depth = validate_positive(hydraulic_depth, "hydraulic_depth")
        gravity = get_gravity()
        
        return velocity / math.sqrt(gravity * hydraulic_depth)


class NormalDepth:
    """
    Normal depth calculations for uniform flow in open channels.
    
    Normal depth occurs when the flow is uniform (constant depth and velocity).
    """
    
    @staticmethod
    def calculate(
        channel: ChannelGeometry,
        discharge: float,
        slope: float,
        manning_n: float,
        tolerance: float = DEFAULT_TOLERANCE,
        max_iterations: int = MAX_ITERATIONS
    ) -> float:
        """
        Calculate normal depth using Newton-Raphson method.
        
        Args:
            channel: Channel geometry object
            discharge: Discharge (m³/s)
            slope: Channel slope (dimensionless)
            manning_n: Manning's roughness coefficient
            tolerance: Convergence tolerance
            max_iterations: Maximum iterations
            
        Returns:
            Normal depth (m)
            
        Raises:
            ConvergenceError: If solution doesn't converge
        """
        discharge = validate_discharge(discharge)
        slope = validate_slope(slope)
        manning_n = validate_manning_n(manning_n)
        
        # Initial guess based on wide channel approximation
        depth = ((discharge * manning_n) / (slope**0.5))**(3/5)
        
        for iteration in range(max_iterations):
            try:
                area = channel.area(depth)
                perimeter = channel.wetted_perimeter(depth)
                
                if perimeter == 0:
                    raise InvalidFlowConditionError("Wetted perimeter is zero")
                
                hydraulic_radius = area / perimeter
                
                # Manning's equation: Q = (1/n) * A * R^(2/3) * S^(1/2)
                calculated_q = ManningEquation.discharge(area, hydraulic_radius, slope, manning_n)
                
                f = calculated_q - discharge
                
                # Calculate derivative using finite differences
                delta = depth * 1e-6
                if delta < 1e-9:
                    delta = 1e-9
                
                area_plus = channel.area(depth + delta)
                perimeter_plus = channel.wetted_perimeter(depth + delta)
                hydraulic_radius_plus = area_plus / perimeter_plus
                calculated_q_plus = ManningEquation.discharge(area_plus, hydraulic_radius_plus, slope, manning_n)
                
                df_dy = (calculated_q_plus - calculated_q) / delta
                
                if abs(df_dy) < 1e-12:
                    raise ConvergenceError("Derivative too small - cannot continue iteration")
                
                depth_new = depth - f / df_dy
                
                if depth_new <= 0:
                    depth_new = depth / 2  # Prevent negative depth
                
                if abs(depth_new - depth) < tolerance:
                    return depth_new
                
                depth = depth_new
                
            except Exception as e:
                if iteration == 0:
                    raise ConvergenceError(f"Normal depth calculation failed: {e}")
                # Try with different initial guess
                depth = depth * 1.2
        
        raise ConvergenceError(f"Normal depth did not converge after {max_iterations} iterations")
