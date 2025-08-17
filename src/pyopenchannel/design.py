"""
File: design.py
Author: Alexius Academia
Date: 2025-08-17

Channel design and optimization for open channel flow.

This module provides tools for:
- Optimal channel design for minimum excavation
- Economic channel design considering construction costs
- Channel sizing for given flow requirements
"""

import math
from typing import Union, Optional, Dict, Tuple, List
from dataclasses import dataclass
from abc import ABC, abstractmethod

from .constants import SIDE_SLOPES
from .geometry import (
    ChannelGeometry,
    RectangularChannel,
    TrapezoidalChannel,
    TriangularChannel,
)
from .hydraulics import ManningEquation, NormalDepth, CriticalDepth
from .validators import (
    validate_positive,
    validate_discharge,
    validate_manning_n,
    validate_slope,
    validate_side_slope,
)
from .exceptions import InvalidGeometryError, InvalidFlowConditionError


@dataclass
class DesignResult:
    """
    Results from channel design optimization.
    
    Attributes:
        channel: Optimized channel geometry
        depth: Design flow depth (m)
        velocity: Design velocity (m/s)
        area: Cross-sectional area (m²)
        perimeter: Wetted perimeter (m)
        hydraulic_radius: Hydraulic radius (m)
        froude_number: Froude number
        excavation_area: Excavation cross-sectional area (m²)
        cost_per_meter: Cost per unit length (if applicable)
        freeboard: Recommended freeboard (m)
    """
    channel: ChannelGeometry
    depth: float
    velocity: float
    area: float
    perimeter: float
    hydraulic_radius: float
    froude_number: float
    excavation_area: float
    cost_per_meter: Optional[float] = None
    freeboard: Optional[float] = None
    
    @property
    def total_depth(self) -> float:
        """Total channel depth including freeboard."""
        if self.freeboard is not None:
            return self.depth + self.freeboard
        return self.depth


class OptimalSections:
    """
    Design optimal channel sections for hydraulic efficiency.
    
    Optimal sections minimize wetted perimeter for a given area,
    resulting in maximum discharge for given slope and roughness.
    """
    
    @staticmethod
    def rectangular(
        discharge: float,
        slope: float,
        manning_n: float,
        max_depth: Optional[float] = None
    ) -> DesignResult:
        """
        Design optimal rectangular section.
        
        For rectangular channels, optimal hydraulic efficiency occurs when
        depth = width/2 (half-square section).
        
        Args:
            discharge: Design discharge (m³/s)
            slope: Channel slope (dimensionless)
            manning_n: Manning's roughness coefficient
            max_depth: Maximum allowable depth (m)
            
        Returns:
            DesignResult with optimal dimensions
        """
        discharge = validate_discharge(discharge)
        slope = validate_slope(slope)
        manning_n = validate_manning_n(manning_n)
        
        # For optimal rectangular: y = b/2, so A = b*y = 2y²
        # From Manning's equation: Q = (manning_factor/n) * A * R^(2/3) * S^(1/2)
        # For optimal rectangular: R = b/4 = y/2
        
        # Import unit-aware Manning factor
        from .units import get_manning_factor
        manning_factor = get_manning_factor()
        
        # Solve: Q = (manning_factor/n) * 2y² * (y/2)^(2/3) * S^(1/2)
        # Q = (manning_factor/n) * 2y² * y^(2/3) / 2^(2/3) * S^(1/2)
        # Q = (manning_factor/n) * 2^(1/3) * y^(8/3) * S^(1/2)
        
        coefficient = (manning_factor / manning_n) * (2**(1/3)) * (slope**0.5)
        optimal_depth = (discharge / coefficient)**(3/8)
        optimal_width = 2 * optimal_depth
        
        # Check depth constraint
        if max_depth is not None and optimal_depth > max_depth:
            # Use constrained depth and calculate required width
            depth = max_depth
            # From Manning's equation, solve for width
            width = OptimalSections._solve_rectangular_width(
                discharge, depth, slope, manning_n
            )
        else:
            depth = optimal_depth
            width = optimal_width
        
        channel = RectangularChannel(width)
        
        return OptimalSections._create_design_result(
            channel, depth, discharge, slope, manning_n
        )
    
    @staticmethod
    def trapezoidal(
        discharge: float,
        slope: float,
        manning_n: float,
        side_slope: float,
        max_depth: Optional[float] = None
    ) -> DesignResult:
        """
        Design optimal trapezoidal section.
        
        For trapezoidal channels, optimal hydraulic efficiency occurs when
        the channel forms half of a regular hexagon.
        
        Args:
            discharge: Design discharge (m³/s)
            slope: Channel slope (dimensionless)
            manning_n: Manning's roughness coefficient
            side_slope: Side slope ratio (horizontal:vertical)
            max_depth: Maximum allowable depth (m)
            
        Returns:
            DesignResult with optimal dimensions
        """
        discharge = validate_discharge(discharge)
        slope = validate_slope(slope)
        manning_n = validate_manning_n(manning_n)
        side_slope = validate_side_slope(side_slope)
        
        # For optimal trapezoidal with given side slope:
        # b = 2y(√(1+m²) - m) where m is side slope
        
        # This requires iterative solution
        optimal_depth = OptimalSections._solve_optimal_trapezoidal_depth(
            discharge, slope, manning_n, side_slope
        )
        
        optimal_bottom_width = 2 * optimal_depth * (
            math.sqrt(1 + side_slope**2) - side_slope
        )
        
        # Ensure positive bottom width
        if optimal_bottom_width <= 0:
            optimal_bottom_width = optimal_depth * 0.1  # Minimum width
        
        # Check depth constraint
        if max_depth is not None and optimal_depth > max_depth:
            depth = max_depth
            # Recalculate bottom width for constrained depth
            bottom_width = OptimalSections._solve_trapezoidal_width(
                discharge, depth, slope, manning_n, side_slope
            )
        else:
            depth = optimal_depth
            bottom_width = optimal_bottom_width
        
        channel = TrapezoidalChannel(bottom_width, side_slope)
        
        return OptimalSections._create_design_result(
            channel, depth, discharge, slope, manning_n
        )
    
    @staticmethod
    def triangular(
        discharge: float,
        slope: float,
        manning_n: float,
        max_depth: Optional[float] = None
    ) -> DesignResult:
        """
        Design optimal triangular section.
        
        For triangular channels, optimal hydraulic efficiency occurs when
        the vertex angle is 90 degrees (side slope = 1:1).
        
        Args:
            discharge: Design discharge (m³/s)
            slope: Channel slope (dimensionless)
            manning_n: Manning's roughness coefficient
            max_depth: Maximum allowable depth (m)
            
        Returns:
            DesignResult with optimal dimensions
        """
        discharge = validate_discharge(discharge)
        slope = validate_slope(slope)
        manning_n = validate_manning_n(manning_n)
        
        # Optimal triangular has side slope = 1 (45° sides)
        optimal_side_slope = 1.0
        
        # For triangular: A = m*y², P = 2y*√(1+m²), R = A/P
        # For optimal (m=1): A = y², P = 2y√2, R = y/(2√2)
        
        # Import unit-aware Manning factor
        from .units import get_manning_factor
        manning_factor = get_manning_factor()
        
        # From Manning's: Q = (manning_factor/n) * y² * (y/(2√2))^(2/3) * S^(1/2)
        coefficient = (manning_factor / manning_n) * (1 / (2 * math.sqrt(2))**(2/3)) * (slope**0.5)
        optimal_depth = (discharge / coefficient)**(3/8)
        
        # Check depth constraint
        if max_depth is not None and optimal_depth > max_depth:
            depth = max_depth
            # Calculate required side slope for constrained depth
            side_slope = OptimalSections._solve_triangular_side_slope(
                discharge, depth, slope, manning_n
            )
        else:
            depth = optimal_depth
            side_slope = optimal_side_slope
        
        channel = TriangularChannel(side_slope)
        
        return OptimalSections._create_design_result(
            channel, depth, discharge, slope, manning_n
        )
    
    @staticmethod
    def _solve_rectangular_width(
        discharge: float,
        depth: float,
        slope: float,
        manning_n: float
    ) -> float:
        """Solve for rectangular channel width given depth."""
        # From Manning's equation: Q = (1/n) * b*y * (b*y/(b+2y))^(2/3) * S^(1/2)
        # This requires iterative solution
        
        # Initial guess
        width = discharge / (depth * math.sqrt(slope))
        
        for _ in range(20):  # Simple iteration
            area = width * depth
            perimeter = width + 2 * depth
            hydraulic_radius = area / perimeter
            
            calculated_q = ManningEquation.discharge(area, hydraulic_radius, slope, manning_n)
            
            if abs(calculated_q - discharge) < 0.001:
                break
            
            # Adjust width
            width *= discharge / calculated_q
        
        return width
    
    @staticmethod
    def _solve_optimal_trapezoidal_depth(
        discharge: float,
        slope: float,
        manning_n: float,
        side_slope: float
    ) -> float:
        """Solve for optimal trapezoidal depth using iteration."""
        # Initial guess
        depth = (discharge / slope**0.5)**(1/3)
        
        for _ in range(20):
            bottom_width = 2 * depth * (math.sqrt(1 + side_slope**2) - side_slope)
            if bottom_width <= 0:
                bottom_width = depth * 0.1
            
            channel = TrapezoidalChannel(bottom_width, side_slope)
            calculated_q = OptimalSections._calculate_discharge(channel, depth, slope, manning_n)
            
            if abs(calculated_q - discharge) < 0.001:
                break
            
            # Adjust depth
            depth *= (discharge / calculated_q)**(3/8)
        
        return depth
    
    @staticmethod
    def _solve_trapezoidal_width(
        discharge: float,
        depth: float,
        slope: float,
        manning_n: float,
        side_slope: float
    ) -> float:
        """Solve for trapezoidal bottom width given depth."""
        # Initial guess
        width = discharge / (depth * math.sqrt(slope))
        
        for _ in range(20):
            channel = TrapezoidalChannel(width, side_slope)
            calculated_q = OptimalSections._calculate_discharge(channel, depth, slope, manning_n)
            
            if abs(calculated_q - discharge) < 0.001:
                break
            
            # Adjust width
            width *= discharge / calculated_q
        
        return max(width, 0.1)  # Ensure positive width
    
    @staticmethod
    def _solve_triangular_side_slope(
        discharge: float,
        depth: float,
        slope: float,
        manning_n: float
    ) -> float:
        """Solve for triangular side slope given depth."""
        # Initial guess
        side_slope = 1.0
        
        for _ in range(20):
            channel = TriangularChannel(side_slope)
            calculated_q = OptimalSections._calculate_discharge(channel, depth, slope, manning_n)
            
            if abs(calculated_q - discharge) < 0.001:
                break
            
            # Adjust side slope
            side_slope *= (discharge / calculated_q)**(1/2)
        
        return side_slope
    
    @staticmethod
    def _calculate_discharge(
        channel: ChannelGeometry,
        depth: float,
        slope: float,
        manning_n: float
    ) -> float:
        """Calculate discharge for given channel and depth."""
        area = channel.area(depth)
        hydraulic_radius = channel.hydraulic_radius(depth)
        return ManningEquation.discharge(area, hydraulic_radius, slope, manning_n)
    
    @staticmethod
    def _create_design_result(
        channel: ChannelGeometry,
        depth: float,
        discharge: float,
        slope: float,
        manning_n: float
    ) -> DesignResult:
        """Create DesignResult from channel parameters."""
        area = channel.area(depth)
        perimeter = channel.wetted_perimeter(depth)
        hydraulic_radius = channel.hydraulic_radius(depth)
        velocity = discharge / area
        
        # Calculate Froude number
        from .units import get_gravity
        gravity = get_gravity()
        
        top_width = channel.top_width(depth)
        hydraulic_depth = area / top_width if top_width > 0 else depth
        froude_number = velocity / math.sqrt(gravity * hydraulic_depth)
        
        # Excavation area (assuming rectangular excavation)
        if isinstance(channel, RectangularChannel):
            excavation_area = channel.width * depth
        elif isinstance(channel, TrapezoidalChannel):
            excavation_area = depth * (channel.bottom_width + channel.side_slope * depth)
        elif isinstance(channel, TriangularChannel):
            excavation_area = channel.side_slope * depth**2
        else:
            excavation_area = area  # Approximation
        
        # Calculate recommended freeboard
        freeboard = ChannelDesigner.calculate_freeboard(discharge, depth, velocity)
        
        return DesignResult(
            channel=channel,
            depth=depth,
            velocity=velocity,
            area=area,
            perimeter=perimeter,
            hydraulic_radius=hydraulic_radius,
            froude_number=froude_number,
            excavation_area=excavation_area,
            freeboard=freeboard
        )


class EconomicSections:
    """
    Design economical channel sections considering construction costs.
    
    Economic sections minimize the total cost including excavation,
    lining, and maintenance costs.
    """
    
    def __init__(
        self,
        excavation_cost_per_m3: float,
        lining_cost_per_m2: Optional[float] = None,
        land_cost_per_m2: Optional[float] = None
    ):
        """
        Initialize economic design parameters.
        
        Args:
            excavation_cost_per_m3: Cost of excavation per cubic meter
            lining_cost_per_m2: Cost of lining per square meter (optional)
            land_cost_per_m2: Cost of land per square meter (optional)
        """
        self.excavation_cost = validate_positive(excavation_cost_per_m3, "excavation_cost_per_m3")
        self.lining_cost = lining_cost_per_m2
        self.land_cost = land_cost_per_m2
        
        if self.lining_cost is not None:
            self.lining_cost = validate_positive(lining_cost_per_m2, "lining_cost_per_m2")
        if self.land_cost is not None:
            self.land_cost = validate_positive(land_cost_per_m2, "land_cost_per_m2")
    
    def design_rectangular(
        self,
        discharge: float,
        slope: float,
        manning_n: float,
        width_range: Tuple[float, float] = (0.5, 20.0),
        num_trials: int = 50
    ) -> DesignResult:
        """
        Design economical rectangular section.
        
        Args:
            discharge: Design discharge (m³/s)
            slope: Channel slope (dimensionless)
            manning_n: Manning's roughness coefficient
            width_range: Range of widths to consider (min, max)
            num_trials: Number of trial widths
            
        Returns:
            DesignResult with most economical dimensions
        """
        discharge = validate_discharge(discharge)
        slope = validate_slope(slope)
        manning_n = validate_manning_n(manning_n)
        
        min_width, max_width = width_range
        min_width = validate_positive(min_width, "min_width")
        max_width = validate_positive(max_width, "max_width")
        
        if min_width >= max_width:
            raise InvalidGeometryError("min_width must be less than max_width")
        
        best_cost = float('inf')
        best_result = None
        
        widths = [min_width + i * (max_width - min_width) / (num_trials - 1) 
                 for i in range(num_trials)]
        
        for width in widths:
            try:
                channel = RectangularChannel(width)
                depth = NormalDepth.calculate(channel, discharge, slope, manning_n)
                
                # Calculate costs
                cost_per_meter = self._calculate_rectangular_cost(width, depth)
                
                if cost_per_meter < best_cost:
                    best_cost = cost_per_meter
                    best_result = OptimalSections._create_design_result(
                        channel, depth, discharge, slope, manning_n
                    )
                    best_result.cost_per_meter = cost_per_meter
                    
            except Exception:
                continue  # Skip invalid configurations
        
        if best_result is None:
            raise InvalidFlowConditionError("No valid economic solution found")
        
        return best_result
    
    def design_trapezoidal(
        self,
        discharge: float,
        slope: float,
        manning_n: float,
        side_slope: float,
        width_range: Tuple[float, float] = (0.5, 20.0),
        num_trials: int = 50
    ) -> DesignResult:
        """
        Design economical trapezoidal section.
        
        Args:
            discharge: Design discharge (m³/s)
            slope: Channel slope (dimensionless)
            manning_n: Manning's roughness coefficient
            side_slope: Side slope ratio (horizontal:vertical)
            width_range: Range of bottom widths to consider (min, max)
            num_trials: Number of trial widths
            
        Returns:
            DesignResult with most economical dimensions
        """
        discharge = validate_discharge(discharge)
        slope = validate_slope(slope)
        manning_n = validate_manning_n(manning_n)
        side_slope = validate_side_slope(side_slope)
        
        min_width, max_width = width_range
        min_width = validate_positive(min_width, "min_width")
        max_width = validate_positive(max_width, "max_width")
        
        best_cost = float('inf')
        best_result = None
        
        widths = [min_width + i * (max_width - min_width) / (num_trials - 1) 
                 for i in range(num_trials)]
        
        for bottom_width in widths:
            try:
                channel = TrapezoidalChannel(bottom_width, side_slope)
                depth = NormalDepth.calculate(channel, discharge, slope, manning_n)
                
                # Calculate costs
                cost_per_meter = self._calculate_trapezoidal_cost(bottom_width, depth, side_slope)
                
                if cost_per_meter < best_cost:
                    best_cost = cost_per_meter
                    best_result = OptimalSections._create_design_result(
                        channel, depth, discharge, slope, manning_n
                    )
                    best_result.cost_per_meter = cost_per_meter
                    
            except Exception:
                continue  # Skip invalid configurations
        
        if best_result is None:
            raise InvalidFlowConditionError("No valid economic solution found")
        
        return best_result
    
    def _calculate_rectangular_cost(self, width: float, depth: float) -> float:
        """Calculate cost per meter for rectangular section."""
        # Excavation cost
        excavation_area = width * depth
        excavation_cost = excavation_area * self.excavation_cost
        
        # Lining cost
        lining_cost = 0.0
        if self.lining_cost is not None:
            wetted_perimeter = width + 2 * depth
            lining_cost = wetted_perimeter * self.lining_cost
        
        # Land cost
        land_cost = 0.0
        if self.land_cost is not None:
            # Assume some additional width for maintenance access
            total_width = width + 2.0  # 1m on each side
            land_cost = total_width * self.land_cost
        
        return excavation_cost + lining_cost + land_cost
    
    def _calculate_trapezoidal_cost(
        self,
        bottom_width: float,
        depth: float,
        side_slope: float
    ) -> float:
        """Calculate cost per meter for trapezoidal section."""
        # Excavation cost
        excavation_area = depth * (bottom_width + side_slope * depth)
        excavation_cost = excavation_area * self.excavation_cost
        
        # Lining cost
        lining_cost = 0.0
        if self.lining_cost is not None:
            side_length = depth * math.sqrt(1 + side_slope**2)
            wetted_perimeter = bottom_width + 2 * side_length
            lining_cost = wetted_perimeter * self.lining_cost
        
        # Land cost
        land_cost = 0.0
        if self.land_cost is not None:
            top_width = bottom_width + 2 * side_slope * depth
            # Assume some additional width for maintenance access
            total_width = top_width + 2.0  # 1m on each side
            land_cost = total_width * self.land_cost
        
        return excavation_cost + lining_cost + land_cost


class ChannelDesigner:
    """
    General channel design utilities and helper functions.
    """
    
    @staticmethod
    def calculate_freeboard(
        discharge: float,
        depth: float,
        velocity: float,
        channel_type: str = "general"
    ) -> float:
        """
        Calculate recommended freeboard.
        
        Args:
            discharge: Design discharge (m³/s)
            depth: Flow depth (m)
            velocity: Flow velocity (m/s)
            channel_type: Type of channel ("general", "concrete", "earth")
            
        Returns:
            Recommended freeboard (m)
        """
        discharge = validate_discharge(discharge)
        depth = validate_positive(depth, "depth")
        velocity = validate_positive(velocity, "velocity")
        
        # Base freeboard on discharge and velocity
        if discharge < 0.5:
            base_freeboard = 0.3
        elif discharge < 5.0:
            base_freeboard = 0.4
        elif discharge < 50.0:
            base_freeboard = 0.5
        else:
            base_freeboard = 0.6
        
        # Adjust for velocity
        if velocity > 3.0:
            base_freeboard += 0.2
        elif velocity > 2.0:
            base_freeboard += 0.1
        
        # Adjust for channel type
        if channel_type.lower() == "earth":
            base_freeboard += 0.2
        elif channel_type.lower() == "concrete":
            base_freeboard *= 0.8
        
        # Minimum freeboard
        return max(base_freeboard, 0.3)
    
    @staticmethod
    def check_velocity_limits(
        velocity: float,
        channel_material: str = "earth"
    ) -> Dict[str, Union[bool, str]]:
        """
        Check if velocity is within acceptable limits.
        
        Args:
            velocity: Flow velocity (m/s)
            channel_material: Channel material type
            
        Returns:
            Dictionary with check results
        """
        velocity = validate_positive(velocity, "velocity")
        
        # Velocity limits by material
        limits = {
            "earth": {"min": 0.3, "max": 1.5},
            "concrete": {"min": 0.6, "max": 6.0},
            "rock": {"min": 0.5, "max": 3.0},
            "grass": {"min": 0.3, "max": 1.2},
        }
        
        material_limits = limits.get(channel_material.lower(), limits["earth"])
        
        result = {
            "velocity": velocity,
            "material": channel_material,
            "min_velocity": material_limits["min"],
            "max_velocity": material_limits["max"],
            "is_acceptable": True,
            "warnings": []
        }
        
        if velocity < material_limits["min"]:
            result["is_acceptable"] = False
            result["warnings"].append(f"Velocity too low - may cause sedimentation")
        
        if velocity > material_limits["max"]:
            result["is_acceptable"] = False
            result["warnings"].append(f"Velocity too high - may cause erosion")
        
        return result
    
    @staticmethod
    def recommend_side_slope(soil_type: str) -> float:
        """
        Recommend side slope based on soil type.
        
        Args:
            soil_type: Type of soil
            
        Returns:
            Recommended side slope ratio (horizontal:vertical)
        """
        return SIDE_SLOPES.get(soil_type.lower(), SIDE_SLOPES["earth_ordinary"])
    
    @staticmethod
    def size_channel_for_capacity(
        target_discharge: float,
        slope: float,
        manning_n: float,
        channel_type: str = "rectangular",
        **kwargs
    ) -> DesignResult:
        """
        Size channel to carry target discharge with optimal efficiency.
        
        Args:
            target_discharge: Required discharge capacity (m³/s)
            slope: Channel slope (dimensionless)
            manning_n: Manning's roughness coefficient
            channel_type: Type of channel to design
            **kwargs: Additional parameters for specific channel types
            
        Returns:
            DesignResult with sized channel
        """
        if channel_type.lower() == "rectangular":
            return OptimalSections.rectangular(target_discharge, slope, manning_n)
        elif channel_type.lower() == "trapezoidal":
            side_slope = kwargs.get("side_slope", 1.5)
            return OptimalSections.trapezoidal(target_discharge, slope, manning_n, side_slope)
        elif channel_type.lower() == "triangular":
            return OptimalSections.triangular(target_discharge, slope, manning_n)
        else:
            raise InvalidGeometryError(f"Unknown channel type: {channel_type}")
