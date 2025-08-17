"""
geometry.py
Author: Alexius Academia
Date: 2025-08-17

Geometric properties of open channel cross-sections.

This module provides classes for different channel geometries including:
- Rectangular channels
- Trapezoidal channels  
- Triangular channels
- Circular channels
- Parabolic channels

Each class calculates geometric properties like area, wetted perimeter,
hydraulic radius, and top width as functions of flow depth.
"""

import math
from abc import ABC, abstractmethod
from typing import Union, Optional

from .constants import PI
from .validators import (
    validate_positive,
    validate_non_negative,
    validate_depth,
    validate_side_slope,
    validate_angle,
)
from .exceptions import InvalidGeometryError


class ChannelGeometry(ABC):
    """Abstract base class for channel geometries."""
    
    @abstractmethod
    def area(self, depth: float) -> float:
        """Calculate cross-sectional area for given depth."""
        pass
    
    @abstractmethod
    def wetted_perimeter(self, depth: float) -> float:
        """Calculate wetted perimeter for given depth."""
        pass
    
    @abstractmethod
    def top_width(self, depth: float) -> float:
        """Calculate top width for given depth."""
        pass
    
    def hydraulic_radius(self, depth: float) -> float:
        """Calculate hydraulic radius (A/P) for given depth."""
        depth = validate_depth(depth)
        area = self.area(depth)
        perimeter = self.wetted_perimeter(depth)
        
        if perimeter == 0:
            raise InvalidGeometryError("Wetted perimeter cannot be zero")
        
        return area / perimeter
    
    def hydraulic_depth(self, depth: float) -> float:
        """Calculate hydraulic depth (A/T) for given depth."""
        depth = validate_depth(depth)
        area = self.area(depth)
        width = self.top_width(depth)
        
        if width == 0:
            raise InvalidGeometryError("Top width cannot be zero")
        
        return area / width


class RectangularChannel(ChannelGeometry):
    """
    Rectangular channel cross-section.
    
    Args:
        width: Channel bottom width (m)
    """
    
    def __init__(self, width: Union[int, float]):
        self.width = validate_positive(width, "width")
    
    def area(self, depth: float) -> float:
        """Calculate cross-sectional area."""
        depth = validate_depth(depth)
        return self.width * depth
    
    def wetted_perimeter(self, depth: float) -> float:
        """Calculate wetted perimeter."""
        depth = validate_depth(depth)
        return self.width + 2 * depth
    
    def top_width(self, depth: float) -> float:
        """Calculate top width."""
        validate_depth(depth)  # Validate but don't use - top width is constant
        return self.width
    
    def __repr__(self) -> str:
        return f"RectangularChannel(width={self.width})"


class TrapezoidalChannel(ChannelGeometry):
    """
    Trapezoidal channel cross-section.
    
    Args:
        bottom_width: Channel bottom width (m)
        side_slope: Side slope ratio (horizontal:vertical)
    """
    
    def __init__(self, bottom_width: Union[int, float], side_slope: Union[int, float]):
        self.bottom_width = validate_positive(bottom_width, "bottom_width")
        self.side_slope = validate_side_slope(side_slope)
    
    def area(self, depth: float) -> float:
        """Calculate cross-sectional area."""
        depth = validate_depth(depth)
        return depth * (self.bottom_width + self.side_slope * depth)
    
    def wetted_perimeter(self, depth: float) -> float:
        """Calculate wetted perimeter."""
        depth = validate_depth(depth)
        side_length = depth * math.sqrt(1 + self.side_slope**2)
        return self.bottom_width + 2 * side_length
    
    def top_width(self, depth: float) -> float:
        """Calculate top width."""
        depth = validate_depth(depth)
        return self.bottom_width + 2 * self.side_slope * depth
    
    def __repr__(self) -> str:
        return f"TrapezoidalChannel(bottom_width={self.bottom_width}, side_slope={self.side_slope})"


class TriangularChannel(ChannelGeometry):
    """
    Triangular channel cross-section.
    
    Args:
        side_slope: Side slope ratio (horizontal:vertical)
    """
    
    def __init__(self, side_slope: Union[int, float]):
        self.side_slope = validate_positive(side_slope, "side_slope")
    
    def area(self, depth: float) -> float:
        """Calculate cross-sectional area."""
        depth = validate_depth(depth)
        return self.side_slope * depth**2
    
    def wetted_perimeter(self, depth: float) -> float:
        """Calculate wetted perimeter."""
        depth = validate_depth(depth)
        side_length = depth * math.sqrt(1 + self.side_slope**2)
        return 2 * side_length
    
    def top_width(self, depth: float) -> float:
        """Calculate top width."""
        depth = validate_depth(depth)
        return 2 * self.side_slope * depth
    
    def __repr__(self) -> str:
        return f"TriangularChannel(side_slope={self.side_slope})"


class CircularChannel(ChannelGeometry):
    """
    Circular channel cross-section.
    
    Args:
        diameter: Channel diameter (m)
    """
    
    def __init__(self, diameter: Union[int, float]):
        self.diameter = validate_positive(diameter, "diameter")
        self.radius = self.diameter / 2
    
    def _central_angle(self, depth: float) -> float:
        """Calculate central angle for given depth."""
        depth = validate_depth(depth)
        
        if depth > self.diameter:
            raise InvalidGeometryError(f"Depth ({depth}) cannot exceed diameter ({self.diameter})")
        
        # Distance from center to water surface
        h = self.radius - depth
        
        if abs(h) >= self.radius:
            if h >= self.radius:  # Empty pipe
                return 0
            else:  # Full pipe
                return 2 * PI
        
        # Central angle (radians)
        theta = 2 * math.acos(h / self.radius)
        return theta
    
    def area(self, depth: float) -> float:
        """Calculate cross-sectional area."""
        depth = validate_depth(depth)
        
        if depth >= self.diameter:
            # Full pipe
            return PI * self.radius**2
        
        theta = self._central_angle(depth)
        return (self.radius**2 / 2) * (theta - math.sin(theta))
    
    def wetted_perimeter(self, depth: float) -> float:
        """Calculate wetted perimeter."""
        depth = validate_depth(depth)
        
        if depth >= self.diameter:
            # Full pipe
            return PI * self.diameter
        
        theta = self._central_angle(depth)
        return self.radius * theta
    
    def top_width(self, depth: float) -> float:
        """Calculate top width."""
        depth = validate_depth(depth)
        
        if depth >= self.diameter:
            # Full pipe
            return 0  # No free surface
        
        if depth == 0:
            return 0
        
        # Distance from center to water surface
        h = self.radius - depth
        
        if abs(h) >= self.radius:
            return 0
        
        # Half-width at water surface
        half_width = math.sqrt(self.radius**2 - h**2)
        return 2 * half_width
    
    def __repr__(self) -> str:
        return f"CircularChannel(diameter={self.diameter})"


class ParabolicChannel(ChannelGeometry):
    """
    Parabolic channel cross-section.
    
    The parabola is defined by y = ax² where 'a' is the shape parameter.
    
    Args:
        shape_parameter: Shape parameter 'a' in y = ax²
    """
    
    def __init__(self, shape_parameter: Union[int, float]):
        self.shape_parameter = validate_positive(shape_parameter, "shape_parameter")
    
    def area(self, depth: float) -> float:
        """Calculate cross-sectional area."""
        depth = validate_non_negative(depth, "depth")
        # For parabola y = ax², area = (2/3) * width * depth
        width = self.top_width(depth)
        return (2/3) * width * depth
    
    def wetted_perimeter(self, depth: float) -> float:
        """Calculate wetted perimeter."""
        depth = validate_non_negative(depth, "depth")
        
        # Half-width at water surface
        b = math.sqrt(depth / self.shape_parameter)
        
        # Wetted perimeter using integration formula
        # P = 2 * integral from 0 to b of sqrt(1 + (2ax)²) dx
        # This is an elliptic integral, approximated here
        
        if b == 0:
            return 0
        
        # Approximation for small slopes
        if 2 * self.shape_parameter * b < 0.5:
            return 2 * b * (1 + (2/3) * (self.shape_parameter * b)**2)
        
        # More accurate formula for larger slopes
        term1 = 2 * self.shape_parameter * b
        term2 = math.sqrt(1 + term1**2)
        term3 = math.log(term1 + term2)
        
        return (b * term2 + term3 / (2 * self.shape_parameter))
    
    def top_width(self, depth: float) -> float:
        """Calculate top width."""
        depth = validate_non_negative(depth, "depth")
        
        if depth == 0:
            return 0
        
        # For parabola y = ax², width = 2 * sqrt(y/a)
        return 2 * math.sqrt(depth / self.shape_parameter)
    
    def __repr__(self) -> str:
        return f"ParabolicChannel(shape_parameter={self.shape_parameter})"


# Factory function for creating channels
def create_channel(channel_type: str, **kwargs) -> ChannelGeometry:
    """
    Factory function to create channel geometries.
    
    Args:
        channel_type: Type of channel ('rectangular', 'trapezoidal', 'triangular', 
                     'circular', 'parabolic')
        **kwargs: Parameters specific to each channel type
        
    Returns:
        ChannelGeometry instance
        
    Raises:
        InvalidGeometryError: If channel type is not recognized
    """
    channel_types = {
        'rectangular': RectangularChannel,
        'trapezoidal': TrapezoidalChannel,
        'triangular': TriangularChannel,
        'circular': CircularChannel,
        'parabolic': ParabolicChannel,
    }
    
    if channel_type.lower() not in channel_types:
        available = ', '.join(channel_types.keys())
        raise InvalidGeometryError(f"Unknown channel type '{channel_type}'. Available: {available}")
    
    channel_class = channel_types[channel_type.lower()]
    return channel_class(**kwargs)
