"""
validators.py
Author: Alexius Academia
Date: 2025-08-17

Input validation utilities for the PyOpenChannel library.

This module contains the following functions:
- validate_positive
- validate_non_negative
- validate_depth
- validate_discharge
- validate_manning_n
- validate_slope
- validate_side_slope
- validate_angle

The functions are used to validate the input parameters for the open channel flow calculations.
"""

from typing import Union, Optional
from .exceptions import (
    InvalidGeometryError,
    InvalidFlowConditionError,
    InvalidRoughnessError,
    InvalidSlopeError,
)


def validate_positive(value: Union[int, float], name: str) -> float:
    """
    Validate that a value is positive.
    
    Args:
        value: The value to validate
        name: Name of the parameter for error messages
        
    Returns:
        The validated value as a float
        
    Raises:
        InvalidGeometryError: If value is not positive
    """
    if not isinstance(value, (int, float)):
        raise InvalidGeometryError(f"{name} must be a number, got {type(value)}")
    
    if value <= 0:
        raise InvalidGeometryError(f"{name} must be positive, got {value}")
    
    return float(value)


def validate_non_negative(value: Union[int, float], name: str) -> float:
    """
    Validate that a value is non-negative.
    
    Args:
        value: The value to validate
        name: Name of the parameter for error messages
        
    Returns:
        The validated value as a float
        
    Raises:
        InvalidGeometryError: If value is negative
    """
    if not isinstance(value, (int, float)):
        raise InvalidGeometryError(f"{name} must be a number, got {type(value)}")
    
    if value < 0:
        raise InvalidGeometryError(f"{name} must be non-negative, got {value}")
    
    return float(value)


def validate_depth(depth: Union[int, float]) -> float:
    """
    Validate flow depth.
    
    Args:
        depth: Flow depth to validate
        
    Returns:
        The validated depth as a float
        
    Raises:
        InvalidFlowConditionError: If depth is not positive
    """
    if not isinstance(depth, (int, float)):
        raise InvalidFlowConditionError(f"Depth must be a number, got {type(depth)}")
    
    if depth <= 0:
        raise InvalidFlowConditionError(f"Flow depth must be positive, got {depth}")
    
    return float(depth)


def validate_discharge(discharge: Union[int, float]) -> float:
    """
    Validate discharge (flow rate).
    
    Args:
        discharge: Discharge to validate
        
    Returns:
        The validated discharge as a float
        
    Raises:
        InvalidFlowConditionError: If discharge is not positive
    """
    if not isinstance(discharge, (int, float)):
        raise InvalidFlowConditionError(f"Discharge must be a number, got {type(discharge)}")
    
    if discharge <= 0:
        raise InvalidFlowConditionError(f"Discharge must be positive, got {discharge}")
    
    return float(discharge)


def validate_manning_n(n: Union[int, float]) -> float:
    """
    Validate Manning's roughness coefficient.
    
    Args:
        n: Manning's roughness coefficient
        
    Returns:
        The validated coefficient as a float
        
    Raises:
        InvalidRoughnessError: If coefficient is not in valid range
    """
    if not isinstance(n, (int, float)):
        raise InvalidRoughnessError(f"Manning's n must be a number, got {type(n)}")
    
    if n <= 0:
        raise InvalidRoughnessError(f"Manning's n must be positive, got {n}")
    
    if n > 0.2:  # Practical upper limit
        raise InvalidRoughnessError(f"Manning's n seems too large (>{0.2}), got {n}")
    
    return float(n)


def validate_slope(slope: Union[int, float], allow_zero: bool = False) -> float:
    """
    Validate channel slope.
    
    Args:
        slope: Channel slope (dimensionless)
        allow_zero: Whether to allow zero slope
        
    Returns:
        The validated slope as a float
        
    Raises:
        InvalidSlopeError: If slope is invalid
    """
    if not isinstance(slope, (int, float)):
        raise InvalidSlopeError(f"Slope must be a number, got {type(slope)}")
    
    if slope < 0:
        raise InvalidSlopeError(f"Slope cannot be negative, got {slope}")
    
    if not allow_zero and slope == 0:
        raise InvalidSlopeError("Slope cannot be zero for this calculation")
    
    if slope > 1:  # Practical upper limit (45 degrees)
        raise InvalidSlopeError(f"Slope seems too steep (>1), got {slope}")
    
    return float(slope)


def validate_side_slope(side_slope: Union[int, float]) -> float:
    """
    Validate side slope (horizontal:vertical ratio).
    
    Args:
        side_slope: Side slope ratio (horizontal:vertical)
        
    Returns:
        The validated side slope as a float
        
    Raises:
        InvalidGeometryError: If side slope is negative
    """
    if not isinstance(side_slope, (int, float)):
        raise InvalidGeometryError(f"Side slope must be a number, got {type(side_slope)}")
    
    if side_slope < 0:
        raise InvalidGeometryError(f"Side slope cannot be negative, got {side_slope}")
    
    return float(side_slope)


def validate_angle(angle: Union[int, float], min_angle: float = 0, max_angle: float = 90) -> float:
    """
    Validate angle in degrees.
    
    Args:
        angle: Angle in degrees
        min_angle: Minimum allowed angle
        max_angle: Maximum allowed angle
        
    Returns:
        The validated angle as a float
        
    Raises:
        InvalidGeometryError: If angle is outside valid range
    """
    if not isinstance(angle, (int, float)):
        raise InvalidGeometryError(f"Angle must be a number, got {type(angle)}")
    
    if angle < min_angle or angle > max_angle:
        raise InvalidGeometryError(
            f"Angle must be between {min_angle}° and {max_angle}°, got {angle}°"
        )
    
    return float(angle)
