"""
units.py
Author: Alexius Academia
Date: 2025-08-17

Unit system and conversion utilities for PyOpenChannel.

This module provides:
- Unit system management (SI/Metric vs US Customary)
- Automatic unit conversions
- Unit-aware calculations
- Input/output unit handling

Supported unit systems:
- SI (Metric): meters, m³/s, m/s, etc.
- US Customary: feet, ft³/s (cfs), ft/s, etc.
"""

from typing import Union
from enum import Enum
import math

from .constants import CONVERSION_FACTORS, GRAVITY as GRAVITY_SI


class UnitSystem(Enum):
    """Enumeration of supported unit systems."""
    SI = "SI"
    US_CUSTOMARY = "US_CUSTOMARY"


class Units:
    """
    Unit system manager for PyOpenChannel calculations.
    
    This class handles unit conversions and maintains consistency
    throughout calculations while allowing users to work in their
    preferred unit system.
    """
    
    def __init__(self, system: UnitSystem = UnitSystem.SI):
        """
        Initialize unit system.
        
        Args:
            system: Unit system to use (SI or US_CUSTOMARY)
        """
        self.system = system
        self._setup_units()
    
    def _setup_units(self):
        """Set up unit definitions and conversion factors."""
        if self.system == UnitSystem.SI:
            self.length_unit = "m"
            self.area_unit = "m²"
            self.volume_unit = "m³"
            self.discharge_unit = "m³/s"
            self.velocity_unit = "m/s"
            self.gravity = GRAVITY_SI  # 9.81 m/s²
            
        elif self.system == UnitSystem.US_CUSTOMARY:
            self.length_unit = "ft"
            self.area_unit = "ft²"
            self.volume_unit = "ft³"
            self.discharge_unit = "ft³/s"
            self.velocity_unit = "ft/s"
            self.gravity = GRAVITY_SI * CONVERSION_FACTORS["m_to_ft"]  # 32.17 ft/s²
    
    def convert_length_to_si(self, value: float) -> float:
        """Convert length from current unit system to SI (meters)."""
        if self.system == UnitSystem.SI:
            return value
        elif self.system == UnitSystem.US_CUSTOMARY:
            return value * CONVERSION_FACTORS["ft_to_m"]
    
    def convert_length_from_si(self, value: float) -> float:
        """Convert length from SI (meters) to current unit system."""
        if self.system == UnitSystem.SI:
            return value
        elif self.system == UnitSystem.US_CUSTOMARY:
            return value * CONVERSION_FACTORS["m_to_ft"]
    
    def convert_area_to_si(self, value: float) -> float:
        """Convert area from current unit system to SI (m²)."""
        if self.system == UnitSystem.SI:
            return value
        elif self.system == UnitSystem.US_CUSTOMARY:
            return value * CONVERSION_FACTORS["ft2_to_m2"]
    
    def convert_area_from_si(self, value: float) -> float:
        """Convert area from SI (m²) to current unit system."""
        if self.system == UnitSystem.SI:
            return value
        elif self.system == UnitSystem.US_CUSTOMARY:
            return value * CONVERSION_FACTORS["m2_to_ft2"]
    
    def convert_discharge_to_si(self, value: float) -> float:
        """Convert discharge from current unit system to SI (m³/s)."""
        if self.system == UnitSystem.SI:
            return value
        elif self.system == UnitSystem.US_CUSTOMARY:
            return value * CONVERSION_FACTORS["cfs_to_cms"]
    
    def convert_discharge_from_si(self, value: float) -> float:
        """Convert discharge from SI (m³/s) to current unit system."""
        if self.system == UnitSystem.SI:
            return value
        elif self.system == UnitSystem.US_CUSTOMARY:
            return value * CONVERSION_FACTORS["cms_to_cfs"]
    
    def convert_velocity_to_si(self, value: float) -> float:
        """Convert velocity from current unit system to SI (m/s)."""
        if self.system == UnitSystem.SI:
            return value
        elif self.system == UnitSystem.US_CUSTOMARY:
            return value * CONVERSION_FACTORS["ft_to_m"]  # ft/s to m/s
    
    def convert_velocity_from_si(self, value: float) -> float:
        """Convert velocity from SI (m/s) to current unit system."""
        if self.system == UnitSystem.SI:
            return value
        elif self.system == UnitSystem.US_CUSTOMARY:
            return value * CONVERSION_FACTORS["m_to_ft"]  # m/s to ft/s
    
    def get_manning_n_factor(self) -> float:
        """
        Get Manning's equation unit conversion factor.
        
        Manning's equation in SI: Q = (1/n) * A * R^(2/3) * S^(1/2)
        Manning's equation in US: Q = (1.49/n) * A * R^(2/3) * S^(1/2)
        
        The factor 1.49 comes from unit conversion:
        1 m^(1/3)/s = 3.28084^(1/3) ft^(1/3)/s ≈ 1.4859 ft^(1/3)/s
        
        Returns:
            Unit conversion factor for Manning's equation
        """
        if self.system == UnitSystem.SI:
            return 1.0
        elif self.system == UnitSystem.US_CUSTOMARY:
            # More precise Manning factor: (m_to_ft)^(1/3)
            return CONVERSION_FACTORS["m_to_ft"]**(1/3)
    
    def format_value(self, value: float, quantity_type: str, precision: int = 3) -> str:
        """
        Format a value with appropriate units and precision.
        
        Args:
            value: Numerical value
            quantity_type: Type of quantity ('length', 'area', 'discharge', 'velocity')
            precision: Number of decimal places
            
        Returns:
            Formatted string with value and units
        """
        unit_map = {
            'length': self.length_unit,
            'area': self.area_unit,
            'discharge': self.discharge_unit,
            'velocity': self.velocity_unit,
            'slope': '',  # Dimensionless
            'manning_n': '',  # Dimensionless
        }
        
        unit = unit_map.get(quantity_type, '')
        if unit:
            return f"{value:.{precision}f} {unit}"
        else:
            return f"{value:.{precision}f}"
    
    def __str__(self) -> str:
        """String representation of the unit system."""
        return f"Units({self.system.value})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"Units(system={self.system.value}, "
                f"length={self.length_unit}, "
                f"discharge={self.discharge_unit})")


class UnitConverter:
    """Utility class for standalone unit conversions."""
    
    @staticmethod
    def length(value: float, from_unit: str, to_unit: str) -> float:
        """
        Convert length between different units.
        
        Args:
            value: Value to convert
            from_unit: Source unit ('m', 'ft', 'in')
            to_unit: Target unit ('m', 'ft', 'in')
            
        Returns:
            Converted value
        """
        # Convert to meters first
        if from_unit == 'm':
            meters = value
        elif from_unit == 'ft':
            meters = value * CONVERSION_FACTORS["ft_to_m"]
        elif from_unit == 'in':
            meters = value * CONVERSION_FACTORS["in_to_m"]
        else:
            raise ValueError(f"Unsupported length unit: {from_unit}")
        
        # Convert from meters to target unit
        if to_unit == 'm':
            return meters
        elif to_unit == 'ft':
            return meters * CONVERSION_FACTORS["m_to_ft"]
        elif to_unit == 'in':
            return meters * CONVERSION_FACTORS["m_to_in"]
        else:
            raise ValueError(f"Unsupported length unit: {to_unit}")
    
    @staticmethod
    def discharge(value: float, from_unit: str, to_unit: str) -> float:
        """
        Convert discharge between different units.
        
        Args:
            value: Value to convert
            from_unit: Source unit ('cms', 'cfs', 'gpm')
            to_unit: Target unit ('cms', 'cfs', 'gpm')
            
        Returns:
            Converted value
        """
        # Convert to m³/s first
        if from_unit == 'cms' or from_unit == 'm3/s':
            cms = value
        elif from_unit == 'cfs' or from_unit == 'ft3/s':
            cms = value * CONVERSION_FACTORS["cfs_to_cms"]
        elif from_unit == 'gpm':
            cms = value * CONVERSION_FACTORS["gpm_to_cms"]
        else:
            raise ValueError(f"Unsupported discharge unit: {from_unit}")
        
        # Convert from m³/s to target unit
        if to_unit == 'cms' or to_unit == 'm3/s':
            return cms
        elif to_unit == 'cfs' or to_unit == 'ft3/s':
            return cms * CONVERSION_FACTORS["cms_to_cfs"]
        elif to_unit == 'gpm':
            return cms * CONVERSION_FACTORS["cms_to_gpm"]
        else:
            raise ValueError(f"Unsupported discharge unit: {to_unit}")
    
    @staticmethod
    def area(value: float, from_unit: str, to_unit: str) -> float:
        """
        Convert area between different units.
        
        Args:
            value: Value to convert
            from_unit: Source unit ('m2', 'ft2')
            to_unit: Target unit ('m2', 'ft2')
            
        Returns:
            Converted value
        """
        # Convert to m² first
        if from_unit == 'm2' or from_unit == 'm²':
            m2 = value
        elif from_unit == 'ft2' or from_unit == 'ft²':
            m2 = value * CONVERSION_FACTORS["ft2_to_m2"]
        else:
            raise ValueError(f"Unsupported area unit: {from_unit}")
        
        # Convert from m² to target unit
        if to_unit == 'm2' or to_unit == 'm²':
            return m2
        elif to_unit == 'ft2' or to_unit == 'ft²':
            return m2 * CONVERSION_FACTORS["m2_to_ft2"]
        else:
            raise ValueError(f"Unsupported area unit: {to_unit}")


# Global unit system instance
_global_units = Units(UnitSystem.SI)


def set_unit_system(system: Union[UnitSystem, str]) -> None:
    """
    Set the global unit system for PyOpenChannel.
    
    Args:
        system: Unit system to use ('SI', 'US_CUSTOMARY', or UnitSystem enum)
    """
    global _global_units
    
    if isinstance(system, str):
        system = UnitSystem(system.upper())
    
    _global_units = Units(system)


def get_unit_system() -> Units:
    """
    Get the current global unit system.
    
    Returns:
        Current Units instance
    """
    return _global_units


def get_gravity() -> float:
    """
    Get gravitational acceleration in current unit system.
    
    Returns:
        Gravity constant (9.81 m/s² or 32.17 ft/s²)
    """
    return _global_units.gravity


def get_manning_factor() -> float:
    """
    Get Manning's equation unit factor for current system.
    
    Returns:
        Manning factor (1.0 for SI, 1.49 for US Customary)
    """
    return _global_units.get_manning_n_factor()


# Convenience functions for common conversions
def ft_to_m(feet: float) -> float:
    """Convert feet to meters."""
    return UnitConverter.length(feet, 'ft', 'm')


def m_to_ft(meters: float) -> float:
    """Convert meters to feet."""
    return UnitConverter.length(meters, 'm', 'ft')


def cfs_to_cms(cfs: float) -> float:
    """Convert cubic feet per second to cubic meters per second."""
    return UnitConverter.discharge(cfs, 'cfs', 'cms')


def cms_to_cfs(cms: float) -> float:
    """Convert cubic meters per second to cubic feet per second."""
    return UnitConverter.discharge(cms, 'cms', 'cfs')


def gpm_to_cms(gpm: float) -> float:
    """Convert gallons per minute to cubic meters per second."""
    return UnitConverter.discharge(gpm, 'gpm', 'cms')


def cms_to_gpm(cms: float) -> float:
    """Convert cubic meters per second to gallons per minute."""
    return UnitConverter.discharge(cms, 'cms', 'gpm')
