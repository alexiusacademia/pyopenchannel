"""
File: test_hydraulics.py
Author: Alexius Academia
Date: 2025-08-17

Tests for the hydraulics module.
"""

# import imp
import pytest
import math
from pyopenchannel.hydraulics import (
    ManningEquation,
    ChezyEquation,
    HydraulicRadius,
    CriticalDepth,
    NormalDepth,
)
from pyopenchannel.geometry import RectangularChannel, TrapezoidalChannel
from pyopenchannel.exceptions import (
    ConvergenceError,
    InvalidFlowConditionError,
    InvalidSlopeError,
    InvalidRoughnessError,
)
from pyopenchannel.constants import GRAVITY
import pyopenchannel as poc


class TestManningEquation:
    """Test Manning's equation calculations."""
    
    def test_discharge_calculation(self):
        """Test discharge calculation using Manning's equation."""
        poc.set_unit_system(poc.UnitSystem.SI)
        depth = 0.989
        base = 1.0
        area = base * depth  # m²
        perimeter = base + 2 * depth  # m
        hydraulic_radius = area / perimeter  # m
        slope = 0.001  # dimensionless
        manning_n = 0.015
        
        discharge = round(poc.ManningEquation.discharge(area, hydraulic_radius, slope, manning_n), 3)
        # discharge = ManningEquation.discharge(area, hydraulic_radius, slope, manning_n)

        # Q = (1/n) * A * R^(2/3) * S^(1/2)
        expected = 1.0
        assert abs(discharge - expected) < 1e-10
    
    def test_velocity_calculation(self):
        """Test velocity calculation using Manning's equation."""
        hydraulic_radius = 1.5  # m
        slope = 0.002  # dimensionless
        manning_n = 0.030
        
        velocity = ManningEquation.velocity(hydraulic_radius, slope, manning_n)
        
        # V = (1/n) * R^(2/3) * S^(1/2)
        expected = (1/0.030) * (1.5**(2/3)) * (0.002**0.5)
        assert abs(velocity - expected) < 1e-10
    
    def test_required_slope(self):
        """Test required slope calculation."""
        discharge = 5.0  # m³/s
        area = 8.0  # m²
        hydraulic_radius = 1.2  # m
        manning_n = 0.025
        
        slope = ManningEquation.required_slope(discharge, area, hydraulic_radius, manning_n)
        
        # S = (Q * n / (A * R^(2/3)))²
        term = (discharge * manning_n) / (area * (hydraulic_radius**(2/3)))
        expected = term**2
        assert abs(slope - expected) < 1e-10
    
    def test_invalid_parameters(self):
        """Test invalid parameters raise exceptions."""
        with pytest.raises(Exception):  # Should raise validation error
            ManningEquation.discharge(-1.0, 2.0, 0.001, 0.025)
        
        with pytest.raises(Exception):  # Should raise validation error
            ManningEquation.discharge(10.0, -1.0, 0.001, 0.025)
        
        with pytest.raises(Exception):  # Should raise validation error
            ManningEquation.discharge(10.0, 2.0, -0.001, 0.025)
        
        with pytest.raises(Exception):  # Should raise validation error
            ManningEquation.discharge(10.0, 2.0, 0.001, -0.025)


class TestChezyEquation:
    """Test Chezy equation calculations."""
    
    def test_discharge_calculation(self):
        """Test discharge calculation using Chezy equation."""
        area = 12.0  # m²
        hydraulic_radius = 2.5  # m
        slope = 0.0015  # dimensionless
        chezy_c = 50.0
        
        discharge = ChezyEquation.discharge(area, hydraulic_radius, slope, chezy_c)
        
        # Q = C * A * sqrt(R * S)
        expected = chezy_c * area * math.sqrt(hydraulic_radius * slope)
        assert abs(discharge - expected) < 1e-10
    
    def test_chezy_from_manning(self):
        """Test conversion from Manning's n to Chezy coefficient."""
        manning_n = 0.025
        hydraulic_radius = 2.0  # m
        
        chezy_c = ChezyEquation.chezy_from_manning(manning_n, hydraulic_radius)
        
        # C = R^(1/6) / n
        expected = (hydraulic_radius**(1/6)) / manning_n
        assert abs(chezy_c - expected) < 1e-10
    
    def test_consistency_with_manning(self):
        """Test that Chezy and Manning equations give consistent results."""
        area = 10.0
        hydraulic_radius = 1.8
        slope = 0.002
        manning_n = 0.030
        
        # Calculate discharge using Manning's equation
        q_manning = ManningEquation.discharge(area, hydraulic_radius, slope, manning_n)
        
        # Calculate Chezy coefficient and discharge using Chezy equation
        chezy_c = ChezyEquation.chezy_from_manning(manning_n, hydraulic_radius)
        q_chezy = ChezyEquation.discharge(area, hydraulic_radius, slope, chezy_c)
        
        # Results should be very close
        assert abs(q_manning - q_chezy) < 1e-10


class TestHydraulicRadius:
    """Test hydraulic radius calculations."""
    
    def test_from_geometry(self):
        """Test hydraulic radius calculation from geometry."""
        channel = RectangularChannel(width=4.0)
        depth = 2.0
        
        hydraulic_radius = HydraulicRadius.from_geometry(channel, depth)
        
        # For rectangular: R = A/P = (b*y)/(b+2y) = (4*2)/(4+2*2) = 8/8 = 1.0
        expected = 1.0
        assert abs(hydraulic_radius - expected) < 1e-10
    
    def test_optimal_rectangular(self):
        """Test optimal hydraulic radius for rectangular channel."""
        width = 8.0
        
        optimal_r = HydraulicRadius.optimal_rectangular(width)
        
        # For optimal rectangular: R = width/4
        expected = width / 4
        assert abs(optimal_r - expected) < 1e-10
    
    def test_optimal_trapezoidal(self):
        """Test optimal hydraulic radius for trapezoidal channel."""
        bottom_width = 4.0
        side_slope = 1.0
        
        optimal_r = HydraulicRadius.optimal_trapezoidal(bottom_width, side_slope)
        
        # This is a more complex calculation - just verify it's positive and reasonable
        assert optimal_r > 0
        assert optimal_r < bottom_width  # Should be less than bottom width


class TestCriticalDepth:
    """Test critical depth calculations."""
    
    def test_rectangular_critical_depth(self):
        """Test critical depth calculation for rectangular channel."""
        channel = RectangularChannel(width=3.0)
        discharge = 6.0  # m³/s
        
        critical_depth = CriticalDepth.calculate(channel, discharge)
        
        # For rectangular: yc = (Q²/(g*b²))^(1/3)
        expected = (discharge**2 / (GRAVITY * channel.width**2))**(1/3)
        assert abs(critical_depth - expected) < 1e-6
    
    def test_trapezoidal_critical_depth(self):
        """Test critical depth calculation for trapezoidal channel."""
        channel = TrapezoidalChannel(bottom_width=2.0, side_slope=1.5)
        discharge = 8.0  # m³/s
        
        critical_depth = CriticalDepth.calculate(channel, discharge)
        
        # Should be positive and reasonable
        assert critical_depth > 0
        assert critical_depth < 10.0  # Reasonable upper bound
        
        # Verify critical flow condition: Q² = g * A³ / T
        area = channel.area(critical_depth)
        top_width = channel.top_width(critical_depth)
        
        lhs = discharge**2
        rhs = GRAVITY * (area**3) / top_width
        
        # Should satisfy critical flow condition within tolerance
        assert abs(lhs - rhs) / lhs < 1e-3
    
    def test_froude_number_calculation(self):
        """Test Froude number calculation."""
        velocity = 2.0  # m/s
        hydraulic_depth = 1.5  # m
        
        froude_number = CriticalDepth.froude_number(velocity, hydraulic_depth)
        
        # Fr = V / sqrt(g * D)
        expected = velocity / math.sqrt(GRAVITY * hydraulic_depth)
        assert abs(froude_number - expected) < 1e-10
    
    def test_critical_froude_number(self):
        """Test that critical depth gives Froude number = 1."""
        channel = RectangularChannel(width=4.0)
        discharge = 10.0
        
        critical_depth = CriticalDepth.calculate(channel, discharge)
        
        # Calculate Froude number at critical depth
        area = channel.area(critical_depth)
        velocity = discharge / area
        hydraulic_depth = area / channel.top_width(critical_depth)
        froude_number = CriticalDepth.froude_number(velocity, hydraulic_depth)
        
        # Should be very close to 1.0
        assert abs(froude_number - 1.0) < 1e-6
    
    def test_zero_discharge(self):
        """Test zero discharge raises exception."""
        channel = RectangularChannel(width=3.0)
        
        with pytest.raises(Exception):  # Should raise validation error
            CriticalDepth.calculate(channel, discharge=0.0)
    
    def test_negative_discharge(self):
        """Test negative discharge raises exception."""
        channel = RectangularChannel(width=3.0)
        
        with pytest.raises(Exception):  # Should raise validation error
            CriticalDepth.calculate(channel, discharge=-5.0)


class TestNormalDepth:
    """Test normal depth calculations."""
    
    def test_rectangular_normal_depth(self):
        """Test normal depth calculation for rectangular channel."""
        channel = RectangularChannel(width=4.0)
        discharge = 8.0  # m³/s
        slope = 0.001  # dimensionless
        manning_n = 0.025
        
        normal_depth = NormalDepth.calculate(channel, discharge, slope, manning_n)
        
        # Verify the result satisfies Manning's equation
        area = channel.area(normal_depth)
        hydraulic_radius = channel.hydraulic_radius(normal_depth)
        calculated_q = ManningEquation.discharge(area, hydraulic_radius, slope, manning_n)
        
        # Should match the input discharge within tolerance
        assert abs(calculated_q - discharge) / discharge < 1e-6
    
    def test_trapezoidal_normal_depth(self):
        """Test normal depth calculation for trapezoidal channel."""
        channel = TrapezoidalChannel(bottom_width=3.0, side_slope=1.5)
        discharge = 12.0  # m³/s
        slope = 0.002  # dimensionless
        manning_n = 0.030
        
        normal_depth = NormalDepth.calculate(channel, discharge, slope, manning_n)
        
        # Should be positive and reasonable
        assert normal_depth > 0
        assert normal_depth < 10.0
        
        # Verify the result satisfies Manning's equation
        area = channel.area(normal_depth)
        hydraulic_radius = channel.hydraulic_radius(normal_depth)
        calculated_q = ManningEquation.discharge(area, hydraulic_radius, slope, manning_n)
        
        # Should match the input discharge within tolerance
        assert abs(calculated_q - discharge) / discharge < 1e-6
    
    def test_steep_slope_normal_depth(self):
        """Test normal depth calculation for steep slope."""
        channel = RectangularChannel(width=2.0)
        discharge = 5.0
        slope = 0.01  # Steep slope (1%)
        manning_n = 0.015
        
        normal_depth = NormalDepth.calculate(channel, discharge, slope, manning_n)
        
        # Should be positive
        assert normal_depth > 0
        
        # For steep slopes, normal depth should be relatively small
        assert normal_depth < 2.0
    
    def test_mild_slope_normal_depth(self):
        """Test normal depth calculation for mild slope."""
        channel = RectangularChannel(width=2.0)
        discharge = 5.0
        slope = 0.0001  # Mild slope (0.01%)
        manning_n = 0.015
        
        normal_depth = NormalDepth.calculate(channel, discharge, slope, manning_n)
        
        # Should be positive
        assert normal_depth > 0
        
        # For mild slopes, normal depth should be relatively large
        assert normal_depth > 1.0
    
    def test_zero_slope(self):
        """Test zero slope raises exception."""
        channel = RectangularChannel(width=3.0)
        
        with pytest.raises(Exception):  # Should raise validation error
            NormalDepth.calculate(channel, discharge=5.0, slope=0.0, manning_n=0.025)
    
    def test_negative_slope(self):
        """Test negative slope raises exception."""
        channel = RectangularChannel(width=3.0)
        
        with pytest.raises(Exception):  # Should raise validation error
            NormalDepth.calculate(channel, discharge=5.0, slope=-0.001, manning_n=0.025)
    
    def test_convergence_failure(self):
        """Test convergence failure handling."""
        # Create a scenario that might not converge easily
        channel = RectangularChannel(width=0.01)  # Very narrow channel
        discharge = 1000.0  # Very high discharge
        slope = 0.000001  # Very mild slope
        manning_n = 0.1  # Very rough surface
        
        # This might raise a convergence error or give a very large depth
        try:
            normal_depth = NormalDepth.calculate(
                channel, discharge, slope, manning_n, max_iterations=10
            )
            # If it converges, depth should be positive
            assert normal_depth > 0
        except ConvergenceError:
            # This is acceptable for extreme cases
            pass


class TestHydraulicsEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_very_small_discharge(self):
        """Test calculations with very small discharge."""
        channel = RectangularChannel(width=1.0)
        discharge = 1e-6  # Very small discharge
        slope = 0.001
        manning_n = 0.025
        
        normal_depth = NormalDepth.calculate(channel, discharge, slope, manning_n)
        critical_depth = CriticalDepth.calculate(channel, discharge)
        
        # Both should be positive but very small
        assert normal_depth > 0
        assert critical_depth > 0
        assert normal_depth < 0.01
        assert critical_depth < 0.01
    
    def test_very_large_discharge(self):
        """Test calculations with very large discharge."""
        channel = RectangularChannel(width=10.0)
        discharge = 1000.0  # Very large discharge
        slope = 0.001
        manning_n = 0.025
        
        normal_depth = NormalDepth.calculate(channel, discharge, slope, manning_n)
        critical_depth = CriticalDepth.calculate(channel, discharge)
        
        # Both should be positive and reasonably large
        assert normal_depth > 0
        assert critical_depth > 0
        assert normal_depth > 1.0
        assert critical_depth > 1.0
    
    def test_extreme_roughness(self):
        """Test calculations with extreme Manning's n values."""
        channel = RectangularChannel(width=3.0)
        discharge = 5.0
        slope = 0.001
        
        # Very smooth surface
        manning_n_smooth = 0.008
        normal_depth_smooth = NormalDepth.calculate(
            channel, discharge, slope, manning_n_smooth
        )
        
        # Very rough surface
        manning_n_rough = 0.1
        normal_depth_rough = NormalDepth.calculate(
            channel, discharge, slope, manning_n_rough
        )
        
        # Rough channel should require greater depth
        assert normal_depth_rough > normal_depth_smooth
        assert normal_depth_smooth > 0
        assert normal_depth_rough > 0
    
    def test_parameter_validation(self):
        """Test parameter validation in hydraulic calculations."""
        channel = RectangularChannel(width=3.0)
        
        # Test various invalid parameter combinations
        with pytest.raises(Exception):
            ManningEquation.discharge(area=0, hydraulic_radius=1, slope=0.001, manning_n=0.025)
        
        with pytest.raises(Exception):
            ManningEquation.discharge(area=10, hydraulic_radius=0, slope=0.001, manning_n=0.025)
        
        with pytest.raises(Exception):
            CriticalDepth.froude_number(velocity=2.0, hydraulic_depth=0)
        
        with pytest.raises(Exception):
            CriticalDepth.froude_number(velocity=-2.0, hydraulic_depth=1.0)
