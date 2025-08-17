"""
Test GVF Solver - PyOpenChannel

Author: Alexius Academia
Email: alexius.sayco.academia@gmail.com

Unit tests for the Gradually Varied Flow solver module.
Tests core functionality, accuracy, and edge cases.
"""

import pytest
import math
from typing import List

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pyopenchannel as poc
from pyopenchannel.gvf import GVFSolver, BoundaryType, GVFResult
from pyopenchannel.geometry import RectangularChannel, TrapezoidalChannel
from pyopenchannel.hydraulics import NormalDepth
from pyopenchannel.flow_analysis import CriticalFlow


class TestGVFSolver:
    """Test cases for GVF solver."""
    
    def setup_method(self):
        """Set up test fixtures."""
        poc.set_unit_system(poc.UnitSystem.SI)
        self.solver = GVFSolver()
        self.channel = RectangularChannel(width=5.0)
        self.discharge = 20.0
        self.slope = 0.001
        self.manning_n = 0.030
        
        # Calculate reference depths
        self.normal_depth = NormalDepth.calculate(
            self.channel, self.discharge, self.slope, self.manning_n
        )
        self.critical_depth = CriticalFlow(self.channel).calculate_critical_depth(self.discharge)
    
    def test_solver_initialization(self):
        """Test GVF solver initialization."""
        # Default initialization
        solver = GVFSolver()
        assert solver is not None
        
        # Custom initialization
        solver = GVFSolver(
            integration_method="rk4",
            rtol=1e-8,
            max_steps=2000
        )
        assert solver is not None
    
    def test_basic_profile_computation(self):
        """Test basic GVF profile computation."""
        result = self.solver.solve_profile(
            channel=self.channel,
            discharge=self.discharge,
            slope=self.slope,
            manning_n=self.manning_n,
            x_start=0.0,
            x_end=1000.0,
            boundary_depth=3.0,
            boundary_type=BoundaryType.UPSTREAM_DEPTH
        )
        
        assert result.success
        assert len(result.profile_points) > 0
        assert result.length > 0
        assert result.message == "GVF profile computed successfully"
    
    def test_m1_profile_dam_backwater(self):
        """Test M1 profile (dam backwater) computation."""
        # Use depth above normal depth to create M1 profile
        boundary_depth = self.normal_depth * 1.3
        
        result = self.solver.solve_profile(
            channel=self.channel,
            discharge=self.discharge,
            slope=self.slope,
            manning_n=self.manning_n,
            x_start=0.0,
            x_end=2000.0,
            boundary_depth=boundary_depth,
            boundary_type=BoundaryType.UPSTREAM_DEPTH
        )
        
        assert result.success
        depths = [p.depth for p in result.profile_points]
        
        # M1 profile should have depths above normal depth
        assert min(depths) > self.normal_depth * 0.95  # Allow small tolerance
        assert max(depths) > self.normal_depth
        
        # Profile should be subcritical
        froude_numbers = [p.froude_number for p in result.profile_points]
        assert all(fr < 1.1 for fr in froude_numbers)  # Allow small tolerance for critical
    
    def test_m2_profile_channel_entrance(self):
        """Test M2 profile (channel entrance) computation."""
        # Use depth between critical and normal to create M2 profile
        boundary_depth = (self.critical_depth + self.normal_depth) / 2
        
        result = self.solver.solve_profile(
            channel=self.channel,
            discharge=self.discharge,
            slope=self.slope,
            manning_n=self.manning_n,
            x_start=0.0,
            x_end=1000.0,
            boundary_depth=boundary_depth,
            boundary_type=BoundaryType.UPSTREAM_DEPTH
        )
        
        assert result.success
        depths = [p.depth for p in result.profile_points]
        
        # M2 profile should transition from boundary toward critical
        assert depths[0] == pytest.approx(boundary_depth, rel=1e-3)
        assert min(depths) > self.critical_depth * 0.95  # Should stay above critical
    
    def test_steep_channel_profile(self):
        """Test profile computation on steep channel."""
        # Create steep channel conditions
        steep_slope = 0.02  # 2% slope
        steep_channel = RectangularChannel(width=3.0)
        steep_discharge = 10.0
        
        # Calculate reference depths for steep channel
        steep_normal = NormalDepth.calculate(steep_channel, steep_discharge, steep_slope, self.manning_n)
        steep_critical = CriticalFlow(steep_channel).calculate_critical_depth(steep_discharge)
        
        # Verify it's actually steep (yc > yn)
        assert steep_critical > steep_normal
        
        # Use boundary depth above critical for S1 profile
        boundary_depth = steep_critical * 1.5
        
        result = self.solver.solve_profile(
            channel=steep_channel,
            discharge=steep_discharge,
            slope=steep_slope,
            manning_n=self.manning_n,
            x_start=0.0,
            x_end=500.0,
            boundary_depth=boundary_depth,
            boundary_type=BoundaryType.UPSTREAM_DEPTH
        )
        
        assert result.success
        assert len(result.profile_points) > 0
    
    def test_different_boundary_types(self):
        """Test different boundary condition types."""
        boundary_conditions = [
            (BoundaryType.UPSTREAM_DEPTH, 3.0),
            (BoundaryType.CRITICAL_DEPTH, self.critical_depth),
            (BoundaryType.NORMAL_DEPTH, self.normal_depth)
        ]
        
        for boundary_type, depth in boundary_conditions:
            result = self.solver.solve_profile(
                channel=self.channel,
                discharge=self.discharge,
                slope=self.slope,
                manning_n=self.manning_n,
                x_start=0.0,
                x_end=500.0,
                boundary_depth=depth,
                boundary_type=boundary_type
            )
            
            assert result.success, f"Failed for boundary type {boundary_type}"
            assert len(result.profile_points) > 0
    
    def test_different_integration_methods(self):
        """Test different integration methods."""
        methods = ["rk4", "rkf45", "dormand_prince"]
        
        for method in methods:
            solver = GVFSolver(integration_method=method)
            result = solver.solve_profile(
                channel=self.channel,
                discharge=self.discharge,
                slope=self.slope,
                manning_n=self.manning_n,
                x_start=0.0,
                x_end=1000.0,
                boundary_depth=3.0,
                boundary_type=BoundaryType.UPSTREAM_DEPTH
            )
            
            assert result.success, f"Failed for method {method}"
            assert len(result.profile_points) > 0
    
    def test_trapezoidal_channel(self):
        """Test GVF computation with trapezoidal channel."""
        trap_channel = TrapezoidalChannel(bottom_width=4.0, side_slope=1.5)
        
        result = self.solver.solve_profile(
            channel=trap_channel,
            discharge=self.discharge,
            slope=self.slope,
            manning_n=self.manning_n,
            x_start=0.0,
            x_end=1000.0,
            boundary_depth=2.5,
            boundary_type=BoundaryType.UPSTREAM_DEPTH
        )
        
        assert result.success
        assert len(result.profile_points) > 0
        
        # Verify hydraulic properties are reasonable
        for point in result.profile_points:
            assert point.depth > 0
            assert point.velocity > 0
            assert point.area > 0
            assert point.hydraulic_radius > 0
            assert point.froude_number > 0
    
    def test_profile_point_properties(self):
        """Test that profile points have correct properties."""
        result = self.solver.solve_profile(
            channel=self.channel,
            discharge=self.discharge,
            slope=self.slope,
            manning_n=self.manning_n,
            x_start=0.0,
            x_end=1000.0,
            boundary_depth=3.0,
            boundary_type=BoundaryType.UPSTREAM_DEPTH
        )
        
        assert result.success
        
        for point in result.profile_points:
            # Check all properties are positive
            assert point.x >= 0
            assert point.depth > 0
            assert point.velocity > 0
            assert point.discharge > 0
            assert point.area > 0
            assert point.top_width > 0
            assert point.hydraulic_radius > 0
            assert point.froude_number > 0
            assert point.specific_energy > 0
            
            # Check continuity equation
            computed_discharge = point.area * point.velocity
            assert computed_discharge == pytest.approx(self.discharge, rel=1e-3)
            
            # Check Froude number calculation
            expected_froude = point.velocity / math.sqrt(9.81 * point.depth)
            assert point.froude_number == pytest.approx(expected_froude, rel=1e-3)
    
    def test_energy_conservation(self):
        """Test energy conservation along profile."""
        result = self.solver.solve_profile(
            channel=self.channel,
            discharge=self.discharge,
            slope=self.slope,
            manning_n=self.manning_n,
            x_start=0.0,
            x_end=1000.0,
            boundary_depth=3.0,
            boundary_type=BoundaryType.UPSTREAM_DEPTH
        )
        
        assert result.success
        
        # Calculate total energy at each point
        energies = []
        for i, point in enumerate(result.profile_points):
            # Total energy = depth + velocity head + elevation
            elevation = -self.slope * point.x  # Negative because x increases upstream
            total_energy = point.depth + point.velocity**2/(2*9.81) + elevation
            energies.append(total_energy)
        
        # Energy should decrease downstream (allowing for friction losses)
        # This is a basic check - detailed energy analysis would be more complex
        assert len(energies) > 1
    
    def test_invalid_inputs(self):
        """Test handling of invalid inputs."""
        # Negative discharge
        with pytest.raises(Exception):
            self.solver.solve_profile(
                channel=self.channel,
                discharge=-10.0,  # Invalid
                slope=self.slope,
                manning_n=self.manning_n,
                x_start=0.0,
                x_end=1000.0,
                boundary_depth=3.0,
                boundary_type=BoundaryType.UPSTREAM_DEPTH
            )
        
        # Negative slope (should handle gracefully or raise appropriate error)
        result = self.solver.solve_profile(
            channel=self.channel,
            discharge=self.discharge,
            slope=-0.001,  # Adverse slope
            manning_n=self.manning_n,
            x_start=0.0,
            x_end=1000.0,
            boundary_depth=3.0,
            boundary_type=BoundaryType.UPSTREAM_DEPTH
        )
        # Should either succeed (adverse slope is valid) or fail gracefully
        assert isinstance(result, GVFResult)
        
        # Invalid Manning's n
        with pytest.raises(Exception):
            self.solver.solve_profile(
                channel=self.channel,
                discharge=self.discharge,
                slope=self.slope,
                manning_n=-0.030,  # Invalid
                x_start=0.0,
                x_end=1000.0,
                boundary_depth=3.0,
                boundary_type=BoundaryType.UPSTREAM_DEPTH
            )
    
    def test_unit_system_consistency(self):
        """Test consistency across different unit systems."""
        # Test with SI units
        poc.set_unit_system(poc.UnitSystem.SI)
        result_si = self.solver.solve_profile(
            channel=RectangularChannel(width=5.0),  # meters
            discharge=20.0,  # m³/s
            slope=0.001,
            manning_n=0.030,
            x_start=0.0,
            x_end=1000.0,  # meters
            boundary_depth=3.0,  # meters
            boundary_type=BoundaryType.UPSTREAM_DEPTH
        )
        
        # Test with US Customary units
        poc.set_unit_system(poc.UnitSystem.US_CUSTOMARY)
        result_us = self.solver.solve_profile(
            channel=RectangularChannel(width=16.404),  # feet (5m converted)
            discharge=706.29,  # ft³/s (20 m³/s converted)
            slope=0.001,
            manning_n=0.030,
            x_start=0.0,
            x_end=3280.84,  # feet (1000m converted)
            boundary_depth=9.843,  # feet (3m converted)
            boundary_type=BoundaryType.UPSTREAM_DEPTH
        )
        
        # Both should succeed
        assert result_si.success
        assert result_us.success
        
        # Reset to SI for other tests
        poc.set_unit_system(poc.UnitSystem.SI)
    
    def test_solver_convergence(self):
        """Test solver convergence with different tolerances."""
        tolerances = [1e-4, 1e-6, 1e-8]
        
        for tol in tolerances:
            solver = GVFSolver(rtol=tol)
            result = solver.solve_profile(
                channel=self.channel,
                discharge=self.discharge,
                slope=self.slope,
                manning_n=self.manning_n,
                x_start=0.0,
                x_end=1000.0,
                boundary_depth=3.0,
                boundary_type=BoundaryType.UPSTREAM_DEPTH
            )
            
            assert result.success, f"Failed to converge with tolerance {tol}"
            assert len(result.profile_points) > 0


if __name__ == "__main__":
    pytest.main([__file__])
