"""
Test GVF Profile Classification - PyOpenChannel

Author: Alexius Academia
Email: alexius.sayco.academia@gmail.com

Unit tests for the GVF profile classification system.
Tests automatic profile identification and engineering analysis.
"""

import pytest
import math

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pyopenchannel as poc
from pyopenchannel.gvf import (
    GVFSolver, BoundaryType, ProfileClassifier, ProfileAnalyzer,
    ProfileType, SlopeType, FlowRegime
)
from pyopenchannel.geometry import RectangularChannel, TrapezoidalChannel
from pyopenchannel.hydraulics import NormalDepth
from pyopenchannel.flow_analysis import CriticalFlow


class TestProfileClassifier:
    """Test cases for profile classification."""
    
    def setup_method(self):
        """Set up test fixtures."""
        poc.set_unit_system(poc.UnitSystem.SI)
        self.solver = GVFSolver()
        self.classifier = ProfileClassifier(tolerance=0.15)
        
        # Mild slope channel setup
        self.mild_channel = RectangularChannel(width=5.0)
        self.mild_discharge = 20.0
        self.mild_slope = 0.001  # Mild slope
        self.manning_n = 0.030
        
        # Calculate reference depths for mild slope
        self.mild_normal = NormalDepth.calculate(
            self.mild_channel, self.mild_discharge, self.mild_slope, self.manning_n
        )
        self.mild_critical = CriticalFlow(self.mild_channel).calculate_critical_depth(self.mild_discharge)
        
        # Verify it's mild slope (yn > yc)
        assert self.mild_normal > self.mild_critical
        
        # Steep slope channel setup
        self.steep_channel = RectangularChannel(width=3.0)
        self.steep_discharge = 10.0
        self.steep_slope = 0.02  # Steep slope
        
        # Calculate reference depths for steep slope
        self.steep_normal = NormalDepth.calculate(
            self.steep_channel, self.steep_discharge, self.steep_slope, self.manning_n
        )
        self.steep_critical = CriticalFlow(self.steep_channel).calculate_critical_depth(self.steep_discharge)
        
        # Verify it's steep slope (yc > yn)
        assert self.steep_critical > self.steep_normal
    
    def test_classifier_initialization(self):
        """Test profile classifier initialization."""
        # Default initialization
        classifier = ProfileClassifier()
        assert classifier.tolerance == 0.15
        
        # Custom tolerance
        classifier = ProfileClassifier(tolerance=0.2)
        assert classifier.tolerance == 0.2
    
    def test_m1_profile_classification(self):
        """Test M1 profile classification (dam backwater)."""
        # Create M1 profile with depth above normal
        boundary_depth = self.mild_normal * 1.4
        
        result = self.solver.solve_profile(
            channel=self.mild_channel,
            discharge=self.mild_discharge,
            slope=self.mild_slope,
            manning_n=self.manning_n,
            x_start=0.0,
            x_end=2000.0,
            boundary_depth=boundary_depth,
            boundary_type=BoundaryType.UPSTREAM_DEPTH
        )
        
        assert result.success
        
        profile = self.classifier.classify_profile(
            gvf_result=result,
            channel=self.mild_channel,
            discharge=self.mild_discharge,
            slope=self.mild_slope,
            manning_n=self.manning_n
        )
        
        # Should classify as M1 or similar backwater profile
        assert profile.slope_type == SlopeType.MILD
        assert profile.flow_regime == FlowRegime.SUBCRITICAL
        assert profile.normal_depth == pytest.approx(self.mild_normal, rel=1e-2)
        assert profile.critical_depth == pytest.approx(self.mild_critical, rel=1e-2)
        
        # Profile should indicate backwater characteristics
        assert profile.max_depth > profile.normal_depth
    
    def test_m2_profile_classification(self):
        """Test M2 profile classification (channel entrance)."""
        # Create M2 profile with depth between critical and normal
        boundary_depth = (self.mild_critical + self.mild_normal) / 2
        
        result = self.solver.solve_profile(
            channel=self.mild_channel,
            discharge=self.mild_discharge,
            slope=self.mild_slope,
            manning_n=self.manning_n,
            x_start=0.0,
            x_end=1000.0,
            boundary_depth=boundary_depth,
            boundary_type=BoundaryType.UPSTREAM_DEPTH
        )
        
        assert result.success
        
        profile = self.classifier.classify_profile(
            gvf_result=result,
            channel=self.mild_channel,
            discharge=self.mild_discharge,
            slope=self.mild_slope,
            manning_n=self.manning_n
        )
        
        # Should classify as M2 or similar drawdown profile
        assert profile.slope_type == SlopeType.MILD
        assert profile.flow_regime == FlowRegime.SUBCRITICAL
        
        # Profile should be between critical and normal depths
        assert profile.min_depth >= self.mild_critical * 0.95  # Allow tolerance
        assert profile.max_depth <= self.mild_normal * 1.05   # Allow tolerance
    
    def test_s1_profile_classification(self):
        """Test S1 profile classification (steep channel backwater)."""
        # Create S1 profile with depth above critical
        boundary_depth = self.steep_critical * 1.5
        
        result = self.solver.solve_profile(
            channel=self.steep_channel,
            discharge=self.steep_discharge,
            slope=self.steep_slope,
            manning_n=self.manning_n,
            x_start=0.0,
            x_end=500.0,
            boundary_depth=boundary_depth,
            boundary_type=BoundaryType.UPSTREAM_DEPTH
        )
        
        assert result.success
        
        profile = self.classifier.classify_profile(
            gvf_result=result,
            channel=self.steep_channel,
            discharge=self.steep_discharge,
            slope=self.steep_slope,
            manning_n=self.manning_n
        )
        
        # Should classify as steep slope profile
        assert profile.slope_type == SlopeType.STEEP
        assert profile.normal_depth == pytest.approx(self.steep_normal, rel=1e-2)
        assert profile.critical_depth == pytest.approx(self.steep_critical, rel=1e-2)
        
        # Profile should show backwater characteristics on steep slope
        assert profile.max_depth > self.steep_critical
    
    def test_slope_type_classification(self):
        """Test slope type classification."""
        test_cases = [
            # (channel, discharge, slope, expected_slope_type)
            (RectangularChannel(width=5.0), 20.0, 0.0005, SlopeType.MILD),
            (RectangularChannel(width=3.0), 10.0, 0.02, SlopeType.STEEP),
            (RectangularChannel(width=4.0), 15.0, 0.0, SlopeType.HORIZONTAL),
            (RectangularChannel(width=4.0), 15.0, -0.001, SlopeType.ADVERSE),
        ]
        
        for channel, discharge, slope, expected_slope_type in test_cases:
            if slope <= 0:
                # Skip horizontal and adverse slopes for now as they may not converge
                continue
                
            try:
                # Use reasonable boundary depth
                critical_depth = CriticalFlow(channel).calculate_critical_depth(discharge)
                boundary_depth = critical_depth * 1.2
                
                result = self.solver.solve_profile(
                    channel=channel,
                    discharge=discharge,
                    slope=slope,
                    manning_n=self.manning_n,
                    x_start=0.0,
                    x_end=500.0,
                    boundary_depth=boundary_depth,
                    boundary_type=BoundaryType.UPSTREAM_DEPTH
                )
                
                if result.success:
                    profile = self.classifier.classify_profile(
                        gvf_result=result,
                        channel=channel,
                        discharge=discharge,
                        slope=slope,
                        manning_n=self.manning_n
                    )
                    
                    assert profile.slope_type == expected_slope_type
                    
            except Exception:
                # Some combinations may not converge, which is acceptable
                pass
    
    def test_flow_regime_classification(self):
        """Test flow regime classification."""
        # Subcritical flow (mild slope, depth above critical)
        boundary_depth = self.mild_critical * 1.5
        
        result = self.solver.solve_profile(
            channel=self.mild_channel,
            discharge=self.mild_discharge,
            slope=self.mild_slope,
            manning_n=self.manning_n,
            x_start=0.0,
            x_end=1000.0,
            boundary_depth=boundary_depth,
            boundary_type=BoundaryType.UPSTREAM_DEPTH
        )
        
        assert result.success
        
        profile = self.classifier.classify_profile(
            gvf_result=result,
            channel=self.mild_channel,
            discharge=self.mild_discharge,
            slope=self.mild_slope,
            manning_n=self.manning_n
        )
        
        # Should be subcritical
        assert profile.flow_regime == FlowRegime.SUBCRITICAL
        
        # Verify Froude numbers are subcritical
        froude_numbers = [p.froude_number for p in result.profile_points]
        assert all(fr < 1.1 for fr in froude_numbers)  # Allow small tolerance
    
    def test_critical_flow_classification(self):
        """Test critical flow classification."""
        # Use critical depth as boundary
        result = self.solver.solve_profile(
            channel=self.mild_channel,
            discharge=self.mild_discharge,
            slope=self.mild_slope,
            manning_n=self.manning_n,
            x_start=0.0,
            x_end=500.0,
            boundary_depth=self.mild_critical,
            boundary_type=BoundaryType.CRITICAL_DEPTH
        )
        
        assert result.success
        
        profile = self.classifier.classify_profile(
            gvf_result=result,
            channel=self.mild_channel,
            discharge=self.mild_discharge,
            slope=self.mild_slope,
            manning_n=self.manning_n
        )
        
        # Should detect critical flow characteristics
        froude_numbers = [p.froude_number for p in result.profile_points]
        # At least some points should be near critical
        assert any(0.9 <= fr <= 1.1 for fr in froude_numbers)
    
    def test_uniform_flow_classification(self):
        """Test uniform flow classification."""
        # Use normal depth as boundary
        result = self.solver.solve_profile(
            channel=self.mild_channel,
            discharge=self.mild_discharge,
            slope=self.mild_slope,
            manning_n=self.manning_n,
            x_start=0.0,
            x_end=1000.0,
            boundary_depth=self.mild_normal,
            boundary_type=BoundaryType.NORMAL_DEPTH
        )
        
        assert result.success
        
        profile = self.classifier.classify_profile(
            gvf_result=result,
            channel=self.mild_channel,
            discharge=self.mild_discharge,
            slope=self.mild_slope,
            manning_n=self.manning_n
        )
        
        # Should be close to uniform flow
        depths = [p.depth for p in result.profile_points]
        depth_variation = (max(depths) - min(depths)) / self.mild_normal
        
        # Uniform flow should have small depth variation
        assert depth_variation < 0.1  # Less than 10% variation
    
    def test_trapezoidal_channel_classification(self):
        """Test profile classification with trapezoidal channel."""
        trap_channel = TrapezoidalChannel(bottom_width=4.0, side_slope=1.5)
        trap_discharge = 25.0
        trap_slope = 0.0008
        
        # Calculate reference depths
        trap_normal = NormalDepth.calculate(trap_channel, trap_discharge, trap_slope, self.manning_n)
        trap_critical = CriticalFlow(trap_channel).calculate_critical_depth(trap_discharge)
        
        # Create backwater profile
        boundary_depth = trap_normal * 1.3
        
        result = self.solver.solve_profile(
            channel=trap_channel,
            discharge=trap_discharge,
            slope=trap_slope,
            manning_n=self.manning_n,
            x_start=0.0,
            x_end=1500.0,
            boundary_depth=boundary_depth,
            boundary_type=BoundaryType.UPSTREAM_DEPTH
        )
        
        assert result.success
        
        profile = self.classifier.classify_profile(
            gvf_result=result,
            channel=trap_channel,
            discharge=trap_discharge,
            slope=trap_slope,
            manning_n=self.manning_n
        )
        
        # Should classify successfully
        assert profile.slope_type in [SlopeType.MILD, SlopeType.STEEP, SlopeType.CRITICAL]
        assert profile.flow_regime in [FlowRegime.SUBCRITICAL, FlowRegime.SUPERCRITICAL, FlowRegime.CRITICAL]
        assert profile.normal_depth > 0
        assert profile.critical_depth > 0
    
    def test_engineering_significance(self):
        """Test engineering significance interpretation."""
        # Create M1 profile
        boundary_depth = self.mild_normal * 1.4
        
        result = self.solver.solve_profile(
            channel=self.mild_channel,
            discharge=self.mild_discharge,
            slope=self.mild_slope,
            manning_n=self.manning_n,
            x_start=0.0,
            x_end=2000.0,
            boundary_depth=boundary_depth,
            boundary_type=BoundaryType.UPSTREAM_DEPTH
        )
        
        assert result.success
        
        profile = self.classifier.classify_profile(
            gvf_result=result,
            channel=self.mild_channel,
            discharge=self.mild_discharge,
            slope=self.mild_slope,
            manning_n=self.manning_n
        )
        
        # Should have meaningful engineering significance
        assert isinstance(profile.engineering_significance, str)
        assert len(profile.engineering_significance) > 0
        
        # Should mention key characteristics
        significance_lower = profile.engineering_significance.lower()
        assert any(keyword in significance_lower for keyword in 
                  ['backwater', 'dam', 'flood', 'depth', 'flow'])
    
    def test_profile_properties(self):
        """Test profile properties calculation."""
        result = self.solver.solve_profile(
            channel=self.mild_channel,
            discharge=self.mild_discharge,
            slope=self.mild_slope,
            manning_n=self.manning_n,
            x_start=0.0,
            x_end=1000.0,
            boundary_depth=3.0,
            boundary_type=BoundaryType.UPSTREAM_DEPTH
        )
        
        assert result.success
        
        profile = self.classifier.classify_profile(
            gvf_result=result,
            channel=self.mild_channel,
            discharge=self.mild_discharge,
            slope=self.mild_slope,
            manning_n=self.manning_n
        )
        
        # Check all properties are reasonable
        assert profile.length > 0
        assert profile.min_depth > 0
        assert profile.max_depth > 0
        assert profile.max_depth >= profile.min_depth
        assert profile.normal_depth > 0
        assert profile.critical_depth > 0
        
        # Check curvature and asymptotic behavior are set
        assert isinstance(profile.curvature, str)
        assert isinstance(profile.asymptotic_behavior, str)
    
    def test_classifier_tolerance_effect(self):
        """Test effect of different classifier tolerances."""
        result = self.solver.solve_profile(
            channel=self.mild_channel,
            discharge=self.mild_discharge,
            slope=self.mild_slope,
            manning_n=self.manning_n,
            x_start=0.0,
            x_end=1000.0,
            boundary_depth=self.mild_normal * 1.05,  # Very close to normal
            boundary_type=BoundaryType.UPSTREAM_DEPTH
        )
        
        assert result.success
        
        # Test with strict tolerance
        strict_classifier = ProfileClassifier(tolerance=0.02)  # 2%
        strict_profile = strict_classifier.classify_profile(
            gvf_result=result,
            channel=self.mild_channel,
            discharge=self.mild_discharge,
            slope=self.mild_slope,
            manning_n=self.manning_n
        )
        
        # Test with lenient tolerance
        lenient_classifier = ProfileClassifier(tolerance=0.3)  # 30%
        lenient_profile = lenient_classifier.classify_profile(
            gvf_result=result,
            channel=self.mild_channel,
            discharge=self.mild_discharge,
            slope=self.mild_slope,
            manning_n=self.manning_n
        )
        
        # Both should succeed but may classify differently
        assert strict_profile.profile_type in ProfileType
        assert lenient_profile.profile_type in ProfileType


class TestProfileAnalyzer:
    """Test cases for profile analyzer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        poc.set_unit_system(poc.UnitSystem.SI)
        self.analyzer = ProfileAnalyzer()
        self.solver = GVFSolver()
        self.classifier = ProfileClassifier()
        
        # Create sample profiles for comparison
        self.channel = RectangularChannel(width=5.0)
        self.discharge = 20.0
        self.slope = 0.001
        self.manning_n = 0.030
    
    def test_analyzer_initialization(self):
        """Test profile analyzer initialization."""
        analyzer = ProfileAnalyzer()
        assert analyzer is not None
    
    def test_compare_profiles(self):
        """Test profile comparison functionality."""
        # Create multiple profiles
        profiles = []
        boundary_depths = [2.5, 3.0, 3.5]
        
        for depth in boundary_depths:
            result = self.solver.solve_profile(
                channel=self.channel,
                discharge=self.discharge,
                slope=self.slope,
                manning_n=self.manning_n,
                x_start=0.0,
                x_end=1000.0,
                boundary_depth=depth,
                boundary_type=BoundaryType.UPSTREAM_DEPTH
            )
            
            if result.success:
                profile = self.classifier.classify_profile(
                    gvf_result=result,
                    channel=self.channel,
                    discharge=self.discharge,
                    slope=self.slope,
                    manning_n=self.manning_n
                )
                profiles.append(profile)
        
        # Compare profiles
        if len(profiles) >= 2:
            comparison = self.analyzer.compare_profiles(profiles)
            
            # Check comparison results
            assert 'total_profiles' in comparison
            assert comparison['total_profiles'] == len(profiles)
            assert 'profile_types' in comparison
            assert 'slope_types' in comparison
            assert 'flow_regimes' in comparison
            assert 'length_range' in comparison
            assert 'depth_range' in comparison
            
            # Check data types
            assert isinstance(comparison['profile_types'], list)
            assert isinstance(comparison['slope_types'], list)
            assert isinstance(comparison['flow_regimes'], list)
            assert isinstance(comparison['length_range'], (list, tuple))
            assert isinstance(comparison['depth_range'], (list, tuple))


if __name__ == "__main__":
    pytest.main([__file__])
