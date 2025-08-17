"""
Test GVF Applications - PyOpenChannel

Author: Alexius Academia
Email: alexius.sayco.academia@gmail.com

Unit tests for the GVF applications module.
Tests high-level engineering applications for professional use.
"""

import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pyopenchannel as poc
from pyopenchannel.gvf import (
    DamAnalysis, BridgeAnalysis, ChuteAnalysis, ChannelTransition,
    DesignCriteria, AnalysisType, AnalysisResult
)
from pyopenchannel.geometry import RectangularChannel, TrapezoidalChannel


class TestDamAnalysis:
    """Test cases for dam backwater analysis."""
    
    def setup_method(self):
        """Set up test fixtures."""
        poc.set_unit_system(poc.UnitSystem.SI)
        self.dam_analyzer = DamAnalysis(design_criteria=DesignCriteria.STANDARD)
        
        # River channel setup
        self.channel = TrapezoidalChannel(bottom_width=10.0, side_slope=2.0)
        self.discharge = 80.0
        self.slope = 0.0006
        self.manning_n = 0.035
        self.dam_height = 3.0
    
    def test_dam_analyzer_initialization(self):
        """Test dam analyzer initialization."""
        # Default initialization
        analyzer = DamAnalysis()
        assert analyzer.design_criteria == DesignCriteria.STANDARD
        
        # Custom initialization
        analyzer = DamAnalysis(design_criteria=DesignCriteria.CONSERVATIVE)
        assert analyzer.design_criteria == DesignCriteria.CONSERVATIVE
    
    def test_basic_dam_analysis(self):
        """Test basic dam backwater analysis."""
        result = self.dam_analyzer.analyze_backwater(
            channel=self.channel,
            discharge=self.discharge,
            slope=self.slope,
            manning_n=self.manning_n,
            dam_height=self.dam_height,
            analysis_distance=5000.0
        )
        
        assert isinstance(result, AnalysisResult)
        assert result.analysis_type == AnalysisType.BACKWATER
        
        if result.success:
            # Check design parameters
            params = result.design_parameters
            assert 'dam_height' in params
            assert 'normal_depth' in params
            assert 'critical_depth' in params
            assert 'max_backwater' in params
            assert 'backwater_extent' in params
            
            # Check values are reasonable
            assert params['dam_height'] == self.dam_height
            assert params['normal_depth'] > 0
            assert params['critical_depth'] > 0
            assert params['max_backwater'] >= 0
            assert params['backwater_extent'] >= 0
            
            # Check recommendations
            assert isinstance(result.recommendations, list)
            assert len(result.recommendations) > 0
            
            # Check compliance notes
            assert isinstance(result.compliance, list)
    
    def test_dam_analysis_with_bridges(self):
        """Test dam analysis with bridge clearance calculations."""
        bridge_locations = [1000.0, 2500.0, 5000.0]
        
        result = self.dam_analyzer.analyze_backwater(
            channel=self.channel,
            discharge=self.discharge,
            slope=self.slope,
            manning_n=self.manning_n,
            dam_height=self.dam_height,
            analysis_distance=8000.0,
            bridge_locations=bridge_locations
        )
        
        assert result.analysis_type == AnalysisType.BACKWATER
        
        if result.success:
            params = result.design_parameters
            
            # Should have bridge clearance data
            assert 'bridge_clearances' in params
            bridge_data = params['bridge_clearances']
            
            # Should have data for each bridge location
            for location in bridge_locations:
                assert location in bridge_data
                bridge_info = bridge_data[location]
                assert 'water_depth' in bridge_info
                assert 'required_elevation' in bridge_info
                assert bridge_info['water_depth'] > 0
                assert bridge_info['required_elevation'] > bridge_info['water_depth']
    
    def test_different_design_criteria(self):
        """Test dam analysis with different design criteria."""
        criteria_list = [DesignCriteria.CONSERVATIVE, DesignCriteria.STANDARD, DesignCriteria.OPTIMIZED]
        
        results = []
        for criteria in criteria_list:
            analyzer = DamAnalysis(design_criteria=criteria)
            result = analyzer.analyze_backwater(
                channel=self.channel,
                discharge=self.discharge,
                slope=self.slope,
                manning_n=self.manning_n,
                dam_height=self.dam_height,
                analysis_distance=3000.0
            )
            results.append((criteria, result))
        
        # All should succeed
        for criteria, result in results:
            assert result.success, f"Failed for criteria {criteria}"
            assert result.analysis_type == AnalysisType.BACKWATER
        
        # Conservative should have higher safety factors
        conservative_result = results[0][1]
        standard_result = results[1][1]
        
        if conservative_result.success and standard_result.success:
            conservative_params = conservative_result.design_parameters
            standard_params = standard_result.design_parameters
            
            # Conservative should have higher design dam depth
            assert conservative_params['design_dam_depth'] >= standard_params['design_dam_depth']


class TestBridgeAnalysis:
    """Test cases for bridge hydraulic analysis."""
    
    def setup_method(self):
        """Set up test fixtures."""
        poc.set_unit_system(poc.UnitSystem.SI)
        self.bridge_analyzer = BridgeAnalysis(design_criteria=DesignCriteria.STANDARD)
        
        # Bridge setup
        self.approach_channel = RectangularChannel(width=8.0)
        self.bridge_opening = RectangularChannel(width=6.0)
        self.discharge = 40.0
        self.slope = 0.001
        self.manning_n = 0.030
    
    def test_bridge_analyzer_initialization(self):
        """Test bridge analyzer initialization."""
        analyzer = BridgeAnalysis()
        assert analyzer.design_criteria == DesignCriteria.STANDARD
        
        analyzer = BridgeAnalysis(design_criteria=DesignCriteria.CONSERVATIVE)
        assert analyzer.design_criteria == DesignCriteria.CONSERVATIVE
    
    def test_basic_bridge_analysis(self):
        """Test basic bridge hydraulic analysis."""
        result = self.bridge_analyzer.analyze_bridge_hydraulics(
            approach_channel=self.approach_channel,
            bridge_opening=self.bridge_opening,
            discharge=self.discharge,
            slope=self.slope,
            manning_n=self.manning_n,
            analysis_distance=1500.0
        )
        
        assert isinstance(result, AnalysisResult)
        assert result.analysis_type == AnalysisType.TRANSITION
        
        if result.success:
            params = result.design_parameters
            
            # Check required parameters
            required_params = [
                'approach_normal_depth', 'bridge_critical_depth', 'bridge_depth',
                'max_upstream_depth', 'backwater_rise', 'required_clearance',
                'estimated_scour_depth', 'contraction_ratio'
            ]
            
            for param in required_params:
                assert param in params, f"Missing parameter: {param}"
                assert params[param] >= 0, f"Negative value for {param}"
            
            # Check logical relationships
            assert params['bridge_depth'] > 0
            assert params['required_clearance'] > params['bridge_depth']
            assert 0 < params['contraction_ratio'] <= 1.0  # Bridge is narrower than approach
            
            # Check recommendations
            assert isinstance(result.recommendations, list)
            assert len(result.recommendations) > 0
    
    def test_bridge_contraction_effects(self):
        """Test bridge contraction effects."""
        # Test different contraction ratios
        bridge_widths = [7.0, 6.0, 5.0, 4.0]  # Increasing contraction
        
        results = []
        for width in bridge_widths:
            bridge_opening = RectangularChannel(width=width)
            result = self.bridge_analyzer.analyze_bridge_hydraulics(
                approach_channel=self.approach_channel,
                bridge_opening=bridge_opening,
                discharge=self.discharge,
                slope=self.slope,
                manning_n=self.manning_n
            )
            results.append((width, result))
        
        successful_results = [(w, r) for w, r in results if r.success]
        
        if len(successful_results) >= 2:
            # More contraction should generally lead to higher backwater
            for i in range(len(successful_results) - 1):
                width1, result1 = successful_results[i]
                width2, result2 = successful_results[i + 1]
                
                if width1 > width2:  # width1 is less contracted
                    backwater1 = result1.design_parameters.get('backwater_rise', 0)
                    backwater2 = result2.design_parameters.get('backwater_rise', 0)
                    # More contraction (smaller width) should generally cause more backwater
                    # Note: This is a general trend, not always strictly true
                    assert backwater2 >= backwater1 * 0.8  # Allow some tolerance


class TestChuteAnalysis:
    """Test cases for chute energy dissipation analysis."""
    
    def setup_method(self):
        """Set up test fixtures."""
        poc.set_unit_system(poc.UnitSystem.SI)
        self.chute_analyzer = ChuteAnalysis(design_criteria=DesignCriteria.STANDARD)
        
        # Chute setup
        self.chute_channel = RectangularChannel(width=4.0)
        self.tailwater_channel = RectangularChannel(width=5.0)
        self.discharge = 20.0
        self.chute_slope = 0.05  # 5% steep slope
        self.tailwater_slope = 0.002  # 0.2% mild slope
        self.manning_n = 0.025
    
    def test_chute_analyzer_initialization(self):
        """Test chute analyzer initialization."""
        analyzer = ChuteAnalysis()
        assert analyzer.design_criteria == DesignCriteria.STANDARD
    
    def test_basic_chute_analysis(self):
        """Test basic chute energy dissipation analysis."""
        result = self.chute_analyzer.analyze_chute(
            chute_channel=self.chute_channel,
            tailwater_channel=self.tailwater_channel,
            discharge=self.discharge,
            chute_slope=self.chute_slope,
            tailwater_slope=self.tailwater_slope,
            manning_n=self.manning_n,
            chute_length=300.0
        )
        
        assert isinstance(result, AnalysisResult)
        assert result.analysis_type == AnalysisType.ENERGY_DISSIPATION
        
        if result.success:
            params = result.design_parameters
            
            # Check required parameters
            required_params = [
                'exit_depth', 'exit_velocity', 'exit_energy', 'exit_froude',
                'energy_dissipated', 'jump_required'
            ]
            
            for param in required_params:
                assert param in params, f"Missing parameter: {param}"
            
            # Check values are reasonable
            assert params['exit_depth'] > 0
            assert params['exit_velocity'] > 0
            assert params['exit_energy'] > 0
            assert params['exit_froude'] > 0
            assert params['energy_dissipated'] >= 0
            assert isinstance(params['jump_required'], bool)
            
            # If jump is required, should have jump parameters
            if params['jump_required']:
                assert 'jump_length' in params
                assert 'sequent_depth' in params
                assert params['jump_length'] > 0
                assert params['sequent_depth'] > params['exit_depth']
            
            # Check recommendations
            assert isinstance(result.recommendations, list)
            assert len(result.recommendations) > 0
    
    def test_hydraulic_jump_detection(self):
        """Test hydraulic jump detection and analysis."""
        # Use conditions that should create supercritical flow requiring jump
        steep_slope = 0.08  # 8% very steep
        
        result = self.chute_analyzer.analyze_chute(
            chute_channel=self.chute_channel,
            tailwater_channel=self.tailwater_channel,
            discharge=self.discharge,
            chute_slope=steep_slope,
            tailwater_slope=self.tailwater_slope,
            manning_n=self.manning_n,
            chute_length=200.0
        )
        
        if result.success:
            params = result.design_parameters
            
            # Should have high exit Froude number
            assert params['exit_froude'] > 1.0  # Supercritical
            
            # Should require hydraulic jump
            if params['jump_required']:
                assert params['sequent_depth'] > params['exit_depth']
                assert params['jump_length'] > 0
                
                # Jump length should be reasonable (typically 4-6 times sequent depth difference)
                depth_diff = params['sequent_depth'] - params['exit_depth']
                expected_length_range = (3 * depth_diff, 10 * depth_diff)
                assert expected_length_range[0] <= params['jump_length'] <= expected_length_range[1]


class TestChannelTransition:
    """Test cases for channel transition analysis."""
    
    def setup_method(self):
        """Set up test fixtures."""
        poc.set_unit_system(poc.UnitSystem.SI)
        self.transition_analyzer = ChannelTransition(design_criteria=DesignCriteria.STANDARD)
        
        # Transition setup
        self.upstream_channel = TrapezoidalChannel(bottom_width=6.0, side_slope=2.0)
        self.downstream_channel = RectangularChannel(width=5.0)
        self.discharge = 25.0
        self.upstream_slope = 0.0008
        self.downstream_slope = 0.0012
        self.manning_n = 0.030
    
    def test_transition_analyzer_initialization(self):
        """Test transition analyzer initialization."""
        analyzer = ChannelTransition()
        assert analyzer.design_criteria == DesignCriteria.STANDARD
    
    def test_basic_transition_analysis(self):
        """Test basic channel transition analysis."""
        result = self.transition_analyzer.analyze_transition(
            upstream_channel=self.upstream_channel,
            downstream_channel=self.downstream_channel,
            discharge=self.discharge,
            upstream_slope=self.upstream_slope,
            downstream_slope=self.downstream_slope,
            manning_n=self.manning_n,
            transition_length=80.0
        )
        
        assert isinstance(result, AnalysisResult)
        assert result.analysis_type == AnalysisType.TRANSITION
        
        if result.success:
            params = result.design_parameters
            
            # Check required parameters
            required_params = [
                'upstream_normal', 'downstream_normal', 'transition_depth',
                'upstream_energy', 'downstream_energy', 'energy_loss',
                'contraction_ratio', 'slope_change'
            ]
            
            for param in required_params:
                assert param in params, f"Missing parameter: {param}"
            
            # Check values are reasonable
            assert params['upstream_normal'] > 0
            assert params['downstream_normal'] > 0
            assert params['transition_depth'] > 0
            assert params['upstream_energy'] > 0
            assert params['downstream_energy'] > 0
            assert params['contraction_ratio'] > 0
            
            # Energy loss can be positive or negative
            assert isinstance(params['energy_loss'], (int, float))
            
            # Slope change should match input
            expected_slope_change = self.downstream_slope - self.upstream_slope
            assert params['slope_change'] == pytest.approx(expected_slope_change, rel=1e-6)
            
            # Check recommendations
            assert isinstance(result.recommendations, list)
    
    def test_contraction_vs_expansion(self):
        """Test different types of transitions."""
        # Test contraction (wide to narrow)
        wide_channel = RectangularChannel(width=8.0)
        narrow_channel = RectangularChannel(width=5.0)
        
        contraction_result = self.transition_analyzer.analyze_transition(
            upstream_channel=wide_channel,
            downstream_channel=narrow_channel,
            discharge=self.discharge,
            upstream_slope=self.upstream_slope,
            downstream_slope=self.downstream_slope,
            manning_n=self.manning_n
        )
        
        # Test expansion (narrow to wide)
        expansion_result = self.transition_analyzer.analyze_transition(
            upstream_channel=narrow_channel,
            downstream_channel=wide_channel,
            discharge=self.discharge,
            upstream_slope=self.upstream_slope,
            downstream_slope=self.downstream_slope,
            manning_n=self.manning_n
        )
        
        if contraction_result.success and expansion_result.success:
            contraction_ratio = contraction_result.design_parameters['contraction_ratio']
            expansion_ratio = expansion_result.design_parameters['contraction_ratio']
            
            # Contraction should have ratio < 1, expansion should have ratio > 1
            assert contraction_ratio < 1.0
            assert expansion_ratio > 1.0
            
            # They should be reciprocals (approximately)
            assert contraction_ratio * expansion_ratio == pytest.approx(1.0, rel=0.1)


class TestAnalysisResult:
    """Test cases for AnalysisResult data structure."""
    
    def test_analysis_result_creation(self):
        """Test AnalysisResult creation and properties."""
        result = AnalysisResult(
            analysis_type=AnalysisType.BACKWATER,
            success=True,
            message="Test analysis completed"
        )
        
        assert result.analysis_type == AnalysisType.BACKWATER
        assert result.success is True
        assert result.message == "Test analysis completed"
        assert result.gvf_result is None
        assert result.profile is None
        assert isinstance(result.design_parameters, dict)
        assert isinstance(result.recommendations, list)
        assert isinstance(result.warnings, list)
        assert isinstance(result.compliance, list)
    
    def test_analysis_result_with_data(self):
        """Test AnalysisResult with actual data."""
        design_params = {
            'param1': 1.0,
            'param2': 2.0
        }
        recommendations = ['Recommendation 1', 'Recommendation 2']
        warnings = ['Warning 1']
        compliance = ['Compliance note 1']
        
        result = AnalysisResult(
            analysis_type=AnalysisType.TRANSITION,
            success=True,
            message="Analysis with data",
            design_parameters=design_params,
            recommendations=recommendations,
            warnings=warnings,
            compliance=compliance
        )
        
        assert result.design_parameters == design_params
        assert result.recommendations == recommendations
        assert result.warnings == warnings
        assert result.compliance == compliance


if __name__ == "__main__":
    pytest.main([__file__])
