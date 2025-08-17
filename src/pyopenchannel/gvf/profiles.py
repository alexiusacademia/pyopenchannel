"""
File: gvf/profiles.py
Author: Alexius Academia
Date: 2025-08-17

Water surface profile classification and analysis.

This module provides comprehensive classification of gradually varied flow profiles:

Profile Types:
- M profiles (Mild slope): M1, M2, M3
- S profiles (Steep slope): S1, S2, S3  
- C profiles (Critical slope): C1, C3
- H profiles (Horizontal): H2, H3
- A profiles (Adverse slope): A2, A3

Classification is based on:
- Channel slope vs critical slope
- Flow depth vs normal depth
- Flow depth vs critical depth
- Flow regime (subcritical/supercritical)

Applications:
- Automatic profile identification
- Engineering documentation
- Design verification
- Educational analysis
"""

import math
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from .solver import ProfilePoint, GVFResult, FlowRegime
from ..geometry import ChannelGeometry
from ..hydraulics import CriticalDepth, NormalDepth
from ..exceptions import InvalidFlowConditionError


class SlopeType(Enum):
    """Channel slope classification."""
    MILD = "mild"           # S₀ < Sc (normal depth > critical depth)
    STEEP = "steep"         # S₀ > Sc (normal depth < critical depth)
    CRITICAL = "critical"   # S₀ = Sc (normal depth = critical depth)
    HORIZONTAL = "horizontal"  # S₀ = 0
    ADVERSE = "adverse"     # S₀ < 0


class ProfileType(Enum):
    """Water surface profile types."""
    # Mild slope profiles
    M1 = "M1"  # Backwater curve (y > yn > yc)
    M2 = "M2"  # Drawdown curve (yn > y > yc)
    M3 = "M3"  # Backwater curve (yn > yc > y)
    
    # Steep slope profiles
    S1 = "S1"  # Backwater curve (y > yc > yn)
    S2 = "S2"  # Drawdown curve (yc > y > yn)
    S3 = "S3"  # Backwater curve (yc > yn > y)
    
    # Critical slope profiles
    C1 = "C1"  # Backwater curve (y > yc = yn)
    C3 = "C3"  # Backwater curve (yc = yn > y)
    
    # Horizontal channel profiles
    H2 = "H2"  # Drawdown curve (y > yc, yn = ∞)
    H3 = "H3"  # Backwater curve (yc > y, yn = ∞)
    
    # Adverse slope profiles
    A2 = "A2"  # Drawdown curve (y > yc, yn imaginary)
    A3 = "A3"  # Backwater curve (yc > y, yn imaginary)
    
    # Special cases
    UNIFORM = "UNIFORM"     # Uniform flow (y = yn)
    CRITICAL_FLOW = "CRITICAL"  # Critical flow (y = yc)
    MIXED = "MIXED"         # Mixed profile types
    UNKNOWN = "UNKNOWN"     # Cannot classify


@dataclass
class ProfileCharacteristics:
    """
    Characteristics of a water surface profile.
    
    Attributes:
        profile_type: Classified profile type
        slope_type: Channel slope classification
        flow_regime: Dominant flow regime
        normal_depth: Normal depth (if exists)
        critical_depth: Critical depth
        depth_range: (min_depth, max_depth) in profile
        length: Profile length
        curvature: Profile curvature (concave up/down)
        asymptotic_behavior: Asymptotic approach to normal/critical depth
        engineering_significance: Engineering interpretation
    """
    profile_type: ProfileType
    slope_type: SlopeType
    flow_regime: FlowRegime
    normal_depth: Optional[float]
    critical_depth: float
    depth_range: Tuple[float, float]
    length: float
    curvature: str  # "concave_up", "concave_down", "mixed"
    asymptotic_behavior: str
    engineering_significance: str


class ProfileClassifier:
    """
    Classifies water surface profiles based on hydraulic theory.
    
    Uses the fundamental relationships between:
    - Normal depth (yn) and critical depth (yc)
    - Channel slope (S₀) and critical slope (Sc)
    - Flow depth (y) relative to yn and yc
    """
    
    def __init__(self, tolerance: float = 0.15):
        """
        Initialize profile classifier.
        
        Args:
            tolerance: Tolerance for depth comparisons (fraction of depth)
        """
        self.tolerance = tolerance
    
    def classify_profile(
        self,
        gvf_result: GVFResult,
        channel: ChannelGeometry,
        discharge: float,
        slope: float,
        manning_n: float
    ) -> ProfileCharacteristics:
        """
        Classify a computed water surface profile.
        
        Args:
            gvf_result: Results from GVF computation
            channel: Channel geometry
            discharge: Discharge
            slope: Channel slope
            manning_n: Manning's roughness coefficient
            
        Returns:
            ProfileCharacteristics with classification results
        """
        if not gvf_result.success or not gvf_result.profile_points:
            return ProfileCharacteristics(
                profile_type=ProfileType.UNKNOWN,
                slope_type=SlopeType.MILD,
                flow_regime=FlowRegime.SUBCRITICAL,
                normal_depth=None,
                critical_depth=0.0,
                depth_range=(0.0, 0.0),
                length=0.0,
                curvature="unknown",
                asymptotic_behavior="unknown",
                engineering_significance="Profile computation failed"
            )
        
        # Extract profile information
        profile_points = gvf_result.profile_points
        depths = [point.depth for point in profile_points]
        min_depth = min(depths)
        max_depth = max(depths)
        length = gvf_result.length
        
        # Compute critical depth
        critical_depth = CriticalDepth.calculate(channel, discharge)
        
        # Compute normal depth (if possible)
        normal_depth = None
        try:
            if slope > 0:
                normal_depth = NormalDepth.calculate(channel, discharge, slope, manning_n)
        except:
            normal_depth = None  # May not exist for adverse/horizontal slopes
        
        # Classify slope type
        slope_type = self._classify_slope(slope, critical_depth, normal_depth, channel, discharge, manning_n)
        
        # Determine dominant flow regime
        flow_regime = self._determine_flow_regime(profile_points)
        
        # Classify profile type
        profile_type = self._classify_profile_type(
            depths, normal_depth, critical_depth, slope_type
        )
        
        # Analyze curvature
        curvature = self._analyze_curvature(profile_points)
        
        # Determine asymptotic behavior
        asymptotic_behavior = self._analyze_asymptotic_behavior(
            profile_points, normal_depth, critical_depth, profile_type
        )
        
        # Generate engineering significance
        engineering_significance = self._generate_engineering_significance(
            profile_type, slope_type, flow_regime
        )
        
        return ProfileCharacteristics(
            profile_type=profile_type,
            slope_type=slope_type,
            flow_regime=flow_regime,
            normal_depth=normal_depth,
            critical_depth=critical_depth,
            depth_range=(min_depth, max_depth),
            length=length,
            curvature=curvature,
            asymptotic_behavior=asymptotic_behavior,
            engineering_significance=engineering_significance
        )
    
    def _classify_slope(
        self,
        slope: float,
        critical_depth: float,
        normal_depth: Optional[float],
        channel: ChannelGeometry,
        discharge: float,
        manning_n: float
    ) -> SlopeType:
        """Classify channel slope type."""
        if abs(slope) < 1e-10:
            return SlopeType.HORIZONTAL
        
        if slope < 0:
            return SlopeType.ADVERSE
        
        if normal_depth is None:
            # Cannot compute normal depth - likely horizontal or adverse
            return SlopeType.HORIZONTAL if slope == 0 else SlopeType.ADVERSE
        
        # Compute critical slope
        try:
            from ..flow_analysis import CriticalFlow
            critical_flow = CriticalFlow(channel)
            critical_slope = critical_flow.calculate_critical_slope(discharge, manning_n)
            
            # Compare slopes with tolerance (use smaller tolerance for slope comparison)
            slope_tolerance = min(self.tolerance, 0.1)  # Max 10% tolerance for slopes
            if abs(slope - critical_slope) / critical_slope < slope_tolerance:
                return SlopeType.CRITICAL
            elif slope < critical_slope:
                return SlopeType.MILD
            else:
                return SlopeType.STEEP
                
        except:
            # Fallback: compare normal and critical depths with more lenient tolerance
            if normal_depth is None:
                return SlopeType.HORIZONTAL if slope == 0 else SlopeType.ADVERSE
            
            depth_tolerance = self.tolerance * 1.5  # More lenient for depth comparison
            if abs(normal_depth - critical_depth) / critical_depth < depth_tolerance:
                return SlopeType.CRITICAL
            elif normal_depth > critical_depth * (1 + depth_tolerance):
                return SlopeType.MILD
            elif normal_depth < critical_depth * (1 - depth_tolerance):
                return SlopeType.STEEP
            else:
                # Close to critical, default to mild for safety
                return SlopeType.MILD
    
    def _determine_flow_regime(self, profile_points: List[ProfilePoint]) -> FlowRegime:
        """Determine dominant flow regime in profile."""
        subcritical_count = sum(1 for p in profile_points if p.froude_number < 0.95)
        supercritical_count = sum(1 for p in profile_points if p.froude_number > 1.05)
        critical_count = len(profile_points) - subcritical_count - supercritical_count
        
        if subcritical_count > supercritical_count and subcritical_count > critical_count:
            return FlowRegime.SUBCRITICAL
        elif supercritical_count > subcritical_count and supercritical_count > critical_count:
            return FlowRegime.SUPERCRITICAL
        elif critical_count > 0:
            return FlowRegime.CRITICAL
        else:
            return FlowRegime.MIXED
    
    def _classify_profile_type(
        self,
        depths: List[float],
        normal_depth: Optional[float],
        critical_depth: float,
        slope_type: SlopeType
    ) -> ProfileType:
        """Classify profile type based on depth relationships."""
        min_depth = min(depths)
        max_depth = max(depths)
        avg_depth = sum(depths) / len(depths)
        depth_range = max_depth - min_depth
        
        # Helper function to compare depths with adaptive tolerance
        def depth_compare(d1: float, d2: float, context: str = "general") -> str:
            """Compare two depths: 'greater', 'less', or 'equal'"""
            # Use adaptive tolerance based on context
            if context == "boundary":
                tolerance = self.tolerance * 2  # More lenient for boundary conditions
            else:
                tolerance = self.tolerance
            
            relative_diff = abs(d1 - d2) / max(d2, 0.1)
            if relative_diff < tolerance:
                return 'equal'
            elif d1 > d2:
                return 'greater'
            else:
                return 'less'
        
        # Helper function to check if depth is in range
        def depth_in_range(depth: float, range_min: float, range_max: float) -> bool:
            """Check if depth is within a range with tolerance"""
            margin = max(0.05, self.tolerance * max(range_max, 0.1))
            return (range_min - margin) <= depth <= (range_max + margin)
        
        # Special cases - check with more lenient criteria
        if normal_depth:
            # Check for uniform flow (all depths near normal AND small depth variation)
            uniform_tolerance = self.tolerance * 1.5  # Slightly more lenient
            uniform_count = sum(1 for d in depths if abs(d - normal_depth) / max(normal_depth, 0.1) < uniform_tolerance)
            depth_variation = (max_depth - min_depth) / max(avg_depth, 0.1)
            
            # Only classify as uniform if most points are near normal AND variation is small
            if (uniform_count > len(depths) * 0.9 and  # 90% of points near normal
                depth_variation < 0.05):  # Less than 5% depth variation
                return ProfileType.UNIFORM
        
        # Check for critical flow
        critical_count = sum(1 for d in depths if abs(d - critical_depth) / max(critical_depth, 0.1) < self.tolerance * 2)
        if critical_count > len(depths) * 0.8:  # 80% of points near critical depth
            return ProfileType.CRITICAL_FLOW
        
        # Classification based on slope type and depth relationships
        if slope_type == SlopeType.MILD:
            if normal_depth is None:
                return ProfileType.UNKNOWN
            
            # For mild slopes: yn > yc
            # M1: y > yn > yc (backwater curve) - depths consistently above normal
            if (avg_depth > normal_depth * (1 + self.tolerance * 0.5) and  # Average clearly above normal
                min_depth > normal_depth * (1 - self.tolerance * 0.5)):    # All points above normal
                return ProfileType.M1
            
            # M2: yn > y > yc (drawdown curve)  
            elif (max_depth < normal_depth * (1 + self.tolerance) and 
                  min_depth > critical_depth * (1 - self.tolerance) and
                  avg_depth < normal_depth):
                return ProfileType.M2
            
            # M3: yn > yc > y (backwater curve in supercritical region)
            elif max_depth < critical_depth * (1 + self.tolerance):
                return ProfileType.M3
            
            # Check if it's transitioning between regions (could be M1 or M2)
            elif depth_in_range(min_depth, critical_depth, normal_depth) or depth_in_range(max_depth, critical_depth, normal_depth):
                # Determine based on trend and average position
                if avg_depth > (normal_depth + critical_depth) / 2:
                    return ProfileType.M1  # Closer to normal depth
                else:
                    return ProfileType.M2  # Closer to critical depth
        
        elif slope_type == SlopeType.STEEP:
            if normal_depth is None:
                return ProfileType.UNKNOWN
            
            # For steep slopes: yc > yn
            # S1: y > yc > yn (backwater curve) - depths consistently above critical
            if (avg_depth > critical_depth * (1 + self.tolerance * 0.5) and  # Average clearly above critical
                min_depth > critical_depth * (1 + self.tolerance * 0.2)):    # All points above critical
                return ProfileType.S1
            
            # S2: yc > y > yn (drawdown curve)
            elif (max_depth < critical_depth * (1 + self.tolerance) and 
                  min_depth > normal_depth * (1 - self.tolerance) and
                  avg_depth < critical_depth):
                return ProfileType.S2
            
            # S3: yc > yn > y (backwater curve in supercritical region)
            elif max_depth < normal_depth * (1 + self.tolerance):
                return ProfileType.S3
            
            # Check if it's transitioning between regions
            elif depth_in_range(min_depth, normal_depth, critical_depth) or depth_in_range(max_depth, normal_depth, critical_depth):
                if avg_depth > (critical_depth + normal_depth) / 2:
                    return ProfileType.S1  # Closer to critical depth
                else:
                    return ProfileType.S2  # Between critical and normal
        
        elif slope_type == SlopeType.CRITICAL:
            # For critical slopes: yc ≈ yn
            if avg_depth > critical_depth * (1 + self.tolerance):
                return ProfileType.C1  # y > yc = yn
            elif avg_depth < critical_depth * (1 - self.tolerance):
                return ProfileType.C3  # yc = yn > y
        
        elif slope_type == SlopeType.HORIZONTAL:
            # For horizontal channels: yn = ∞
            if avg_depth > critical_depth * (1 + self.tolerance):
                return ProfileType.H2  # y > yc, yn = ∞
            elif avg_depth < critical_depth * (1 - self.tolerance):
                return ProfileType.H3  # yc > y, yn = ∞
        
        elif slope_type == SlopeType.ADVERSE:
            # For adverse slopes: yn is imaginary
            if avg_depth > critical_depth * (1 + self.tolerance):
                return ProfileType.A2  # y > yc, yn imaginary
            elif avg_depth < critical_depth * (1 - self.tolerance):
                return ProfileType.A3  # yc > y, yn imaginary
        
        # If we can't classify definitively, try to make an educated guess
        if normal_depth and critical_depth:
            if slope_type == SlopeType.MILD:
                # For mild slopes, if depth is generally above normal, likely M1
                if avg_depth > normal_depth:
                    return ProfileType.M1
                elif avg_depth > critical_depth:
                    return ProfileType.M2
                else:
                    return ProfileType.M3
            elif slope_type == SlopeType.STEEP:
                # For steep slopes, if depth is generally above critical, likely S1
                if avg_depth > critical_depth:
                    return ProfileType.S1
                elif avg_depth > normal_depth:
                    return ProfileType.S2
                else:
                    return ProfileType.S3
        
        return ProfileType.UNKNOWN
    
    def _analyze_curvature(self, profile_points: List[ProfilePoint]) -> str:
        """Analyze profile curvature using second derivatives."""
        if len(profile_points) < 3:
            return "insufficient_data"
        
        # Compute approximate second derivatives
        second_derivatives = []
        for i in range(1, len(profile_points) - 1):
            p_prev = profile_points[i-1]
            p_curr = profile_points[i]
            p_next = profile_points[i+1]
            
            # Approximate second derivative
            dx1 = p_curr.x - p_prev.x
            dx2 = p_next.x - p_curr.x
            dy1 = p_curr.depth - p_prev.depth
            dy2 = p_next.depth - p_curr.depth
            
            if dx1 > 0 and dx2 > 0:
                d2y_dx2 = (dy2/dx2 - dy1/dx1) / ((dx1 + dx2)/2)
                second_derivatives.append(d2y_dx2)
        
        if not second_derivatives:
            return "unknown"
        
        # Analyze curvature
        positive_count = sum(1 for d2 in second_derivatives if d2 > 1e-6)
        negative_count = sum(1 for d2 in second_derivatives if d2 < -1e-6)
        
        if positive_count > negative_count * 2:
            return "concave_up"
        elif negative_count > positive_count * 2:
            return "concave_down"
        else:
            return "mixed"
    
    def _analyze_asymptotic_behavior(
        self,
        profile_points: List[ProfilePoint],
        normal_depth: Optional[float],
        critical_depth: float,
        profile_type: ProfileType
    ) -> str:
        """Analyze asymptotic approach to normal or critical depth."""
        if len(profile_points) < 2:
            return "insufficient_data"
        
        # Check approach to normal depth
        if normal_depth:
            start_diff = abs(profile_points[0].depth - normal_depth)
            end_diff = abs(profile_points[-1].depth - normal_depth)
            
            if end_diff < start_diff * 0.5:
                return f"approaches_normal_depth ({normal_depth:.3f}m)"
        
        # Check approach to critical depth
        start_diff_crit = abs(profile_points[0].depth - critical_depth)
        end_diff_crit = abs(profile_points[-1].depth - critical_depth)
        
        if end_diff_crit < start_diff_crit * 0.5:
            return f"approaches_critical_depth ({critical_depth:.3f}m)"
        
        # Analyze trend
        depth_change = profile_points[-1].depth - profile_points[0].depth
        if abs(depth_change) < 0.01:
            return "nearly_horizontal"
        elif depth_change > 0:
            return "increasing_depth"
        else:
            return "decreasing_depth"
    
    def _generate_engineering_significance(
        self,
        profile_type: ProfileType,
        slope_type: SlopeType,
        flow_regime: FlowRegime
    ) -> str:
        """Generate engineering interpretation of the profile."""
        significance_map = {
            ProfileType.M1: "Backwater curve upstream of dam or obstruction. Subcritical flow with increasing depth upstream.",
            ProfileType.M2: "Drawdown curve at channel entrance or gate opening. Transition from normal to critical depth.",
            ProfileType.M3: "Backwater curve downstream of gate or under bridge. Supercritical flow transitioning to normal depth.",
            ProfileType.S1: "Backwater curve upstream of obstruction in steep channel. Flow backed up above critical depth.",
            ProfileType.S2: "Drawdown curve in steep channel. Transition from critical to normal depth.",
            ProfileType.S3: "Backwater curve at channel entrance in steep slope. Supercritical flow development.",
            ProfileType.C1: "Backwater curve on critical slope. Slight obstruction causes significant depth increase.",
            ProfileType.C3: "Drawdown curve on critical slope. Flow acceleration from rest.",
            ProfileType.H2: "Drawdown curve in horizontal channel. Flow accelerating toward critical depth.",
            ProfileType.H3: "Backwater curve in horizontal channel. Flow decelerating from supercritical.",
            ProfileType.A2: "Drawdown curve on adverse slope. Flow maintained by upstream momentum.",
            ProfileType.A3: "Backwater curve on adverse slope. Flow deceleration and possible reversal.",
            ProfileType.UNIFORM: "Uniform flow condition. Depth equals normal depth throughout.",
            ProfileType.CRITICAL_FLOW: "Critical flow condition. Froude number equals 1.0.",
            ProfileType.MIXED: "Mixed profile with multiple flow regimes. Complex hydraulic conditions.",
            ProfileType.UNKNOWN: "Profile type cannot be determined. May require additional analysis."
        }
        
        base_significance = significance_map.get(profile_type, "Unknown profile characteristics.")
        
        # Add slope and regime context
        slope_context = f" Channel slope: {slope_type.value}."
        regime_context = f" Flow regime: {flow_regime.value}."
        
        return base_significance + slope_context + regime_context


class WaterSurfaceProfile:
    """
    Represents a complete water surface profile with classification.
    
    Combines GVF computation results with profile classification
    to provide comprehensive hydraulic analysis.
    """
    
    def __init__(
        self,
        gvf_result: GVFResult,
        characteristics: ProfileCharacteristics
    ):
        """
        Initialize water surface profile.
        
        Args:
            gvf_result: GVF computation results
            characteristics: Profile classification results
        """
        self.gvf_result = gvf_result
        self.characteristics = characteristics
    
    @property
    def profile_points(self) -> List[ProfilePoint]:
        """Get profile points."""
        return self.gvf_result.profile_points
    
    @property
    def profile_type(self) -> ProfileType:
        """Get profile type."""
        return self.characteristics.profile_type
    
    @property
    def engineering_description(self) -> str:
        """Get engineering description of the profile."""
        return self.characteristics.engineering_significance
    
    def get_depth_at_distance(self, x: float) -> Optional[float]:
        """
        Get depth at specified distance using interpolation.
        
        Args:
            x: Distance along channel
            
        Returns:
            Interpolated depth or None if outside range
        """
        points = self.profile_points
        if not points:
            return None
        
        # Check bounds
        x_min = min(p.x for p in points)
        x_max = max(p.x for p in points)
        
        if x < x_min or x > x_max:
            return None
        
        # Find surrounding points
        for i in range(len(points) - 1):
            if points[i].x <= x <= points[i+1].x:
                # Linear interpolation
                x1, y1 = points[i].x, points[i].depth
                x2, y2 = points[i+1].x, points[i+1].depth
                
                if x2 == x1:
                    return y1
                
                return y1 + (y2 - y1) * (x - x1) / (x2 - x1)
        
        return None
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive profile summary."""
        return {
            'profile_type': self.characteristics.profile_type.value,
            'slope_type': self.characteristics.slope_type.value,
            'flow_regime': self.characteristics.flow_regime.value,
            'normal_depth': self.characteristics.normal_depth,
            'critical_depth': self.characteristics.critical_depth,
            'depth_range': self.characteristics.depth_range,
            'length': self.characteristics.length,
            'curvature': self.characteristics.curvature,
            'asymptotic_behavior': self.characteristics.asymptotic_behavior,
            'engineering_significance': self.characteristics.engineering_significance,
            'computation_successful': self.gvf_result.success,
            'total_points': len(self.profile_points),
            'integration_steps': self.gvf_result.integration_result.total_steps,
            'events_detected': len(self.gvf_result.events_detected)
        }


class ProfileAnalyzer:
    """
    High-level analyzer for water surface profiles.
    
    Provides comprehensive analysis including:
    - Profile classification
    - Engineering interpretation
    - Design recommendations
    - Comparison with standard profiles
    """
    
    def __init__(self):
        """Initialize profile analyzer."""
        self.classifier = ProfileClassifier()
    
    def analyze_profile(
        self,
        gvf_result: GVFResult,
        channel: ChannelGeometry,
        discharge: float,
        slope: float,
        manning_n: float
    ) -> WaterSurfaceProfile:
        """
        Perform complete profile analysis.
        
        Args:
            gvf_result: GVF computation results
            channel: Channel geometry
            discharge: Discharge
            slope: Channel slope
            manning_n: Manning's roughness coefficient
            
        Returns:
            WaterSurfaceProfile with complete analysis
        """
        # Classify the profile
        characteristics = self.classifier.classify_profile(
            gvf_result, channel, discharge, slope, manning_n
        )
        
        # Create comprehensive profile object
        profile = WaterSurfaceProfile(gvf_result, characteristics)
        
        return profile
    
    def compare_profiles(
        self,
        profiles: List[WaterSurfaceProfile]
    ) -> Dict[str, Any]:
        """
        Compare multiple water surface profiles.
        
        Args:
            profiles: List of profiles to compare
            
        Returns:
            Comparison analysis
        """
        if not profiles:
            return {"error": "No profiles provided"}
        
        # Extract profile types
        profile_types = [p.profile_type.value for p in profiles]
        
        # Find common characteristics
        slope_types = [p.characteristics.slope_type.value for p in profiles]
        flow_regimes = [p.characteristics.flow_regime.value for p in profiles]
        
        # Compute statistics
        lengths = [p.characteristics.length for p in profiles]
        max_depths = [p.characteristics.depth_range[1] for p in profiles]
        
        return {
            'profile_count': len(profiles),
            'profile_types': profile_types,
            'unique_types': list(set(profile_types)),
            'slope_types': list(set(slope_types)),
            'flow_regimes': list(set(flow_regimes)),
            'length_range': (min(lengths), max(lengths)),
            'depth_range': (min(max_depths), max(max_depths)),
            'average_length': sum(lengths) / len(lengths),
            'average_max_depth': sum(max_depths) / len(max_depths)
        }
