"""
applications.py - GVF Engineering Applications
===============================================

Pre-built application classes for common hydraulic engineering scenarios.
These classes provide high-level interfaces for specific engineering applications,
combining GVF analysis with domain-specific knowledge and design guidelines.

Applications included:
- DamAnalysis: Backwater analysis for dams and reservoirs
- BridgeAnalysis: Bridge hydraulics and clearance design
- ChuteAnalysis: Steep channel and energy dissipation design
- ChannelTransition: Geometry and slope transition analysis

Author: Alexius Academia
Date: 2025-08-17
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import math

from .solver import GVFSolver, GVFResult, BoundaryType
from .profiles import ProfileClassifier, WaterSurfaceProfile
from ..geometry import ChannelGeometry, RectangularChannel, TrapezoidalChannel
from ..hydraulics import NormalDepth
from ..flow_analysis import CriticalFlow
from ..exceptions import PyOpenChannelError


class AnalysisType(Enum):
    """Types of hydraulic analysis."""
    BACKWATER = "backwater"
    DRAWDOWN = "drawdown"
    TRANSITION = "transition"
    ENERGY_DISSIPATION = "energy_dissipation"
    FLOOD_ROUTING = "flood_routing"


class DesignCriteria(Enum):
    """Design criteria standards."""
    CONSERVATIVE = "conservative"
    STANDARD = "standard"
    OPTIMIZED = "optimized"


@dataclass
class AnalysisResult:
    """
    Base class for application analysis results.
    
    Attributes:
        analysis_type: Type of analysis performed
        success: Whether analysis completed successfully
        message: Status message or error description
        gvf_result: Raw GVF computation results
        profile: Classified water surface profile
        design_parameters: Key design parameters
        recommendations: Engineering recommendations
        warnings: Design warnings and considerations
        compliance: Regulatory compliance notes
    """
    analysis_type: AnalysisType
    success: bool
    message: str
    gvf_result: Optional[GVFResult] = None
    profile: Optional[WaterSurfaceProfile] = None
    design_parameters: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    compliance: List[str] = field(default_factory=list)


class DamAnalysis:
    """
    Dam backwater analysis for flood studies and reservoir operations.
    
    This class provides comprehensive analysis of dam-induced backwater effects,
    including flood elevation mapping, bridge clearance analysis, and
    environmental impact assessment.
    
    Features:
    - Multiple dam height scenarios
    - Flood zone delineation
    - Bridge clearance calculations
    - Sediment deposition analysis
    - Environmental impact assessment
    - Regulatory compliance documentation
    """
    
    def __init__(self, design_criteria: DesignCriteria = DesignCriteria.STANDARD):
        """
        Initialize dam analysis.
        
        Args:
            design_criteria: Design conservatism level
        """
        self.design_criteria = design_criteria
        self.solver = GVFSolver()
        self.classifier = ProfileClassifier()
        
        # Design factors based on criteria
        self.safety_factors = {
            DesignCriteria.CONSERVATIVE: {"freeboard": 1.5, "backwater": 1.3},
            DesignCriteria.STANDARD: {"freeboard": 1.2, "backwater": 1.1},
            DesignCriteria.OPTIMIZED: {"freeboard": 1.1, "backwater": 1.05}
        }
    
    def analyze_backwater(
        self,
        channel: ChannelGeometry,
        discharge: float,
        slope: float,
        manning_n: float,
        dam_height: float,
        analysis_distance: float = 10000.0,
        bridge_locations: Optional[List[float]] = None
    ) -> AnalysisResult:
        """
        Analyze dam backwater effects.
        
        Args:
            channel: Channel geometry
            discharge: Design discharge (m³/s or ft³/s)
            slope: Channel slope
            manning_n: Manning's roughness coefficient
            dam_height: Dam height above channel bottom (m or ft)
            analysis_distance: Distance upstream to analyze (m or ft)
            bridge_locations: List of bridge distances upstream (m or ft)
            
        Returns:
            Comprehensive analysis results
        """
        try:
            # Calculate reference depths
            normal_depth = NormalDepth.calculate(channel, discharge, slope, manning_n)
            critical_depth = CriticalFlow(channel).calculate_critical_depth(discharge)
            
            # Dam creates boundary condition
            dam_depth = normal_depth + dam_height
            safety_factor = self.safety_factors[self.design_criteria]["backwater"]
            design_dam_depth = dam_depth * safety_factor
            
            # Solve GVF profile
            gvf_result = self.solver.solve_profile(
                channel=channel,
                discharge=discharge,
                slope=slope,
                manning_n=manning_n,
                x_start=0.0,
                x_end=analysis_distance,
                boundary_depth=design_dam_depth,
                boundary_type=BoundaryType.UPSTREAM_DEPTH
            )
            
            # Classify profile
            profile = self.classifier.classify_profile(
                gvf_result=gvf_result,
                channel=channel,
                discharge=discharge,
                slope=slope,
                manning_n=manning_n
            )
            
            # Extract analysis data
            depths = [p.depth for p in gvf_result.profile_points]
            distances = [p.x for p in gvf_result.profile_points]
            velocities = [p.velocity for p in gvf_result.profile_points]
            
            # Find backwater extent
            backwater_extent = self._find_backwater_extent(depths, distances, normal_depth)
            max_backwater = max(depths) - normal_depth
            
            # Design parameters
            design_params = {
                "dam_height": dam_height,
                "dam_depth": dam_depth,
                "design_dam_depth": design_dam_depth,
                "normal_depth": normal_depth,
                "critical_depth": critical_depth,
                "max_backwater": max_backwater,
                "backwater_extent": backwater_extent,
                "safety_factor": safety_factor,
                "flood_elevation": max(depths)
            }
            
            # Bridge analysis
            if bridge_locations:
                bridge_analysis = self._analyze_bridge_clearances(
                    bridge_locations, distances, depths, design_params
                )
                design_params.update(bridge_analysis)
            
            # Generate recommendations
            recommendations = self._generate_dam_recommendations(design_params, profile)
            warnings = self._generate_dam_warnings(design_params, profile)
            compliance = self._generate_dam_compliance(design_params)
            
            return AnalysisResult(
                analysis_type=AnalysisType.BACKWATER,
                success=True,
                message="Dam backwater analysis completed successfully",
                gvf_result=gvf_result,
                profile=profile,
                design_parameters=design_params,
                recommendations=recommendations,
                warnings=warnings,
                compliance=compliance
            )
            
        except Exception as e:
            return AnalysisResult(
                analysis_type=AnalysisType.BACKWATER,
                success=False,
                message=f"Dam analysis failed: {str(e)}",
                design_parameters={"error": str(e)}
            )
    
    def _find_backwater_extent(self, depths: List[float], distances: List[float], 
                              normal_depth: float, tolerance: float = 0.05) -> float:
        """Find the extent of backwater influence."""
        for i, depth in enumerate(depths):
            if abs(depth - normal_depth) / normal_depth < tolerance:
                return distances[i]
        return max(distances)  # Backwater extends beyond analysis distance
    
    def _analyze_bridge_clearances(self, bridge_locations: List[float], 
                                  distances: List[float], depths: List[float],
                                  design_params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze bridge clearance requirements."""
        freeboard_factor = self.safety_factors[self.design_criteria]["freeboard"]
        bridge_analysis = {"bridge_clearances": {}}
        
        for bridge_loc in bridge_locations:
            # Find water depth at bridge location
            bridge_depth = design_params["normal_depth"]  # Default
            for i, dist in enumerate(distances):
                if dist >= bridge_loc:
                    bridge_depth = depths[i]
                    break
            
            required_elevation = bridge_depth * freeboard_factor
            bridge_analysis["bridge_clearances"][bridge_loc] = {
                "water_depth": bridge_depth,
                "required_elevation": required_elevation,
                "freeboard_factor": freeboard_factor
            }
        
        return bridge_analysis
    
    def _generate_dam_recommendations(self, params: Dict[str, Any], 
                                    profile: WaterSurfaceProfile) -> List[str]:
        """Generate engineering recommendations."""
        recommendations = []
        
        if params["max_backwater"] > 2.0:
            recommendations.append("Significant backwater effect - consider environmental impact assessment")
        
        if params["backwater_extent"] > 5000:
            recommendations.append("Extensive backwater zone - detailed flood mapping required")
        
        if profile.profile_type.value == "M1":
            recommendations.append("M1 profile confirmed - typical dam backwater curve")
        
        recommendations.append(f"Design flood elevation: {params['flood_elevation']:.2f} m")
        recommendations.append("Consider sediment deposition in backwater zone")
        
        return recommendations
    
    def _generate_dam_warnings(self, params: Dict[str, Any], 
                             profile: WaterSurfaceProfile) -> List[str]:
        """Generate design warnings."""
        warnings = []
        
        if params["backwater_extent"] > 10000:
            warnings.append("Backwater extends >10km - may affect multiple communities")
        
        if params["max_backwater"] > params["normal_depth"]:
            warnings.append("Backwater depth exceeds normal depth - significant impact")
        
        return warnings
    
    def _generate_dam_compliance(self, params: Dict[str, Any]) -> List[str]:
        """Generate regulatory compliance notes."""
        compliance = []
        
        compliance.append("Floodplain mapping required per local regulations")
        
        if params["dam_height"] > 3.0:
            compliance.append("Dam safety analysis required for structures >3m")
        
        if params["backwater_extent"] > 2000:
            compliance.append("Environmental impact assessment recommended")
        
        return compliance


class BridgeAnalysis:
    """
    Bridge hydraulics analysis for clearance design and scour assessment.
    
    This class provides comprehensive bridge hydraulic analysis including
    backwater effects, clearance requirements, scour potential, and
    design optimization.
    
    Features:
    - Bridge opening optimization
    - Clearance calculations with safety factors
    - Scour depth estimation
    - Approach and departure analysis
    - Multiple span configurations
    - Debris and ice considerations
    """
    
    def __init__(self, design_criteria: DesignCriteria = DesignCriteria.STANDARD):
        """Initialize bridge analysis."""
        self.design_criteria = design_criteria
        self.solver = GVFSolver()
        self.classifier = ProfileClassifier()
        
        # Bridge design factors
        self.design_factors = {
            DesignCriteria.CONSERVATIVE: {"clearance": 2.0, "contraction": 0.7, "scour": 2.5},
            DesignCriteria.STANDARD: {"clearance": 1.5, "contraction": 0.8, "scour": 2.0},
            DesignCriteria.OPTIMIZED: {"clearance": 1.2, "contraction": 0.9, "scour": 1.5}
        }
    
    def analyze_bridge_hydraulics(
        self,
        approach_channel: ChannelGeometry,
        bridge_opening: ChannelGeometry,
        discharge: float,
        slope: float,
        manning_n: float,
        analysis_distance: float = 2000.0
    ) -> AnalysisResult:
        """
        Analyze bridge hydraulics and design requirements.
        
        Args:
            approach_channel: Upstream channel geometry
            bridge_opening: Bridge opening geometry
            discharge: Design discharge
            slope: Channel slope
            manning_n: Manning's roughness coefficient
            analysis_distance: Analysis distance upstream
            
        Returns:
            Bridge analysis results
        """
        try:
            # Calculate reference conditions
            approach_normal = NormalDepth.calculate(approach_channel, discharge, slope, manning_n)
            approach_critical = CriticalFlow(approach_channel).calculate_critical_depth(discharge)
            bridge_critical = CriticalFlow(bridge_opening).calculate_critical_depth(discharge)
            
            # Bridge boundary condition - avoid choking
            contraction_factor = self.design_factors[self.design_criteria]["contraction"]
            bridge_depth = max(bridge_critical * 1.1, approach_normal * contraction_factor)
            
            # Analyze upstream backwater
            gvf_result = self.solver.solve_profile(
                channel=approach_channel,
                discharge=discharge,
                slope=slope,
                manning_n=manning_n,
                x_start=0.0,
                x_end=analysis_distance,
                boundary_depth=bridge_depth,
                boundary_type=BoundaryType.UPSTREAM_DEPTH
            )
            
            # Classify profile
            profile = self.classifier.classify_profile(
                gvf_result=gvf_result,
                channel=approach_channel,
                discharge=discharge,
                slope=slope,
                manning_n=manning_n
            )
            
            # Calculate design parameters
            depths = [p.depth for p in gvf_result.profile_points]
            velocities = [p.velocity for p in gvf_result.profile_points]
            
            max_depth = max(depths)
            max_velocity = max(velocities)
            clearance_factor = self.design_factors[self.design_criteria]["clearance"]
            scour_factor = self.design_factors[self.design_criteria]["scour"]
            
            # Scour estimation (simplified)
            scour_depth = self._estimate_scour_depth(bridge_depth, max_velocity, bridge_opening)
            
            design_params = {
                "approach_normal_depth": approach_normal,
                "approach_critical_depth": approach_critical,
                "bridge_critical_depth": bridge_critical,
                "bridge_depth": bridge_depth,
                "max_upstream_depth": max_depth,
                "max_velocity": max_velocity,
                "backwater_rise": max_depth - approach_normal,
                "required_clearance": max_depth * clearance_factor,
                "estimated_scour_depth": scour_depth * scour_factor,
                "contraction_ratio": self._calculate_contraction_ratio(approach_channel, bridge_opening),
                "froude_number_bridge": velocities[0] / math.sqrt(9.81 * bridge_depth)
            }
            
            # Generate analysis outputs
            recommendations = self._generate_bridge_recommendations(design_params, profile)
            warnings = self._generate_bridge_warnings(design_params)
            compliance = self._generate_bridge_compliance(design_params)
            
            return AnalysisResult(
                analysis_type=AnalysisType.TRANSITION,
                success=True,
                message="Bridge hydraulic analysis completed successfully",
                gvf_result=gvf_result,
                profile=profile,
                design_parameters=design_params,
                recommendations=recommendations,
                warnings=warnings,
                compliance=compliance
            )
            
        except Exception as e:
            return AnalysisResult(
                analysis_type=AnalysisType.TRANSITION,
                success=False,
                message=f"Bridge analysis failed: {str(e)}",
                design_parameters={"error": str(e)}
            )
    
    def _estimate_scour_depth(self, depth: float, velocity: float, 
                            bridge_opening: ChannelGeometry) -> float:
        """Estimate scour depth using simplified equations."""
        # Simplified scour estimation - in practice, use detailed scour equations
        if hasattr(bridge_opening, 'width'):
            pier_width = bridge_opening.width * 0.1  # Assume pier width is 10% of opening
        else:
            pier_width = 1.0  # Default pier width
        
        # Simplified Froehlich equation
        froude = velocity / math.sqrt(9.81 * depth)
        scour_depth = 2.27 * pier_width * (froude ** 0.43) * (depth / pier_width) ** 0.65
        
        return max(scour_depth, 0.5)  # Minimum 0.5m scour
    
    def _calculate_contraction_ratio(self, approach: ChannelGeometry, 
                                   bridge: ChannelGeometry) -> float:
        """Calculate channel contraction ratio."""
        # Use representative depths for area calculation
        depth = 2.0  # Representative depth
        approach_area = approach.area(depth)
        bridge_area = bridge.area(depth)
        return bridge_area / approach_area
    
    def _generate_bridge_recommendations(self, params: Dict[str, Any], 
                                       profile: WaterSurfaceProfile) -> List[str]:
        """Generate bridge design recommendations."""
        recommendations = []
        
        recommendations.append(f"Minimum bridge clearance: {params['required_clearance']:.2f} m")
        recommendations.append(f"Design scour depth: {params['estimated_scour_depth']:.2f} m")
        
        if params["backwater_rise"] > 0.5:
            recommendations.append("Significant backwater - consider larger opening")
        
        if params["contraction_ratio"] < 0.7:
            recommendations.append("Severe contraction - evaluate for hydraulic efficiency")
        
        if params["froude_number_bridge"] > 0.8:
            recommendations.append("High Froude number at bridge - check for choking")
        
        recommendations.append("Install scour monitoring and protection")
        recommendations.append("Consider debris and ice loading in design")
        
        return recommendations
    
    def _generate_bridge_warnings(self, params: Dict[str, Any]) -> List[str]:
        """Generate bridge design warnings."""
        warnings = []
        
        if params["froude_number_bridge"] > 0.9:
            warnings.append("Critical flow conditions at bridge - risk of choking")
        
        if params["estimated_scour_depth"] > 3.0:
            warnings.append("Deep scour predicted - detailed scour analysis required")
        
        if params["backwater_rise"] > 1.0:
            warnings.append("Significant backwater may affect upstream properties")
        
        return warnings
    
    def _generate_bridge_compliance(self, params: Dict[str, Any]) -> List[str]:
        """Generate regulatory compliance notes."""
        compliance = []
        
        compliance.append("Design per AASHTO LRFD Bridge Design Specifications")
        compliance.append("Scour analysis per HEC-18 guidelines")
        
        if params["backwater_rise"] > 0.3:
            compliance.append("Hydraulic analysis required for permit approval")
        
        compliance.append("Consider environmental impact on aquatic habitat")
        
        return compliance


class ChuteAnalysis:
    """
    Steep channel and chute analysis for energy dissipation design.
    
    This class provides analysis of steep channels, chutes, and spillways
    including energy dissipation, hydraulic jump design, and erosion protection.
    
    Features:
    - Steep channel profile analysis
    - Energy dissipation calculations
    - Hydraulic jump location and design
    - Erosion protection requirements
    - Stilling basin design
    - Flip bucket analysis
    """
    
    def __init__(self, design_criteria: DesignCriteria = DesignCriteria.STANDARD):
        """Initialize chute analysis."""
        self.design_criteria = design_criteria
        self.solver = GVFSolver()
        self.classifier = ProfileClassifier()
        
        # Energy dissipation factors
        self.energy_factors = {
            DesignCriteria.CONSERVATIVE: {"dissipation": 0.85, "jump": 1.3},
            DesignCriteria.STANDARD: {"dissipation": 0.90, "jump": 1.2},
            DesignCriteria.OPTIMIZED: {"dissipation": 0.95, "jump": 1.1}
        }
    
    def analyze_chute(
        self,
        chute_channel: ChannelGeometry,
        tailwater_channel: ChannelGeometry,
        discharge: float,
        chute_slope: float,
        tailwater_slope: float,
        manning_n: float,
        chute_length: float = 500.0
    ) -> AnalysisResult:
        """
        Analyze steep chute and energy dissipation.
        
        Args:
            chute_channel: Steep chute geometry
            tailwater_channel: Downstream channel geometry
            discharge: Design discharge
            chute_slope: Steep chute slope
            tailwater_slope: Downstream channel slope
            manning_n: Manning's roughness coefficient
            chute_length: Length of steep chute
            
        Returns:
            Chute analysis results
        """
        try:
            # Calculate reference depths
            chute_normal = NormalDepth.calculate(chute_channel, discharge, chute_slope, manning_n)
            chute_critical = CriticalFlow(chute_channel).calculate_critical_depth(discharge)
            tailwater_normal = NormalDepth.calculate(tailwater_channel, discharge, tailwater_slope, manning_n)
            tailwater_critical = CriticalFlow(tailwater_channel).calculate_critical_depth(discharge)
            
            # Analyze chute profile (typically supercritical)
            chute_boundary = min(chute_normal * 1.1, chute_critical * 0.9)
            
            chute_result = self.solver.solve_profile(
                channel=chute_channel,
                discharge=discharge,
                slope=chute_slope,
                manning_n=manning_n,
                x_start=0.0,
                x_end=chute_length,
                boundary_depth=chute_boundary,
                boundary_type=BoundaryType.UPSTREAM_DEPTH
            )
            
            # Classify chute profile
            chute_profile = self.classifier.classify_profile(
                gvf_result=chute_result,
                channel=chute_channel,
                discharge=discharge,
                slope=chute_slope,
                manning_n=manning_n
            )
            
            # Calculate energy conditions
            chute_depths = [p.depth for p in chute_result.profile_points]
            chute_velocities = [p.velocity for p in chute_result.profile_points]
            
            # Energy at chute exit
            exit_depth = chute_depths[-1]
            exit_velocity = chute_velocities[-1]
            exit_energy = exit_depth + (exit_velocity ** 2) / (2 * 9.81)
            
            # Hydraulic jump analysis
            jump_analysis = self._analyze_hydraulic_jump(
                exit_depth, exit_velocity, tailwater_normal, tailwater_channel
            )
            
            # Energy dissipation calculation
            dissipation_factor = self.energy_factors[self.design_criteria]["dissipation"]
            energy_dissipated = (exit_energy - tailwater_normal) * dissipation_factor
            
            design_params = {
                "chute_normal_depth": chute_normal,
                "chute_critical_depth": chute_critical,
                "tailwater_normal_depth": tailwater_normal,
                "tailwater_critical_depth": tailwater_critical,
                "exit_depth": exit_depth,
                "exit_velocity": exit_velocity,
                "exit_energy": exit_energy,
                "exit_froude": exit_velocity / math.sqrt(9.81 * exit_depth),
                "energy_dissipated": energy_dissipated,
                "jump_required": jump_analysis["jump_required"],
                "jump_length": jump_analysis.get("jump_length", 0),
                "sequent_depth": jump_analysis.get("sequent_depth", 0)
            }
            
            # Generate analysis outputs
            recommendations = self._generate_chute_recommendations(design_params, chute_profile)
            warnings = self._generate_chute_warnings(design_params)
            compliance = self._generate_chute_compliance(design_params)
            
            return AnalysisResult(
                analysis_type=AnalysisType.ENERGY_DISSIPATION,
                success=True,
                message="Chute analysis completed successfully",
                gvf_result=chute_result,
                profile=chute_profile,
                design_parameters=design_params,
                recommendations=recommendations,
                warnings=warnings,
                compliance=compliance
            )
            
        except Exception as e:
            return AnalysisResult(
                analysis_type=AnalysisType.ENERGY_DISSIPATION,
                success=False,
                message=f"Chute analysis failed: {str(e)}",
                design_parameters={"error": str(e)}
            )
    
    def _analyze_hydraulic_jump(self, y1: float, v1: float, tailwater_depth: float,
                              channel: ChannelGeometry) -> Dict[str, Any]:
        """Analyze hydraulic jump characteristics."""
        froude1 = v1 / math.sqrt(9.81 * y1)
        
        if froude1 <= 1.0:
            return {"jump_required": False, "message": "Subcritical flow - no jump"}
        
        # Calculate sequent depth for rectangular channel (simplified)
        if isinstance(channel, RectangularChannel):
            y2 = (y1 / 2) * (-1 + math.sqrt(1 + 8 * froude1 ** 2))
        else:
            # Approximate for other geometries
            y2 = y1 * (1 + froude1) / 2
        
        # Jump length (empirical)
        jump_length = 6 * (y2 - y1)
        
        jump_required = y2 > tailwater_depth * 1.1
        
        return {
            "jump_required": jump_required,
            "sequent_depth": y2,
            "jump_length": jump_length,
            "froude_upstream": froude1,
            "tailwater_adequate": tailwater_depth >= y2 * 0.9
        }
    
    def _generate_chute_recommendations(self, params: Dict[str, Any], 
                                      profile: WaterSurfaceProfile) -> List[str]:
        """Generate chute design recommendations."""
        recommendations = []
        
        if params["exit_froude"] > 3.0:
            recommendations.append("High-velocity flow - robust energy dissipation required")
        
        if params["jump_required"]:
            recommendations.append(f"Hydraulic jump required - design stilling basin length: {params['jump_length']:.1f} m")
            recommendations.append("Install baffle blocks and end sill for jump stabilization")
        else:
            recommendations.append("No hydraulic jump required - consider flip bucket design")
        
        recommendations.append("Install erosion protection downstream of energy dissipator")
        recommendations.append("Design for cavitation prevention at high velocities")
        
        if params["energy_dissipated"] > 10.0:
            recommendations.append("High energy dissipation - consider stepped spillway design")
        
        return recommendations
    
    def _generate_chute_warnings(self, params: Dict[str, Any]) -> List[str]:
        """Generate chute design warnings."""
        warnings = []
        
        if params["exit_velocity"] > 15.0:
            warnings.append("Very high exit velocity - cavitation risk")
        
        if params["exit_froude"] > 5.0:
            warnings.append("Extremely supercritical flow - special design considerations")
        
        if params["energy_dissipated"] > 20.0:
            warnings.append("Extreme energy dissipation - detailed model studies recommended")
        
        return warnings
    
    def _generate_chute_compliance(self, params: Dict[str, Any]) -> List[str]:
        """Generate regulatory compliance notes."""
        compliance = []
        
        compliance.append("Design per USBR spillway design standards")
        compliance.append("Consider environmental impact of energy dissipation")
        
        if params["exit_velocity"] > 10.0:
            compliance.append("High-velocity flow - fish passage considerations")
        
        return compliance


class ChannelTransition:
    """
    Channel transition analysis for geometry and slope changes.
    
    This class provides analysis of channel transitions including
    contractions, expansions, slope changes, and geometry modifications.
    
    Features:
    - Contraction and expansion analysis
    - Slope transition effects
    - Energy loss calculations
    - Transition length optimization
    - Multiple geometry combinations
    """
    
    def __init__(self, design_criteria: DesignCriteria = DesignCriteria.STANDARD):
        """Initialize transition analysis."""
        self.design_criteria = design_criteria
        self.solver = GVFSolver()
        self.classifier = ProfileClassifier()
    
    def analyze_transition(
        self,
        upstream_channel: ChannelGeometry,
        downstream_channel: ChannelGeometry,
        discharge: float,
        upstream_slope: float,
        downstream_slope: float,
        manning_n: float,
        transition_length: float = 100.0
    ) -> AnalysisResult:
        """
        Analyze channel transition effects.
        
        Args:
            upstream_channel: Upstream channel geometry
            downstream_channel: Downstream channel geometry
            discharge: Design discharge
            upstream_slope: Upstream channel slope
            downstream_slope: Downstream channel slope
            manning_n: Manning's roughness coefficient
            transition_length: Length of transition zone
            
        Returns:
            Transition analysis results
        """
        try:
            # Calculate reference conditions
            upstream_normal = NormalDepth.calculate(upstream_channel, discharge, upstream_slope, manning_n)
            downstream_normal = NormalDepth.calculate(downstream_channel, discharge, downstream_slope, manning_n)
            upstream_critical = CriticalFlow(upstream_channel).calculate_critical_depth(discharge)
            downstream_critical = CriticalFlow(downstream_channel).calculate_critical_depth(discharge)
            
            # Determine transition boundary condition
            transition_depth = self._determine_transition_depth(
                upstream_normal, downstream_normal, upstream_critical, downstream_critical
            )
            
            # Analyze upstream effects
            upstream_result = self.solver.solve_profile(
                channel=upstream_channel,
                discharge=discharge,
                slope=upstream_slope,
                manning_n=manning_n,
                x_start=0.0,
                x_end=500.0,
                boundary_depth=transition_depth,
                boundary_type=BoundaryType.UPSTREAM_DEPTH
            )
            
            upstream_profile = self.classifier.classify_profile(
                gvf_result=upstream_result,
                channel=upstream_channel,
                discharge=discharge,
                slope=upstream_slope,
                manning_n=manning_n
            )
            
            # Calculate energy conditions
            upstream_depths = [p.depth for p in upstream_result.profile_points]
            upstream_velocities = [p.velocity for p in upstream_result.profile_points]
            
            # Energy analysis
            upstream_energy = upstream_depths[0] + (upstream_velocities[0] ** 2) / (2 * 9.81)
            downstream_energy = downstream_normal + (discharge / downstream_channel.area(downstream_normal)) ** 2 / (2 * 9.81)
            energy_loss = upstream_energy - downstream_energy
            
            design_params = {
                "upstream_normal": upstream_normal,
                "downstream_normal": downstream_normal,
                "upstream_critical": upstream_critical,
                "downstream_critical": downstream_critical,
                "transition_depth": transition_depth,
                "upstream_energy": upstream_energy,
                "downstream_energy": downstream_energy,
                "energy_loss": energy_loss,
                "contraction_ratio": self._calculate_area_ratio(upstream_channel, downstream_channel),
                "slope_change": downstream_slope - upstream_slope
            }
            
            # Generate analysis outputs
            recommendations = self._generate_transition_recommendations(design_params, upstream_profile)
            warnings = self._generate_transition_warnings(design_params)
            compliance = self._generate_transition_compliance(design_params)
            
            return AnalysisResult(
                analysis_type=AnalysisType.TRANSITION,
                success=True,
                message="Channel transition analysis completed successfully",
                gvf_result=upstream_result,
                profile=upstream_profile,
                design_parameters=design_params,
                recommendations=recommendations,
                warnings=warnings,
                compliance=compliance
            )
            
        except Exception as e:
            return AnalysisResult(
                analysis_type=AnalysisType.TRANSITION,
                success=False,
                message=f"Transition analysis failed: {str(e)}",
                design_parameters={"error": str(e)}
            )
    
    def _determine_transition_depth(self, yn1: float, yn2: float, yc1: float, yc2: float) -> float:
        """Determine appropriate transition boundary depth."""
        # Use the higher of the critical depths as a starting point
        base_depth = max(yc1, yc2)
        
        # Adjust based on normal depth relationships
        if yn1 > yn2:  # Contraction
            return min(yn1 * 1.1, base_depth * 1.2)
        else:  # Expansion
            return max(yn2 * 0.9, base_depth * 1.1)
    
    def _calculate_area_ratio(self, upstream: ChannelGeometry, downstream: ChannelGeometry) -> float:
        """Calculate area ratio at representative depth."""
        depth = 2.0  # Representative depth
        return downstream.area(depth) / upstream.area(depth)
    
    def _generate_transition_recommendations(self, params: Dict[str, Any], 
                                           profile: WaterSurfaceProfile) -> List[str]:
        """Generate transition design recommendations."""
        recommendations = []
        
        if params["energy_loss"] > 0.2:
            recommendations.append("Significant energy loss - consider gradual transition")
        
        if params["contraction_ratio"] < 0.8:
            recommendations.append("Significant contraction - check for choking conditions")
        elif params["contraction_ratio"] > 1.2:
            recommendations.append("Channel expansion - design for flow separation control")
        
        if abs(params["slope_change"]) > 0.005:
            recommendations.append("Significant slope change - consider critical depth at break")
        
        recommendations.append("Design transition length per standard guidelines (4:1 to 6:1)")
        
        return recommendations
    
    def _generate_transition_warnings(self, params: Dict[str, Any]) -> List[str]:
        """Generate transition warnings."""
        warnings = []
        
        if params["energy_loss"] > 0.5:
            warnings.append("High energy loss - detailed hydraulic design required")
        
        if params["contraction_ratio"] < 0.6:
            warnings.append("Severe contraction - risk of flow instability")
        
        return warnings
    
    def _generate_transition_compliance(self, params: Dict[str, Any]) -> List[str]:
        """Generate compliance notes."""
        compliance = []
        
        compliance.append("Design per standard hydraulic design guidelines")
        
        if abs(params["slope_change"]) > 0.01:
            compliance.append("Significant slope change - structural design considerations")
        
        return compliance
