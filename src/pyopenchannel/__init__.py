"""
File: openchannel/__init__.py

Date: 2025-08-17

PyOpenChannel - A Python library for open channel flow analysis and design.

This library provides tools for:
- Geometric analysis of open channel cross-sections
- Hydraulic calculations (normal depth, critical depth, etc.)
- Flow analysis and design
- Energy and momentum calculations
- Gradually varied flow analysis
"""

__version__ = "0.4.0"
__author__ = "Alexius Academia"
__email__ = "alexius.sayco.academia@gmail.com"

# Core imports for easy access
from .geometry import (
    RectangularChannel,
    TrapezoidalChannel,
    TriangularChannel,
    CircularChannel,
    ParabolicChannel,
)

from .hydraulics import (
    ManningEquation,
    ChezyEquation,
    HydraulicRadius,
    CriticalDepth,
    NormalDepth,
)

from .flow_analysis import (
    UniformFlow,
    CriticalFlow,
    GraduallyVariedFlow,
    EnergyEquation,
    MomentumEquation,
)

from .design import (
    ChannelDesigner,
    OptimalSections,
    EconomicSections,
)

from .constants import (
    GRAVITY,
    WATER_DENSITY,
    KINEMATIC_VISCOSITY,
    CONVERSION_FACTORS,
)

from .exceptions import (
    PyOpenChannelError,
    InvalidGeometryError,
    ConvergenceError,
    InvalidFlowConditionError,
)

from .units import (
    Units,
    UnitSystem,
    UnitConverter,
    set_unit_system,
    get_unit_system,
    get_gravity,
    get_manning_factor,
    ft_to_m,
    m_to_ft,
    cfs_to_cms,
    cms_to_cfs,
    gpm_to_cms,
    cms_to_gpm,
)

from .gvf import (
    GVFSolver,
    GVFEquation,
    GVFResult,
    ProfilePoint,
    BoundaryType,
    FlowRegime,
    ProfileType,
    ProfileClassifier,
    WaterSurfaceProfile,
    ProfileAnalyzer,
    SlopeType,
    ProfileCharacteristics,
    DamAnalysis,
    BridgeAnalysis,
    ChuteAnalysis,
    ChannelTransition,
    AnalysisResult,
    AnalysisType,
    DesignCriteria,
)

# RVF (Rapidly Varied Flow) Components
from .rvf import (
    RVFSolver,
    RVFAnalyzer,
    RVFResult,
    RVFRegime,
    JumpType,
    TransitionType,
    ShockProperties,
    GateFlowSolver,
    GateFlowAnalyzer,
    GateFlowResult,
    GateGeometry,
    GateType,
    FlowCondition,
    CavitationRisk,
    WeirFlowSolver,
    WeirFlowAnalyzer,
    WeirFlowResult,
    WeirGeometry,
    WeirType,
    WeirCondition,
    AerationLevel,
)

__all__ = [
    # Geometry classes
    "RectangularChannel",
    "TrapezoidalChannel", 
    "TriangularChannel",
    "CircularChannel",
    "ParabolicChannel",
    
    # Hydraulics classes
    "ManningEquation",
    "ChezyEquation",
    "HydraulicRadius",
    "CriticalDepth",
    "NormalDepth",
    
    # Flow analysis classes
    "UniformFlow",
    "CriticalFlow", 
    "GraduallyVariedFlow",
    "EnergyEquation",
    "MomentumEquation",
    
    # Design classes
    "ChannelDesigner",
    "OptimalSections",
    "EconomicSections",
    
    # Constants
    "GRAVITY",
    "WATER_DENSITY", 
    "KINEMATIC_VISCOSITY",
    "CONVERSION_FACTORS",
    
    # Exceptions
    "PyOpenChannelError",
    "InvalidGeometryError",
    "ConvergenceError", 
    "InvalidFlowConditionError",
    
    # Units
    "Units",
    "UnitSystem",
    "UnitConverter",
    "set_unit_system",
    "get_unit_system",
    "get_gravity",
    "get_manning_factor",
    "ft_to_m",
    "m_to_ft",
    "cfs_to_cms",
    "cms_to_cfs",
    "gpm_to_cms",
    "cms_to_gpm",
    
    # GVF (Gradually Varied Flow)
    "GVFSolver",
    "GVFEquation",
    "GVFResult",
    "ProfilePoint",
    "BoundaryType",
    "FlowRegime",
    "ProfileType",
    "ProfileClassifier",
    "WaterSurfaceProfile",
    "ProfileAnalyzer",
    "SlopeType",
    "ProfileCharacteristics",
    
    # GVF Applications
    "DamAnalysis",
    "BridgeAnalysis",
    "ChuteAnalysis",
    "ChannelTransition",
    "AnalysisResult",
    "AnalysisType",
    "DesignCriteria",
    
    # RVF Components
    "RVFSolver",
    "RVFAnalyzer",
    "RVFResult",
    "RVFRegime",
    "JumpType",
    "TransitionType",
    "ShockProperties",
    
    # Gate Flow Components
    "GateFlowSolver",
    "GateFlowAnalyzer",
    "GateFlowResult",
    "GateGeometry",
    "GateType",
    "FlowCondition",
    "CavitationRisk",
    
    # Weir Flow Components
    "WeirFlowSolver",
    "WeirFlowAnalyzer",
    "WeirFlowResult",
    "WeirGeometry",
    "WeirType",
    "WeirCondition",
    "AerationLevel",
]


def main() -> None:
    """Main entry point for the package."""
    print("PyOpenChannel - Open Channel Flow Analysis Library")
    print(f"Version: {__version__}")
    print("Use 'import pyopenchannel' to get started!")