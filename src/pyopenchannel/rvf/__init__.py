"""
RVF (Rapidly Varied Flow) Module - PyOpenChannel

Author: Alexius Academia
Email: alexius.sayco.academia@gmail.com

This module provides comprehensive analysis tools for rapidly varied flow
conditions where flow depth changes rapidly over short distances.

Key Features:
- Hydraulic jump analysis (all types)
- Shock wave and surge analysis  
- Automatic GVF-RVF transition detection
- Energy-momentum balance methods
- Professional design recommendations

Components:
- RVFSolver: Core solver for rapid flow transitions
- HydraulicJump: Comprehensive jump analysis
- ShockWave: Wave propagation analysis
- RVFAnalyzer: High-level analysis tools
"""

# Core RVF components
from .core import (
    RVFSolver,
    RVFAnalyzer,
    RVFResult,
    RVFRegime,
    JumpType,
    TransitionType,
    ShockProperties
)

# Gate flow components
from .gates import (
    GateFlowSolver,
    GateFlowAnalyzer,
    GateFlowResult,
    GateGeometry,
    GateType,
    FlowCondition,
    CavitationRisk
)

# Weir flow components
from .weirs import (
    WeirFlowSolver,
    WeirFlowAnalyzer,
    WeirFlowResult,
    WeirGeometry,
    WeirType,
    WeirCondition,
    AerationLevel
)

# Version info
__version__ = "0.5.0-dev"

# Public API
__all__ = [
    # Core classes
    "RVFSolver",
    "RVFAnalyzer", 
    "RVFResult",
    
    # Gate flow classes
    "GateFlowSolver",
    "GateFlowAnalyzer",
    "GateFlowResult",
    "GateGeometry",
    
    # Weir flow classes
    "WeirFlowSolver",
    "WeirFlowAnalyzer", 
    "WeirFlowResult",
    "WeirGeometry",
    
    # Enums
    "RVFRegime",
    "JumpType", 
    "TransitionType",
    "GateType",
    "FlowCondition",
    "CavitationRisk",
    "WeirType",
    "WeirCondition",
    "AerationLevel",
    
    # Data structures
    "ShockProperties",
]
