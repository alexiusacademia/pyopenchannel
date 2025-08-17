"""
File: gvf/__init__.py
Author: Alexius Academia
Date: 2025-08-17

Gradually Varied Flow (GVF) analysis module.

This module provides comprehensive tools for gradually varied flow analysis:
- GVF differential equation solver
- Water surface profile computation
- Profile classification (M1, M2, S1, S2, etc.)
- Backwater analysis
- Critical depth transitions
- Dam and bridge hydraulics
- Channel transitions

The core GVF equation solved is:
dy/dx = (S₀ - Sf) / (1 - Fr²)

Where:
- y = flow depth
- x = distance along channel
- S₀ = channel slope
- Sf = friction slope (from Manning's equation)
- Fr = Froude number

Applications:
- Dam backwater analysis
- Bridge hydraulics
- Channel design
- Flood routing
- Environmental flow analysis
"""

from .solver import (
    GVFSolver,
    GVFEquation,
    GVFResult,
    ProfilePoint,
    BoundaryType,
    FlowRegime,
)

from .profiles import (
    ProfileType,
    ProfileClassifier,
    WaterSurfaceProfile,
    ProfileAnalyzer,
    SlopeType,
    ProfileCharacteristics,
)

# Applications (to be implemented)
from .applications import (
    DamAnalysis,
    BridgeAnalysis,
    ChuteAnalysis,
    ChannelTransition,
    AnalysisResult,
    AnalysisType,
    DesignCriteria,
)

__all__ = [
    # Core GVF solver
    "GVFSolver",
    "GVFEquation", 
    "GVFResult",
    "ProfilePoint",
    "BoundaryType",
    "FlowRegime",
    
    # Profile analysis
    "ProfileType",
    "ProfileClassifier",
    "WaterSurfaceProfile",
    "ProfileAnalyzer",
    "SlopeType",
    "ProfileCharacteristics",
    
    # Applications
    "DamAnalysis",
    "BridgeAnalysis", 
    "ChuteAnalysis",
    "ChannelTransition",
    "AnalysisResult",
    "AnalysisType",
    "DesignCriteria",
]
