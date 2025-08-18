"""
File: numerical/fvm/__init__.py
Author: Alexius Academia
Date: 2025-08-17

Finite Volume Method (FVM) implementation for PyOpenChannel.

This module provides high-accuracy FVM solvers for rapidly varied flow
analysis, particularly hydraulic jumps and shock-dominated flows.

Features:
- 1D shallow water equation solver
- Shock-capturing numerical schemes (Roe, HLL, HLLC)
- Adaptive mesh refinement
- Conservative formulation
- Professional-grade accuracy

Applications:
- Detailed hydraulic jump analysis
- Pressure distribution for structure design
- Velocity profiles for scour analysis
- Turbulence modeling
- Air entrainment analysis
"""

from .core import (
    FVMGrid,
    FVMCell,
    FVMScheme,
    FVMResult,
    ConservativeVariables,
    FluxVector,
    SourceTerms,
)

from .schemes import (
    RoeScheme,
    HLLScheme,
    HLLCScheme,
    LaxFriedrichsScheme,
)

from .grid import (
    UniformGrid,
    AdaptiveGrid,
    GridRefinement,
    GridQualityMetrics,
)

from .boundary import (
    BoundaryCondition,
    BoundaryManager,
    BoundaryData,
    BoundaryType,
    DirichletBC,
    NeumannBC,
    TransmissiveBC,
    ReflectiveBC,
    InletBC,
    OutletBC,
    CriticalBC,
    RatingCurveBC,
)

from .solver import (
    FVMSolver,
    ShallowWaterSolver,
    TimeIntegrator,
    ConvergenceCriteria,
    SolutionMonitor,
    TimeIntegrationMethod,
    SolutionStatus,
)

from .validation import (
    AnalyticalSolution,
    ValidationSuite,
    ErrorMetrics,
    ConvergenceStudy,
    ValidationResult,
    ValidationCase,
    SteadyUniformFlowSolution,
    DamBreakSolution,
    RiemannProblemSolution,
    ManufacturedSolution,
)

__all__ = [
    # Core classes
    "FVMGrid",
    "FVMCell", 
    "FVMScheme",
    "FVMResult",
    "ConservativeVariables",
    "FluxVector",
    "SourceTerms",
    
    # Numerical schemes
    "RoeScheme",
    "HLLScheme", 
    "HLLCScheme",
    "LaxFriedrichsScheme",
    
    # Grid management
    "UniformGrid",
    "AdaptiveGrid",
    "GridRefinement",
    "GridQualityMetrics",
    
    # Boundary conditions
    "BoundaryCondition",
    "BoundaryManager",
    "BoundaryData",
    "BoundaryType",
    "DirichletBC",
    "NeumannBC", 
    "TransmissiveBC",
    "ReflectiveBC",
    "InletBC",
    "OutletBC",
    "CriticalBC",
    "RatingCurveBC",
    
    # Solvers
    "FVMSolver",
    "ShallowWaterSolver", 
    "TimeIntegrator",
    "ConvergenceCriteria",
    "SolutionMonitor",
    "TimeIntegrationMethod",
    "SolutionStatus",
    
    # Validation
    "AnalyticalSolution",
    "ValidationSuite",
    "ErrorMetrics",
    "ConvergenceStudy",
    "ValidationResult",
    "ValidationCase",
    "SteadyUniformFlowSolution",
    "DamBreakSolution",
    "RiemannProblemSolution",
    "ManufacturedSolution",
]
