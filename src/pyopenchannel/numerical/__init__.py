"""
File: numerical/__init__.py
Author: Alexius Academia
Date: 2025-08-17

High-accuracy numerical methods for hydraulic computations.

This module provides robust numerical methods optimized for accuracy:
- Runge-Kutta integration methods (RK4, RK45, Dormand-Prince)
- Adaptive step size control
- Finite difference methods
- Root finding algorithms
- Error estimation and control

Designed for gradually varied flow analysis, backwater computations,
and other hydraulic applications requiring high precision.
"""

from .integration import (
    RungeKutta4,
    RungeKuttaFehlberg45,
    DormandPrince,
    AdaptiveIntegrator,
    IntegrationResult,
    EventType,
    Event,
    EventDetector,
    AnalyticalValidator,
)

from .finite_difference import (
    FiniteDifferenceMethod,
    CentralDifference,
    BackwardDifference,
    ForwardDifference,
)

# Root finding methods (to be implemented)
# from .root_finding import (
#     NewtonRaphson,
#     Bisection,
#     Secant,
#     BrentMethod,
# )

__all__ = [
    # Integration methods
    "RungeKutta4",
    "RungeKuttaFehlberg45", 
    "DormandPrince",
    "AdaptiveIntegrator",
    "IntegrationResult",
    
    # Event detection and validation
    "EventType",
    "Event", 
    "EventDetector",
    "AnalyticalValidator",
    
    # Finite difference methods
    "FiniteDifferenceMethod",
    "CentralDifference",
    "BackwardDifference", 
    "ForwardDifference",
    
    # Root finding methods (to be implemented)
    # "NewtonRaphson",
    # "Bisection", 
    # "Secant",
    # "BrentMethod",
]
