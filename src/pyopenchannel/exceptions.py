"""
exceptions.py
Author: Alexius Academia
Date: 2025-08-17

Custom exceptions for the PyOpenChannel library.

This module contains the following exceptions:
- PyOpenChannelError
- InvalidGeometryError
- ConvergenceError
- InvalidFlowConditionError
- NegativeDepthError
- NegativeDischargeError
- SupercriticalFlowError
- SubcriticalFlowError
- InvalidRoughnessError
- InvalidSlopeError
- ChannelCapacityExceededError

The exceptions are used to handle errors and invalid input parameters in the open channel flow calculations.
"""


class PyOpenChannelError(Exception):
    """Base exception class for all PyOpenChannel errors."""
    pass


class InvalidGeometryError(PyOpenChannelError):
    """Raised when channel geometry parameters are invalid."""
    pass


class ConvergenceError(PyOpenChannelError):
    """Raised when iterative calculations fail to converge."""
    pass


class InvalidFlowConditionError(PyOpenChannelError):
    """Raised when flow conditions are physically impossible or invalid."""
    pass


class NegativeDepthError(InvalidFlowConditionError):
    """Raised when calculated depth is negative."""
    pass


class NegativeDischargeError(InvalidFlowConditionError):
    """Raised when calculated discharge is negative."""
    pass


class SupercriticalFlowError(InvalidFlowConditionError):
    """Raised when flow conditions indicate supercritical flow where not expected."""
    pass


class SubcriticalFlowError(InvalidFlowConditionError):
    """Raised when flow conditions indicate subcritical flow where not expected."""
    pass


class InvalidRoughnessError(PyOpenChannelError):
    """Raised when Manning's roughness coefficient is invalid."""
    pass


class InvalidSlopeError(PyOpenChannelError):
    """Raised when channel slope is invalid (negative or zero where positive required)."""
    pass


class ChannelCapacityExceededError(PyOpenChannelError):
    """Raised when flow exceeds channel capacity."""
    pass
