"""
File: numerical/fvm/boundary.py
Author: Alexius Academia
Date: 2025-08-17

Boundary condition implementation for FVM.

This module provides comprehensive boundary condition handling for
shallow water equations:
- Dirichlet (prescribed values)
- Neumann (prescribed gradients)
- Transmissive (non-reflecting outflow)
- Reflective (solid walls)
- Inlet/Outlet conditions
- Critical flow conditions
- Rating curve conditions

All boundary conditions maintain conservation properties and
handle subcritical/supercritical flow transitions properly.
"""

import numpy as np
import math
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum

from .core import (
    ConservativeVariables, 
    FluxVector, 
    FVMCell, 
    CellType,
    get_gravity
)
from ...exceptions import InvalidFlowConditionError, ConvergenceError


class BoundaryType(Enum):
    """Types of boundary conditions."""
    DIRICHLET = "dirichlet"           # Prescribed values
    NEUMANN = "neumann"               # Prescribed gradients
    TRANSMISSIVE = "transmissive"     # Non-reflecting outflow
    REFLECTIVE = "reflective"         # Solid wall
    INLET = "inlet"                   # Flow inlet
    OUTLET = "outlet"                 # Flow outlet
    CRITICAL = "critical"             # Critical flow condition
    RATING_CURVE = "rating_curve"     # Stage-discharge relation
    WEIR = "weir"                     # Weir overflow
    GATE = "gate"                     # Gate flow


class FlowDirection(Enum):
    """Flow direction at boundary."""
    INFLOW = "inflow"
    OUTFLOW = "outflow"
    UNKNOWN = "unknown"


@dataclass
class BoundaryData:
    """Data structure for boundary condition parameters."""
    # Primary variables
    depth: Optional[float] = None
    velocity: Optional[float] = None
    discharge: Optional[float] = None
    
    # Gradients (for Neumann BC)
    depth_gradient: Optional[float] = None
    velocity_gradient: Optional[float] = None
    
    # Special parameters
    critical_depth: Optional[float] = None
    normal_depth: Optional[float] = None
    water_elevation: Optional[float] = None
    
    # Rating curve parameters
    rating_curve: Optional[Callable[[float], float]] = None
    stage_discharge_data: Optional[Dict[str, List[float]]] = None
    
    # Structure parameters (weirs, gates)
    crest_elevation: Optional[float] = None
    gate_opening: Optional[float] = None
    discharge_coefficient: Optional[float] = None
    
    # Time-dependent parameters
    time_series: Optional[Dict[str, List[Tuple[float, float]]]] = None
    
    def validate(self, bc_type: BoundaryType) -> bool:
        """Validate boundary data for given BC type."""
        if bc_type == BoundaryType.DIRICHLET:
            return (self.depth is not None or self.velocity is not None or 
                    self.time_series is not None)
        elif bc_type == BoundaryType.INLET:
            return (self.discharge is not None or 
                    (self.depth is not None and self.velocity is not None) or
                    self.time_series is not None)
        elif bc_type == BoundaryType.OUTLET:
            return (self.depth is not None or self.discharge is not None or
                    self.time_series is not None)
        elif bc_type == BoundaryType.CRITICAL:
            return (self.discharge is not None or self.time_series is not None)
        elif bc_type == BoundaryType.RATING_CURVE:
            return self.rating_curve is not None or self.stage_discharge_data is not None
        else:
            return True


class BoundaryCondition(ABC):
    """
    Abstract base class for boundary conditions.
    
    Defines the interface for applying boundary conditions
    to ghost cells and calculating boundary fluxes.
    """
    
    def __init__(
        self, 
        bc_type: BoundaryType, 
        location: str,  # "left" or "right"
        data: BoundaryData
    ):
        """
        Initialize boundary condition.
        
        Args:
            bc_type: Type of boundary condition
            location: Boundary location ("left" or "right")
            data: Boundary condition data
        """
        self.bc_type = bc_type
        self.location = location
        self.data = data
        self.gravity = get_gravity()
        
        # Validate data
        if not data.validate(bc_type):
            raise InvalidFlowConditionError(
                f"Invalid data for {bc_type.value} boundary condition"
            )
    
    @abstractmethod
    def apply_boundary_condition(
        self, 
        interior_cell: FVMCell, 
        ghost_cell: Optional[FVMCell],
        time: float = 0.0
    ) -> ConservativeVariables:
        """
        Apply boundary condition to ghost cell.
        
        Args:
            interior_cell: Adjacent interior cell
            ghost_cell: Ghost cell to update
            time: Current time (for time-dependent BC)
            
        Returns:
            Conservative variables for ghost cell
        """
        pass
    
    @abstractmethod
    def calculate_boundary_flux(
        self, 
        interior_cell: FVMCell, 
        time: float = 0.0
    ) -> FluxVector:
        """
        Calculate flux at boundary.
        
        Args:
            interior_cell: Adjacent interior cell
            time: Current time
            
        Returns:
            Boundary flux vector
        """
        pass
    
    def determine_flow_direction(self, interior_cell: FVMCell) -> FlowDirection:
        """Determine flow direction at boundary."""
        if self.location == "left":
            # Left boundary: positive velocity = inflow
            return FlowDirection.INFLOW if interior_cell.U.u > 0 else FlowDirection.OUTFLOW
        else:
            # Right boundary: negative velocity = inflow
            return FlowDirection.INFLOW if interior_cell.U.u < 0 else FlowDirection.OUTFLOW
    
    def get_time_dependent_value(self, parameter: str, time: float) -> Optional[float]:
        """Get time-dependent parameter value."""
        if self.data.time_series is None or parameter not in self.data.time_series:
            return None
        
        time_data = self.data.time_series[parameter]
        if not time_data:
            return None
        
        # Linear interpolation
        times = [t for t, v in time_data]
        values = [v for t, v in time_data]
        
        return np.interp(time, times, values)


class DirichletBC(BoundaryCondition):
    """
    Dirichlet boundary condition - prescribed values.
    
    Directly specifies depth and/or velocity at the boundary.
    Simple but can cause reflection if not physically consistent.
    """
    
    def __init__(self, location: str, data: BoundaryData):
        """Initialize Dirichlet BC."""
        super().__init__(BoundaryType.DIRICHLET, location, data)
    
    def apply_boundary_condition(
        self, 
        interior_cell: FVMCell, 
        ghost_cell: Optional[FVMCell],
        time: float = 0.0
    ) -> ConservativeVariables:
        """Apply Dirichlet boundary condition."""
        # Get prescribed values (with time dependence if available)
        prescribed_depth = self.get_time_dependent_value("depth", time) or self.data.depth
        prescribed_velocity = self.get_time_dependent_value("velocity", time) or self.data.velocity
        
        # Use prescribed values or extrapolate from interior
        if prescribed_depth is not None:
            depth = prescribed_depth
        else:
            depth = interior_cell.U.h
        
        if prescribed_velocity is not None:
            velocity = prescribed_velocity
        else:
            velocity = interior_cell.U.u
        
        # Ensure positive depth
        depth = max(depth, 1e-12)
        
        return ConservativeVariables(h=depth, hu=depth * velocity)
    
    def calculate_boundary_flux(
        self, 
        interior_cell: FVMCell, 
        time: float = 0.0
    ) -> FluxVector:
        """Calculate boundary flux for Dirichlet BC."""
        # Create ghost cell state
        ghost_U = self.apply_boundary_condition(interior_cell, None, time)
        
        # Use physical flux at boundary
        return FluxVector.from_conservative(ghost_U)


class NeumannBC(BoundaryCondition):
    """
    Neumann boundary condition - prescribed gradients.
    
    Specifies gradients of depth and/or velocity at boundary.
    Good for natural boundaries where gradients are known.
    """
    
    def __init__(self, location: str, data: BoundaryData):
        """Initialize Neumann BC."""
        super().__init__(BoundaryType.NEUMANN, location, data)
    
    def apply_boundary_condition(
        self, 
        interior_cell: FVMCell, 
        ghost_cell: Optional[FVMCell],
        time: float = 0.0
    ) -> ConservativeVariables:
        """Apply Neumann boundary condition."""
        # Distance from interior to ghost cell center
        # Use cell width as default distance if ghost cell not provided
        if ghost_cell is not None:
            dx = abs(ghost_cell.x_center - interior_cell.x_center)
        else:
            dx = interior_cell.dx  # Use cell width as default
        
        # Apply gradients
        if self.data.depth_gradient is not None:
            if self.location == "left":
                depth = interior_cell.U.h - self.data.depth_gradient * dx
            else:
                depth = interior_cell.U.h + self.data.depth_gradient * dx
        else:
            depth = interior_cell.U.h
        
        if self.data.velocity_gradient is not None:
            if self.location == "left":
                velocity = interior_cell.U.u - self.data.velocity_gradient * dx
            else:
                velocity = interior_cell.U.u + self.data.velocity_gradient * dx
        else:
            velocity = interior_cell.U.u
        
        # Ensure positive depth
        depth = max(depth, 1e-12)
        
        return ConservativeVariables(h=depth, hu=depth * velocity)
    
    def calculate_boundary_flux(
        self, 
        interior_cell: FVMCell, 
        time: float = 0.0
    ) -> FluxVector:
        """Calculate boundary flux for Neumann BC."""
        # Create ghost cell state
        ghost_U = self.apply_boundary_condition(interior_cell, None, time)
        
        # Average flux between interior and ghost
        interior_flux = FluxVector.from_conservative(interior_cell.U)
        ghost_flux = FluxVector.from_conservative(ghost_U)
        
        return FluxVector(
            mass_flux=0.5 * (interior_flux.mass_flux + ghost_flux.mass_flux),
            momentum_flux=0.5 * (interior_flux.momentum_flux + ghost_flux.momentum_flux)
        )


class TransmissiveBC(BoundaryCondition):
    """
    Transmissive boundary condition - non-reflecting outflow.
    
    Allows waves to pass through boundary without reflection.
    Excellent for outflow boundaries in supercritical flow.
    """
    
    def __init__(self, location: str):
        """Initialize transmissive BC."""
        data = BoundaryData()  # No specific data needed
        super().__init__(BoundaryType.TRANSMISSIVE, location, data)
    
    def apply_boundary_condition(
        self, 
        interior_cell: FVMCell, 
        ghost_cell: Optional[FVMCell],
        time: float = 0.0
    ) -> ConservativeVariables:
        """Apply transmissive boundary condition."""
        # Simply extrapolate from interior (zero gradient)
        return interior_cell.U.copy()
    
    def calculate_boundary_flux(
        self, 
        interior_cell: FVMCell, 
        time: float = 0.0
    ) -> FluxVector:
        """Calculate boundary flux for transmissive BC."""
        # Use interior flux directly
        return FluxVector.from_conservative(interior_cell.U)


class ReflectiveBC(BoundaryCondition):
    """
    Reflective boundary condition - solid wall.
    
    Reflects flow back into domain. Used for solid boundaries
    like channel walls or closed ends.
    """
    
    def __init__(self, location: str):
        """Initialize reflective BC."""
        data = BoundaryData()
        super().__init__(BoundaryType.REFLECTIVE, location, data)
    
    def apply_boundary_condition(
        self, 
        interior_cell: FVMCell, 
        ghost_cell: Optional[FVMCell],
        time: float = 0.0
    ) -> ConservativeVariables:
        """Apply reflective boundary condition."""
        # Reflect velocity, keep depth
        return ConservativeVariables(
            h=interior_cell.U.h,
            hu=-interior_cell.U.hu  # Reverse velocity
        )
    
    def calculate_boundary_flux(
        self, 
        interior_cell: FVMCell, 
        time: float = 0.0
    ) -> FluxVector:
        """Calculate boundary flux for reflective BC."""
        # Zero flux at solid wall
        return FluxVector(mass_flux=0.0, momentum_flux=0.0)


class InletBC(BoundaryCondition):
    """
    Inlet boundary condition.
    
    Handles flow entering the domain. Can specify discharge,
    depth, or both depending on flow regime.
    """
    
    def __init__(self, location: str, data: BoundaryData):
        """Initialize inlet BC."""
        super().__init__(BoundaryType.INLET, location, data)
    
    def apply_boundary_condition(
        self, 
        interior_cell: FVMCell, 
        ghost_cell: Optional[FVMCell],
        time: float = 0.0
    ) -> ConservativeVariables:
        """Apply inlet boundary condition."""
        # Get time-dependent values
        prescribed_discharge = self.get_time_dependent_value("discharge", time) or self.data.discharge
        prescribed_depth = self.get_time_dependent_value("depth", time) or self.data.depth
        prescribed_velocity = self.get_time_dependent_value("velocity", time) or self.data.velocity
        
        # Determine flow regime
        interior_froude = interior_cell.U.froude
        
        if prescribed_discharge is not None and prescribed_depth is not None:
            # Both specified - use directly
            depth = prescribed_depth
            velocity = prescribed_discharge / depth
        elif prescribed_discharge is not None:
            # Discharge specified - determine depth based on flow regime
            if interior_froude < 1.0:
                # Subcritical - depth controlled by downstream
                depth = interior_cell.U.h
                velocity = prescribed_discharge / depth
            else:
                # Supercritical - calculate critical depth
                depth = self._calculate_critical_depth(prescribed_discharge)
                velocity = prescribed_discharge / depth
        elif prescribed_depth is not None and prescribed_velocity is not None:
            # Depth and velocity specified
            depth = prescribed_depth
            velocity = prescribed_velocity
        else:
            # Fallback to interior values
            depth = interior_cell.U.h
            velocity = interior_cell.U.u
        
        # Ensure positive depth
        depth = max(depth, 1e-12)
        
        return ConservativeVariables(h=depth, hu=depth * velocity)
    
    def calculate_boundary_flux(
        self, 
        interior_cell: FVMCell, 
        time: float = 0.0
    ) -> FluxVector:
        """Calculate boundary flux for inlet BC."""
        ghost_U = self.apply_boundary_condition(interior_cell, None, time)
        return FluxVector.from_conservative(ghost_U)
    
    def _calculate_critical_depth(self, discharge: float) -> float:
        """Calculate critical depth for given discharge."""
        # For rectangular channel: hc = (Q²/g)^(1/3)
        # This is simplified - should use actual channel geometry
        if discharge <= 0:
            return 1e-12
        
        return (discharge**2 / self.gravity)**(1/3)


class OutletBC(BoundaryCondition):
    """
    Outlet boundary condition.
    
    Handles flow leaving the domain. Can specify depth
    (for subcritical) or allow free outflow (for supercritical).
    """
    
    def __init__(self, location: str, data: BoundaryData):
        """Initialize outlet BC."""
        super().__init__(BoundaryType.OUTLET, location, data)
    
    def apply_boundary_condition(
        self, 
        interior_cell: FVMCell, 
        ghost_cell: Optional[FVMCell],
        time: float = 0.0
    ) -> ConservativeVariables:
        """Apply outlet boundary condition."""
        # Get prescribed values
        prescribed_depth = self.get_time_dependent_value("depth", time) or self.data.depth
        prescribed_elevation = self.get_time_dependent_value("water_elevation", time) or self.data.water_elevation
        
        # Determine flow regime
        interior_froude = interior_cell.U.froude
        
        if interior_froude > 1.0:
            # Supercritical - information flows downstream, use interior values
            return interior_cell.U.copy()
        else:
            # Subcritical - downstream controls depth
            if prescribed_depth is not None:
                depth = prescribed_depth
            elif prescribed_elevation is not None:
                # Convert elevation to depth (need bed elevation)
                bed_elevation = getattr(interior_cell, 'bed_elevation', 0.0)
                depth = prescribed_elevation - bed_elevation
            else:
                # Use critical depth as default
                discharge = interior_cell.U.hu
                depth = self._calculate_critical_depth(abs(discharge))
            
            # Velocity from continuity
            velocity = interior_cell.U.hu / depth if depth > 1e-12 else 0.0
            
            # Ensure positive depth
            depth = max(depth, 1e-12)
            
            return ConservativeVariables(h=depth, hu=depth * velocity)
    
    def calculate_boundary_flux(
        self, 
        interior_cell: FVMCell, 
        time: float = 0.0
    ) -> FluxVector:
        """Calculate boundary flux for outlet BC."""
        ghost_U = self.apply_boundary_condition(interior_cell, None, time)
        return FluxVector.from_conservative(ghost_U)
    
    def _calculate_critical_depth(self, discharge: float) -> float:
        """Calculate critical depth for given discharge."""
        if discharge <= 0:
            return 1e-12
        return (discharge**2 / self.gravity)**(1/3)


class CriticalBC(BoundaryCondition):
    """
    Critical flow boundary condition.
    
    Enforces critical flow condition at boundary.
    Useful for weirs, spillways, and channel contractions.
    """
    
    def __init__(self, location: str, data: BoundaryData):
        """Initialize critical BC."""
        super().__init__(BoundaryType.CRITICAL, location, data)
    
    def apply_boundary_condition(
        self, 
        interior_cell: FVMCell, 
        ghost_cell: Optional[FVMCell],
        time: float = 0.0
    ) -> ConservativeVariables:
        """Apply critical flow boundary condition."""
        # Get discharge
        prescribed_discharge = self.get_time_dependent_value("discharge", time) or self.data.discharge
        
        if prescribed_discharge is not None:
            discharge = prescribed_discharge
        else:
            discharge = abs(interior_cell.U.hu)
        
        # Calculate critical depth
        critical_depth = self._calculate_critical_depth(discharge)
        
        # Critical velocity
        critical_velocity = discharge / critical_depth if critical_depth > 1e-12 else 0.0
        
        return ConservativeVariables(h=critical_depth, hu=critical_depth * critical_velocity)
    
    def calculate_boundary_flux(
        self, 
        interior_cell: FVMCell, 
        time: float = 0.0
    ) -> FluxVector:
        """Calculate boundary flux for critical BC."""
        ghost_U = self.apply_boundary_condition(interior_cell, None, time)
        return FluxVector.from_conservative(ghost_U)
    
    def _calculate_critical_depth(self, discharge: float) -> float:
        """Calculate critical depth for given discharge."""
        if discharge <= 0:
            return 1e-12
        return (discharge**2 / self.gravity)**(1/3)


class RatingCurveBC(BoundaryCondition):
    """
    Rating curve boundary condition.
    
    Uses stage-discharge relationship to determine boundary conditions.
    Common for natural channels and calibrated structures.
    """
    
    def __init__(self, location: str, data: BoundaryData):
        """Initialize rating curve BC."""
        super().__init__(BoundaryType.RATING_CURVE, location, data)
        
        # Create interpolation function if data provided
        if data.stage_discharge_data is not None:
            stages = data.stage_discharge_data["stages"]
            discharges = data.stage_discharge_data["discharges"]
            self.rating_function = lambda h: np.interp(h, stages, discharges)
        elif data.rating_curve is not None:
            self.rating_function = data.rating_curve
        else:
            raise InvalidFlowConditionError("Rating curve data or function required")
    
    def apply_boundary_condition(
        self, 
        interior_cell: FVMCell, 
        ghost_cell: Optional[FVMCell],
        time: float = 0.0
    ) -> ConservativeVariables:
        """Apply rating curve boundary condition."""
        # Use interior depth to get discharge from rating curve
        depth = interior_cell.U.h
        discharge = self.rating_function(depth)
        
        # Calculate velocity
        velocity = discharge / depth if depth > 1e-12 else 0.0
        
        return ConservativeVariables(h=depth, hu=depth * velocity)
    
    def calculate_boundary_flux(
        self, 
        interior_cell: FVMCell, 
        time: float = 0.0
    ) -> FluxVector:
        """Calculate boundary flux for rating curve BC."""
        ghost_U = self.apply_boundary_condition(interior_cell, None, time)
        return FluxVector.from_conservative(ghost_U)


class BoundaryManager:
    """
    Manages all boundary conditions for FVM simulation.
    
    Coordinates application of boundary conditions and
    ensures consistency across the domain.
    """
    
    def __init__(self):
        """Initialize boundary manager."""
        self.left_bc: Optional[BoundaryCondition] = None
        self.right_bc: Optional[BoundaryCondition] = None
        self.bc_history: List[Dict[str, Any]] = []
    
    def set_left_boundary(self, bc: BoundaryCondition):
        """Set left boundary condition."""
        if bc.location != "left":
            raise ValueError("Boundary condition location must be 'left'")
        self.left_bc = bc
    
    def set_right_boundary(self, bc: BoundaryCondition):
        """Set right boundary condition."""
        if bc.location != "right":
            raise ValueError("Boundary condition location must be 'right'")
        self.right_bc = bc
    
    def apply_all_boundaries(
        self, 
        cells: List[FVMCell], 
        time: float = 0.0
    ) -> Tuple[ConservativeVariables, ConservativeVariables]:
        """
        Apply all boundary conditions.
        
        Args:
            cells: List of all cells (including boundary cells)
            time: Current time
            
        Returns:
            (left_ghost_state, right_ghost_state)
        """
        if len(cells) < 2:
            raise ValueError("Need at least 2 cells for boundary conditions")
        
        # Left boundary
        left_ghost_state = None
        if self.left_bc is not None:
            left_interior = cells[0]  # First cell is left boundary
            left_ghost_state = self.left_bc.apply_boundary_condition(
                left_interior, None, time
            )
        
        # Right boundary
        right_ghost_state = None
        if self.right_bc is not None:
            right_interior = cells[-1]  # Last cell is right boundary
            right_ghost_state = self.right_bc.apply_boundary_condition(
                right_interior, None, time
            )
        
        return left_ghost_state, right_ghost_state
    
    def calculate_boundary_fluxes(
        self, 
        cells: List[FVMCell], 
        time: float = 0.0
    ) -> Tuple[Optional[FluxVector], Optional[FluxVector]]:
        """
        Calculate fluxes at boundaries.
        
        Args:
            cells: List of all cells
            time: Current time
            
        Returns:
            (left_boundary_flux, right_boundary_flux)
        """
        left_flux = None
        right_flux = None
        
        if self.left_bc is not None and len(cells) > 0:
            left_flux = self.left_bc.calculate_boundary_flux(cells[0], time)
        
        if self.right_bc is not None and len(cells) > 0:
            right_flux = self.right_bc.calculate_boundary_flux(cells[-1], time)
        
        return left_flux, right_flux
    
    def validate_boundary_setup(self) -> List[str]:
        """Validate boundary condition setup."""
        warnings = []
        
        if self.left_bc is None:
            warnings.append("No left boundary condition specified")
        
        if self.right_bc is None:
            warnings.append("No right boundary condition specified")
        
        # Check for potential issues
        if (self.left_bc and self.right_bc and 
            self.left_bc.bc_type == BoundaryType.DIRICHLET and
            self.right_bc.bc_type == BoundaryType.DIRICHLET):
            warnings.append("Both boundaries are Dirichlet - may cause over-specification")
        
        return warnings
    
    def get_boundary_summary(self) -> Dict[str, Any]:
        """Get summary of boundary conditions."""
        return {
            'left_bc': {
                'type': self.left_bc.bc_type.value if self.left_bc else None,
                'data': self.left_bc.data if self.left_bc else None
            },
            'right_bc': {
                'type': self.right_bc.bc_type.value if self.right_bc else None,
                'data': self.right_bc.data if self.right_bc else None
            },
            'validation_warnings': self.validate_boundary_setup()
        }
