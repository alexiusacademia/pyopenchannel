"""
File: numerical/fvm/core.py
Author: Alexius Academia
Date: 2025-08-17

Core FVM classes and data structures.

This module provides the fundamental building blocks for finite volume
method implementation:
- Grid cells and connectivity
- Conservative variables and fluxes
- Base classes for numerical schemes
- Result data structures
"""

import numpy as np
import math
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum

from ...exceptions import ConvergenceError, InvalidFlowConditionError
from ...units import get_gravity


class CellType(Enum):
    """Types of finite volume cells."""
    INTERIOR = "interior"
    BOUNDARY_LEFT = "boundary_left"
    BOUNDARY_RIGHT = "boundary_right"
    GHOST = "ghost"


class FluxType(Enum):
    """Types of numerical fluxes."""
    PHYSICAL = "physical"
    NUMERICAL = "numerical"
    DIFFUSIVE = "diffusive"


@dataclass
class ConservativeVariables:
    """
    Conservative variables for shallow water equations.
    
    U = [h, hu] where:
    h = water depth
    hu = momentum (depth × velocity)
    """
    h: float    # Water depth
    hu: float   # Momentum
    
    def __post_init__(self):
        """Validate conservative variables."""
        # Fix any negative depths due to numerical precision
        if self.h < 1e-12:
            self.h = 1e-12
    
    @property
    def u(self) -> float:
        """Velocity from momentum."""
        return self.hu / self.h if self.h > 1e-12 else 0.0
    
    @property
    def froude(self) -> float:
        """Froude number."""
        if self.h <= 1e-12:
            return 0.0
        return abs(self.u) / math.sqrt(get_gravity() * self.h)
    
    @property
    def is_supercritical(self) -> bool:
        """Check if flow is supercritical."""
        return self.froude > 1.0
    
    @property
    def is_subcritical(self) -> bool:
        """Check if flow is subcritical."""
        return self.froude < 1.0
    
    @property
    def is_critical(self) -> bool:
        """Check if flow is critical."""
        return abs(self.froude - 1.0) < 0.01
    
    def copy(self) -> 'ConservativeVariables':
        """Create a copy of conservative variables."""
        return ConservativeVariables(h=self.h, hu=self.hu)
    
    def __add__(self, other: 'ConservativeVariables') -> 'ConservativeVariables':
        """Add conservative variables."""
        return ConservativeVariables(
            h=self.h + other.h,
            hu=self.hu + other.hu
        )
    
    def __sub__(self, other: 'ConservativeVariables') -> 'ConservativeVariables':
        """Subtract conservative variables."""
        return ConservativeVariables(
            h=self.h - other.h,
            hu=self.hu - other.hu
        )
    
    def __mul__(self, scalar: float) -> 'ConservativeVariables':
        """Multiply by scalar."""
        return ConservativeVariables(
            h=self.h * scalar,
            hu=self.hu * scalar
        )
    
    def __rmul__(self, scalar: float) -> 'ConservativeVariables':
        """Right multiply by scalar."""
        return self.__mul__(scalar)


@dataclass
class FluxVector:
    """
    Flux vector for shallow water equations.
    
    F = [hu, hu² + gh²/2] where:
    First component = mass flux
    Second component = momentum flux
    """
    mass_flux: float      # hu
    momentum_flux: float  # hu² + gh²/2
    
    @classmethod
    def from_conservative(cls, U: ConservativeVariables) -> 'FluxVector':
        """Calculate flux from conservative variables."""
        g = get_gravity()
        
        mass_flux = U.hu
        momentum_flux = U.hu * U.u + 0.5 * g * U.h**2
        
        return cls(mass_flux=mass_flux, momentum_flux=momentum_flux)
    
    def copy(self) -> 'FluxVector':
        """Create a copy of flux vector."""
        return FluxVector(
            mass_flux=self.mass_flux,
            momentum_flux=self.momentum_flux
        )
    
    def __add__(self, other: 'FluxVector') -> 'FluxVector':
        """Add flux vectors."""
        return FluxVector(
            mass_flux=self.mass_flux + other.mass_flux,
            momentum_flux=self.momentum_flux + other.momentum_flux
        )
    
    def __sub__(self, other: 'FluxVector') -> 'FluxVector':
        """Subtract flux vectors."""
        return FluxVector(
            mass_flux=self.mass_flux - other.mass_flux,
            momentum_flux=self.momentum_flux - other.momentum_flux
        )
    
    def __mul__(self, scalar: float) -> 'FluxVector':
        """Multiply by scalar."""
        return FluxVector(
            mass_flux=self.mass_flux * scalar,
            momentum_flux=self.momentum_flux * scalar
        )


@dataclass
class SourceTerms:
    """
    Source terms for shallow water equations.
    
    S = [0, -gh(Sf + Sb)] where:
    First component = 0 (mass conservation)
    Second component = friction and bed slope terms
    """
    mass_source: float = 0.0      # Always zero for mass conservation
    momentum_source: float = 0.0  # -gh(Sf + Sb)
    
    @classmethod
    def from_friction_slope(
        cls, 
        U: ConservativeVariables, 
        friction_slope: float, 
        bed_slope: float = 0.0
    ) -> 'SourceTerms':
        """Calculate source terms from friction and bed slope."""
        g = get_gravity()
        
        # Momentum source: -gh(Sf + Sb)
        momentum_source = -g * U.h * (friction_slope + bed_slope)
        
        return cls(mass_source=0.0, momentum_source=momentum_source)


@dataclass
class FVMCell:
    """
    Finite volume cell with conservative variables and properties.
    """
    index: int                           # Cell index
    x_center: float                      # Cell center coordinate
    dx: float                           # Cell width
    cell_type: CellType                 # Cell type
    
    # Flow variables
    U: ConservativeVariables            # Conservative variables
    U_old: Optional[ConservativeVariables] = None  # Previous time step
    
    # Fluxes
    flux_left: Optional[FluxVector] = None   # Left face flux
    flux_right: Optional[FluxVector] = None  # Right face flux
    
    # Source terms
    source: Optional[SourceTerms] = None
    
    # Cell properties
    bed_elevation: float = 0.0
    manning_n: float = 0.030
    
    @property
    def x_left(self) -> float:
        """Left face coordinate."""
        return self.x_center - 0.5 * self.dx
    
    @property
    def x_right(self) -> float:
        """Right face coordinate."""
        return self.x_center + 0.5 * self.dx
    
    @property
    def water_elevation(self) -> float:
        """Water surface elevation."""
        return self.bed_elevation + self.U.h
    
    @property
    def specific_energy(self) -> float:
        """Specific energy."""
        return self.U.h + self.U.u**2 / (2 * get_gravity())
    
    def calculate_friction_slope(self) -> float:
        """Calculate friction slope using Manning's equation."""
        if self.U.h <= 1e-12 or abs(self.U.u) <= 1e-12:
            return 0.0
        
        # Manning's equation: Sf = n²u²/R^(4/3)
        # For wide rectangular channel: R ≈ h
        hydraulic_radius = self.U.h  # Simplified for wide channel
        
        # Get Manning factor for unit consistency
        from ...units import get_manning_factor
        manning_factor = get_manning_factor()
        
        friction_slope = (
            manning_factor * self.manning_n**2 * self.U.u**2 / 
            hydraulic_radius**(4/3)
        )
        
        return friction_slope
    
    def update_source_terms(self, bed_slope: float = 0.0):
        """Update source terms based on current flow state."""
        friction_slope = self.calculate_friction_slope()
        self.source = SourceTerms.from_friction_slope(
            self.U, friction_slope, bed_slope
        )
    
    def copy(self) -> 'FVMCell':
        """Create a copy of the cell."""
        return FVMCell(
            index=self.index,
            x_center=self.x_center,
            dx=self.dx,
            cell_type=self.cell_type,
            U=self.U.copy(),
            U_old=self.U_old.copy() if self.U_old else None,
            flux_left=self.flux_left.copy() if self.flux_left else None,
            flux_right=self.flux_right.copy() if self.flux_right else None,
            source=self.source,
            bed_elevation=self.bed_elevation,
            manning_n=self.manning_n
        )


class FVMGrid:
    """
    1D finite volume grid for shallow water equations.
    
    Manages cell connectivity, boundary conditions, and grid quality.
    """
    
    def __init__(
        self,
        x_min: float,
        x_max: float,
        num_cells: int,
        bed_elevations: Optional[np.ndarray] = None,
        manning_n: float = 0.030
    ):
        """
        Initialize FVM grid.
        
        Args:
            x_min: Minimum x coordinate
            x_max: Maximum x coordinate  
            num_cells: Number of cells
            bed_elevations: Bed elevation at each cell (optional)
            manning_n: Manning's roughness coefficient
        """
        self.x_min = x_min
        self.x_max = x_max
        self.num_cells = num_cells
        self.manning_n = manning_n
        
        # Calculate uniform grid spacing
        self.dx = (x_max - x_min) / num_cells
        
        # Create cells
        self.cells: List[FVMCell] = []
        self._create_cells(bed_elevations)
        
        # Grid quality metrics
        self.min_cell_size = self.dx
        self.max_cell_size = self.dx
        self.aspect_ratio = 1.0
        
    def _create_cells(self, bed_elevations: Optional[np.ndarray]):
        """Create finite volume cells."""
        for i in range(self.num_cells):
            # Cell center coordinate
            x_center = self.x_min + (i + 0.5) * self.dx
            
            # Determine cell type
            if i == 0:
                cell_type = CellType.BOUNDARY_LEFT
            elif i == self.num_cells - 1:
                cell_type = CellType.BOUNDARY_RIGHT
            else:
                cell_type = CellType.INTERIOR
            
            # Bed elevation
            bed_elev = bed_elevations[i] if bed_elevations is not None else 0.0
            
            # Initialize with zero flow
            U = ConservativeVariables(h=0.1, hu=0.0)  # Small initial depth
            
            # Create cell
            cell = FVMCell(
                index=i,
                x_center=x_center,
                dx=self.dx,
                cell_type=cell_type,
                U=U,
                bed_elevation=bed_elev,
                manning_n=self.manning_n
            )
            
            self.cells.append(cell)
    
    def get_cell(self, index: int) -> FVMCell:
        """Get cell by index."""
        if 0 <= index < self.num_cells:
            return self.cells[index]
        else:
            raise IndexError(f"Cell index {index} out of range [0, {self.num_cells-1}]")
    
    def get_interior_cells(self) -> List[FVMCell]:
        """Get all interior cells."""
        return [cell for cell in self.cells if cell.cell_type == CellType.INTERIOR]
    
    def get_boundary_cells(self) -> List[FVMCell]:
        """Get all boundary cells."""
        return [cell for cell in self.cells 
                if cell.cell_type in [CellType.BOUNDARY_LEFT, CellType.BOUNDARY_RIGHT]]
    
    def get_x_coordinates(self) -> np.ndarray:
        """Get cell center coordinates."""
        return np.array([cell.x_center for cell in self.cells])
    
    def get_depths(self) -> np.ndarray:
        """Get water depths."""
        return np.array([cell.U.h for cell in self.cells])
    
    def get_velocities(self) -> np.ndarray:
        """Get velocities."""
        return np.array([cell.U.u for cell in self.cells])
    
    def get_froude_numbers(self) -> np.ndarray:
        """Get Froude numbers."""
        return np.array([cell.U.froude for cell in self.cells])
    
    def get_water_elevations(self) -> np.ndarray:
        """Get water surface elevations."""
        return np.array([cell.water_elevation for cell in self.cells])
    
    def calculate_grid_quality(self) -> Dict[str, float]:
        """Calculate grid quality metrics."""
        cell_sizes = [cell.dx for cell in self.cells]
        
        return {
            'min_cell_size': min(cell_sizes),
            'max_cell_size': max(cell_sizes),
            'mean_cell_size': np.mean(cell_sizes),
            'cell_size_ratio': max(cell_sizes) / min(cell_sizes),
            'total_length': self.x_max - self.x_min,
            'num_cells': self.num_cells
        }


class FVMScheme(ABC):
    """
    Abstract base class for finite volume numerical schemes.
    
    Defines the interface for calculating numerical fluxes at cell faces.
    """
    
    def __init__(self, name: str):
        """Initialize scheme with name."""
        self.name = name
        self.gravity = get_gravity()
    
    @abstractmethod
    def calculate_flux(
        self, 
        U_left: ConservativeVariables, 
        U_right: ConservativeVariables
    ) -> FluxVector:
        """
        Calculate numerical flux at cell interface.
        
        Args:
            U_left: Conservative variables in left cell
            U_right: Conservative variables in right cell
            
        Returns:
            Numerical flux vector
        """
        pass
    
    def calculate_wave_speeds(
        self, 
        U: ConservativeVariables
    ) -> Tuple[float, float]:
        """
        Calculate characteristic wave speeds.
        
        Args:
            U: Conservative variables
            
        Returns:
            (lambda_minus, lambda_plus) - left and right wave speeds
        """
        if U.h <= 1e-12:
            return (0.0, 0.0)
        
        c = math.sqrt(self.gravity * U.h)  # Shallow water wave speed
        u = U.u
        
        lambda_minus = u - c
        lambda_plus = u + c
        
        return (lambda_minus, lambda_plus)
    
    def calculate_max_wave_speed(
        self, 
        U_left: ConservativeVariables, 
        U_right: ConservativeVariables
    ) -> float:
        """Calculate maximum wave speed for CFL condition."""
        lambda_L_minus, lambda_L_plus = self.calculate_wave_speeds(U_left)
        lambda_R_minus, lambda_R_plus = self.calculate_wave_speeds(U_right)
        
        return max(
            abs(lambda_L_minus), abs(lambda_L_plus),
            abs(lambda_R_minus), abs(lambda_R_plus)
        )


@dataclass
class FVMResult:
    """
    Result of FVM simulation.
    
    Contains the complete solution including flow field, convergence
    information, and computational metrics.
    """
    # Grid and solution
    grid: FVMGrid
    time_final: float
    
    # Solution arrays
    x_coordinates: np.ndarray
    depths: np.ndarray
    velocities: np.ndarray
    froude_numbers: np.ndarray
    water_elevations: np.ndarray
    
    # Convergence information
    converged: bool
    iterations: int
    final_residual: float
    
    # Computational metrics
    computation_time: float
    memory_usage: float
    
    # Additional properties
    properties: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def num_points(self) -> int:
        """Number of solution points."""
        return len(self.x_coordinates)
    
    @property
    def solution_quality(self) -> str:
        """Assess solution quality."""
        if not self.converged:
            return "Failed to converge"
        elif self.final_residual < 1e-8:
            return "Excellent"
        elif self.final_residual < 1e-6:
            return "Good"
        elif self.final_residual < 1e-4:
            return "Acceptable"
        else:
            return "Poor"
    
    def get_solution_at_x(self, x: float) -> Dict[str, float]:
        """Get solution at specific x coordinate using interpolation."""
        if x < self.x_coordinates[0] or x > self.x_coordinates[-1]:
            raise ValueError(f"x = {x} is outside solution domain")
        
        # Linear interpolation
        depth = np.interp(x, self.x_coordinates, self.depths)
        velocity = np.interp(x, self.x_coordinates, self.velocities)
        froude = np.interp(x, self.x_coordinates, self.froude_numbers)
        elevation = np.interp(x, self.x_coordinates, self.water_elevations)
        
        return {
            'depth': depth,
            'velocity': velocity,
            'froude': froude,
            'water_elevation': elevation,
            'specific_energy': depth + velocity**2 / (2 * self.grid.gravity)
        }
    
    def calculate_mass_conservation_error(self) -> float:
        """Calculate mass conservation error."""
        # For steady flow, mass flux should be constant
        mass_fluxes = self.depths * self.velocities
        
        if len(mass_fluxes) < 2:
            return 0.0
        
        mean_flux = np.mean(mass_fluxes)
        if abs(mean_flux) < 1e-12:
            return 0.0
        
        max_deviation = np.max(np.abs(mass_fluxes - mean_flux))
        return max_deviation / abs(mean_flux)
    
    def calculate_energy_loss(self) -> float:
        """Calculate total energy loss across domain."""
        if len(self.depths) < 2:
            return 0.0
        
        # Specific energy at upstream and downstream
        g = get_gravity()
        E_upstream = self.depths[0] + self.velocities[0]**2 / (2 * g)
        E_downstream = self.depths[-1] + self.velocities[-1]**2 / (2 * g)
        
        return E_upstream - E_downstream
