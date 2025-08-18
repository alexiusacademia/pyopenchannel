"""
File: numerical/fvm/grid.py
Author: Alexius Academia
Date: 2025-08-17

Grid generation and management for FVM.

This module provides:
- Uniform grid generation
- Adaptive mesh refinement
- Grid quality assessment
- Grid coarsening and refinement
"""

import numpy as np
import math
from typing import List, Dict, Optional, Tuple, Callable, Any
from dataclasses import dataclass
from enum import Enum

from .core import FVMGrid, FVMCell, CellType, ConservativeVariables
from ...exceptions import InvalidFlowConditionError


class RefinementCriterion(Enum):
    """Criteria for mesh refinement."""
    GRADIENT_BASED = "gradient_based"
    CURVATURE_BASED = "curvature_based"
    ERROR_BASED = "error_based"
    SHOCK_DETECTION = "shock_detection"
    USER_DEFINED = "user_defined"


@dataclass
class RefinementParameters:
    """Parameters controlling mesh refinement."""
    max_refinement_level: int = 3
    min_cells: int = 50
    max_cells: int = 2000
    refinement_threshold: float = 0.1
    coarsening_threshold: float = 0.01
    refinement_ratio: int = 2
    buffer_cells: int = 2


class GridQualityMetrics:
    """
    Grid quality assessment tools.
    
    Provides metrics to evaluate grid quality and identify
    areas needing refinement or coarsening.
    """
    
    @staticmethod
    def calculate_aspect_ratio(grid: FVMGrid) -> float:
        """Calculate maximum aspect ratio."""
        # For 1D grid, aspect ratio is always 1
        return 1.0
    
    @staticmethod
    def calculate_skewness(grid: FVMGrid) -> float:
        """Calculate grid skewness."""
        # For uniform 1D grid, skewness is 0
        return 0.0
    
    @staticmethod
    def calculate_smoothness(grid: FVMGrid) -> float:
        """Calculate grid smoothness (cell size variation)."""
        if grid.num_cells < 2:
            return 1.0
        
        cell_sizes = [cell.dx for cell in grid.cells]
        max_size = max(cell_sizes)
        min_size = min(cell_sizes)
        
        return min_size / max_size if max_size > 0 else 1.0
    
    @staticmethod
    def calculate_orthogonality(grid: FVMGrid) -> float:
        """Calculate grid orthogonality."""
        # For 1D Cartesian grid, orthogonality is perfect
        return 1.0
    
    @staticmethod
    def assess_overall_quality(grid: FVMGrid) -> Dict[str, float]:
        """Assess overall grid quality."""
        return {
            'aspect_ratio': GridQualityMetrics.calculate_aspect_ratio(grid),
            'skewness': GridQualityMetrics.calculate_skewness(grid),
            'smoothness': GridQualityMetrics.calculate_smoothness(grid),
            'orthogonality': GridQualityMetrics.calculate_orthogonality(grid),
            'num_cells': grid.num_cells,
            'min_cell_size': min(cell.dx for cell in grid.cells),
            'max_cell_size': max(cell.dx for cell in grid.cells)
        }


class UniformGrid(FVMGrid):
    """
    Uniform 1D finite volume grid.
    
    Simple uniform spacing - good for initial implementation
    and cases without strong gradients.
    """
    
    def __init__(
        self,
        x_min: float,
        x_max: float,
        num_cells: int,
        bed_elevations: Optional[np.ndarray] = None,
        manning_n: float = 0.030
    ):
        """Initialize uniform grid."""
        super().__init__(x_min, x_max, num_cells, bed_elevations, manning_n)
        self.grid_type = "uniform"
    
    def refine_uniformly(self, refinement_factor: int = 2) -> 'UniformGrid':
        """Create uniformly refined grid."""
        new_num_cells = self.num_cells * refinement_factor
        
        # Interpolate bed elevations if present
        new_bed_elevations = None
        if any(cell.bed_elevation != 0.0 for cell in self.cells):
            old_x = self.get_x_coordinates()
            old_bed = np.array([cell.bed_elevation for cell in self.cells])
            
            new_x = np.linspace(self.x_min, self.x_max, new_num_cells)
            new_bed_elevations = np.interp(new_x, old_x, old_bed)
        
        return UniformGrid(
            self.x_min, self.x_max, new_num_cells, 
            new_bed_elevations, self.manning_n
        )


class AdaptiveGrid(FVMGrid):
    """
    Adaptive 1D finite volume grid.
    
    Supports local refinement and coarsening based on solution
    gradients and user-defined criteria.
    """
    
    def __init__(
        self,
        x_min: float,
        x_max: float,
        initial_num_cells: int,
        refinement_params: Optional[RefinementParameters] = None,
        bed_elevations: Optional[np.ndarray] = None,
        manning_n: float = 0.030
    ):
        """Initialize adaptive grid."""
        super().__init__(x_min, x_max, initial_num_cells, bed_elevations, manning_n)
        self.grid_type = "adaptive"
        self.refinement_params = refinement_params or RefinementParameters()
        self.refinement_history: List[Dict] = []
        
        # Track refinement levels
        self.cell_levels = [0] * self.num_cells
        self.max_level = 0
    
    def identify_refinement_cells(
        self, 
        criterion: RefinementCriterion = RefinementCriterion.GRADIENT_BASED
    ) -> List[int]:
        """
        Identify cells that need refinement.
        
        Args:
            criterion: Refinement criterion to use
            
        Returns:
            List of cell indices to refine
        """
        if criterion == RefinementCriterion.GRADIENT_BASED:
            return self._gradient_based_refinement()
        elif criterion == RefinementCriterion.SHOCK_DETECTION:
            return self._shock_detection_refinement()
        elif criterion == RefinementCriterion.CURVATURE_BASED:
            return self._curvature_based_refinement()
        else:
            return []
    
    def _gradient_based_refinement(self) -> List[int]:
        """Identify cells for refinement based on solution gradients."""
        refine_cells = []
        
        if self.num_cells < 3:
            return refine_cells
        
        # Calculate gradients
        depths = self.get_depths()
        velocities = self.get_velocities()
        
        for i in range(1, self.num_cells - 1):
            # Depth gradient
            depth_grad = abs(depths[i+1] - depths[i-1]) / (2 * self.cells[i].dx)
            depth_normalized = depth_grad / (depths[i] + 1e-12)
            
            # Velocity gradient
            vel_grad = abs(velocities[i+1] - velocities[i-1]) / (2 * self.cells[i].dx)
            vel_normalized = vel_grad / (abs(velocities[i]) + 1e-12)
            
            # Refinement criterion
            if (depth_normalized > self.refinement_params.refinement_threshold or
                vel_normalized > self.refinement_params.refinement_threshold):
                
                if self.cell_levels[i] < self.refinement_params.max_refinement_level:
                    refine_cells.append(i)
        
        return refine_cells
    
    def _shock_detection_refinement(self) -> List[int]:
        """Identify cells for refinement based on shock detection."""
        refine_cells = []
        
        if self.num_cells < 3:
            return refine_cells
        
        froude_numbers = self.get_froude_numbers()
        
        for i in range(1, self.num_cells - 1):
            # Check for supercritical to subcritical transition (shock)
            if (froude_numbers[i-1] > 1.1 and froude_numbers[i+1] < 0.9):
                # Potential shock location
                if self.cell_levels[i] < self.refinement_params.max_refinement_level:
                    # Refine this cell and neighbors
                    for j in range(max(0, i-1), min(self.num_cells, i+2)):
                        if j not in refine_cells:
                            refine_cells.append(j)
        
        return refine_cells
    
    def _curvature_based_refinement(self) -> List[int]:
        """Identify cells for refinement based on solution curvature."""
        refine_cells = []
        
        if self.num_cells < 4:
            return refine_cells
        
        depths = self.get_depths()
        
        for i in range(2, self.num_cells - 2):
            # Second derivative (curvature) approximation
            d2h_dx2 = (depths[i+1] - 2*depths[i] + depths[i-1]) / (self.cells[i].dx**2)
            curvature = abs(d2h_dx2) / (depths[i] + 1e-12)
            
            if curvature > self.refinement_params.refinement_threshold:
                if self.cell_levels[i] < self.refinement_params.max_refinement_level:
                    refine_cells.append(i)
        
        return refine_cells
    
    def identify_coarsening_cells(self) -> List[int]:
        """Identify cells that can be coarsened."""
        coarsen_cells = []
        
        if self.num_cells < 6:  # Need minimum cells
            return coarsen_cells
        
        depths = self.get_depths()
        velocities = self.get_velocities()
        
        for i in range(2, self.num_cells - 2):
            # Only coarsen if not at base level
            if self.cell_levels[i] > 0:
                # Check if gradients are small
                depth_grad = abs(depths[i+1] - depths[i-1]) / (2 * self.cells[i].dx)
                depth_normalized = depth_grad / (depths[i] + 1e-12)
                
                vel_grad = abs(velocities[i+1] - velocities[i-1]) / (2 * self.cells[i].dx)
                vel_normalized = vel_grad / (abs(velocities[i]) + 1e-12)
                
                if (depth_normalized < self.refinement_params.coarsening_threshold and
                    vel_normalized < self.refinement_params.coarsening_threshold):
                    coarsen_cells.append(i)
        
        return coarsen_cells
    
    def refine_cells(self, cell_indices: List[int]) -> 'AdaptiveGrid':
        """
        Refine specified cells.
        
        Args:
            cell_indices: Indices of cells to refine
            
        Returns:
            New refined grid
        """
        if not cell_indices:
            return self
        
        # Create new cell list
        new_cells = []
        new_levels = []
        
        for i, cell in enumerate(self.cells):
            if i in cell_indices:
                # Split this cell into multiple cells
                refined_cells = self._split_cell(cell, self.refinement_params.refinement_ratio)
                new_cells.extend(refined_cells)
                
                # Update refinement levels
                new_level = self.cell_levels[i] + 1
                new_levels.extend([new_level] * len(refined_cells))
                
                self.max_level = max(self.max_level, new_level)
            else:
                # Keep original cell
                new_cells.append(cell)
                new_levels.append(self.cell_levels[i])
        
        # Create new grid
        new_grid = self._create_grid_from_cells(new_cells)
        new_grid.cell_levels = new_levels
        new_grid.max_level = self.max_level
        
        # Record refinement
        self.refinement_history.append({
            'type': 'refinement',
            'cells_refined': len(cell_indices),
            'total_cells': len(new_cells)
        })
        
        return new_grid
    
    def _split_cell(self, cell: FVMCell, ratio: int) -> List[FVMCell]:
        """Split a cell into multiple smaller cells."""
        new_cells = []
        new_dx = cell.dx / ratio
        
        for j in range(ratio):
            # New cell center
            x_offset = (j - (ratio - 1) / 2) * new_dx
            new_x_center = cell.x_center + x_offset
            
            # Create new cell
            new_cell = FVMCell(
                index=cell.index,  # Will be updated later
                x_center=new_x_center,
                dx=new_dx,
                cell_type=cell.cell_type,
                U=cell.U.copy(),
                bed_elevation=cell.bed_elevation,
                manning_n=cell.manning_n
            )
            
            new_cells.append(new_cell)
        
        return new_cells
    
    def _create_grid_from_cells(self, cells: List[FVMCell]) -> 'AdaptiveGrid':
        """Create new grid from list of cells."""
        # Sort cells by x-coordinate
        cells.sort(key=lambda c: c.x_center)
        
        # Update cell indices and types
        for i, cell in enumerate(cells):
            cell.index = i
            
            if i == 0:
                cell.cell_type = CellType.BOUNDARY_LEFT
            elif i == len(cells) - 1:
                cell.cell_type = CellType.BOUNDARY_RIGHT
            else:
                cell.cell_type = CellType.INTERIOR
        
        # Create new grid
        x_min = cells[0].x_center - 0.5 * cells[0].dx
        x_max = cells[-1].x_center + 0.5 * cells[-1].dx
        
        new_grid = AdaptiveGrid(
            x_min, x_max, len(cells), 
            self.refinement_params, None, self.manning_n
        )
        
        # Replace cells
        new_grid.cells = cells
        new_grid.num_cells = len(cells)
        
        return new_grid


class GridRefinement:
    """
    High-level grid refinement controller.
    
    Manages the refinement process including:
    - Criterion evaluation
    - Refinement decisions
    - Grid quality maintenance
    """
    
    def __init__(self, parameters: Optional[RefinementParameters] = None):
        """Initialize grid refinement controller."""
        self.parameters = parameters or RefinementParameters()
        self.refinement_history: List[Dict] = []
    
    def adapt_grid(
        self, 
        grid: AdaptiveGrid, 
        criterion: RefinementCriterion = RefinementCriterion.GRADIENT_BASED
    ) -> AdaptiveGrid:
        """
        Adapt grid based on solution and criteria.
        
        Args:
            grid: Current grid
            criterion: Refinement criterion
            
        Returns:
            Adapted grid
        """
        # Check if adaptation is needed
        if grid.num_cells >= self.parameters.max_cells:
            return grid
        
        # Identify cells for refinement
        refine_cells = grid.identify_refinement_cells(criterion)
        
        # Identify cells for coarsening
        coarsen_cells = grid.identify_coarsening_cells()
        
        # Apply refinement first
        if refine_cells and grid.num_cells < self.parameters.max_cells:
            grid = grid.refine_cells(refine_cells)
        
        # Apply coarsening if grid is too fine
        if coarsen_cells and grid.num_cells > self.parameters.min_cells:
            grid = self._coarsen_cells(grid, coarsen_cells)
        
        return grid
    
    def _coarsen_cells(self, grid: AdaptiveGrid, cell_indices: List[int]) -> AdaptiveGrid:
        """Coarsen specified cells (simplified implementation)."""
        # For now, return original grid
        # Full implementation would merge adjacent cells
        return grid
    
    def assess_refinement_quality(self, grid: AdaptiveGrid) -> Dict[str, Any]:
        """Assess quality of current refinement."""
        quality_metrics = GridQualityMetrics.assess_overall_quality(grid)
        
        return {
            'grid_quality': quality_metrics,
            'refinement_efficiency': self._calculate_refinement_efficiency(grid),
            'memory_usage': self._estimate_memory_usage(grid),
            'computational_cost': self._estimate_computational_cost(grid)
        }
    
    def _calculate_refinement_efficiency(self, grid: AdaptiveGrid) -> float:
        """Calculate refinement efficiency metric."""
        if not hasattr(grid, 'cell_levels'):
            return 1.0
        
        # Efficiency based on level distribution
        level_counts = {}
        for level in grid.cell_levels:
            level_counts[level] = level_counts.get(level, 0) + 1
        
        # Prefer balanced refinement
        total_cells = sum(level_counts.values())
        efficiency = 1.0
        
        for level, count in level_counts.items():
            fraction = count / total_cells
            if level > 0:
                efficiency *= (1.0 - 0.1 * level * fraction)
        
        return max(0.0, efficiency)
    
    def _estimate_memory_usage(self, grid: AdaptiveGrid) -> float:
        """Estimate memory usage in MB."""
        # Rough estimate: ~1KB per cell
        return grid.num_cells * 1e-3
    
    def _estimate_computational_cost(self, grid: AdaptiveGrid) -> float:
        """Estimate relative computational cost."""
        # Cost scales roughly linearly with number of cells
        return float(grid.num_cells) / 100.0  # Normalized to 100 cells = 1.0
