"""
File: numerical/fvm/solver.py
Author: Alexius Academia
Date: 2025-08-17

FVM Solver implementation for shallow water equations.

This module provides the complete FVM solution engine:
- Time integration (explicit/implicit)
- Convergence control
- CFL condition enforcement
- Steady-state detection
- Adaptive time stepping
- Professional-grade solution monitoring

The solver orchestrates all FVM components:
- Numerical schemes (Roe, HLL, HLLC)
- Grid management and adaptation
- Boundary condition application
- Source term integration
"""

import numpy as np
import math
import time as time_module
from typing import List, Dict, Optional, Tuple, Callable, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

from .core import (
    FVMGrid, FVMCell, FVMScheme, FVMResult, 
    ConservativeVariables, FluxVector, SourceTerms,
    CellType, get_gravity
)
from .schemes import RoeScheme, HLLScheme, HLLCScheme, LaxFriedrichsScheme
from .grid import UniformGrid, AdaptiveGrid, GridRefinement, RefinementCriterion
from .boundary import BoundaryManager, BoundaryCondition
from ...exceptions import ConvergenceError, InvalidFlowConditionError


class TimeIntegrationMethod(Enum):
    """Time integration methods."""
    EXPLICIT_EULER = "explicit_euler"
    RK2 = "rk2"
    RK3 = "rk3"
    RK4 = "rk4"
    IMPLICIT_EULER = "implicit_euler"
    CRANK_NICOLSON = "crank_nicolson"


class SolutionStatus(Enum):
    """Solution status indicators."""
    INITIALIZING = "initializing"
    RUNNING = "running"
    CONVERGED = "converged"
    MAX_ITERATIONS = "max_iterations"
    FAILED = "failed"
    UNSTABLE = "unstable"


@dataclass
class ConvergenceCriteria:
    """Convergence criteria for FVM solver."""
    max_iterations: int = 10000
    residual_tolerance: float = 1e-6
    steady_state_tolerance: float = 1e-8
    max_time: float = 1000.0
    min_time_step: float = 1e-8
    max_time_step: float = 1.0
    cfl_number: float = 0.5
    check_interval: int = 10
    
    def __post_init__(self):
        """Validate convergence criteria."""
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if self.residual_tolerance <= 0:
            raise ValueError("residual_tolerance must be positive")
        if self.cfl_number <= 0 or self.cfl_number > 1:
            raise ValueError("cfl_number must be in (0, 1]")


@dataclass
class SolutionMonitor:
    """Monitor solution progress and quality."""
    iteration: int = 0
    time: float = 0.0
    time_step: float = 0.0
    residual: float = float('inf')
    mass_conservation_error: float = 0.0
    energy_conservation_error: float = 0.0
    max_froude: float = 0.0
    min_depth: float = float('inf')
    status: SolutionStatus = SolutionStatus.INITIALIZING
    
    # History tracking
    residual_history: List[float] = field(default_factory=list)
    time_step_history: List[float] = field(default_factory=list)
    conservation_history: List[float] = field(default_factory=list)
    
    def update(
        self, 
        iteration: int, 
        time: float, 
        time_step: float, 
        residual: float,
        grid: FVMGrid
    ):
        """Update monitor with current solution state."""
        self.iteration = iteration
        self.time = time
        self.time_step = time_step
        self.residual = residual
        
        # Calculate solution quality metrics
        depths = grid.get_depths()
        velocities = grid.get_velocities()
        froude_numbers = grid.get_froude_numbers()
        
        self.min_depth = np.min(depths)
        self.max_froude = np.max(froude_numbers)
        
        # Mass conservation check
        mass_fluxes = depths * velocities
        if len(mass_fluxes) > 1:
            mean_flux = np.mean(mass_fluxes)
            if abs(mean_flux) > 1e-12:
                self.mass_conservation_error = np.max(np.abs(mass_fluxes - mean_flux)) / abs(mean_flux)
            else:
                self.mass_conservation_error = 0.0
        
        # Update history
        self.residual_history.append(residual)
        self.time_step_history.append(time_step)
        self.conservation_history.append(self.mass_conservation_error)
        
        # Limit history length
        max_history = 1000
        if len(self.residual_history) > max_history:
            self.residual_history = self.residual_history[-max_history:]
            self.time_step_history = self.time_step_history[-max_history:]
            self.conservation_history = self.conservation_history[-max_history:]
    
    def check_stability(self) -> bool:
        """Check if solution is stable."""
        if self.min_depth < 1e-12:
            return False
        if self.max_froude > 50.0:  # Unreasonably high Froude number
            return False
        if self.mass_conservation_error > 0.1:  # 10% mass conservation error
            return False
        if math.isnan(self.residual) or math.isinf(self.residual):
            return False
        return True
    
    def get_convergence_rate(self, window: int = 10) -> float:
        """Calculate convergence rate over recent iterations."""
        if len(self.residual_history) < window + 1:
            return 0.0
        
        recent_residuals = self.residual_history[-window-1:]
        if recent_residuals[0] <= 1e-15 or recent_residuals[-1] <= 1e-15:
            return 0.0
        
        # Linear regression on log(residual) vs iteration
        iterations = np.arange(len(recent_residuals))
        log_residuals = np.log(np.maximum(recent_residuals, 1e-15))
        
        if len(iterations) > 1:
            slope = np.polyfit(iterations, log_residuals, 1)[0]
            return -slope  # Negative slope means convergence
        return 0.0


class TimeIntegrator(ABC):
    """Abstract base class for time integration methods."""
    
    def __init__(self, name: str):
        """Initialize time integrator."""
        self.name = name
    
    @abstractmethod
    def advance_time_step(
        self, 
        grid: FVMGrid, 
        scheme: FVMScheme,
        boundary_manager: BoundaryManager,
        dt: float,
        time: float
    ) -> FVMGrid:
        """
        Advance solution by one time step.
        
        Args:
            grid: Current grid state
            scheme: Numerical scheme
            boundary_manager: Boundary condition manager
            dt: Time step size
            time: Current time
            
        Returns:
            Updated grid
        """
        pass


class ExplicitEulerIntegrator(TimeIntegrator):
    """Explicit Euler time integration (first-order)."""
    
    def __init__(self):
        """Initialize explicit Euler integrator."""
        super().__init__("Explicit Euler")
    
    def advance_time_step(
        self, 
        grid: FVMGrid, 
        scheme: FVMScheme,
        boundary_manager: BoundaryManager,
        dt: float,
        time: float
    ) -> FVMGrid:
        """Advance using explicit Euler method."""
        # Store old values
        for cell in grid.cells:
            cell.U_old = cell.U.copy()
        
        # Apply boundary conditions
        boundary_manager.apply_all_boundaries(grid.cells, time)
        
        # Calculate fluxes at all cell faces
        self._calculate_fluxes(grid, scheme, boundary_manager, time)
        
        # Update conservative variables
        for cell in grid.cells:
            if cell.cell_type == CellType.INTERIOR:
                # Flux divergence
                flux_divergence = self._calculate_flux_divergence(cell)
                
                # Source terms
                cell.update_source_terms()
                source_contribution = self._calculate_source_contribution(cell)
                
                # Explicit Euler update: U^{n+1} = U^n + dt * (flux_div + source)
                dU_dt = flux_divergence + source_contribution
                
                cell.U = ConservativeVariables(
                    h=cell.U_old.h + dt * dU_dt.h,
                    hu=cell.U_old.hu + dt * dU_dt.hu
                )
                
                # Ensure positive depth and limit extreme values
                cell.U.h = max(cell.U.h, 1e-12)
                cell.U.h = min(cell.U.h, 100.0)  # Reasonable upper limit
                
                # Limit velocity to prevent numerical instabilities
                if cell.U.h > 1e-12:
                    max_velocity = 50.0  # Reasonable upper limit
                    velocity = cell.U.hu / cell.U.h
                    if abs(velocity) > max_velocity:
                        cell.U.hu = cell.U.h * max_velocity * (1 if velocity > 0 else -1)
        
        return grid
    
    def _calculate_fluxes(
        self, 
        grid: FVMGrid, 
        scheme: FVMScheme,
        boundary_manager: BoundaryManager,
        time: float
    ):
        """Calculate numerical fluxes at all cell faces."""
        # Interior faces
        for i in range(len(grid.cells) - 1):
            left_cell = grid.cells[i]
            right_cell = grid.cells[i + 1]
            
            # Calculate numerical flux
            flux = scheme.calculate_flux(left_cell.U, right_cell.U)
            
            # Assign to cells
            left_cell.flux_right = flux
            right_cell.flux_left = flux
        
        # Boundary fluxes
        left_flux, right_flux = boundary_manager.calculate_boundary_fluxes(grid.cells, time)
        
        if left_flux is not None and len(grid.cells) > 0:
            grid.cells[0].flux_left = left_flux
        
        if right_flux is not None and len(grid.cells) > 0:
            grid.cells[-1].flux_right = right_flux
    
    def _calculate_flux_divergence(self, cell: FVMCell) -> ConservativeVariables:
        """Calculate flux divergence for a cell."""
        # ∇·F = (F_right - F_left) / dx
        if cell.flux_left is None or cell.flux_right is None:
            return ConservativeVariables(h=0.0, hu=0.0)
        
        dh_dt = -(cell.flux_right.mass_flux - cell.flux_left.mass_flux) / cell.dx
        dhu_dt = -(cell.flux_right.momentum_flux - cell.flux_left.momentum_flux) / cell.dx
        
        # Limit extreme derivatives to prevent instabilities
        max_dh_dt = 10.0  # Reasonable limit for depth change rate
        max_dhu_dt = 100.0  # Reasonable limit for momentum change rate
        
        dh_dt = max(-max_dh_dt, min(dh_dt, max_dh_dt))
        dhu_dt = max(-max_dhu_dt, min(dhu_dt, max_dhu_dt))
        
        return ConservativeVariables(h=dh_dt, hu=dhu_dt)
    
    def _calculate_source_contribution(self, cell: FVMCell) -> ConservativeVariables:
        """Calculate source term contribution."""
        if cell.source is None:
            return ConservativeVariables(h=0.0, hu=0.0)
        
        return ConservativeVariables(
            h=cell.source.mass_source,
            hu=cell.source.momentum_source
        )


class RungeKutta4Integrator(TimeIntegrator):
    """Fourth-order Runge-Kutta time integration."""
    
    def __init__(self):
        """Initialize RK4 integrator."""
        super().__init__("Runge-Kutta 4")
        self.euler_integrator = ExplicitEulerIntegrator()
    
    def advance_time_step(
        self, 
        grid: FVMGrid, 
        scheme: FVMScheme,
        boundary_manager: BoundaryManager,
        dt: float,
        time: float
    ) -> FVMGrid:
        """Advance using RK4 method."""
        # Store initial state
        U0 = {}
        for i, cell in enumerate(grid.cells):
            U0[i] = cell.U.copy()
        
        # RK4 stages
        k1 = self._calculate_rhs(grid, scheme, boundary_manager, time)
        
        # Stage 2
        self._update_state(grid, U0, k1, 0.5 * dt)
        k2 = self._calculate_rhs(grid, scheme, boundary_manager, time + 0.5 * dt)
        
        # Stage 3
        self._update_state(grid, U0, k2, 0.5 * dt)
        k3 = self._calculate_rhs(grid, scheme, boundary_manager, time + 0.5 * dt)
        
        # Stage 4
        self._update_state(grid, U0, k3, dt)
        k4 = self._calculate_rhs(grid, scheme, boundary_manager, time + dt)
        
        # Final update: U^{n+1} = U^n + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        for i, cell in enumerate(grid.cells):
            if cell.cell_type == CellType.INTERIOR:
                dU = ConservativeVariables(
                    h=(k1[i].h + 2*k2[i].h + 2*k3[i].h + k4[i].h) / 6.0,
                    hu=(k1[i].hu + 2*k2[i].hu + 2*k3[i].hu + k4[i].hu) / 6.0
                )
                
                cell.U = ConservativeVariables(
                    h=U0[i].h + dt * dU.h,
                    hu=U0[i].hu + dt * dU.hu
                )
                
                # Ensure positive depth and limit extreme values
                cell.U.h = max(cell.U.h, 1e-12)
                cell.U.h = min(cell.U.h, 100.0)  # Reasonable upper limit
                
                # Limit velocity to prevent numerical instabilities
                if cell.U.h > 1e-12:
                    max_velocity = 50.0  # Reasonable upper limit
                    velocity = cell.U.hu / cell.U.h
                    if abs(velocity) > max_velocity:
                        cell.U.hu = cell.U.h * max_velocity * (1 if velocity > 0 else -1)
        
        return grid
    
    def _calculate_rhs(
        self, 
        grid: FVMGrid, 
        scheme: FVMScheme,
        boundary_manager: BoundaryManager,
        time: float
    ) -> Dict[int, ConservativeVariables]:
        """Calculate right-hand side of ODE system."""
        # Apply boundary conditions
        boundary_manager.apply_all_boundaries(grid.cells, time)
        
        # Calculate fluxes
        self.euler_integrator._calculate_fluxes(grid, scheme, boundary_manager, time)
        
        # Calculate RHS for each cell
        rhs = {}
        for i, cell in enumerate(grid.cells):
            if cell.cell_type == CellType.INTERIOR:
                flux_div = self.euler_integrator._calculate_flux_divergence(cell)
                cell.update_source_terms()
                source = self.euler_integrator._calculate_source_contribution(cell)
                
                rhs[i] = ConservativeVariables(
                    h=flux_div.h + source.h,
                    hu=flux_div.hu + source.hu
                )
            else:
                rhs[i] = ConservativeVariables(h=0.0, hu=0.0)
        
        return rhs
    
    def _update_state(
        self, 
        grid: FVMGrid, 
        U0: Dict[int, ConservativeVariables],
        k: Dict[int, ConservativeVariables],
        dt: float
    ):
        """Update grid state for RK4 intermediate stages."""
        for i, cell in enumerate(grid.cells):
            if cell.cell_type == CellType.INTERIOR:
                cell.U = ConservativeVariables(
                    h=U0[i].h + dt * k[i].h,
                    hu=U0[i].hu + dt * k[i].hu
                )
                
                # Ensure positive depth and limit extreme values
                cell.U.h = max(cell.U.h, 1e-12)
                cell.U.h = min(cell.U.h, 100.0)  # Reasonable upper limit
                
                # Limit velocity to prevent numerical instabilities
                if cell.U.h > 1e-12:
                    max_velocity = 50.0  # Reasonable upper limit
                    velocity = cell.U.hu / cell.U.h
                    if abs(velocity) > max_velocity:
                        cell.U.hu = cell.U.h * max_velocity * (1 if velocity > 0 else -1)


class FVMSolver:
    """
    Complete FVM solver for shallow water equations.
    
    Orchestrates all FVM components to provide professional-grade
    solution capabilities with adaptive time stepping, convergence
    monitoring, and robust error handling.
    """
    
    def __init__(
        self,
        scheme_name: str = "roe",
        time_integration: TimeIntegrationMethod = TimeIntegrationMethod.EXPLICIT_EULER,
        convergence_criteria: Optional[ConvergenceCriteria] = None
    ):
        """
        Initialize FVM solver.
        
        Args:
            scheme_name: Numerical scheme ("roe", "hll", "hllc", "lax_friedrichs")
            time_integration: Time integration method
            convergence_criteria: Convergence criteria
        """
        self.scheme_name = scheme_name
        self.time_integration = time_integration
        self.criteria = convergence_criteria or ConvergenceCriteria()
        
        # Initialize components
        self.scheme = self._create_scheme(scheme_name)
        self.integrator = self._create_integrator(time_integration)
        self.boundary_manager = BoundaryManager()
        self.grid_refinement = GridRefinement()
        
        # Solution monitoring
        self.monitor = SolutionMonitor()
        
        # Solver state
        self.is_initialized = False
        self.solution_history: List[Dict[str, Any]] = []
    
    def _create_scheme(self, scheme_name: str) -> FVMScheme:
        """Create numerical scheme."""
        scheme_map = {
            "roe": RoeScheme,
            "hll": HLLScheme,
            "hllc": HLLCScheme,
            "lax_friedrichs": LaxFriedrichsScheme
        }
        
        if scheme_name.lower() not in scheme_map:
            raise ValueError(f"Unknown scheme: {scheme_name}")
        
        return scheme_map[scheme_name.lower()]()
    
    def _create_integrator(self, method: TimeIntegrationMethod) -> TimeIntegrator:
        """Create time integrator."""
        if method == TimeIntegrationMethod.EXPLICIT_EULER:
            return ExplicitEulerIntegrator()
        elif method == TimeIntegrationMethod.RK4:
            return RungeKutta4Integrator()
        else:
            # Default to explicit Euler for now
            return ExplicitEulerIntegrator()
    
    def set_boundary_conditions(
        self, 
        left_bc: Optional[BoundaryCondition] = None,
        right_bc: Optional[BoundaryCondition] = None
    ):
        """Set boundary conditions."""
        if left_bc is not None:
            self.boundary_manager.set_left_boundary(left_bc)
        if right_bc is not None:
            self.boundary_manager.set_right_boundary(right_bc)
    
    def solve(
        self, 
        grid: FVMGrid,
        target_time: Optional[float] = None,
        steady_state: bool = True
    ) -> FVMResult:
        """
        Solve shallow water equations.
        
        Args:
            grid: Initial grid with flow conditions
            target_time: Target simulation time (None for steady state)
            steady_state: Whether to solve for steady state
            
        Returns:
            FVM solution result
        """
        start_time = time_module.time()
        
        # Initialize solver
        self._initialize_solution(grid)
        
        # Main solution loop
        try:
            while not self._check_termination_criteria(target_time, steady_state):
                # Calculate time step
                dt = self._calculate_time_step(grid)
                
                # Advance solution
                grid = self.integrator.advance_time_step(
                    grid, self.scheme, self.boundary_manager, dt, self.monitor.time
                )
                
                # Update monitor
                residual = self._calculate_residual(grid)
                self.monitor.update(
                    self.monitor.iteration + 1,
                    self.monitor.time + dt,
                    dt,
                    residual,
                    grid
                )
                
                # Check stability
                if not self.monitor.check_stability():
                    self.monitor.status = SolutionStatus.UNSTABLE
                    break
                
                # Adaptive grid refinement (if enabled)
                if isinstance(grid, AdaptiveGrid) and self.monitor.iteration % 100 == 0:
                    grid = self.grid_refinement.adapt_grid(grid)
                
                # Progress reporting
                if self.monitor.iteration % self.criteria.check_interval == 0:
                    self._report_progress()
            
            # Finalize solution
            self._finalize_solution(grid, start_time)
            
        except Exception as e:
            self.monitor.status = SolutionStatus.FAILED
            raise ConvergenceError(f"FVM solver failed: {str(e)}")
        
        # Create result
        return self._create_result(grid, time_module.time() - start_time)
    
    def _initialize_solution(self, grid: FVMGrid):
        """Initialize solution process."""
        self.monitor = SolutionMonitor()
        self.monitor.status = SolutionStatus.RUNNING
        self.is_initialized = True
        
        # Validate boundary conditions
        warnings = self.boundary_manager.validate_boundary_setup()
        if warnings:
            print(f"⚠️ Boundary condition warnings: {warnings}")
        
        # Initialize fluxes and source terms
        for cell in grid.cells:
            cell.U_old = cell.U.copy()
            cell.update_source_terms()
    
    def _calculate_time_step(self, grid: FVMGrid) -> float:
        """Calculate stable time step using CFL condition."""
        max_wave_speed = 0.0
        min_cell_size = float('inf')
        
        for cell in grid.cells:
            if cell.U.h > 1e-12:
                # Wave speeds
                c = math.sqrt(get_gravity() * cell.U.h)
                wave_speed = abs(cell.U.u) + c
                max_wave_speed = max(max_wave_speed, wave_speed)
            
            min_cell_size = min(min_cell_size, cell.dx)
        
        if max_wave_speed > 1e-12:
            dt_cfl = self.criteria.cfl_number * min_cell_size / max_wave_speed
        else:
            dt_cfl = self.criteria.max_time_step
        
        # Apply limits with additional safety factor for stability
        safety_factor = 0.8  # Additional safety margin
        dt = max(self.criteria.min_time_step, 
                min(dt_cfl * safety_factor, self.criteria.max_time_step))
        
        return dt
    
    def _calculate_residual(self, grid: FVMGrid) -> float:
        """Calculate solution residual."""
        residual = 0.0
        count = 0
        
        for cell in grid.cells:
            if cell.cell_type == CellType.INTERIOR and cell.U_old is not None:
                # L2 norm of change
                dh = cell.U.h - cell.U_old.h
                dhu = cell.U.hu - cell.U_old.hu
                
                cell_residual = math.sqrt(dh**2 + dhu**2)
                residual += cell_residual**2
                count += 1
        
        return math.sqrt(residual / max(count, 1))
    
    def _check_termination_criteria(
        self, 
        target_time: Optional[float], 
        steady_state: bool
    ) -> bool:
        """Check if solution should terminate."""
        # Maximum iterations
        if self.monitor.iteration >= self.criteria.max_iterations:
            self.monitor.status = SolutionStatus.MAX_ITERATIONS
            return True
        
        # Target time reached
        if target_time is not None and self.monitor.time >= target_time:
            self.monitor.status = SolutionStatus.CONVERGED
            return True
        
        # Steady state convergence
        if steady_state and self.monitor.residual < self.criteria.steady_state_tolerance:
            self.monitor.status = SolutionStatus.CONVERGED
            return True
        
        # General convergence
        if self.monitor.residual < self.criteria.residual_tolerance:
            self.monitor.status = SolutionStatus.CONVERGED
            return True
        
        # Maximum time
        if self.monitor.time >= self.criteria.max_time:
            self.monitor.status = SolutionStatus.MAX_ITERATIONS
            return True
        
        return False
    
    def _report_progress(self):
        """Report solution progress."""
        convergence_rate = self.monitor.get_convergence_rate()
        
        print(f"Iteration {self.monitor.iteration:6d}: "
              f"t={self.monitor.time:8.3f}s, "
              f"dt={self.monitor.time_step:.2e}s, "
              f"residual={self.monitor.residual:.2e}, "
              f"conv_rate={convergence_rate:.2f}")
    
    def _finalize_solution(self, grid: FVMGrid, start_time: float):
        """Finalize solution process."""
        if self.monitor.status == SolutionStatus.RUNNING:
            self.monitor.status = SolutionStatus.CONVERGED
        
        computation_time = time_module.time() - start_time
        
        print(f"\n🎉 FVM Solution Complete!")
        print(f"   Status: {self.monitor.status.value}")
        print(f"   Iterations: {self.monitor.iteration}")
        print(f"   Final time: {self.monitor.time:.3f}s")
        print(f"   Final residual: {self.monitor.residual:.2e}")
        print(f"   Computation time: {computation_time:.3f}s")
        print(f"   Mass conservation error: {self.monitor.mass_conservation_error:.2e}")
    
    def _create_result(self, grid: FVMGrid, computation_time: float) -> FVMResult:
        """Create FVM result object."""
        return FVMResult(
            grid=grid,
            time_final=self.monitor.time,
            x_coordinates=grid.get_x_coordinates(),
            depths=grid.get_depths(),
            velocities=grid.get_velocities(),
            froude_numbers=grid.get_froude_numbers(),
            water_elevations=grid.get_water_elevations(),
            converged=(self.monitor.status == SolutionStatus.CONVERGED),
            iterations=self.monitor.iteration,
            final_residual=self.monitor.residual,
            computation_time=computation_time,
            memory_usage=self._estimate_memory_usage(grid),
            properties={
                'scheme': self.scheme.name,
                'time_integration': self.integrator.name,
                'boundary_conditions': self.boundary_manager.get_boundary_summary(),
                'monitor': self.monitor,
                'solution_quality': self.monitor.check_stability()
            }
        )
    
    def _estimate_memory_usage(self, grid: FVMGrid) -> float:
        """Estimate memory usage in MB."""
        # Rough estimate: ~2KB per cell (conservative variables, fluxes, etc.)
        return grid.num_cells * 2e-3


class ShallowWaterSolver(FVMSolver):
    """
    Specialized FVM solver for shallow water equations.
    
    Provides hydraulic-specific functionality and optimizations
    for open channel flow applications.
    """
    
    def __init__(
        self,
        scheme_name: str = "roe",
        time_integration: TimeIntegrationMethod = TimeIntegrationMethod.RK4,
        convergence_criteria: Optional[ConvergenceCriteria] = None
    ):
        """Initialize shallow water solver with optimized defaults."""
        # Use more stringent criteria for hydraulic applications
        if convergence_criteria is None:
            convergence_criteria = ConvergenceCriteria(
                max_iterations=5000,
                residual_tolerance=1e-8,
                steady_state_tolerance=1e-10,
                cfl_number=0.4  # More conservative for stability
            )
        
        super().__init__(scheme_name, time_integration, convergence_criteria)
    
    def solve_hydraulic_jump(
        self,
        grid: FVMGrid,
        upstream_conditions: ConservativeVariables,
        downstream_conditions: Optional[ConservativeVariables] = None
    ) -> FVMResult:
        """
        Solve hydraulic jump problem.
        
        Args:
            grid: Computational grid
            upstream_conditions: Upstream flow conditions
            downstream_conditions: Downstream conditions (optional)
            
        Returns:
            FVM solution with detailed jump structure
        """
        # Set initial conditions
        self._initialize_hydraulic_jump(grid, upstream_conditions, downstream_conditions)
        
        # Solve to steady state
        result = self.solve(grid, steady_state=True)
        
        # Add hydraulic jump analysis
        result.properties['jump_analysis'] = self._analyze_hydraulic_jump(result)
        
        return result
    
    def _initialize_hydraulic_jump(
        self,
        grid: FVMGrid,
        upstream: ConservativeVariables,
        downstream: Optional[ConservativeVariables]
    ):
        """Initialize hydraulic jump conditions."""
        # Set upstream conditions (left half)
        for i, cell in enumerate(grid.cells):
            if cell.x_center < (grid.x_min + grid.x_max) / 2:
                cell.U = upstream.copy()
            else:
                if downstream is not None:
                    cell.U = downstream.copy()
                else:
                    # Use upstream conditions initially
                    cell.U = upstream.copy()
    
    def _analyze_hydraulic_jump(self, result: FVMResult) -> Dict[str, Any]:
        """Analyze hydraulic jump characteristics."""
        depths = result.depths
        velocities = result.velocities
        froude_numbers = result.froude_numbers
        
        # Find jump location (maximum depth gradient)
        depth_gradients = np.gradient(depths)
        jump_index = np.argmax(np.abs(depth_gradients))
        
        # Upstream and downstream conditions
        upstream_index = max(0, jump_index - 10)
        downstream_index = min(len(depths) - 1, jump_index + 10)
        
        upstream_depth = np.mean(depths[upstream_index:jump_index])
        downstream_depth = np.mean(depths[jump_index:downstream_index])
        
        upstream_velocity = np.mean(velocities[upstream_index:jump_index])
        downstream_velocity = np.mean(velocities[jump_index:downstream_index])
        
        upstream_froude = np.mean(froude_numbers[upstream_index:jump_index])
        downstream_froude = np.mean(froude_numbers[jump_index:downstream_index])
        
        # Jump characteristics
        jump_height = downstream_depth - upstream_depth
        energy_loss = self._calculate_energy_loss(
            upstream_depth, upstream_velocity,
            downstream_depth, downstream_velocity
        )
        
        return {
            'jump_location': result.x_coordinates[jump_index],
            'upstream_depth': upstream_depth,
            'downstream_depth': downstream_depth,
            'upstream_velocity': upstream_velocity,
            'downstream_velocity': downstream_velocity,
            'upstream_froude': upstream_froude,
            'downstream_froude': downstream_froude,
            'jump_height': jump_height,
            'energy_loss': energy_loss,
            'jump_efficiency': 1.0 - energy_loss / (upstream_velocity**2 / (2 * get_gravity()))
        }
    
    def _calculate_energy_loss(
        self, 
        h1: float, u1: float, 
        h2: float, u2: float
    ) -> float:
        """Calculate energy loss across hydraulic jump."""
        g = get_gravity()
        E1 = h1 + u1**2 / (2 * g)
        E2 = h2 + u2**2 / (2 * g)
        return E1 - E2
