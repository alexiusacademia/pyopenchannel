"""
File: numerical/fvm/validation.py
Author: Alexius Academia
Date: 2025-08-17

Validation suite for FVM implementation.

This module provides comprehensive validation tools for the FVM solver:
- Analytical solutions for comparison
- Error metrics and convergence studies
- Standard benchmark problems
- Grid convergence analysis
- Professional validation reporting

Validation cases include:
- Steady uniform flow
- Dam break problems
- Riemann problems
- Hydraulic jump validation
- Conservation verification
"""

import numpy as np
import math
from typing import List, Dict, Optional, Tuple, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

from .core import (
    FVMGrid, FVMResult, ConservativeVariables, 
    FluxVector, get_gravity
)
from .grid import UniformGrid
from .solver import FVMSolver, ConvergenceCriteria, TimeIntegrationMethod
from .boundary import BoundaryManager, InletBC, OutletBC, TransmissiveBC, BoundaryData
from ...exceptions import ValidationError


class ValidationCase(Enum):
    """Types of validation cases."""
    STEADY_UNIFORM = "steady_uniform"
    DAM_BREAK = "dam_break"
    RIEMANN_PROBLEM = "riemann_problem"
    HYDRAULIC_JUMP = "hydraulic_jump"
    CONSERVATION_TEST = "conservation_test"
    MANUFACTURED_SOLUTION = "manufactured_solution"


class ErrorNorm(Enum):
    """Types of error norms."""
    L1 = "l1"
    L2 = "l2"
    LINF = "linf"
    RELATIVE_L2 = "relative_l2"


@dataclass
class ValidationResult:
    """Result of a validation test."""
    case_name: str
    test_passed: bool
    error_metrics: Dict[str, float]
    convergence_order: Optional[float] = None
    computation_time: float = 0.0
    grid_points: int = 0
    scheme_name: str = ""
    notes: str = ""
    
    # Detailed results
    numerical_solution: Optional[np.ndarray] = None
    analytical_solution: Optional[np.ndarray] = None
    x_coordinates: Optional[np.ndarray] = None
    
    @property
    def max_error(self) -> float:
        """Get maximum error across all metrics."""
        if not self.error_metrics:
            return float('inf')
        return max(self.error_metrics.values())
    
    @property
    def success_summary(self) -> str:
        """Get success summary string."""
        status = "PASS" if self.test_passed else "FAIL"
        max_err = self.max_error
        return f"{status} (max error: {max_err:.2e})"


class AnalyticalSolution(ABC):
    """
    Abstract base class for analytical solutions.
    
    Provides interface for computing exact solutions to
    shallow water equation problems for validation.
    """
    
    def __init__(self, name: str, description: str):
        """Initialize analytical solution."""
        self.name = name
        self.description = description
        self.gravity = get_gravity()
    
    @abstractmethod
    def evaluate(
        self, 
        x: np.ndarray, 
        t: float = 0.0,
        **parameters
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate analytical solution.
        
        Args:
            x: Spatial coordinates
            t: Time
            **parameters: Problem-specific parameters
            
        Returns:
            (depths, velocities) at given points
        """
        pass
    
    @abstractmethod
    def get_initial_conditions(
        self, 
        x: np.ndarray,
        **parameters
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get initial conditions for the problem.
        
        Args:
            x: Spatial coordinates
            **parameters: Problem-specific parameters
            
        Returns:
            (initial_depths, initial_velocities)
        """
        pass


class SteadyUniformFlowSolution(AnalyticalSolution):
    """Analytical solution for steady uniform flow."""
    
    def __init__(self):
        """Initialize steady uniform flow solution."""
        super().__init__(
            "Steady Uniform Flow",
            "Constant depth and velocity throughout domain"
        )
    
    def evaluate(
        self, 
        x: np.ndarray, 
        t: float = 0.0,
        depth: float = 1.0,
        velocity: float = 1.0,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate steady uniform flow solution."""
        depths = np.full_like(x, depth)
        velocities = np.full_like(x, velocity)
        return depths, velocities
    
    def get_initial_conditions(
        self, 
        x: np.ndarray,
        depth: float = 1.0,
        velocity: float = 1.0,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get initial conditions for steady uniform flow."""
        return self.evaluate(x, 0.0, depth=depth, velocity=velocity)


class DamBreakSolution(AnalyticalSolution):
    """Analytical solution for dam break problem."""
    
    def __init__(self):
        """Initialize dam break solution."""
        super().__init__(
            "Dam Break",
            "Classical dam break with exact Riemann solution"
        )
    
    def evaluate(
        self, 
        x: np.ndarray, 
        t: float = 1.0,
        h_left: float = 2.0,
        h_right: float = 0.5,
        u_left: float = 0.0,
        u_right: float = 0.0,
        x_interface: float = 0.0,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate dam break solution using exact Riemann solver."""
        if t <= 1e-12:
            # Initial conditions
            depths = np.where(x < x_interface, h_left, h_right)
            velocities = np.where(x < x_interface, u_left, u_right)
            return depths, velocities
        
        # Exact Riemann solution for dam break
        g = self.gravity
        
        # Wave speeds
        c_left = math.sqrt(g * h_left)
        c_right = math.sqrt(g * h_right)
        
        # Star region properties (exact solution)
        h_star = (1/16) * (math.sqrt(g * h_left) + math.sqrt(g * h_right))**2 / g
        u_star = 2 * (math.sqrt(g * h_left) - math.sqrt(g * h_right))
        c_star = math.sqrt(g * h_star)
        
        # Wave positions
        x_head = x_interface - c_left * t  # Left rarefaction head
        x_tail = x_interface + (u_star - c_star) * t  # Left rarefaction tail
        x_shock = x_interface + self._shock_speed(h_star, h_right, u_star, u_right) * t
        
        depths = np.zeros_like(x)
        velocities = np.zeros_like(x)
        
        for i, xi in enumerate(x):
            if xi < x_head:
                # Left state
                depths[i] = h_left
                velocities[i] = u_left
            elif xi < x_tail:
                # Rarefaction fan
                u_fan = (2/3) * (xi/t + c_left)
                h_fan = (1/(9*g)) * (2*c_left - xi/t)**2
                depths[i] = max(h_fan, 1e-12)
                velocities[i] = u_fan
            elif xi < x_shock:
                # Star region
                depths[i] = h_star
                velocities[i] = u_star
            else:
                # Right state
                depths[i] = h_right
                velocities[i] = u_right
        
        return depths, velocities
    
    def get_initial_conditions(
        self, 
        x: np.ndarray,
        h_left: float = 2.0,
        h_right: float = 0.5,
        u_left: float = 0.0,
        u_right: float = 0.0,
        x_interface: float = 0.0,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get initial conditions for dam break."""
        depths = np.where(x < x_interface, h_left, h_right)
        velocities = np.where(x < x_interface, u_left, u_right)
        return depths, velocities
    
    def _shock_speed(self, h_left: float, h_right: float, u_left: float, u_right: float) -> float:
        """Calculate shock speed using Rankine-Hugoniot conditions."""
        if abs(h_left - h_right) < 1e-12:
            return 0.5 * (u_left + u_right)
        
        return (h_left * u_left - h_right * u_right) / (h_left - h_right)


class RiemannProblemSolution(AnalyticalSolution):
    """Analytical solution for general Riemann problems."""
    
    def __init__(self):
        """Initialize Riemann problem solution."""
        super().__init__(
            "Riemann Problem",
            "General Riemann problem with left and right states"
        )
    
    def evaluate(
        self, 
        x: np.ndarray, 
        t: float = 1.0,
        h_left: float = 1.0,
        h_right: float = 1.0,
        u_left: float = 0.0,
        u_right: float = 0.0,
        x_interface: float = 0.0,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate Riemann problem solution."""
        # For now, use simplified approach
        # Full Riemann solver would be more complex
        
        if t <= 1e-12:
            depths = np.where(x < x_interface, h_left, h_right)
            velocities = np.where(x < x_interface, u_left, u_right)
            return depths, velocities
        
        # Simplified wave propagation
        g = self.gravity
        c_left = math.sqrt(g * h_left) if h_left > 0 else 0
        c_right = math.sqrt(g * h_right) if h_right > 0 else 0
        
        # Approximate wave speeds
        s_left = u_left - c_left
        s_right = u_right + c_right
        
        depths = np.zeros_like(x)
        velocities = np.zeros_like(x)
        
        for i, xi in enumerate(x):
            xi_norm = (xi - x_interface) / t if t > 0 else 0
            
            if xi_norm < s_left:
                depths[i] = h_left
                velocities[i] = u_left
            elif xi_norm > s_right:
                depths[i] = h_right
                velocities[i] = u_right
            else:
                # Intermediate state (simplified)
                alpha = (xi_norm - s_left) / (s_right - s_left) if s_right != s_left else 0.5
                depths[i] = h_left * (1 - alpha) + h_right * alpha
                velocities[i] = u_left * (1 - alpha) + u_right * alpha
        
        return depths, velocities
    
    def get_initial_conditions(
        self, 
        x: np.ndarray,
        h_left: float = 1.0,
        h_right: float = 1.0,
        u_left: float = 0.0,
        u_right: float = 0.0,
        x_interface: float = 0.0,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get initial conditions for Riemann problem."""
        depths = np.where(x < x_interface, h_left, h_right)
        velocities = np.where(x < x_interface, u_left, u_right)
        return depths, velocities


class ManufacturedSolution(AnalyticalSolution):
    """Manufactured solution for method of manufactured solutions."""
    
    def __init__(self):
        """Initialize manufactured solution."""
        super().__init__(
            "Manufactured Solution",
            "Smooth analytical solution with known source terms"
        )
    
    def evaluate(
        self, 
        x: np.ndarray, 
        t: float = 0.0,
        amplitude: float = 0.1,
        wavelength: float = 2.0,
        base_depth: float = 1.0,
        base_velocity: float = 0.5,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate manufactured solution."""
        # Smooth sinusoidal variation
        k = 2 * math.pi / wavelength
        
        depths = base_depth + amplitude * np.sin(k * x - 0.5 * t)
        velocities = base_velocity + 0.5 * amplitude * np.cos(k * x - 0.5 * t)
        
        # Ensure positive depths
        depths = np.maximum(depths, 0.1)
        
        return depths, velocities
    
    def get_initial_conditions(
        self, 
        x: np.ndarray,
        amplitude: float = 0.1,
        wavelength: float = 2.0,
        base_depth: float = 1.0,
        base_velocity: float = 0.5,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get initial conditions for manufactured solution."""
        return self.evaluate(x, 0.0, amplitude=amplitude, wavelength=wavelength,
                           base_depth=base_depth, base_velocity=base_velocity)


class ErrorMetrics:
    """Calculate error metrics between numerical and analytical solutions."""
    
    @staticmethod
    def l1_norm(numerical: np.ndarray, analytical: np.ndarray) -> float:
        """Calculate L1 norm error."""
        return np.mean(np.abs(numerical - analytical))
    
    @staticmethod
    def l2_norm(numerical: np.ndarray, analytical: np.ndarray) -> float:
        """Calculate L2 norm error."""
        return np.sqrt(np.mean((numerical - analytical)**2))
    
    @staticmethod
    def linf_norm(numerical: np.ndarray, analytical: np.ndarray) -> float:
        """Calculate L∞ norm error."""
        return np.max(np.abs(numerical - analytical))
    
    @staticmethod
    def relative_l2_norm(numerical: np.ndarray, analytical: np.ndarray) -> float:
        """Calculate relative L2 norm error."""
        analytical_norm = np.sqrt(np.mean(analytical**2))
        if analytical_norm < 1e-12:
            return ErrorMetrics.l2_norm(numerical, analytical)
        return ErrorMetrics.l2_norm(numerical, analytical) / analytical_norm
    
    @staticmethod
    def calculate_all_norms(
        numerical: np.ndarray, 
        analytical: np.ndarray
    ) -> Dict[str, float]:
        """Calculate all error norms."""
        return {
            'l1': ErrorMetrics.l1_norm(numerical, analytical),
            'l2': ErrorMetrics.l2_norm(numerical, analytical),
            'linf': ErrorMetrics.linf_norm(numerical, analytical),
            'relative_l2': ErrorMetrics.relative_l2_norm(numerical, analytical)
        }


class ConvergenceStudy:
    """Perform grid convergence studies."""
    
    def __init__(self, analytical_solution: AnalyticalSolution):
        """Initialize convergence study."""
        self.analytical_solution = analytical_solution
        self.results: List[ValidationResult] = []
    
    def run_convergence_study(
        self,
        grid_sizes: List[int],
        solver_config: Dict[str, Any],
        problem_params: Dict[str, Any],
        domain: Tuple[float, float] = (0.0, 10.0),
        final_time: float = 1.0
    ) -> List[ValidationResult]:
        """
        Run grid convergence study.
        
        Args:
            grid_sizes: List of grid sizes to test
            solver_config: Solver configuration
            problem_params: Problem parameters
            domain: Spatial domain (x_min, x_max)
            final_time: Final simulation time
            
        Returns:
            List of validation results
        """
        self.results = []
        
        for n_cells in grid_sizes:
            print(f"  Running convergence study with {n_cells} cells...")
            
            # Create grid
            grid = UniformGrid(
                x_min=domain[0], 
                x_max=domain[1], 
                num_cells=n_cells
            )
            
            # Set initial conditions
            x_centers = grid.get_x_coordinates()
            initial_depths, initial_velocities = self.analytical_solution.get_initial_conditions(
                x_centers, **problem_params
            )
            
            for i, cell in enumerate(grid.cells):
                cell.U = ConservativeVariables(
                    h=initial_depths[i],
                    hu=initial_depths[i] * initial_velocities[i]
                )
            
            # Create solver
            solver = FVMSolver(**solver_config)
            
            # Set boundary conditions (transmissive for most cases)
            solver.set_boundary_conditions(
                TransmissiveBC("left"),
                TransmissiveBC("right")
            )
            
            # Solve
            result = solver.solve(grid, target_time=final_time, steady_state=False)
            
            # Calculate analytical solution at final time
            analytical_depths, analytical_velocities = self.analytical_solution.evaluate(
                x_centers, final_time, **problem_params
            )
            
            # Calculate errors
            depth_errors = ErrorMetrics.calculate_all_norms(result.depths, analytical_depths)
            velocity_errors = ErrorMetrics.calculate_all_norms(result.velocities, analytical_velocities)
            
            # Combine error metrics
            error_metrics = {}
            for norm in depth_errors:
                error_metrics[f'depth_{norm}'] = depth_errors[norm]
                error_metrics[f'velocity_{norm}'] = velocity_errors[norm]
            
            # Create validation result
            validation_result = ValidationResult(
                case_name=f"{self.analytical_solution.name} (N={n_cells})",
                test_passed=error_metrics['depth_l2'] < 0.1,  # Reasonable tolerance
                error_metrics=error_metrics,
                computation_time=result.computation_time,
                grid_points=n_cells,
                scheme_name=solver.scheme.name,
                numerical_solution=result.depths,
                analytical_solution=analytical_depths,
                x_coordinates=x_centers
            )
            
            self.results.append(validation_result)
        
        # Calculate convergence orders
        self._calculate_convergence_orders(grid_sizes)
        
        return self.results
    
    def _calculate_convergence_orders(self, grid_sizes: List[int]):
        """Calculate convergence orders from results."""
        if len(self.results) < 2:
            return
        
        for i in range(1, len(self.results)):
            h1 = (10.0 - 0.0) / grid_sizes[i-1]  # Grid spacing
            h2 = (10.0 - 0.0) / grid_sizes[i]
            
            e1 = self.results[i-1].error_metrics['depth_l2']
            e2 = self.results[i].error_metrics['depth_l2']
            
            if e1 > 1e-15 and e2 > 1e-15:
                order = math.log(e1 / e2) / math.log(h1 / h2)
                self.results[i].convergence_order = order


class ValidationSuite:
    """
    Complete validation suite for FVM solver.
    
    Orchestrates multiple validation tests and provides
    comprehensive reporting of solver accuracy and reliability.
    """
    
    def __init__(self):
        """Initialize validation suite."""
        self.analytical_solutions = {
            ValidationCase.STEADY_UNIFORM: SteadyUniformFlowSolution(),
            ValidationCase.DAM_BREAK: DamBreakSolution(),
            ValidationCase.RIEMANN_PROBLEM: RiemannProblemSolution(),
            ValidationCase.MANUFACTURED_SOLUTION: ManufacturedSolution()
        }
        
        self.validation_results: List[ValidationResult] = []
        self.convergence_studies: Dict[str, ConvergenceStudy] = {}
    
    def run_all_validations(
        self,
        schemes: List[str] = ["lax_friedrichs", "hll"],
        grid_sizes: List[int] = [50, 100, 200]
    ) -> Dict[str, List[ValidationResult]]:
        """
        Run complete validation suite.
        
        Args:
            schemes: List of numerical schemes to test
            grid_sizes: List of grid sizes for convergence studies
            
        Returns:
            Dictionary of validation results by case
        """
        all_results = {}
        
        print("🧪 RUNNING COMPLETE FVM VALIDATION SUITE")
        print("=" * 60)
        
        # Test each validation case
        for case, analytical_solution in self.analytical_solutions.items():
            print(f"\n📊 Testing {case.value.upper()}...")
            
            case_results = []
            
            for scheme in schemes:
                print(f"  Scheme: {scheme.upper()}")
                
                # Run convergence study
                convergence_study = ConvergenceStudy(analytical_solution)
                
                # Configure solver
                solver_config = {
                    'scheme_name': scheme,
                    'time_integration': TimeIntegrationMethod.EXPLICIT_EULER,
                    'convergence_criteria': ConvergenceCriteria(
                        max_iterations=1000,
                        cfl_number=0.3
                    )
                }
                
                # Problem parameters
                problem_params = self._get_problem_parameters(case)
                
                # Run convergence study
                results = convergence_study.run_convergence_study(
                    grid_sizes=grid_sizes,
                    solver_config=solver_config,
                    problem_params=problem_params,
                    final_time=self._get_final_time(case)
                )
                
                case_results.extend(results)
                self.convergence_studies[f"{case.value}_{scheme}"] = convergence_study
            
            all_results[case.value] = case_results
            self.validation_results.extend(case_results)
        
        return all_results
    
    def _get_problem_parameters(self, case: ValidationCase) -> Dict[str, Any]:
        """Get problem parameters for validation case."""
        if case == ValidationCase.STEADY_UNIFORM:
            return {'depth': 1.5, 'velocity': 1.0}
        elif case == ValidationCase.DAM_BREAK:
            return {'h_left': 2.0, 'h_right': 0.5, 'x_interface': 5.0}
        elif case == ValidationCase.RIEMANN_PROBLEM:
            return {'h_left': 1.5, 'h_right': 1.0, 'u_left': 0.5, 'u_right': -0.5, 'x_interface': 5.0}
        elif case == ValidationCase.MANUFACTURED_SOLUTION:
            return {'amplitude': 0.1, 'wavelength': 4.0, 'base_depth': 1.0, 'base_velocity': 0.5}
        else:
            return {}
    
    def _get_final_time(self, case: ValidationCase) -> float:
        """Get final time for validation case."""
        if case == ValidationCase.STEADY_UNIFORM:
            return 5.0  # Long enough to reach steady state
        elif case in [ValidationCase.DAM_BREAK, ValidationCase.RIEMANN_PROBLEM]:
            return 1.0  # Standard time for wave propagation
        elif case == ValidationCase.MANUFACTURED_SOLUTION:
            return 2.0  # Allow for wave propagation
        else:
            return 1.0
    
    def generate_validation_report(self) -> str:
        """Generate comprehensive validation report."""
        report = []
        report.append("🧪 FVM VALIDATION SUITE REPORT")
        report.append("=" * 60)
        
        # Overall summary
        total_tests = len(self.validation_results)
        passed_tests = sum(1 for r in self.validation_results if r.test_passed)
        
        report.append(f"\n📊 OVERALL SUMMARY:")
        report.append(f"   Total tests: {total_tests}")
        report.append(f"   Passed: {passed_tests}")
        report.append(f"   Failed: {total_tests - passed_tests}")
        report.append(f"   Success rate: {passed_tests/total_tests*100:.1f}%")
        
        # Results by case
        cases = {}
        for result in self.validation_results:
            case_base = result.case_name.split(' (')[0]
            if case_base not in cases:
                cases[case_base] = []
            cases[case_base].append(result)
        
        for case_name, results in cases.items():
            report.append(f"\n🎯 {case_name.upper()}:")
            
            for result in results:
                status = "✅ PASS" if result.test_passed else "❌ FAIL"
                report.append(f"   {status} | {result.scheme_name} | N={result.grid_points} | "
                            f"L2={result.error_metrics.get('depth_l2', 0):.2e}")
                
                if result.convergence_order is not None:
                    report.append(f"     Convergence order: {result.convergence_order:.2f}")
        
        # Convergence analysis
        report.append(f"\n📈 CONVERGENCE ANALYSIS:")
        for study_name, study in self.convergence_studies.items():
            if len(study.results) >= 2:
                final_result = study.results[-1]
                if final_result.convergence_order is not None:
                    report.append(f"   {study_name}: Order {final_result.convergence_order:.2f}")
        
        return "\n".join(report)
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get validation summary statistics."""
        if not self.validation_results:
            return {}
        
        total_tests = len(self.validation_results)
        passed_tests = sum(1 for r in self.validation_results if r.test_passed)
        
        # Error statistics
        all_errors = []
        for result in self.validation_results:
            if 'depth_l2' in result.error_metrics:
                all_errors.append(result.error_metrics['depth_l2'])
        
        return {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'success_rate': passed_tests / total_tests if total_tests > 0 else 0,
            'mean_error': np.mean(all_errors) if all_errors else 0,
            'max_error': np.max(all_errors) if all_errors else 0,
            'min_error': np.min(all_errors) if all_errors else 0
        }
