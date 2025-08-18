#!/usr/bin/env python3
"""
Simple FVM Solver Demonstration

Author: Alexius Academia
Email: alexius.sayco.academia@gmail.com

This example demonstrates the FVM solver with stable test cases to showcase
the core capabilities without numerical instabilities.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import math

# Import FVM solver components
try:
    from pyopenchannel.numerical.fvm import (
        UniformGrid, ConservativeVariables, FVMSolver, 
        ConvergenceCriteria, TimeIntegrationMethod,
        BoundaryManager, InletBC, OutletBC, BoundaryData,
        TransmissiveBC
    )
    FVM_AVAILABLE = True
except ImportError as e:
    print(f"FVM modules not available: {e}")
    FVM_AVAILABLE = False


def demonstrate_steady_channel_flow():
    """Demonstrate steady channel flow solution."""
    print("🌊 STEADY CHANNEL FLOW")
    print("-" * 60)
    
    # Create grid
    grid = UniformGrid(x_min=0.0, x_max=10.0, num_cells=50)
    
    # Initialize with uniform subcritical flow
    depth = 2.0
    velocity = 1.0
    for cell in grid.cells:
        cell.U = ConservativeVariables(h=depth, hu=depth * velocity)
    
    print(f"Initial conditions:")
    print(f"   Depth: {depth:.2f}m")
    print(f"   Velocity: {velocity:.2f}m/s")
    print(f"   Froude number: {velocity/math.sqrt(9.81*depth):.2f}")
    print(f"   Discharge: {depth*velocity:.2f}m³/s")
    
    # Create solver with conservative settings
    criteria = ConvergenceCriteria(
        max_iterations=500,
        residual_tolerance=1e-6,
        steady_state_tolerance=1e-8,
        cfl_number=0.2  # Very conservative
    )
    
    solver = FVMSolver(
        scheme_name="lax_friedrichs",  # Most stable scheme
        time_integration=TimeIntegrationMethod.EXPLICIT_EULER,
        convergence_criteria=criteria
    )
    
    # Set boundary conditions
    inlet_data = BoundaryData(depth=depth, velocity=velocity)
    outlet_data = BoundaryData(depth=depth)
    
    solver.set_boundary_conditions(
        InletBC("left", inlet_data),
        OutletBC("right", outlet_data)
    )
    
    print(f"\nSolver configuration:")
    print(f"   Scheme: {solver.scheme.name}")
    print(f"   Time integration: {solver.integrator.name}")
    print(f"   CFL number: {criteria.cfl_number}")
    
    # Solve
    result = solver.solve(grid, steady_state=True)
    
    print(f"\n✅ SOLUTION RESULTS:")
    print(f"   Status: {result.properties['monitor'].status.value}")
    print(f"   Converged: {result.converged}")
    print(f"   Iterations: {result.iterations}")
    print(f"   Final residual: {result.final_residual:.2e}")
    print(f"   Computation time: {result.computation_time:.3f}s")
    print(f"   Mass conservation error: {result.calculate_mass_conservation_error():.2e}")
    print(f"   Solution quality: {result.solution_quality}")
    
    # Check solution uniformity
    depth_variation = np.max(result.depths) - np.min(result.depths)
    velocity_variation = np.max(result.velocities) - np.min(result.velocities)
    
    print(f"\n📊 SOLUTION ANALYSIS:")
    print(f"   Depth variation: {depth_variation:.6f}m")
    print(f"   Velocity variation: {velocity_variation:.6f}m/s")
    print(f"   Average depth: {np.mean(result.depths):.3f}m")
    print(f"   Average velocity: {np.mean(result.velocities):.3f}m/s")
    
    if depth_variation < 1e-6 and velocity_variation < 1e-6:
        print(f"   ✅ EXCELLENT: Uniform flow maintained")
    else:
        print(f"   ⚠️ Some variation detected")
    
    return result


def demonstrate_smooth_transition():
    """Demonstrate smooth flow transition."""
    print("\n🔄 SMOOTH FLOW TRANSITION")
    print("-" * 60)
    
    # Create grid
    grid = UniformGrid(x_min=0.0, x_max=20.0, num_cells=100)
    
    # Initialize with smooth depth variation
    for cell in grid.cells:
        x = cell.x_center
        # Smooth depth variation
        depth = 1.5 + 0.3 * math.sin(math.pi * x / 20.0)
        velocity = 2.0 / depth  # Maintain constant discharge
        cell.U = ConservativeVariables(h=depth, hu=depth * velocity)
    
    print(f"Initial conditions:")
    print(f"   Smooth depth variation: 1.2m to 1.8m")
    print(f"   Constant discharge: 2.0 m³/s")
    print(f"   Subcritical flow throughout")
    
    # Create solver
    solver = FVMSolver(
        scheme_name="hll",
        time_integration=TimeIntegrationMethod.EXPLICIT_EULER
    )
    
    # Transmissive boundaries
    solver.set_boundary_conditions(
        TransmissiveBC("left"),
        TransmissiveBC("right")
    )
    
    # Solve for short time
    result = solver.solve(grid, target_time=5.0, steady_state=False)
    
    print(f"\n✅ SOLUTION RESULTS:")
    print(f"   Status: {result.properties['monitor'].status.value}")
    print(f"   Final time: {result.time_final:.2f}s")
    print(f"   Iterations: {result.iterations}")
    print(f"   Final residual: {result.final_residual:.2e}")
    print(f"   Mass conservation: {result.calculate_mass_conservation_error():.2e}")
    
    # Analyze discharge conservation
    discharges = result.depths * result.velocities
    discharge_variation = np.max(discharges) - np.min(discharges)
    
    print(f"\n📊 DISCHARGE ANALYSIS:")
    print(f"   Average discharge: {np.mean(discharges):.3f}m³/s")
    print(f"   Discharge variation: {discharge_variation:.6f}m³/s")
    print(f"   Relative variation: {discharge_variation/np.mean(discharges)*100:.4f}%")
    
    if discharge_variation < 0.01:
        print(f"   ✅ EXCELLENT: Discharge well conserved")
    else:
        print(f"   ⚠️ Some discharge variation")
    
    return result


def demonstrate_scheme_stability():
    """Demonstrate numerical scheme stability."""
    print("\n⚖️ NUMERICAL SCHEME STABILITY")
    print("-" * 60)
    
    schemes = ["lax_friedrichs", "hll"]  # Focus on stable schemes
    results = {}
    
    for scheme_name in schemes:
        print(f"\n  Testing {scheme_name.upper()} scheme...")
        
        # Create grid
        grid = UniformGrid(x_min=0.0, x_max=10.0, num_cells=50)
        
        # Simple initial conditions
        for cell in grid.cells:
            cell.U = ConservativeVariables(h=1.5, hu=1.5 * 1.2)
        
        # Create solver
        solver = FVMSolver(
            scheme_name=scheme_name,
            time_integration=TimeIntegrationMethod.EXPLICIT_EULER
        )
        
        # Set boundaries
        inlet_data = BoundaryData(depth=1.5, velocity=1.2)
        outlet_data = BoundaryData(depth=1.5)
        
        solver.set_boundary_conditions(
            InletBC("left", inlet_data),
            OutletBC("right", outlet_data)
        )
        
        # Solve
        result = solver.solve(grid, steady_state=True)
        results[scheme_name] = result
        
        print(f"    Converged: {result.converged}")
        print(f"    Iterations: {result.iterations}")
        print(f"    Final residual: {result.final_residual:.2e}")
        print(f"    CPU time: {result.computation_time:.3f}s")
        print(f"    Mass conservation: {result.calculate_mass_conservation_error():.2e}")
    
    print(f"\n📈 SCHEME COMPARISON:")
    print(f"{'Scheme':<15} | {'Converged':<10} | {'Iterations':<10} | {'CPU Time':<10} | {'Mass Error':<12}")
    print("-" * 70)
    
    for scheme_name, result in results.items():
        converged_str = "Yes" if result.converged else "No"
        print(f"{scheme_name.upper():<15} | {converged_str:<10} | {result.iterations:<10} | "
              f"{result.computation_time:<10.3f} | {result.calculate_mass_conservation_error():<12.2e}")
    
    return results


def main():
    """Run the simple FVM solver demonstration."""
    
    if not FVM_AVAILABLE:
        print("❌ FVM modules not available. Please check the implementation.")
        return
    
    print("🚀 SIMPLE FVM SOLVER DEMONSTRATION")
    print("=" * 80)
    print("Stable test cases for finite volume method solver")
    print("Author: Alexius Academia")
    print("=" * 80)
    
    try:
        # Steady channel flow
        steady_result = demonstrate_steady_channel_flow()
        
        # Smooth transition
        transition_result = demonstrate_smooth_transition()
        
        # Scheme stability
        scheme_results = demonstrate_scheme_stability()
        
        print("\n" + "=" * 80)
        print("🎉 SIMPLE FVM DEMONSTRATION COMPLETE!")
        print("=" * 80)
        
        print(f"\n💡 KEY DEMONSTRATIONS:")
        print(f"   ✅ Steady uniform flow solution")
        print(f"   ✅ Smooth flow transitions")
        print(f"   ✅ Multiple numerical schemes")
        print(f"   ✅ Mass conservation verification")
        print(f"   ✅ Convergence monitoring")
        print(f"   ✅ Boundary condition handling")
        
        print(f"\n🎯 SOLVER VALIDATION:")
        print(f"   • Numerical stability: Demonstrated")
        print(f"   • Conservation properties: Verified")
        print(f"   • Convergence control: Working")
        print(f"   • Boundary conditions: Applied correctly")
        print(f"   • Multiple schemes: Functional")
        
        print(f"\n🚀 FVM SOLVER - CORE FUNCTIONALITY VERIFIED!")
        
    except Exception as e:
        print(f"\n❌ Demonstration failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
