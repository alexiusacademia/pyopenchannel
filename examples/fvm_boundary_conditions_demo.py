#!/usr/bin/env python3
"""
FVM Boundary Conditions Demonstration

Author: Alexius Academia
Email: alexius.sayco.academia@gmail.com

This example demonstrates the comprehensive boundary condition system
for the FVM implementation, showing different BC types and their applications.

This demonstrates the implemented boundary condition classes - 
the full FVM solver integration will come next!
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from typing import List, Dict, Any

# Import FVM boundary condition classes
try:
    from pyopenchannel.numerical.fvm.boundary import (
        BoundaryType, BoundaryData, BoundaryManager,
        DirichletBC, NeumannBC, TransmissiveBC, ReflectiveBC,
        InletBC, OutletBC, CriticalBC, RatingCurveBC,
        FlowDirection
    )
    from pyopenchannel.numerical.fvm.core import (
        ConservativeVariables, FVMCell, CellType, FluxVector
    )
    FVM_AVAILABLE = True
except ImportError as e:
    print(f"FVM modules not available: {e}")
    FVM_AVAILABLE = False


def create_test_cell(x_center: float, depth: float, velocity: float) -> FVMCell:
    """Create a test FVM cell."""
    U = ConservativeVariables(h=depth, hu=depth * velocity)
    
    return FVMCell(
        index=0,
        x_center=x_center,
        dx=1.0,
        cell_type=CellType.INTERIOR,
        U=U,
        bed_elevation=0.0,
        manning_n=0.030
    )


def demonstrate_dirichlet_bc():
    """Demonstrate Dirichlet boundary conditions."""
    print("🔧 DIRICHLET BOUNDARY CONDITIONS")
    print("-" * 60)
    
    # Create boundary data
    data = BoundaryData(depth=2.0, velocity=1.5)
    
    # Create left Dirichlet BC
    left_bc = DirichletBC("left", data)
    
    # Create test interior cell
    interior_cell = create_test_cell(1.0, 1.8, 1.2)
    
    print(f"Interior cell: h={interior_cell.U.h:.2f}m, u={interior_cell.U.u:.2f}m/s")
    print(f"Prescribed: h={data.depth:.2f}m, u={data.velocity:.2f}m/s")
    
    # Apply boundary condition
    ghost_state = left_bc.apply_boundary_condition(interior_cell, None)
    boundary_flux = left_bc.calculate_boundary_flux(interior_cell)
    
    print(f"Ghost cell: h={ghost_state.h:.2f}m, u={ghost_state.u:.2f}m/s")
    print(f"Boundary flux: mass={boundary_flux.mass_flux:.3f}, momentum={boundary_flux.momentum_flux:.3f}")
    print(f"✅ Dirichlet BC enforces prescribed values exactly")
    print()


def demonstrate_neumann_bc():
    """Demonstrate Neumann boundary conditions."""
    print("📐 NEUMANN BOUNDARY CONDITIONS")
    print("-" * 60)
    
    # Create boundary data with gradients
    data = BoundaryData(depth_gradient=0.1, velocity_gradient=-0.05)
    
    # Create right Neumann BC
    right_bc = NeumannBC("right", data)
    
    # Create test interior cell
    interior_cell = create_test_cell(10.0, 1.5, 2.0)
    
    # Create ghost cell
    ghost_cell = create_test_cell(11.0, 0.0, 0.0)  # Will be updated
    
    print(f"Interior cell: h={interior_cell.U.h:.2f}m, u={interior_cell.U.u:.2f}m/s")
    print(f"Prescribed gradients: dh/dx={data.depth_gradient:.3f}, du/dx={data.velocity_gradient:.3f}")
    
    # Apply boundary condition
    ghost_state = right_bc.apply_boundary_condition(interior_cell, ghost_cell)
    boundary_flux = right_bc.calculate_boundary_flux(interior_cell)
    
    print(f"Ghost cell: h={ghost_state.h:.2f}m, u={ghost_state.u:.2f}m/s")
    print(f"Boundary flux: mass={boundary_flux.mass_flux:.3f}, momentum={boundary_flux.momentum_flux:.3f}")
    print(f"✅ Neumann BC applies specified gradients")
    print()


def demonstrate_inlet_outlet_bc():
    """Demonstrate inlet and outlet boundary conditions."""
    print("🌊 INLET/OUTLET BOUNDARY CONDITIONS")
    print("-" * 60)
    
    # Inlet BC with prescribed discharge
    inlet_data = BoundaryData(discharge=5.0)
    inlet_bc = InletBC("left", inlet_data)
    
    # Outlet BC with prescribed depth
    outlet_data = BoundaryData(depth=1.2)
    outlet_bc = OutletBC("right", outlet_data)
    
    # Test cells
    inlet_interior = create_test_cell(1.0, 1.8, 1.0)  # Subcritical
    outlet_interior = create_test_cell(10.0, 1.5, 2.5)  # Supercritical
    
    print("INLET BOUNDARY:")
    print(f"  Interior: h={inlet_interior.U.h:.2f}m, u={inlet_interior.U.u:.2f}m/s, Fr={inlet_interior.U.froude:.2f}")
    print(f"  Prescribed discharge: {inlet_data.discharge:.1f} m³/s")
    
    inlet_ghost = inlet_bc.apply_boundary_condition(inlet_interior, None)
    print(f"  Ghost: h={inlet_ghost.h:.2f}m, u={inlet_ghost.u:.2f}m/s")
    print(f"  Calculated discharge: {inlet_ghost.hu:.2f} m³/s")
    
    print("\nOUTLET BOUNDARY:")
    print(f"  Interior: h={outlet_interior.U.h:.2f}m, u={outlet_interior.U.u:.2f}m/s, Fr={outlet_interior.U.froude:.2f}")
    print(f"  Prescribed depth: {outlet_data.depth:.1f} m")
    
    outlet_ghost = outlet_bc.apply_boundary_condition(outlet_interior, None)
    print(f"  Ghost: h={outlet_ghost.h:.2f}m, u={outlet_ghost.u:.2f}m/s")
    
    if outlet_interior.U.froude > 1.0:
        print(f"  ✅ Supercritical flow: Uses interior values (information flows downstream)")
    else:
        print(f"  ✅ Subcritical flow: Uses prescribed depth (downstream control)")
    print()


def demonstrate_critical_bc():
    """Demonstrate critical flow boundary condition."""
    print("⚡ CRITICAL FLOW BOUNDARY CONDITION")
    print("-" * 60)
    
    # Critical BC with prescribed discharge
    data = BoundaryData(discharge=8.0)
    critical_bc = CriticalBC("right", data)
    
    # Test interior cell
    interior_cell = create_test_cell(10.0, 1.8, 2.0)
    
    print(f"Interior: h={interior_cell.U.h:.2f}m, u={interior_cell.U.u:.2f}m/s, Fr={interior_cell.U.froude:.2f}")
    print(f"Prescribed discharge: {data.discharge:.1f} m³/s")
    
    # Apply critical BC
    critical_state = critical_bc.apply_boundary_condition(interior_cell, None)
    
    print(f"Critical state: h={critical_state.h:.2f}m, u={critical_state.u:.2f}m/s, Fr={critical_state.froude:.2f}")
    print(f"✅ Critical condition enforced: Fr ≈ 1.0")
    print()


def demonstrate_rating_curve_bc():
    """Demonstrate rating curve boundary condition."""
    print("📊 RATING CURVE BOUNDARY CONDITION")
    print("-" * 60)
    
    # Create rating curve data (stage-discharge relationship)
    stages = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    discharges = [0.1, 1.2, 3.5, 7.2, 12.5, 19.8]
    
    data = BoundaryData(stage_discharge_data={
        "stages": stages,
        "discharges": discharges
    })
    
    rating_bc = RatingCurveBC("right", data)
    
    # Test different depths
    test_depths = [1.2, 1.8, 2.3]
    
    print("Stage-Discharge Relationship:")
    for stage, discharge in zip(stages, discharges):
        print(f"  h={stage:.1f}m → Q={discharge:.1f}m³/s")
    
    print("\nTesting Rating Curve BC:")
    for depth in test_depths:
        interior_cell = create_test_cell(10.0, depth, 1.0)
        rating_state = rating_bc.apply_boundary_condition(interior_cell, None)
        
        print(f"  h={depth:.1f}m → Q={rating_state.hu:.1f}m³/s, u={rating_state.u:.2f}m/s")
    
    print("✅ Rating curve interpolates stage-discharge relationship")
    print()


def demonstrate_transmissive_reflective_bc():
    """Demonstrate transmissive and reflective boundary conditions."""
    print("🔄 TRANSMISSIVE & REFLECTIVE BOUNDARY CONDITIONS")
    print("-" * 60)
    
    # Create BCs
    transmissive_bc = TransmissiveBC("right")
    reflective_bc = ReflectiveBC("left")
    
    # Test cell with outflow
    outflow_cell = create_test_cell(10.0, 1.5, 2.0)
    
    print("TRANSMISSIVE BC (Non-reflecting outflow):")
    print(f"  Interior: h={outflow_cell.U.h:.2f}m, u={outflow_cell.U.u:.2f}m/s")
    
    transmissive_state = transmissive_bc.apply_boundary_condition(outflow_cell, None)
    transmissive_flux = transmissive_bc.calculate_boundary_flux(outflow_cell)
    
    print(f"  Ghost: h={transmissive_state.h:.2f}m, u={transmissive_state.u:.2f}m/s")
    print(f"  Flux: mass={transmissive_flux.mass_flux:.3f}")
    print("  ✅ Extrapolates interior values (zero gradient)")
    
    print("\nREFLECTIVE BC (Solid wall):")
    print(f"  Interior: h={outflow_cell.U.h:.2f}m, u={outflow_cell.U.u:.2f}m/s")
    
    reflective_state = reflective_bc.apply_boundary_condition(outflow_cell, None)
    reflective_flux = reflective_bc.calculate_boundary_flux(outflow_cell)
    
    print(f"  Ghost: h={reflective_state.h:.2f}m, u={reflective_state.u:.2f}m/s")
    print(f"  Flux: mass={reflective_flux.mass_flux:.3f}")
    print("  ✅ Reflects velocity, zero flux at wall")
    print()


def demonstrate_boundary_manager():
    """Demonstrate the boundary manager system."""
    print("🎛️ BOUNDARY MANAGER SYSTEM")
    print("-" * 60)
    
    # Create boundary manager
    manager = BoundaryManager()
    
    # Set up boundaries
    inlet_data = BoundaryData(discharge=10.0)
    outlet_data = BoundaryData(depth=1.5)
    
    left_bc = InletBC("left", inlet_data)
    right_bc = OutletBC("right", outlet_data)
    
    manager.set_left_boundary(left_bc)
    manager.set_right_boundary(right_bc)
    
    # Create test cells
    cells = [
        create_test_cell(1.0, 2.0, 1.8),   # Left boundary cell
        create_test_cell(5.0, 1.8, 2.2),   # Interior cell
        create_test_cell(10.0, 1.6, 2.5)   # Right boundary cell
    ]
    
    print("Boundary Setup:")
    summary = manager.get_boundary_summary()
    print(f"  Left BC: {summary['left_bc']['type']}")
    print(f"  Right BC: {summary['right_bc']['type']}")
    
    # Apply boundaries
    left_ghost, right_ghost = manager.apply_all_boundaries(cells)
    left_flux, right_flux = manager.calculate_boundary_fluxes(cells)
    
    print(f"\nBoundary States:")
    print(f"  Left ghost: h={left_ghost.h:.2f}m, u={left_ghost.u:.2f}m/s")
    print(f"  Right ghost: h={right_ghost.h:.2f}m, u={right_ghost.u:.2f}m/s")
    
    print(f"\nBoundary Fluxes:")
    print(f"  Left flux: mass={left_flux.mass_flux:.3f}")
    print(f"  Right flux: mass={right_flux.mass_flux:.3f}")
    
    # Validation
    warnings = manager.validate_boundary_setup()
    if warnings:
        print(f"\n⚠️ Warnings: {warnings}")
    else:
        print(f"\n✅ Boundary setup validated successfully")
    print()


def demonstrate_time_dependent_bc():
    """Demonstrate time-dependent boundary conditions."""
    print("⏰ TIME-DEPENDENT BOUNDARY CONDITIONS")
    print("-" * 60)
    
    # Create time series data
    time_series = {
        "discharge": [(0.0, 5.0), (10.0, 8.0), (20.0, 12.0), (30.0, 6.0)],
        "depth": [(0.0, 1.5), (15.0, 2.0), (30.0, 1.8)]
    }
    
    data = BoundaryData(time_series=time_series)
    inlet_bc = InletBC("left", data)
    
    # Test at different times
    test_times = [0.0, 5.0, 15.0, 25.0, 30.0]
    interior_cell = create_test_cell(1.0, 1.8, 1.0)
    
    print("Time Series Data:")
    print("  Discharge: t=0→5.0, t=10→8.0, t=20→12.0, t=30→6.0 m³/s")
    print("  Depth: t=0→1.5, t=15→2.0, t=30→1.8 m")
    
    print(f"\nTime-Dependent BC Response:")
    for t in test_times:
        ghost_state = inlet_bc.apply_boundary_condition(interior_cell, None, time=t)
        
        # Get interpolated values
        discharge = inlet_bc.get_time_dependent_value("discharge", t)
        depth = inlet_bc.get_time_dependent_value("depth", t)
        
        print(f"  t={t:4.1f}s: Q={discharge:4.1f}m³/s, h_prescribed={depth:4.1f}m → "
              f"h={ghost_state.h:.2f}m, u={ghost_state.u:.2f}m/s")
    
    print("✅ Boundary conditions interpolate smoothly over time")
    print()


def main():
    """Run the FVM boundary conditions demonstration."""
    
    if not FVM_AVAILABLE:
        print("❌ FVM modules not available. Please check the implementation.")
        return
    
    print("🚀 FVM BOUNDARY CONDITIONS DEMONSTRATION")
    print("=" * 80)
    print("Comprehensive boundary condition system for professional FVM analysis")
    print("Author: Alexius Academia")
    print("=" * 80)
    
    try:
        # Demonstrate different BC types
        demonstrate_dirichlet_bc()
        demonstrate_neumann_bc()
        demonstrate_inlet_outlet_bc()
        demonstrate_critical_bc()
        demonstrate_rating_curve_bc()
        demonstrate_transmissive_reflective_bc()
        demonstrate_boundary_manager()
        demonstrate_time_dependent_bc()
        
        print("=" * 80)
        print("🎉 BOUNDARY CONDITIONS DEMONSTRATION COMPLETE!")
        print("=" * 80)
        
        print(f"\n💡 KEY FEATURES DEMONSTRATED:")
        print(f"   ✅ 8 boundary condition types")
        print(f"   ✅ Flow regime awareness (subcritical/supercritical)")
        print(f"   ✅ Time-dependent conditions")
        print(f"   ✅ Rating curve interpolation")
        print(f"   ✅ Automatic validation")
        print(f"   ✅ Professional boundary manager")
        
        print(f"\n🎯 BOUNDARY CONDITION TYPES:")
        print(f"   • Dirichlet: Prescribed values")
        print(f"   • Neumann: Prescribed gradients")
        print(f"   • Inlet: Flow entering domain")
        print(f"   • Outlet: Flow leaving domain")
        print(f"   • Critical: Critical flow condition")
        print(f"   • Rating Curve: Stage-discharge relationship")
        print(f"   • Transmissive: Non-reflecting outflow")
        print(f"   • Reflective: Solid wall")
        
        print(f"\n🚀 READY FOR FVM SOLVER INTEGRATION!")
        
    except Exception as e:
        print(f"\n❌ Demonstration failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
