#!/usr/bin/env python3
"""
File: solve_for_unknowns.py
Author: Alexius Academia
Date: 2025-08-17

Demonstration of solving for different unknowns in PyOpenChannel.

This example shows how you can solve for different variables depending on
what information is known:
- Given Q, find normal depth
- Given depth, find discharge
- Given Q and depth, find required slope
- Given Q and slope, find required Manning's n
- Design channel dimensions for given capacity
"""

import sys
import os

# Add the src directory to the path so we can import pyopenchannel
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pyopenchannel as poc


def solve_for_normal_depth():
    """Example: Given Q, slope, n, and geometry → find normal depth."""
    print("=" * 60)
    print("SOLVE FOR NORMAL DEPTH")
    print("=" * 60)
    print("Known: Channel geometry, discharge, slope, Manning's n")
    print("Unknown: Normal depth")
    print()
    
    # Known parameters
    channel = poc.RectangularChannel(width=4.0)
    discharge = 10.0  # m³/s
    slope = 0.002     # dimensionless
    manning_n = 0.025
    
    print(f"Channel width: {channel.width} m")
    print(f"Discharge: {discharge} m³/s")
    print(f"Slope: {slope} ({slope*100:.1f}%)")
    print(f"Manning's n: {manning_n}")
    print()
    
    # Solve for normal depth
    normal_depth = poc.NormalDepth.calculate(channel, discharge, slope, manning_n)
    
    print(f"→ Normal depth: {normal_depth:.3f} m")
    
    # Verify the solution
    area = channel.area(normal_depth)
    hydraulic_radius = channel.hydraulic_radius(normal_depth)
    calculated_q = poc.ManningEquation.discharge(area, hydraulic_radius, slope, manning_n)
    
    print(f"Verification: Calculated Q = {calculated_q:.3f} m³/s (should match {discharge})")
    print()


def solve_for_discharge():
    """Example: Given depth, slope, n, and geometry → find discharge."""
    print("=" * 60)
    print("SOLVE FOR DISCHARGE")
    print("=" * 60)
    print("Known: Channel geometry, depth, slope, Manning's n")
    print("Unknown: Discharge")
    print()
    
    # Known parameters
    channel = poc.RectangularChannel(width=1.0)
    depth = 0.989       # m
    slope = 0.001     # dimensionless
    manning_n = 0.015
    
    print(f"Channel: Rectangular, width = {channel.width} m")
    print(f"Flow depth: {depth} m")
    print(f"Slope: {slope} ({slope*100:.1f}%)")
    print(f"Manning's n: {manning_n}")
    print()
    
    # Calculate discharge using Manning's equation
    area = channel.area(depth)
    hydraulic_radius = channel.hydraulic_radius(depth)
    discharge = poc.ManningEquation.discharge(area, hydraulic_radius, slope, manning_n)
    
    print(f"→ Discharge: {discharge:.3f} m³/s")
    
    # Additional flow properties
    velocity = discharge / area
    top_width = channel.top_width(depth)
    hydraulic_depth = area / top_width
    froude_number = poc.CriticalDepth.froude_number(velocity, hydraulic_depth)
    
    print(f"Flow velocity: {velocity:.3f} m/s")
    print(f"Froude number: {froude_number:.3f}")
    print(f"Flow regime: {'Subcritical' if froude_number < 1.0 else 'Supercritical'}")
    print()


def solve_for_required_slope():
    """Example: Given Q, depth, n, and geometry → find required slope."""
    print("=" * 60)
    print("SOLVE FOR REQUIRED SLOPE")
    print("=" * 60)
    print("Known: Channel geometry, discharge, depth, Manning's n")
    print("Unknown: Required slope")
    print()
    
    # Known parameters
    channel = poc.RectangularChannel(width=5.0)
    discharge = 15.0  # m³/s
    depth = 2.5       # m (desired flow depth)
    manning_n = 0.020
    
    print(f"Channel width: {channel.width} m")
    print(f"Discharge: {discharge} m³/s")
    print(f"Desired depth: {depth} m")
    print(f"Manning's n: {manning_n}")
    print()
    
    # Calculate required slope
    area = channel.area(depth)
    hydraulic_radius = channel.hydraulic_radius(depth)
    required_slope = poc.ManningEquation.required_slope(discharge, area, hydraulic_radius, manning_n)
    
    print(f"→ Required slope: {required_slope:.6f} ({required_slope*100:.4f}%)")
    
    # Verify the solution
    calculated_q = poc.ManningEquation.discharge(area, hydraulic_radius, required_slope, manning_n)
    print(f"Verification: Calculated Q = {calculated_q:.3f} m³/s (should match {discharge})")
    print()


def solve_for_manning_n():
    """Example: Given Q, depth, slope, and geometry → find required Manning's n."""
    print("=" * 60)
    print("SOLVE FOR REQUIRED MANNING'S N")
    print("=" * 60)
    print("Known: Channel geometry, discharge, depth, slope")
    print("Unknown: Required Manning's roughness coefficient")
    print()
    
    # Known parameters
    channel = poc.CircularChannel(diameter=1.5)
    discharge = 1.2   # m³/s
    depth = 0.8       # m
    slope = 0.003     # dimensionless
    
    print(f"Channel: Circular pipe, diameter = {channel.diameter} m")
    print(f"Discharge: {discharge} m³/s")
    print(f"Flow depth: {depth} m")
    print(f"Slope: {slope} ({slope*100:.1f}%)")
    print()
    
    # Calculate required Manning's n
    area = channel.area(depth)
    hydraulic_radius = channel.hydraulic_radius(depth)
    
    # From Manning's equation: n = (A * R^(2/3) * S^(1/2)) / Q
    required_n = (area * (hydraulic_radius**(2/3)) * (slope**0.5)) / discharge
    
    print(f"→ Required Manning's n: {required_n:.4f}")
    
    # Compare with typical values
    print("\nComparison with typical values:")
    print("  Smooth concrete pipe: 0.012")
    print("  Concrete pipe: 0.013")
    print("  Corrugated metal pipe: 0.025")
    
    if required_n < 0.015:
        print("  → Very smooth surface required")
    elif required_n < 0.025:
        print("  → Smooth to moderate surface")
    else:
        print("  → Rough surface")
    
    # Verify the solution
    calculated_q = poc.ManningEquation.discharge(area, hydraulic_radius, slope, required_n)
    print(f"\nVerification: Calculated Q = {calculated_q:.3f} m³/s (should match {discharge})")
    print()


def solve_for_critical_conditions():
    """Example: Solve for critical depth and critical slope."""
    print("=" * 60)
    print("SOLVE FOR CRITICAL CONDITIONS")
    print("=" * 60)
    print("Known: Channel geometry, discharge")
    print("Unknown: Critical depth, critical slope")
    print()
    
    # Known parameters
    channel = poc.TrapezoidalChannel(bottom_width=2.5, side_slope=2.0)
    discharge = 12.0  # m³/s
    manning_n = 0.025
    
    print(f"Channel: Trapezoidal, bottom width = {channel.bottom_width} m, side slope = {channel.side_slope}:1")
    print(f"Discharge: {discharge} m³/s")
    print(f"Manning's n: {manning_n}")
    print()
    
    # Solve for critical depth
    critical_flow = poc.CriticalFlow(channel)
    critical_depth = critical_flow.calculate_critical_depth(discharge)
    
    print(f"→ Critical depth: {critical_depth:.3f} m")
    
    # Solve for critical slope
    critical_slope = critical_flow.calculate_critical_slope(discharge, manning_n)
    
    print(f"→ Critical slope: {critical_slope:.6f} ({critical_slope*100:.4f}%)")
    
    # Additional critical flow properties
    critical_velocity = critical_flow.calculate_critical_velocity(critical_depth)
    area_c = channel.area(critical_depth)
    top_width_c = channel.top_width(critical_depth)
    
    print(f"Critical velocity: {critical_velocity:.3f} m/s")
    print(f"Critical area: {area_c:.3f} m²")
    print(f"Critical top width: {top_width_c:.3f} m")
    
    # Verify Froude number = 1.0
    hydraulic_depth_c = area_c / top_width_c
    froude_c = critical_velocity / (poc.GRAVITY * hydraulic_depth_c)**0.5
    print(f"Froude number: {froude_c:.3f} (should be 1.0)")
    print()


def solve_for_channel_dimensions():
    """Example: Design channel dimensions for given capacity."""
    print("=" * 60)
    print("SOLVE FOR CHANNEL DIMENSIONS")
    print("=" * 60)
    print("Known: Required discharge, slope, Manning's n, channel type")
    print("Unknown: Channel dimensions")
    print()
    
    # Design requirements
    discharge = 20.0  # m³/s
    slope = 0.0015    # dimensionless
    manning_n = 0.030
    
    print(f"Required discharge: {discharge} m³/s")
    print(f"Available slope: {slope} ({slope*100:.2f}%)")
    print(f"Manning's n: {manning_n}")
    print()
    
    # Design optimal rectangular channel
    print("OPTIMAL RECTANGULAR DESIGN:")
    rect_result = poc.OptimalSections.rectangular(discharge, slope, manning_n)
    print(f"→ Width: {rect_result.channel.width:.3f} m")
    print(f"→ Depth: {rect_result.depth:.3f} m")
    print(f"→ Velocity: {rect_result.velocity:.3f} m/s")
    print()
    
    # Design optimal trapezoidal channel
    print("OPTIMAL TRAPEZOIDAL DESIGN (side slope 1.5:1):")
    trap_result = poc.OptimalSections.trapezoidal(discharge, slope, manning_n, side_slope=1.5)
    print(f"→ Bottom width: {trap_result.channel.bottom_width:.3f} m")
    print(f"→ Depth: {trap_result.depth:.3f} m")
    print(f"→ Top width: {trap_result.channel.top_width(trap_result.depth):.3f} m")
    print(f"→ Velocity: {trap_result.velocity:.3f} m/s")
    print()
    
    # Compare excavation areas
    print("COMPARISON:")
    print(f"Rectangular excavation: {rect_result.excavation_area:.3f} m²")
    print(f"Trapezoidal excavation: {trap_result.excavation_area:.3f} m²")
    
    if trap_result.excavation_area < rect_result.excavation_area:
        savings = (rect_result.excavation_area - trap_result.excavation_area) / rect_result.excavation_area * 100
        print(f"→ Trapezoidal saves {savings:.1f}% excavation")
    else:
        extra = (trap_result.excavation_area - rect_result.excavation_area) / rect_result.excavation_area * 100
        print(f"→ Trapezoidal requires {extra:.1f}% more excavation")
    print()


def solve_alternate_depths():
    """Example: Given specific energy, find alternate depths."""
    print("=" * 60)
    print("SOLVE FOR ALTERNATE DEPTHS")
    print("=" * 60)
    print("Known: Channel geometry, discharge, specific energy")
    print("Unknown: Alternate depths (subcritical and supercritical)")
    print()
    
    # Known parameters
    channel = poc.RectangularChannel(width=6.0)
    discharge = 18.0  # m³/s
    specific_energy = 2.5  # m
    
    print(f"Channel width: {channel.width} m")
    print(f"Discharge: {discharge} m³/s")
    print(f"Specific energy: {specific_energy} m")
    print()
    
    # Find critical conditions first
    critical_depth = poc.CriticalDepth.calculate(channel, discharge)
    min_energy = poc.EnergyEquation.minimum_specific_energy(channel, discharge)
    
    print(f"Critical depth: {critical_depth:.3f} m")
    print(f"Minimum specific energy: {min_energy:.3f} m")
    print()
    
    if specific_energy > min_energy:
        # Solve for alternate depths
        subcritical_depth, supercritical_depth = poc.EnergyEquation.alternate_depths(
            channel, discharge, specific_energy
        )
        
        print(f"→ Subcritical depth: {subcritical_depth:.3f} m")
        print(f"→ Supercritical depth: {supercritical_depth:.3f} m")
        
        # Calculate corresponding velocities and Froude numbers
        area_sub = channel.area(subcritical_depth)
        area_super = channel.area(supercritical_depth)
        velocity_sub = discharge / area_sub
        velocity_super = discharge / area_super
        
        fr_sub = poc.CriticalDepth.froude_number(velocity_sub, subcritical_depth)
        fr_super = poc.CriticalDepth.froude_number(velocity_super, supercritical_depth)
        
        print(f"Subcritical: V = {velocity_sub:.3f} m/s, Fr = {fr_sub:.3f}")
        print(f"Supercritical: V = {velocity_super:.3f} m/s, Fr = {fr_super:.3f}")
        
        # Verify specific energies
        E_sub = poc.EnergyEquation.specific_energy(subcritical_depth, velocity_sub)
        E_super = poc.EnergyEquation.specific_energy(supercritical_depth, velocity_super)
        
        print(f"\nVerification:")
        print(f"Subcritical E = {E_sub:.3f} m (should match {specific_energy})")
        print(f"Supercritical E = {E_super:.3f} m (should match {specific_energy})")
    else:
        print(f"ERROR: Given specific energy ({specific_energy} m) is less than minimum ({min_energy:.3f} m)")
    
    print()


def main():
    """Run all examples demonstrating solving for different unknowns."""
    print("PyOpenChannel - Solving for Different Unknowns")
    print("=" * 60)
    print("This example demonstrates the flexibility of PyOpenChannel")
    print("to solve for different variables depending on known information.")
    print()
    
    try:
        solve_for_normal_depth()
        solve_for_discharge()
        solve_for_required_slope()
        solve_for_manning_n()
        solve_for_critical_conditions()
        solve_for_channel_dimensions()
        solve_alternate_depths()
        
        print("=" * 60)
        print("SUMMARY: PyOpenChannel Flexibility")
        print("=" * 60)
        print("✅ Solve for normal depth (given Q, S, n)")
        print("✅ Solve for discharge (given depth, S, n)")
        print("✅ Solve for required slope (given Q, depth, n)")
        print("✅ Solve for required Manning's n (given Q, depth, S)")
        print("✅ Solve for critical conditions (given Q)")
        print("✅ Solve for channel dimensions (given Q, S, n)")
        print("✅ Solve for alternate depths (given Q, E)")
        print()
        print("The library provides maximum flexibility for hydraulic analysis!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
