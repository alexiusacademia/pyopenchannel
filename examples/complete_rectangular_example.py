#!/usr/bin/env python3
"""
Complete Rectangular Channel Example - PyOpenChannel

This comprehensive example demonstrates the full workflow for rectangular channel analysis:
1. Unit system setup
2. Channel geometry definition
3. Hydraulic calculations (Manning's equation, critical depth, normal depth)
4. Flow analysis (uniform flow, critical flow, energy, momentum)
5. Channel design (optimal sections, economic design)
6. Advanced calculations (alternate depths, hydraulic jumps)
7. Unit system switching and comparisons

This example shows the complete capabilities of PyOpenChannel for rectangular channels.
"""

import sys
import os

# Add the src directory to the path so we can import pyopenchannel
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pyopenchannel as poc
from pyopenchannel.units import UnitSystem


def section_header(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"{title.center(70)}")
    print("=" * 70)


def subsection_header(title):
    """Print a formatted subsection header."""
    print(f"\n{title}")
    print("-" * len(title))


def main():
    """Complete rectangular channel analysis workflow."""
    print("PyOpenChannel - Complete Rectangular Channel Example")
    print("=" * 70)
    print("This example demonstrates the full workflow for rectangular channel analysis")
    print("including unit systems, geometry, hydraulics, flow analysis, and design.")
    
    # =================================================================
    # STEP 1: UNIT SYSTEM SETUP
    # =================================================================
    section_header("STEP 1: UNIT SYSTEM SETUP")
    
    print("PyOpenChannel supports both SI (metric) and US Customary unit systems.")
    print("Let's start with SI units (default) and later switch to US units.")
    print()
    
    # Set SI units (this is the default, but shown for clarity)
    poc.set_unit_system(UnitSystem.SI)
    units = poc.get_unit_system()
    
    print(f"Current unit system: {units}")
    print(f"Length unit: {units.length_unit}")
    print(f"Discharge unit: {units.discharge_unit}")
    print(f"Velocity unit: {units.velocity_unit}")
    print(f"Gravity constant: {poc.get_gravity():.2f} {units.length_unit}/s²")
    print(f"Manning factor: {poc.get_manning_factor():.3f}")
    
    # =================================================================
    # STEP 2: CHANNEL GEOMETRY DEFINITION
    # =================================================================
    section_header("STEP 2: CHANNEL GEOMETRY DEFINITION")
    
    # Define rectangular channel
    channel_width = 1.0  # meters
    channel = poc.RectangularChannel(width=channel_width)
    
    print(f"Rectangular channel created:")
    print(f"  Width: {channel.width} {units.length_unit}")
    print(f"  Channel type: {type(channel).__name__}")
    print()
    
    # Demonstrate geometry calculations at different depths
    test_depths = [0.5, 1.0, 1.5, 2.0, 2.5]  # meters
    
    print("Geometric properties at various depths:")
    print(f"{'Depth':<8} {'Area':<8} {'Wetted P':<10} {'Top Width':<10} {'Hyd. Radius':<12}")
    print(f"({units.length_unit})"+" "*7 + f"({units.area_unit})"+" "*5 + f"({units.length_unit})"+" "*8 + f"({units.length_unit})"+" "*8 + f"({units.length_unit})")
    print("-" * 60)
    
    for depth in test_depths:
        area = channel.area(depth)
        perimeter = channel.wetted_perimeter(depth)
        top_width = channel.top_width(depth)
        hydraulic_radius = channel.hydraulic_radius(depth)
        
        print(f"{depth:<8.1f} {area:<8.2f} {perimeter:<10.2f} {top_width:<10.1f} {hydraulic_radius:<12.3f}")
    
    # =================================================================
    # STEP 3: HYDRAULIC CALCULATIONS
    # =================================================================
    section_header("STEP 3: HYDRAULIC CALCULATIONS")
    
    # Define flow conditions
    discharge = 1.0  # m³/s
    slope = 0.001     # dimensionless (0.2%)
    manning_n = 0.015 # concrete channel
    
    print("Flow conditions:")
    print(f"  Design discharge: {discharge} {units.discharge_unit}")
    print(f"  Channel slope: {slope} ({slope*100:.1f}%)")
    print(f"  Manning's roughness coefficient: {manning_n}")
    print()
    
    subsection_header("3.1 Normal Depth Calculation")
    
    # Calculate normal depth
    normal_depth = poc.NormalDepth.calculate(channel, discharge, slope, manning_n)
    
    print(f"Normal depth: {normal_depth:.3f} {units.length_unit}")
    
    # Verify by calculating discharge back
    area_normal = channel.area(normal_depth)
    hydraulic_radius_normal = channel.hydraulic_radius(normal_depth)
    calculated_discharge = poc.ManningEquation.discharge(
        area_normal, hydraulic_radius_normal, slope, manning_n
    )
    
    print(f"Verification - calculated discharge: {calculated_discharge:.3f} {units.discharge_unit}")
    print(f"Difference: {abs(calculated_discharge - discharge):.6f} {units.discharge_unit}")
    
    subsection_header("3.2 Critical Depth Calculation")
    
    # Calculate critical depth
    critical_depth = poc.CriticalDepth.calculate(channel, discharge)
    
    print(f"Critical depth: {critical_depth:.3f} {units.length_unit}")
    
    # Calculate critical velocity
    area_critical = channel.area(critical_depth)
    critical_velocity = discharge / area_critical
    
    print(f"Critical velocity: {critical_velocity:.3f} {units.velocity_unit}")
    
    # Verify Froude number = 1.0 at critical depth
    hydraulic_depth_critical = area_critical / channel.top_width(critical_depth)
    froude_critical = poc.CriticalDepth.froude_number(critical_velocity, hydraulic_depth_critical)
    
    print(f"Froude number at critical depth: {froude_critical:.6f} (should be 1.0)")
    
    subsection_header("3.3 Flow Regime Classification")
    
    # Compare normal and critical depths
    if normal_depth > critical_depth:
        flow_regime = "Subcritical"
        slope_type = "Mild"
    elif normal_depth < critical_depth:
        flow_regime = "Supercritical"
        slope_type = "Steep"
    else:
        flow_regime = "Critical"
        slope_type = "Critical"
    
    print(f"Normal depth ({normal_depth:.3f} {units.length_unit}) vs Critical depth ({critical_depth:.3f} {units.length_unit})")
    print(f"Flow regime: {flow_regime}")
    print(f"Slope classification: {slope_type}")
    
    subsection_header("3.4 Manning's Equation Applications")
    
    # Calculate velocity using Manning's equation
    velocity_manning = poc.ManningEquation.velocity(hydraulic_radius_normal, slope, manning_n)
    velocity_continuity = discharge / area_normal
    
    print(f"Velocity from Manning's equation: {velocity_manning:.3f} {units.velocity_unit}")
    print(f"Velocity from continuity: {velocity_continuity:.3f} {units.velocity_unit}")
    print(f"Difference: {abs(velocity_manning - velocity_continuity):.6f} {units.velocity_unit}")
    
    # Calculate required slope for different depths
    print(f"\nRequired slope for different depths (Q = {discharge} {units.discharge_unit}):")
    print(f"{'Depth':<8} {'Required Slope':<15} {'Percentage'}")
    print(f"({units.length_unit})")
    print("-" * 35)
    
    for depth in [1.0, 1.5, 2.0, 2.5, 3.0]:
        area = channel.area(depth)
        hydraulic_radius = channel.hydraulic_radius(depth)
        required_slope = poc.ManningEquation.required_slope(
            discharge, area, hydraulic_radius, manning_n
        )
        print(f"{depth:<8.1f} {required_slope:<15.6f} {required_slope*100:.4f}%")
    
    # =================================================================
    # STEP 4: FLOW ANALYSIS
    # =================================================================
    section_header("STEP 4: FLOW ANALYSIS")
    
    subsection_header("4.1 Uniform Flow Analysis")
    
    # Create uniform flow analyzer
    uniform_flow = poc.UniformFlow(channel, slope, manning_n)
    flow_state = uniform_flow.calculate_flow_state(discharge)
    
    print("Complete flow state at normal depth:")
    print(f"  Depth: {flow_state.depth:.3f} {units.length_unit}")
    print(f"  Velocity: {flow_state.velocity:.3f} {units.velocity_unit}")
    print(f"  Discharge: {flow_state.discharge:.3f} {units.discharge_unit}")
    print(f"  Area: {flow_state.area:.3f} {units.area_unit}")
    print(f"  Top width: {flow_state.top_width:.3f} {units.length_unit}")
    print(f"  Hydraulic radius: {flow_state.hydraulic_radius:.3f} {units.length_unit}")
    print(f"  Froude number: {flow_state.froude_number:.3f}")
    print(f"  Specific energy: {flow_state.specific_energy:.3f} {units.length_unit}")
    print(f"  Momentum function: {flow_state.momentum:.3f} {units.length_unit}³")
    print()
    
    # Flow classification
    if flow_state.is_subcritical:
        print("Flow classification: Subcritical (Fr < 1)")
    elif flow_state.is_supercritical:
        print("Flow classification: Supercritical (Fr > 1)")
    else:
        print("Flow classification: Critical (Fr ≈ 1)")
    
    subsection_header("4.2 Critical Flow Analysis")
    
    # Create critical flow analyzer
    critical_flow = poc.CriticalFlow(channel)
    
    # Calculate critical properties
    critical_depth_calc = critical_flow.calculate_critical_depth(discharge)
    critical_velocity_calc = critical_flow.calculate_critical_velocity(critical_depth_calc)
    critical_discharge_check = critical_flow.calculate_critical_discharge(critical_depth_calc)
    critical_slope = critical_flow.calculate_critical_slope(discharge, manning_n)
    
    print("Critical flow properties:")
    print(f"  Critical depth: {critical_depth_calc:.3f} {units.length_unit}")
    print(f"  Critical velocity: {critical_velocity_calc:.3f} {units.velocity_unit}")
    print(f"  Critical discharge (verification): {critical_discharge_check:.3f} {units.discharge_unit}")
    print(f"  Critical slope: {critical_slope:.6f} ({critical_slope*100:.4f}%)")
    print()
    
    # Compare with actual slope
    if slope > critical_slope:
        print(f"Actual slope ({slope:.6f}) > Critical slope ({critical_slope:.6f})")
        print("Channel slope is STEEP - normal flow will be supercritical")
    elif slope < critical_slope:
        print(f"Actual slope ({slope:.6f}) < Critical slope ({critical_slope:.6f})")
        print("Channel slope is MILD - normal flow will be subcritical")
    else:
        print("Channel slope equals critical slope - normal flow will be critical")
    
    subsection_header("4.3 Energy Analysis")
    
    # Calculate specific energies
    specific_energy_normal = poc.EnergyEquation.specific_energy(flow_state.depth, flow_state.velocity)
    min_specific_energy = poc.EnergyEquation.minimum_specific_energy(channel, discharge)
    
    print("Energy analysis:")
    print(f"  Specific energy at normal depth: {specific_energy_normal:.3f} {units.length_unit}")
    print(f"  Minimum specific energy: {min_specific_energy:.3f} {units.length_unit}")
    print(f"  Energy above minimum: {specific_energy_normal - min_specific_energy:.3f} {units.length_unit}")
    print()
    
    # Calculate alternate depths for a given specific energy
    test_energy = min_specific_energy + 0.5  # Add 0.5 m above minimum
    
    try:
        subcritical_depth, supercritical_depth = poc.EnergyEquation.alternate_depths(
            channel, discharge, test_energy
        )
        
        print(f"Alternate depths for specific energy = {test_energy:.3f} {units.length_unit}:")
        print(f"  Subcritical depth: {subcritical_depth:.3f} {units.length_unit}")
        print(f"  Supercritical depth: {supercritical_depth:.3f} {units.length_unit}")
        
        # Calculate corresponding velocities and Froude numbers
        area_sub = channel.area(subcritical_depth)
        area_super = channel.area(supercritical_depth)
        velocity_sub = discharge / area_sub
        velocity_super = discharge / area_super
        
        hydraulic_depth_sub = area_sub / channel.top_width(subcritical_depth)
        hydraulic_depth_super = area_super / channel.top_width(supercritical_depth)
        
        froude_sub = poc.CriticalDepth.froude_number(velocity_sub, hydraulic_depth_sub)
        froude_super = poc.CriticalDepth.froude_number(velocity_super, hydraulic_depth_super)
        
        print(f"  Subcritical: V = {velocity_sub:.3f} {units.velocity_unit}, Fr = {froude_sub:.3f}")
        print(f"  Supercritical: V = {velocity_super:.3f} {units.velocity_unit}, Fr = {froude_super:.3f}")
        
    except Exception as e:
        print(f"Alternate depths calculation error: {e}")
    
    subsection_header("4.4 Momentum Analysis")
    
    # Calculate momentum function
    momentum_normal = poc.MomentumEquation.momentum_function(channel, flow_state.depth, discharge)
    
    print("Momentum analysis:")
    print(f"  Momentum function at normal depth: {momentum_normal:.3f} {units.length_unit}³")
    print()
    
    # Demonstrate hydraulic jump (if flow is supercritical)
    if flow_state.froude_number > 1.0:
        try:
            conjugate_depth = poc.MomentumEquation.conjugate_depths(
                channel, discharge, flow_state.depth
            )
            
            print("Hydraulic jump analysis:")
            print(f"  Upstream depth (supercritical): {flow_state.depth:.3f} {units.length_unit}")
            print(f"  Downstream depth (subcritical): {conjugate_depth:.3f} {units.length_unit}")
            print(f"  Jump height: {conjugate_depth - flow_state.depth:.3f} {units.length_unit}")
            
        except Exception as e:
            print(f"Hydraulic jump calculation: {e}")
    else:
        print("Flow is subcritical - no hydraulic jump possible")
        print("(Hydraulic jumps only occur from supercritical to subcritical flow)")
    
    # =================================================================
    # STEP 5: CHANNEL DESIGN
    # =================================================================
    section_header("STEP 5: CHANNEL DESIGN")
    
    subsection_header("5.1 Optimal Channel Design")
    
    # Design optimal rectangular channel for hydraulic efficiency
    optimal_result = poc.OptimalSections.rectangular(discharge, slope, manning_n)
    
    print("Optimal rectangular channel design (minimum wetted perimeter):")
    print(f"  Optimal width: {optimal_result.channel.width:.3f} {units.length_unit}")
    print(f"  Flow depth: {optimal_result.depth:.3f} {units.length_unit}")
    print(f"  Flow velocity: {optimal_result.velocity:.3f} {units.velocity_unit}")
    print(f"  Hydraulic radius: {optimal_result.hydraulic_radius:.3f} {units.length_unit}")
    print(f"  Froude number: {optimal_result.froude_number:.3f}")
    print(f"  Excavation area: {optimal_result.excavation_area:.3f} {units.area_unit}")
    print(f"  Recommended freeboard: {optimal_result.freeboard:.3f} {units.length_unit}")
    print(f"  Total channel depth: {optimal_result.total_depth:.3f} {units.length_unit}")
    print()
    
    # Verify optimal condition (width/depth = 2 for rectangular)
    width_to_depth_ratio = optimal_result.channel.width / optimal_result.depth
    print(f"Width to depth ratio: {width_to_depth_ratio:.3f}")
    print("(Optimal rectangular channel has width/depth = 2.0)")
    print()
    
    # Compare with our original channel
    print("Comparison with original channel:")
    print(f"  Original width: {channel.width:.3f} {units.length_unit}")
    print(f"  Optimal width: {optimal_result.channel.width:.3f} {units.length_unit}")
    print(f"  Original excavation: {channel.width * normal_depth:.3f} {units.area_unit}")
    print(f"  Optimal excavation: {optimal_result.excavation_area:.3f} {units.area_unit}")
    
    excavation_savings = ((channel.width * normal_depth - optimal_result.excavation_area) / 
                         (channel.width * normal_depth) * 100)
    print(f"  Excavation savings: {excavation_savings:.1f}%")
    
    subsection_header("5.2 Economic Channel Design")
    
    # Economic design considering costs
    economic_designer = poc.EconomicSections(
        excavation_cost_per_m3=30.0,  # $/m³
        lining_cost_per_m2=45.0,      # $/m²
        land_cost_per_m2=120.0        # $/m²
    )
    
    try:
        economic_result = economic_designer.design_rectangular(
            discharge, slope, manning_n, width_range=(2.0, 8.0)
        )
        
        print("Economic channel design (minimum total cost):")
        print(f"  Economic width: {economic_result.channel.width:.3f} {units.length_unit}")
        print(f"  Flow depth: {economic_result.depth:.3f} {units.length_unit}")
        print(f"  Flow velocity: {economic_result.velocity:.3f} {units.velocity_unit}")
        print(f"  Cost per meter: ${economic_result.cost_per_meter:.2f}/m")
        print(f"  Total channel depth: {economic_result.total_depth:.3f} {units.length_unit}")
        print()
        
        print("Cost comparison:")
        print(f"  Hydraulic optimal excavation: {optimal_result.excavation_area:.3f} {units.area_unit}")
        print(f"  Economic optimal excavation: {economic_result.excavation_area:.3f} {units.area_unit}")
        
        area_difference = ((economic_result.excavation_area - optimal_result.excavation_area) / 
                          optimal_result.excavation_area * 100)
        print(f"  Economic vs hydraulic optimal: {area_difference:+.1f}% excavation")
        
    except Exception as e:
        print(f"Economic design calculation error: {e}")
    
    subsection_header("5.3 Design Recommendations")
    
    # Velocity checks
    velocity_check = poc.ChannelDesigner.check_velocity_limits(
        flow_state.velocity, "concrete"
    )
    
    print("Design recommendations:")
    print(f"  Flow velocity: {flow_state.velocity:.3f} {units.velocity_unit}")
    print(f"  Velocity check: {'PASS' if velocity_check['is_acceptable'] else 'FAIL'}")
    
    if velocity_check['warnings']:
        for warning in velocity_check['warnings']:
            print(f"    Warning: {warning}")
    
    # Freeboard calculation
    recommended_freeboard = poc.ChannelDesigner.calculate_freeboard(
        discharge, flow_state.depth, flow_state.velocity, "concrete"
    )
    
    print(f"  Recommended freeboard: {recommended_freeboard:.3f} {units.length_unit}")
    print(f"  Total channel depth needed: {flow_state.depth + recommended_freeboard:.3f} {units.length_unit}")
    
    # =================================================================
    # STEP 6: UNIT SYSTEM SWITCHING
    # =================================================================
    section_header("STEP 6: UNIT SYSTEM SWITCHING")
    
    print("Now let's switch to US Customary units and repeat key calculations...")
    print()
    
    # Switch to US Customary units
    poc.set_unit_system('US_CUSTOMARY')
    units_us = poc.get_unit_system()
    
    print(f"Switched to: {units_us}")
    print(f"Length unit: {units_us.length_unit}")
    print(f"Discharge unit: {units_us.discharge_unit}")
    print(f"Gravity constant: {poc.get_gravity():.2f} {units_us.length_unit}/s²")
    print(f"Manning factor: {poc.get_manning_factor():.3f}")
    print()
    
    # Convert our channel and flow parameters to US units
    channel_width_ft = poc.m_to_ft(channel_width)
    discharge_cfs = poc.cms_to_cfs(discharge)
    
    # Create US channel
    channel_us = poc.RectangularChannel(width=channel_width_ft)
    
    print("Channel and flow parameters in US Customary units:")
    print(f"  Channel width: {channel_us.width:.3f} {units_us.length_unit}")
    print(f"  Discharge: {discharge_cfs:.3f} {units_us.discharge_unit}")
    print(f"  Slope: {slope} ({slope*100:.1f}%) - dimensionless, same in both systems")
    print(f"  Manning's n: {manning_n} - dimensionless, same in both systems")
    print()
    
    # Repeat key calculations in US units
    subsection_header("Key Calculations in US Customary Units")
    
    # Normal depth
    normal_depth_us = poc.NormalDepth.calculate(channel_us, discharge_cfs, slope, manning_n)
    print(f"Normal depth: {normal_depth_us:.3f} {units_us.length_unit}")
    
    # Critical depth
    critical_depth_us = poc.CriticalDepth.calculate(channel_us, discharge_cfs)
    print(f"Critical depth: {critical_depth_us:.3f} {units_us.length_unit}")
    
    # Flow state
    uniform_flow_us = poc.UniformFlow(channel_us, slope, manning_n)
    flow_state_us = uniform_flow_us.calculate_flow_state(discharge_cfs)
    
    print(f"Flow velocity: {flow_state_us.velocity:.3f} {units_us.velocity_unit}")
    print(f"Froude number: {flow_state_us.froude_number:.3f}")
    print(f"Specific energy: {flow_state_us.specific_energy:.3f} {units_us.length_unit}")
    print()
    
    # Optimal design in US units
    optimal_result_us = poc.OptimalSections.rectangular(discharge_cfs, slope, manning_n)
    
    print("Optimal design in US Customary units:")
    print(f"  Optimal width: {optimal_result_us.channel.width:.3f} {units_us.length_unit}")
    print(f"  Flow depth: {optimal_result_us.depth:.3f} {units_us.length_unit}")
    print(f"  Flow velocity: {optimal_result_us.velocity:.3f} {units_us.velocity_unit}")
    print()
    
    # =================================================================
    # STEP 7: UNIT CONSISTENCY VERIFICATION
    # =================================================================
    section_header("STEP 7: UNIT CONSISTENCY VERIFICATION")
    
    print("Verifying that results are physically consistent between unit systems...")
    print()
    
    # Convert US results back to SI for comparison
    normal_depth_us_to_si = poc.ft_to_m(normal_depth_us)
    critical_depth_us_to_si = poc.ft_to_m(critical_depth_us)
    velocity_us_to_si = poc.ft_to_m(flow_state_us.velocity)  # ft/s to m/s
    optimal_width_us_to_si = poc.ft_to_m(optimal_result_us.channel.width)
    optimal_depth_us_to_si = poc.ft_to_m(optimal_result_us.depth)
    
    print("Consistency check (US results converted back to SI):")
    print(f"{'Parameter':<20} {'SI Original':<12} {'US Converted':<12} {'Difference':<12}")
    print("-" * 60)
    print(f"{'Normal depth (m)':<20} {normal_depth:<12.6f} {normal_depth_us_to_si:<12.6f} {abs(normal_depth - normal_depth_us_to_si):<12.8f}")
    print(f"{'Critical depth (m)':<20} {critical_depth:<12.6f} {critical_depth_us_to_si:<12.6f} {abs(critical_depth - critical_depth_us_to_si):<12.8f}")
    print(f"{'Velocity (m/s)':<20} {flow_state.velocity:<12.6f} {velocity_us_to_si:<12.6f} {abs(flow_state.velocity - velocity_us_to_si):<12.8f}")
    print(f"{'Froude number':<20} {flow_state.froude_number:<12.6f} {flow_state_us.froude_number:<12.6f} {abs(flow_state.froude_number - flow_state_us.froude_number):<12.8f}")
    print(f"{'Optimal width (m)':<20} {optimal_result.channel.width:<12.6f} {optimal_width_us_to_si:<12.6f} {abs(optimal_result.channel.width - optimal_width_us_to_si):<12.8f}")
    print(f"{'Optimal depth (m)':<20} {optimal_result.depth:<12.6f} {optimal_depth_us_to_si:<12.6f} {abs(optimal_result.depth - optimal_depth_us_to_si):<12.8f}")
    print()
    
    # Check if differences are within engineering tolerance
    tolerance = 1e-5
    checks = [
        abs(normal_depth - normal_depth_us_to_si) < tolerance,
        abs(critical_depth - critical_depth_us_to_si) < tolerance,
        abs(flow_state.velocity - velocity_us_to_si) < tolerance,
        abs(flow_state.froude_number - flow_state_us.froude_number) < tolerance,
        abs(optimal_result.channel.width - optimal_width_us_to_si) < tolerance,
        abs(optimal_result.depth - optimal_depth_us_to_si) < tolerance,
    ]
    
    if all(checks):
        print("✅ UNIT CONSISTENCY CHECK: PASSED")
        print("All calculations are physically consistent between unit systems!")
    else:
        print("❌ UNIT CONSISTENCY CHECK: FAILED")
        print("Some calculations show inconsistencies between unit systems.")
    
    # Reset to SI units
    poc.set_unit_system('SI')
    
    # =================================================================
    # SUMMARY
    # =================================================================
    section_header("SUMMARY")
    
    print("This example demonstrated the complete PyOpenChannel workflow:")
    print()
    print("✅ Unit System Management:")
    print("   • SI and US Customary unit support")
    print("   • Automatic unit conversions")
    print("   • Consistent results across unit systems")
    print()
    print("✅ Channel Geometry:")
    print("   • Rectangular channel definition")
    print("   • Geometric property calculations")
    print("   • Area, perimeter, hydraulic radius, top width")
    print()
    print("✅ Hydraulic Calculations:")
    print("   • Manning's equation applications")
    print("   • Normal depth and critical depth")
    print("   • Flow regime classification")
    print("   • Velocity and discharge calculations")
    print()
    print("✅ Flow Analysis:")
    print("   • Uniform flow analysis")
    print("   • Critical flow conditions")
    print("   • Energy equation applications")
    print("   • Momentum equation and hydraulic jumps")
    print()
    print("✅ Channel Design:")
    print("   • Optimal hydraulic design")
    print("   • Economic optimization")
    print("   • Design recommendations")
    print("   • Freeboard and velocity checks")
    print()
    print("PyOpenChannel provides a complete solution for rectangular channel")
    print("analysis and design with professional engineering accuracy!")
    
    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
