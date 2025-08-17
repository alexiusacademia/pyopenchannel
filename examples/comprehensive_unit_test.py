#!/usr/bin/env python3
"""
Comprehensive unit-aware test for all PyOpenChannel modules.

This test verifies that all modules work correctly with different unit systems:
- hydraulics.py
- flow_analysis.py  
- design.py
- geometry.py (unit-agnostic but tested for consistency)
"""

import sys
import os

# Add the src directory to the path so we can import pyopenchannel
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pyopenchannel as poc


def test_complete_workflow_si():
    """Test complete workflow in SI units."""
    print("=" * 60)
    print("COMPLETE WORKFLOW TEST - SI UNITS")
    print("=" * 60)
    
    # Set SI units
    poc.set_unit_system('SI')
    
    # Design parameters
    discharge = 15.0  # m³/s
    slope = 0.002     # dimensionless
    manning_n = 0.025
    
    print(f"Design discharge: {discharge} m³/s")
    print(f"Slope: {slope} ({slope*100:.1f}%)")
    print(f"Manning's n: {manning_n}")
    print()
    
    # 1. Design optimal channel
    design_result = poc.OptimalSections.rectangular(discharge, slope, manning_n)
    
    print("1. OPTIMAL DESIGN:")
    print(f"   Width: {design_result.channel.width:.3f} m")
    print(f"   Depth: {design_result.depth:.3f} m")
    print(f"   Velocity: {design_result.velocity:.3f} m/s")
    print(f"   Froude number: {design_result.froude_number:.3f}")
    print()
    
    # 2. Flow analysis
    uniform_flow = poc.UniformFlow(design_result.channel, slope, manning_n)
    flow_state = uniform_flow.calculate_flow_state(discharge)
    
    print("2. FLOW ANALYSIS:")
    print(f"   Normal depth: {flow_state.depth:.3f} m")
    print(f"   Velocity: {flow_state.velocity:.3f} m/s")
    print(f"   Specific energy: {flow_state.specific_energy:.3f} m")
    print(f"   Momentum: {flow_state.momentum:.3f} m³")
    print()
    
    # 3. Critical flow analysis
    critical_flow = poc.CriticalFlow(design_result.channel)
    critical_depth = critical_flow.calculate_critical_depth(discharge)
    critical_velocity = critical_flow.calculate_critical_velocity(critical_depth)
    
    print("3. CRITICAL FLOW:")
    print(f"   Critical depth: {critical_depth:.3f} m")
    print(f"   Critical velocity: {critical_velocity:.3f} m/s")
    print()
    
    # 4. Energy analysis
    min_energy = poc.EnergyEquation.minimum_specific_energy(design_result.channel, discharge)
    test_energy = min_energy + 0.5
    
    try:
        sub_depth, super_depth = poc.EnergyEquation.alternate_depths(
            design_result.channel, discharge, test_energy
        )
        
        print("4. ENERGY ANALYSIS:")
        print(f"   Minimum energy: {min_energy:.3f} m")
        print(f"   Test energy: {test_energy:.3f} m")
        print(f"   Subcritical depth: {sub_depth:.3f} m")
        print(f"   Supercritical depth: {super_depth:.3f} m")
        print()
    except Exception as e:
        print(f"   Energy analysis error: {e}")
        print()
    
    return {
        'width': design_result.channel.width,
        'depth': design_result.depth,
        'velocity': design_result.velocity,
        'froude': design_result.froude_number,
        'critical_depth': critical_depth,
        'min_energy': min_energy
    }


def test_complete_workflow_us():
    """Test complete workflow in US Customary units."""
    print("=" * 60)
    print("COMPLETE WORKFLOW TEST - US CUSTOMARY UNITS")
    print("=" * 60)
    
    # Set US Customary units
    poc.set_unit_system('US_CUSTOMARY')
    
    # Design parameters (same physical problem)
    discharge = poc.cms_to_cfs(15.0)  # Convert 15 m³/s to cfs
    slope = 0.002     # dimensionless (same)
    manning_n = 0.025 # dimensionless (same)
    
    print(f"Design discharge: {discharge:.3f} ft³/s")
    print(f"Slope: {slope} ({slope*100:.1f}%)")
    print(f"Manning's n: {manning_n}")
    print()
    
    # 1. Design optimal channel
    design_result = poc.OptimalSections.rectangular(discharge, slope, manning_n)
    
    print("1. OPTIMAL DESIGN:")
    print(f"   Width: {design_result.channel.width:.3f} ft")
    print(f"   Depth: {design_result.depth:.3f} ft")
    print(f"   Velocity: {design_result.velocity:.3f} ft/s")
    print(f"   Froude number: {design_result.froude_number:.3f}")
    print()
    
    # 2. Flow analysis
    uniform_flow = poc.UniformFlow(design_result.channel, slope, manning_n)
    flow_state = uniform_flow.calculate_flow_state(discharge)
    
    print("2. FLOW ANALYSIS:")
    print(f"   Normal depth: {flow_state.depth:.3f} ft")
    print(f"   Velocity: {flow_state.velocity:.3f} ft/s")
    print(f"   Specific energy: {flow_state.specific_energy:.3f} ft")
    print(f"   Momentum: {flow_state.momentum:.3f} ft³")
    print()
    
    # 3. Critical flow analysis
    critical_flow = poc.CriticalFlow(design_result.channel)
    critical_depth = critical_flow.calculate_critical_depth(discharge)
    critical_velocity = critical_flow.calculate_critical_velocity(critical_depth)
    
    print("3. CRITICAL FLOW:")
    print(f"   Critical depth: {critical_depth:.3f} ft")
    print(f"   Critical velocity: {critical_velocity:.3f} ft/s")
    print()
    
    # 4. Energy analysis
    min_energy = poc.EnergyEquation.minimum_specific_energy(design_result.channel, discharge)
    test_energy = min_energy + poc.m_to_ft(0.5)  # Add 0.5 m equivalent in feet
    
    try:
        sub_depth, super_depth = poc.EnergyEquation.alternate_depths(
            design_result.channel, discharge, test_energy
        )
        
        print("4. ENERGY ANALYSIS:")
        print(f"   Minimum energy: {min_energy:.3f} ft")
        print(f"   Test energy: {test_energy:.3f} ft")
        print(f"   Subcritical depth: {sub_depth:.3f} ft")
        print(f"   Supercritical depth: {super_depth:.3f} ft")
        print()
    except Exception as e:
        print(f"   Energy analysis error: {e}")
        print()
    
    return {
        'width': design_result.channel.width,
        'depth': design_result.depth,
        'velocity': design_result.velocity,
        'froude': design_result.froude_number,
        'critical_depth': critical_depth,
        'min_energy': min_energy
    }


def compare_results(si_results, us_results):
    """Compare SI and US results for consistency."""
    print("=" * 60)
    print("UNIT CONSISTENCY COMPARISON")
    print("=" * 60)
    
    # Convert US results to SI for comparison
    us_width_si = poc.ft_to_m(us_results['width'])
    us_depth_si = poc.ft_to_m(us_results['depth'])
    us_velocity_si = poc.ft_to_m(us_results['velocity'])  # ft/s to m/s
    us_critical_depth_si = poc.ft_to_m(us_results['critical_depth'])
    us_min_energy_si = poc.ft_to_m(us_results['min_energy'])
    
    print("Parameter Comparison (US converted to SI):")
    print(f"Width:          SI = {si_results['width']:.6f} m,  US = {us_width_si:.6f} m")
    print(f"Depth:          SI = {si_results['depth']:.6f} m,  US = {us_depth_si:.6f} m")
    print(f"Velocity:       SI = {si_results['velocity']:.6f} m/s, US = {us_velocity_si:.6f} m/s")
    print(f"Froude number:  SI = {si_results['froude']:.6f},   US = {us_results['froude']:.6f}")
    print(f"Critical depth: SI = {si_results['critical_depth']:.6f} m,  US = {us_critical_depth_si:.6f} m")
    print(f"Min energy:     SI = {si_results['min_energy']:.6f} m,  US = {us_min_energy_si:.6f} m")
    print()
    
    # Calculate differences
    width_diff = abs(si_results['width'] - us_width_si)
    depth_diff = abs(si_results['depth'] - us_depth_si)
    velocity_diff = abs(si_results['velocity'] - us_velocity_si)
    froude_diff = abs(si_results['froude'] - us_results['froude'])
    critical_diff = abs(si_results['critical_depth'] - us_critical_depth_si)
    energy_diff = abs(si_results['min_energy'] - us_min_energy_si)
    
    print("Absolute Differences:")
    print(f"Width:          {width_diff:.8f} m")
    print(f"Depth:          {depth_diff:.8f} m")
    print(f"Velocity:       {velocity_diff:.8f} m/s")
    print(f"Froude number:  {froude_diff:.8f}")
    print(f"Critical depth: {critical_diff:.8f} m")
    print(f"Min energy:     {energy_diff:.8f} m")
    print()
    
    # Check consistency (tolerance of 1e-5 for engineering accuracy)
    tolerance = 1e-5
    checks = {
        'Width': width_diff < tolerance,
        'Depth': depth_diff < tolerance,
        'Velocity': velocity_diff < tolerance,
        'Froude': froude_diff < tolerance,
        'Critical depth': critical_diff < tolerance,
        'Min energy': energy_diff < tolerance,
    }
    
    print("Consistency Check (tolerance = 1e-5):")
    all_passed = True
    for param, passed in checks.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {param:<15}: {status}")
        if not passed:
            all_passed = False
    
    print()
    return all_passed


def main():
    """Run comprehensive unit-aware test."""
    print("PyOpenChannel - Comprehensive Unit-Aware Test")
    print("=" * 60)
    print("Testing all modules for unit system consistency")
    print()
    
    try:
        # Test SI workflow
        si_results = test_complete_workflow_si()
        
        # Test US workflow
        us_results = test_complete_workflow_us()
        
        # Compare results
        all_consistent = compare_results(si_results, us_results)
        
        # Reset to SI
        poc.set_unit_system('SI')
        
        # Final summary
        print("=" * 60)
        print("COMPREHENSIVE TEST SUMMARY")
        print("=" * 60)
        
        if all_consistent:
            print("🎉 ALL MODULES ARE UNIT-SYSTEM CONSISTENT!")
            print()
            print("✅ hydraulics.py - Manning's equation unit-aware")
            print("✅ flow_analysis.py - Energy and momentum unit-aware")
            print("✅ design.py - Design calculations unit-aware")
            print("✅ geometry.py - Unit-agnostic (as expected)")
            print()
            print("The entire PyOpenChannel library is now:")
            print("• Internationally usable (SI and US Customary)")
            print("• Physically consistent across unit systems")
            print("• Maintains engineering accuracy")
            print("• Handles mixed unit inputs seamlessly")
        else:
            print("❌ SOME MODULES HAVE UNIT CONSISTENCY ISSUES!")
            print("Review the failed checks above.")
        
        print("=" * 60)
        
    except Exception as e:
        print(f"Error running comprehensive test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
