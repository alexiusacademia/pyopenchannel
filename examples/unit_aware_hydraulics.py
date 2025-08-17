#!/usr/bin/env python3
"""
File: unit_aware_hydraulics.py
Author: Alexius Academia
Date: 2025-08-17

Test unit-aware hydraulics calculations.

This example verifies that hydraulic calculations work correctly
in both SI and US Customary unit systems.
"""

import sys
import os

# Add the src directory to the path so we can import pyopenchannel
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pyopenchannel as poc


def test_manning_equation_units():
    """Test Manning's equation with different unit systems."""
    print("=" * 60)
    print("MANNING'S EQUATION UNIT CONSISTENCY TEST")
    print("=" * 60)
    
    # Test parameters (same physical channel)
    width_m = 3.0  # meters
    depth_m = 2.0  # meters
    slope = 0.002  # dimensionless
    manning_n = 0.025
    
    # Test in SI units
    print("SI UNITS TEST:")
    poc.set_unit_system('SI')
    
    channel_si = poc.RectangularChannel(width=width_m)
    area_si = channel_si.area(depth_m)
    hydraulic_radius_si = channel_si.hydraulic_radius(depth_m)
    
    discharge_si = poc.ManningEquation.discharge(area_si, hydraulic_radius_si, slope, manning_n)
    velocity_si = poc.ManningEquation.velocity(hydraulic_radius_si, slope, manning_n)
    
    print(f"  Manning factor: {poc.get_manning_factor():.2f}")
    print(f"  Gravity: {poc.get_gravity():.2f} m/s²")
    print(f"  Area: {area_si:.3f} m²")
    print(f"  Hydraulic radius: {hydraulic_radius_si:.3f} m")
    print(f"  Discharge: {discharge_si:.3f} m³/s")
    print(f"  Velocity: {velocity_si:.3f} m/s")
    print()
    
    # Test in US Customary units (same physical channel)
    print("US CUSTOMARY UNITS TEST:")
    poc.set_unit_system('US_CUSTOMARY')
    
    width_ft = poc.m_to_ft(width_m)
    depth_ft = poc.m_to_ft(depth_m)
    
    channel_us = poc.RectangularChannel(width=width_ft)
    area_us = channel_us.area(depth_ft)
    hydraulic_radius_us = channel_us.hydraulic_radius(depth_ft)
    
    discharge_us = poc.ManningEquation.discharge(area_us, hydraulic_radius_us, slope, manning_n)
    velocity_us = poc.ManningEquation.velocity(hydraulic_radius_us, slope, manning_n)
    
    print(f"  Manning factor: {poc.get_manning_factor():.2f}")
    print(f"  Gravity: {poc.get_gravity():.2f} ft/s²")
    print(f"  Area: {area_us:.3f} ft²")
    print(f"  Hydraulic radius: {hydraulic_radius_us:.3f} ft")
    print(f"  Discharge: {discharge_us:.3f} ft³/s")
    print(f"  Velocity: {velocity_us:.3f} ft/s")
    print()
    
    # Verify consistency by converting US results back to SI
    print("CONSISTENCY CHECK:")
    discharge_converted = poc.cfs_to_cms(discharge_us)
    velocity_converted = poc.ft_to_m(velocity_us)  # ft/s to m/s
    
    print(f"  US discharge converted to SI: {discharge_converted:.3f} m³/s")
    print(f"  SI discharge: {discharge_si:.3f} m³/s")
    print(f"  Difference: {abs(discharge_converted - discharge_si):.6f} m³/s")
    print()
    
    print(f"  US velocity converted to SI: {velocity_converted:.3f} m/s")
    print(f"  SI velocity: {velocity_si:.3f} m/s")
    print(f"  Difference: {abs(velocity_converted - velocity_si):.6f} m/s")
    print()
    
    # Check if results are consistent (within tolerance)
    discharge_consistent = abs(discharge_converted - discharge_si) < 0.001
    velocity_consistent = abs(velocity_converted - velocity_si) < 0.001
    
    print(f"  Discharge consistent: {'✅ PASS' if discharge_consistent else '❌ FAIL'}")
    print(f"  Velocity consistent: {'✅ PASS' if velocity_consistent else '❌ FAIL'}")
    print()
    
    # Reset to SI
    poc.set_unit_system('SI')
    
    return discharge_consistent and velocity_consistent


def test_critical_depth_units():
    """Test critical depth calculations with different unit systems."""
    print("=" * 60)
    print("CRITICAL DEPTH UNIT CONSISTENCY TEST")
    print("=" * 60)
    
    # Test parameters
    width_m = 4.0  # meters
    discharge_cms = 15.0  # m³/s
    
    # Test in SI units
    print("SI UNITS TEST:")
    poc.set_unit_system('SI')
    
    channel_si = poc.RectangularChannel(width=width_m)
    critical_depth_si = poc.CriticalDepth.calculate(channel_si, discharge_cms)
    
    # Calculate Froude number at critical depth (should be 1.0)
    area_c_si = channel_si.area(critical_depth_si)
    velocity_c_si = discharge_cms / area_c_si
    hydraulic_depth_c_si = area_c_si / channel_si.top_width(critical_depth_si)
    froude_si = poc.CriticalDepth.froude_number(velocity_c_si, hydraulic_depth_c_si)
    
    print(f"  Gravity: {poc.get_gravity():.2f} m/s²")
    print(f"  Critical depth: {critical_depth_si:.3f} m")
    print(f"  Critical velocity: {velocity_c_si:.3f} m/s")
    print(f"  Froude number: {froude_si:.6f}")
    print()
    
    # Test in US Customary units (same physical problem)
    print("US CUSTOMARY UNITS TEST:")
    poc.set_unit_system('US_CUSTOMARY')
    
    width_ft = poc.m_to_ft(width_m)
    discharge_cfs = poc.cms_to_cfs(discharge_cms)
    
    channel_us = poc.RectangularChannel(width=width_ft)
    critical_depth_us = poc.CriticalDepth.calculate(channel_us, discharge_cfs)
    
    # Calculate Froude number at critical depth (should be 1.0)
    area_c_us = channel_us.area(critical_depth_us)
    velocity_c_us = discharge_cfs / area_c_us
    hydraulic_depth_c_us = area_c_us / channel_us.top_width(critical_depth_us)
    froude_us = poc.CriticalDepth.froude_number(velocity_c_us, hydraulic_depth_c_us)
    
    print(f"  Gravity: {poc.get_gravity():.2f} ft/s²")
    print(f"  Critical depth: {critical_depth_us:.3f} ft")
    print(f"  Critical velocity: {velocity_c_us:.3f} ft/s")
    print(f"  Froude number: {froude_us:.6f}")
    print()
    
    # Verify consistency
    print("CONSISTENCY CHECK:")
    critical_depth_converted = poc.ft_to_m(critical_depth_us)
    
    print(f"  US critical depth converted to SI: {critical_depth_converted:.3f} m")
    print(f"  SI critical depth: {critical_depth_si:.3f} m")
    print(f"  Difference: {abs(critical_depth_converted - critical_depth_si):.6f} m")
    print()
    
    # Check Froude numbers (should both be ~1.0)
    froude_si_ok = abs(froude_si - 1.0) < 0.01
    froude_us_ok = abs(froude_us - 1.0) < 0.01
    depth_consistent = abs(critical_depth_converted - critical_depth_si) < 0.001
    
    print(f"  SI Froude ≈ 1.0: {'✅ PASS' if froude_si_ok else '❌ FAIL'}")
    print(f"  US Froude ≈ 1.0: {'✅ PASS' if froude_us_ok else '❌ FAIL'}")
    print(f"  Critical depth consistent: {'✅ PASS' if depth_consistent else '❌ FAIL'}")
    print()
    
    # Reset to SI
    poc.set_unit_system('SI')
    
    return froude_si_ok and froude_us_ok and depth_consistent


def test_normal_depth_units():
    """Test normal depth calculations with different unit systems."""
    print("=" * 60)
    print("NORMAL DEPTH UNIT CONSISTENCY TEST")
    print("=" * 60)
    
    # Test parameters
    width_m = 3.0  # meters
    discharge_cms = 10.0  # m³/s
    slope = 0.002
    manning_n = 0.025
    
    # Test in SI units
    print("SI UNITS TEST:")
    poc.set_unit_system('SI')
    
    channel_si = poc.RectangularChannel(width=width_m)
    normal_depth_si = poc.NormalDepth.calculate(channel_si, discharge_cms, slope, manning_n)
    
    print(f"  Normal depth: {normal_depth_si:.3f} m")
    
    # Verify by calculating discharge back
    area_si = channel_si.area(normal_depth_si)
    hydraulic_radius_si = channel_si.hydraulic_radius(normal_depth_si)
    calculated_q_si = poc.ManningEquation.discharge(area_si, hydraulic_radius_si, slope, manning_n)
    
    print(f"  Verification discharge: {calculated_q_si:.3f} m³/s (should be {discharge_cms})")
    print()
    
    # Test in US Customary units
    print("US CUSTOMARY UNITS TEST:")
    poc.set_unit_system('US_CUSTOMARY')
    
    width_ft = poc.m_to_ft(width_m)
    discharge_cfs = poc.cms_to_cfs(discharge_cms)
    
    channel_us = poc.RectangularChannel(width=width_ft)
    normal_depth_us = poc.NormalDepth.calculate(channel_us, discharge_cfs, slope, manning_n)
    
    print(f"  Normal depth: {normal_depth_us:.3f} ft")
    
    # Verify by calculating discharge back
    area_us = channel_us.area(normal_depth_us)
    hydraulic_radius_us = channel_us.hydraulic_radius(normal_depth_us)
    calculated_q_us = poc.ManningEquation.discharge(area_us, hydraulic_radius_us, slope, manning_n)
    
    print(f"  Verification discharge: {calculated_q_us:.3f} ft³/s (should be {discharge_cfs:.3f})")
    print()
    
    # Verify consistency
    print("CONSISTENCY CHECK:")
    normal_depth_converted = poc.ft_to_m(normal_depth_us)
    
    print(f"  US normal depth converted to SI: {normal_depth_converted:.3f} m")
    print(f"  SI normal depth: {normal_depth_si:.3f} m")
    print(f"  Difference: {abs(normal_depth_converted - normal_depth_si):.6f} m")
    print()
    
    # Check consistency
    depth_consistent = abs(normal_depth_converted - normal_depth_si) < 0.001
    si_verification_ok = abs(calculated_q_si - discharge_cms) < 0.001
    us_verification_ok = abs(calculated_q_us - discharge_cfs) < 0.001
    
    print(f"  Normal depth consistent: {'✅ PASS' if depth_consistent else '❌ FAIL'}")
    print(f"  SI verification: {'✅ PASS' if si_verification_ok else '❌ FAIL'}")
    print(f"  US verification: {'✅ PASS' if us_verification_ok else '❌ FAIL'}")
    print()
    
    # Reset to SI
    poc.set_unit_system('SI')
    
    return depth_consistent and si_verification_ok and us_verification_ok


def main():
    """Run all unit-aware hydraulics tests."""
    print("PyOpenChannel - Unit-Aware Hydraulics Test")
    print("=" * 60)
    print("Testing hydraulic calculations for unit system consistency")
    print()
    
    try:
        # Run all tests
        manning_ok = test_manning_equation_units()
        critical_ok = test_critical_depth_units()
        normal_ok = test_normal_depth_units()
        
        # Summary
        print("=" * 60)
        print("UNIT-AWARE HYDRAULICS TEST SUMMARY")
        print("=" * 60)
        
        print(f"Manning's Equation: {'✅ PASS' if manning_ok else '❌ FAIL'}")
        print(f"Critical Depth: {'✅ PASS' if critical_ok else '❌ FAIL'}")
        print(f"Normal Depth: {'✅ PASS' if normal_ok else '❌ FAIL'}")
        print()
        
        all_passed = manning_ok and critical_ok and normal_ok
        
        if all_passed:
            print("🎉 ALL TESTS PASSED!")
            print("✅ Hydraulic calculations are unit-system consistent")
            print("✅ Manning factors applied correctly")
            print("✅ Gravity constants used properly")
            print("✅ Results are physically consistent across unit systems")
        else:
            print("❌ SOME TESTS FAILED!")
            print("Unit-aware hydraulics need debugging")
        
        print("=" * 60)
        
    except Exception as e:
        print(f"Error running tests: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
