#!/usr/bin/env python3
"""
Quick demonstration of unit system flexibility in PyOpenChannel.
"""

import sys
import os

# Add the src directory to the path so we can import pyopenchannel
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pyopenchannel as poc


def main():
    """Quick unit system demonstration."""
    print("PyOpenChannel - Quick Unit System Demo")
    print("=" * 50)
    
    # Same physical problem, different units
    
    # SCENARIO 1: Work in SI units (default)
    print("SCENARIO 1: SI Units (meters, m³/s)")
    print("-" * 30)
    
    channel = poc.RectangularChannel(width=3.0)  # 3 meters
    discharge = 10.0  # m³/s
    slope = 0.002
    manning_n = 0.025
    
    depth = poc.NormalDepth.calculate(channel, discharge, slope, manning_n)
    print(f"Width: {channel.width:.1f} m")
    print(f"Discharge: {discharge:.1f} m³/s")
    print(f"Normal depth: {depth:.3f} m")
    print()
    
    # SCENARIO 2: Convert to US units
    print("SCENARIO 2: US Units (feet, cfs)")
    print("-" * 30)
    
    # Convert the same physical values
    width_ft = poc.m_to_ft(3.0)
    discharge_cfs = poc.cms_to_cfs(10.0)
    depth_ft = poc.m_to_ft(depth)
    
    print(f"Width: {width_ft:.1f} ft")
    print(f"Discharge: {discharge_cfs:.1f} cfs")
    print(f"Normal depth: {depth_ft:.3f} ft")
    print()
    
    # SCENARIO 3: Mixed units (common in practice)
    print("SCENARIO 3: Mixed Units (input in ft and cfs)")
    print("-" * 30)
    
    # Engineer provides: 8 ft wide channel, 200 cfs flow
    width_input_ft = 8.0
    discharge_input_cfs = 200.0
    
    # Convert to SI for calculation
    width_si = poc.ft_to_m(width_input_ft)
    discharge_si = poc.cfs_to_cms(discharge_input_cfs)
    
    channel_mixed = poc.RectangularChannel(width=width_si)
    depth_mixed = poc.NormalDepth.calculate(channel_mixed, discharge_si, slope, manning_n)
    
    # Show results in both unit systems
    print(f"Input: {width_input_ft} ft wide, {discharge_input_cfs} cfs")
    print(f"Normal depth: {poc.m_to_ft(depth_mixed):.3f} ft ({depth_mixed:.3f} m)")
    print()
    
    # SCENARIO 4: Quick conversions
    print("SCENARIO 4: Quick Conversions")
    print("-" * 30)
    
    common_discharges_cfs = [10, 50, 100, 500]
    print("Common discharge conversions:")
    for cfs in common_discharges_cfs:
        cms = poc.cfs_to_cms(cfs)
        gpm = poc.cms_to_gpm(cms)
        print(f"  {cfs:3d} cfs = {cms:5.2f} m³/s = {gpm:6.0f} gpm")
    
    print()
    print("Key Benefits:")
    print("✓ Work in your preferred units")
    print("✓ Easy conversions between systems")
    print("✓ Handle mixed unit inputs")
    print("✓ Consistent calculations regardless of units")


if __name__ == "__main__":
    main()
