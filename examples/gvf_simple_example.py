#!/usr/bin/env python3
"""
PyOpenChannel - Simple GVF Example
===================================

A clean, straightforward example demonstrating the core GVF capabilities
without complex dependencies or extensive plotting.

Perfect for:
- Quick testing and validation
- Learning the basic API
- Integration into other projects
- Automated testing scenarios

Author: Alexius Academia
Date: 2025-08-17
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pyopenchannel as poc
from pyopenchannel.gvf import GVFSolver, BoundaryType, ProfileClassifier
from pyopenchannel.geometry import RectangularChannel
from pyopenchannel.hydraulics import NormalDepth
from pyopenchannel.flow_analysis import CriticalFlow


def main():
    """Simple demonstration of GVF capabilities."""
    
    print("PyOpenChannel - Simple GVF Example")
    print("=" * 50)
    
    # Setup
    poc.set_unit_system(poc.UnitSystem.SI)
    solver = GVFSolver()
    classifier = ProfileClassifier()
    
    # Channel definition
    channel = RectangularChannel(width=4.0)  # 4m wide
    discharge = 15.0    # m³/s
    slope = 0.001       # 0.1% mild slope
    manning_n = 0.030   # Concrete channel
    
    print(f"\nChannel: {channel.width}m wide rectangular")
    print(f"Discharge: {discharge} m³/s")
    print(f"Slope: {slope:.3f} ({slope*100:.1f}%)")
    print(f"Manning's n: {manning_n}")
    
    # Reference depths
    critical_flow = CriticalFlow(channel)
    normal_depth = NormalDepth.calculate(channel, discharge, slope, manning_n)
    critical_depth = critical_flow.calculate_critical_depth(discharge)
    
    print(f"\nReference Depths:")
    print(f"  Normal depth: {normal_depth:.3f} m")
    print(f"  Critical depth: {critical_depth:.3f} m")
    print(f"  Slope type: {'Mild' if normal_depth > critical_depth else 'Steep'}")
    
    # Example 1: M1 Profile (Dam Backwater)
    print(f"\n" + "=" * 50)
    print("EXAMPLE 1: M1 PROFILE (DAM BACKWATER)")
    print("=" * 50)
    
    dam_depth = 3.5  # m - above normal depth
    print(f"Dam creates depth: {dam_depth:.1f} m")
    
    try:
        result = solver.solve_profile(
            channel=channel,
            discharge=discharge,
            slope=slope,
            manning_n=manning_n,
            x_start=-1500.0,  # 1.5 km upstream of the dam
            x_end=0.0,        # dam (downstream control)
            boundary_depth=dam_depth,
            boundary_type=BoundaryType.DOWNSTREAM_DEPTH
        )
        
        # Classify profile
        profile = classifier.classify_profile(
            gvf_result=result,
            channel=channel,
            discharge=discharge,
            slope=slope,
            manning_n=manning_n
        )
        
        print(f"✅ Success! Profile computed with {len(result.profile_points)} points")
        print(f"Profile type: {profile.profile_type.value}")
        print(f"Flow regime: {profile.flow_regime.value}")
        print(f"Profile length: {result.length:.0f} m")
        
        # Show key points
        depths = [p.depth for p in result.profile_points]
        velocities = [p.velocity for p in result.profile_points]
        distances = [p.x for p in result.profile_points]
        
        print(f"\nProfile Summary:")
        print(f"  Depth range: {min(depths):.3f} - {max(depths):.3f} m")
        print(f"  Velocity range: {min(velocities):.3f} - {max(velocities):.3f} m/s")
        print(f"  Backwater above normal: {max(depths) - normal_depth:.3f} m")
        
        # Show profile points (dam, middle, upstream end)
        print(f"\nKey Profile Points:")
        print(f"  {'Upstream (m)':<12} {'Depth (m)':<10} {'Velocity (m/s)':<12} {'Froude #':<10}")
        print(f"  {'-'*50}")
        
        indices = [0, len(result.profile_points)//2, -1]
        for i in indices:
            p = result.profile_points[i]
            print(f"  {abs(p.x):<12.0f} {p.depth:<10.3f} {p.velocity:<12.3f} {p.froude_number:<10.3f}")
        
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    # Example 2: M2 Profile (Free Overfall Drawdown)
    print(f"\n" + "=" * 50)
    print("EXAMPLE 2: M2 PROFILE (FREE OVERFALL DRAWDOWN)")
    print("=" * 50)
    
    # The channel ends in a free overfall, where the flow passes through
    # critical depth. Start just above yc (the GVF equation is singular at yc)
    # and compute upstream; the depth rises toward normal depth.
    brink_depth = 1.02 * critical_depth  # m - between critical and normal
    print(f"Depth near the brink: {brink_depth:.3f} m (just above yc, below yn)")
    
    try:
        result2 = solver.solve_profile(
            channel=channel,
            discharge=discharge,
            slope=slope,
            manning_n=manning_n,
            x_start=-800.0,  # 800 m upstream of the brink
            x_end=0.0,       # brink (downstream control)
            boundary_depth=brink_depth,
            boundary_type=BoundaryType.DOWNSTREAM_DEPTH
        )
        
        profile2 = classifier.classify_profile(
            gvf_result=result2,
            channel=channel,
            discharge=discharge,
            slope=slope,
            manning_n=manning_n
        )
        
        print(f"✅ Success! Profile computed with {len(result2.profile_points)} points")
        print(f"Profile type: {profile2.profile_type.value}")
        print(f"Flow regime: {profile2.flow_regime.value}")
        
        depths2 = [p.depth for p in result2.profile_points]
        print(f"Depth going upstream from the brink: {depths2[0]:.3f} → {depths2[-1]:.3f} m "
              f"({abs(result2.profile_points[-1].x):.0f} m upstream, yn = {normal_depth:.3f} m)")
        print(f"Profile behavior: {'Drawdown toward the brink' if depths2[-1] > depths2[0] else 'Backwater'}")
        
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    # Example 3: Different Channel Geometry
    print(f"\n" + "=" * 50)
    print("EXAMPLE 3: DIFFERENT GEOMETRY (WIDER CHANNEL)")
    print("=" * 50)
    
    wide_channel = RectangularChannel(width=8.0)  # 8m wide
    wide_discharge = 30.0  # m³/s
    
    # Calculate new reference depths
    wide_normal = NormalDepth.calculate(wide_channel, wide_discharge, slope, manning_n)
    wide_critical = CriticalFlow(wide_channel).calculate_critical_depth(wide_discharge)
    
    print(f"Wide channel: {wide_channel.width}m, Q = {wide_discharge} m³/s")
    print(f"Normal depth: {wide_normal:.3f} m")
    print(f"Critical depth: {wide_critical:.3f} m")
    
    try:
        result3 = solver.solve_profile(
            channel=wide_channel,
            discharge=wide_discharge,
            slope=slope,
            manning_n=manning_n,
            x_start=-1000.0,  # 1 km upstream of the control
            x_end=0.0,        # downstream control
            boundary_depth=wide_normal + 1.0,  # 1m above normal
            boundary_type=BoundaryType.DOWNSTREAM_DEPTH
        )
        
        profile3 = classifier.classify_profile(
            gvf_result=result3,
            channel=wide_channel,
            discharge=wide_discharge,
            slope=slope,
            manning_n=manning_n
        )
        
        print(f"✅ Success! Profile type: {profile3.profile_type.value}")
        
        depths3 = [p.depth for p in result3.profile_points]
        velocities3 = [p.velocity for p in result3.profile_points]
        
        print(f"Depth range: {min(depths3):.3f} - {max(depths3):.3f} m")
        print(f"Velocity range: {min(velocities3):.3f} - {max(velocities3):.3f} m/s")
        
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    # Summary
    print(f"\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    print(f"\n✅ GVF System Capabilities Demonstrated:")
    print(f"  • M1 profiles (dam backwater)")
    print(f"  • M2 profiles (drawdown to a free overfall)")
    print(f"  • Automatic profile classification")
    print(f"  • Multiple channel geometries")
    print(f"  • Professional hydraulic analysis")
    
    print(f"\n🎯 Key Benefits:")
    print(f"  • Easy-to-use API")
    print(f"  • Automatic profile identification")
    print(f"  • Engineering-grade accuracy")
    print(f"  • Comprehensive results")
    
    print(f"\n📚 Next Steps:")
    print(f"  • Try different boundary conditions")
    print(f"  • Experiment with channel shapes")
    print(f"  • Use profile classification features")
    print(f"  • Integrate with your projects")
    
    print(f"\n" + "=" * 50)
    print("Simple GVF Example Completed!")
    print("=" * 50)


if __name__ == "__main__":
    main()
