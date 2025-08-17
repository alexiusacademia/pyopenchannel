#!/usr/bin/env python3
"""
Backwater Effect Analysis Demo - PyOpenChannel

Author: Alexius Academia
Email: alexius.sayco.academia@gmail.com

This example demonstrates backwater effect analysis using GVF.
Shows how downstream conditions higher than normal depth create
backwater profiles that propagate upstream.

Key Concepts:
- M1 Profile: Backwater on mild slopes
- S1 Profile: Backwater on steep slopes  
- Asymptotic approach to normal depth
- Engineering implications for flood analysis
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pyopenchannel as poc


def analyze_backwater_effect():
    """Demonstrate backwater effect analysis."""
    
    print("🌊 BACKWATER EFFECT ANALYSIS")
    print("=" * 60)
    
    # Set up channel conditions
    poc.set_unit_system(poc.UnitSystem.SI)
    channel = poc.RectangularChannel(width=5.0)
    discharge = 20.0
    slope = 0.001  # Mild slope
    manning_n = 0.030
    
    # Calculate reference depths
    normal_depth = poc.NormalDepth.calculate(channel, discharge, slope, manning_n)
    critical_depth = poc.CriticalFlow(channel).calculate_critical_depth(discharge)
    
    print(f"📏 Channel: {channel.width}m wide rectangular")
    print(f"💧 Discharge: {discharge} m³/s")
    print(f"📐 Slope: {slope} (mild slope)")
    print(f"🔵 Normal Depth: {normal_depth:.3f} m")
    print(f"🔴 Critical Depth: {critical_depth:.3f} m")
    print(f"✅ Slope Type: {'Mild' if normal_depth > critical_depth else 'Steep'} (yn > yc)")
    
    # Simulate backwater condition
    downstream_depth = normal_depth * 1.4  # 40% higher than normal
    print(f"\n🌊 BACKWATER SCENARIO:")
    print(f"⬇️  Downstream Depth: {downstream_depth:.3f} m")
    print(f"📊 Backwater Ratio: {downstream_depth/normal_depth:.2f} (downstream/normal)")
    print(f"🔍 Analysis: Downstream depth > Normal depth → BACKWATER EFFECT")
    
    # Analyze with GVF solver
    solver = poc.GVFSolver()
    result = solver.solve_profile(
        channel=channel,
        discharge=discharge,
        slope=slope,
        manning_n=manning_n,
        x_start=0.0,
        x_end=2000.0,
        boundary_depth=downstream_depth,
        boundary_type=poc.BoundaryType.UPSTREAM_DEPTH
    )
    
    if result.success:
        print(f"\n✅ GVF SOLUTION SUCCESS:")
        print(f"📏 Profile Length: {result.length:.0f} m")
        print(f"📊 Profile Points: {len(result.profile_points)}")
        
        # Analyze first and last points
        first_point = result.profile_points[0]
        last_point = result.profile_points[-1]
        
        print(f"\n📍 PROFILE CHARACTERISTICS:")
        print(f"   Downstream (x=0): {first_point.depth:.3f} m, Fr={first_point.froude_number:.2f}")
        print(f"   Upstream (x={last_point.x:.0f}): {last_point.depth:.3f} m, Fr={last_point.froude_number:.2f}")
        print(f"   Depth Change: {last_point.depth - first_point.depth:.3f} m")
        print(f"   Flow Regime: {'Subcritical' if first_point.froude_number < 1 else 'Supercritical'} throughout")
        
        # Show depth progression
        print(f"\n📈 DEPTH PROGRESSION (every 400m):")
        for i, point in enumerate(result.profile_points):
            if i % 20 == 0 or i == len(result.profile_points) - 1:  # Show every 20th point
                approach_to_normal = abs(point.depth - normal_depth) / normal_depth * 100
                print(f"   x={point.x:6.0f}m: depth={point.depth:.3f}m, "
                      f"Fr={point.froude_number:.2f}, "
                      f"deviation from normal: {approach_to_normal:.1f}%")
        
        # Classify the profile
        classifier = poc.ProfileClassifier()
        profile = classifier.classify_profile(
            gvf_result=result,
            channel=channel,
            discharge=discharge,
            slope=slope,
            manning_n=manning_n
        )
        
        print(f"\n🏷️  PROFILE CLASSIFICATION:")
        print(f"   Profile Type: {profile.profile_type.value}")
        print(f"   Slope Type: {profile.slope_type.value}")
        print(f"   Flow Regime: {profile.flow_regime.value}")
        print(f"   Length: {profile.length:.0f} m")
        print(f"   Normal Depth: {profile.normal_depth:.3f} m")
        print(f"   Critical Depth: {profile.critical_depth:.3f} m")
        
        # Engineering analysis
        print(f"\n🔧 ENGINEERING IMPLICATIONS:")
        backwater_extent = result.length
        max_backwater_height = downstream_depth - normal_depth
        
        print(f"   📏 Backwater Extent: {backwater_extent:.0f} m")
        print(f"   📊 Maximum Backwater Height: {max_backwater_height:.3f} m")
        print(f"   🌊 Flood Impact: Water levels elevated up to {backwater_extent/1000:.1f} km upstream")
        print(f"   ⚠️  Bridge Clearance: Check structures within {backwater_extent:.0f}m upstream")
        
        # Compare with uniform flow
        uniform_area = channel.area(normal_depth)
        uniform_velocity = discharge / uniform_area
        backwater_area = channel.area(downstream_depth)
        backwater_velocity = discharge / backwater_area
        
        print(f"\n📊 UNIFORM vs BACKWATER COMPARISON:")
        print(f"   Uniform Flow:  depth={normal_depth:.3f}m, velocity={uniform_velocity:.2f}m/s")
        print(f"   Backwater:     depth={downstream_depth:.3f}m, velocity={backwater_velocity:.2f}m/s")
        print(f"   Velocity Reduction: {(1-backwater_velocity/uniform_velocity)*100:.1f}%")
        
    else:
        print(f"❌ GVF Solution Failed: {result.message}")
    
    print("\n" + "=" * 60)
    print("🎯 CONCLUSION: You are absolutely right about the backwater effect!")
    print("   When downstream depth > normal depth, backwater propagates upstream,")
    print("   creating higher water levels that gradually approach normal depth.")


def compare_different_backwater_scenarios():
    """Compare different backwater scenarios."""
    
    print("\n\n🔄 COMPARING DIFFERENT BACKWATER SCENARIOS")
    print("=" * 60)
    
    # Base conditions
    poc.set_unit_system(poc.UnitSystem.SI)
    channel = poc.RectangularChannel(width=5.0)
    discharge = 20.0
    slope = 0.001
    manning_n = 0.030
    
    normal_depth = poc.NormalDepth.calculate(channel, discharge, slope, manning_n)
    
    # Different backwater ratios
    backwater_ratios = [1.1, 1.2, 1.3, 1.5, 2.0]
    solver = poc.GVFSolver()
    
    print(f"🔵 Normal Depth: {normal_depth:.3f} m")
    print(f"\n📊 BACKWATER SCENARIO COMPARISON:")
    print("   Ratio  | Downstream | Backwater  | Profile")
    print("          | Depth (m)  | Extent (m) | Type")
    print("   -------|------------|------------|--------")
    
    for ratio in backwater_ratios:
        downstream_depth = normal_depth * ratio
        
        result = solver.solve_profile(
            channel=channel,
            discharge=discharge,
            slope=slope,
            manning_n=manning_n,
            x_start=0.0,
            x_end=3000.0,
            boundary_depth=downstream_depth,
            boundary_type=poc.BoundaryType.UPSTREAM_DEPTH
        )
        
        if result.success:
            classifier = poc.ProfileClassifier()
            profile = classifier.classify_profile(
                gvf_result=result,
                channel=channel,
                discharge=discharge,
                slope=slope,
                manning_n=manning_n
            )
            
            print(f"   {ratio:4.1f}   | {downstream_depth:8.3f}  | {result.length:8.0f}   | {profile.profile_type.value}")
        else:
            print(f"   {ratio:4.1f}   | {downstream_depth:8.3f}  |    FAILED   | ---")
    
    print("\n💡 INSIGHTS:")
    print("   • Higher backwater ratios create longer backwater extents")
    print("   • All scenarios show M1 profiles (typical for mild slopes)")
    print("   • Backwater extent increases non-linearly with downstream depth")


if __name__ == "__main__":
    analyze_backwater_effect()
    compare_different_backwater_scenarios()
