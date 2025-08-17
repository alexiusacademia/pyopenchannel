#!/usr/bin/env python3
"""
PyOpenChannel - Channel Transition Analysis Example
====================================================

This example demonstrates GVF analysis for channel transitions, including:
- Width changes (contractions and expansions)
- Slope changes (mild to steep, steep to mild)
- Cross-section changes (rectangular to trapezoidal)
- Bridge contractions and culvert analysis
- Energy losses and hydraulic jumps

Author: Alexius Academia
Date: 2025-08-17
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pyopenchannel as poc
from pyopenchannel.gvf import GVFSolver, BoundaryType, ProfileClassifier
from pyopenchannel.geometry import RectangularChannel, TrapezoidalChannel
from pyopenchannel.hydraulics import NormalDepth
from pyopenchannel.flow_analysis import CriticalFlow


def analyze_transition(name, upstream_channel, downstream_channel, discharge, 
                      upstream_slope, downstream_slope, manning_n, transition_depth):
    """Analyze a channel transition."""
    
    print(f"\n🔍 Analyzing: {name}")
    print(f"   {'─'*50}")
    
    solver = GVFSolver()
    classifier = ProfileClassifier()
    
    # Upstream analysis
    try:
        upstream_normal = NormalDepth.calculate(upstream_channel, discharge, upstream_slope, manning_n)
        upstream_critical = CriticalFlow(upstream_channel).calculate_critical_depth(discharge)
        
        print(f"   Upstream Channel:")
        print(f"     • Geometry: {type(upstream_channel).__name__}")
        if hasattr(upstream_channel, 'width'):
            print(f"     • Width: {upstream_channel.width} m")
        elif hasattr(upstream_channel, 'bottom_width'):
            print(f"     • Bottom width: {upstream_channel.bottom_width} m, Side slope: {upstream_channel.side_slope}:1")
        print(f"     • Slope: {upstream_slope:.4f} ({upstream_slope*100:.2f}%)")
        print(f"     • Normal depth: {upstream_normal:.3f} m")
        print(f"     • Critical depth: {upstream_critical:.3f} m")
        print(f"     • Slope type: {'Mild' if upstream_normal > upstream_critical else 'Steep'}")
        
        # Solve upstream profile
        upstream_result = solver.solve_profile(
            channel=upstream_channel,
            discharge=discharge,
            slope=upstream_slope,
            manning_n=manning_n,
            x_start=0.0,
            x_end=500.0,  # 500m upstream
            boundary_depth=transition_depth,
            boundary_type=BoundaryType.UPSTREAM_DEPTH
        )
        
        upstream_profile = classifier.classify_profile(
            gvf_result=upstream_result,
            channel=upstream_channel,
            discharge=discharge,
            slope=upstream_slope,
            manning_n=manning_n
        )
        
        upstream_depths = [p.depth for p in upstream_result.profile_points]
        upstream_velocities = [p.velocity for p in upstream_result.profile_points]
        
        print(f"   ✅ Upstream profile: {upstream_profile.profile_type.value}")
        print(f"     • Depth range: {min(upstream_depths):.3f} - {max(upstream_depths):.3f} m")
        print(f"     • Velocity range: {min(upstream_velocities):.3f} - {max(upstream_velocities):.3f} m/s")
        
    except Exception as e:
        print(f"   ❌ Upstream analysis failed: {e}")
        return
    
    # Downstream analysis
    try:
        downstream_normal = NormalDepth.calculate(downstream_channel, discharge, downstream_slope, manning_n)
        downstream_critical = CriticalFlow(downstream_channel).calculate_critical_depth(discharge)
        
        print(f"\n   Downstream Channel:")
        print(f"     • Geometry: {type(downstream_channel).__name__}")
        if hasattr(downstream_channel, 'width'):
            print(f"     • Width: {downstream_channel.width} m")
        elif hasattr(downstream_channel, 'bottom_width'):
            print(f"     • Bottom width: {downstream_channel.bottom_width} m, Side slope: {downstream_channel.side_slope}:1")
        print(f"     • Slope: {downstream_slope:.4f} ({downstream_slope*100:.2f}%)")
        print(f"     • Normal depth: {downstream_normal:.3f} m")
        print(f"     • Critical depth: {downstream_critical:.3f} m")
        print(f"     • Slope type: {'Mild' if downstream_normal > downstream_critical else 'Steep'}")
        
        # For downstream, use a reasonable boundary depth
        downstream_boundary = min(downstream_normal * 1.1, transition_depth * 0.9)
        
        downstream_result = solver.solve_profile(
            channel=downstream_channel,
            discharge=discharge,
            slope=downstream_slope,
            manning_n=manning_n,
            x_start=0.0,
            x_end=300.0,  # 300m downstream
            boundary_depth=downstream_boundary,
            boundary_type=BoundaryType.UPSTREAM_DEPTH
        )
        
        downstream_profile = classifier.classify_profile(
            gvf_result=downstream_result,
            channel=downstream_channel,
            discharge=discharge,
            slope=downstream_slope,
            manning_n=manning_n
        )
        
        downstream_depths = [p.depth for p in downstream_result.profile_points]
        downstream_velocities = [p.velocity for p in downstream_result.profile_points]
        
        print(f"   ✅ Downstream profile: {downstream_profile.profile_type.value}")
        print(f"     • Depth range: {min(downstream_depths):.3f} - {max(downstream_depths):.3f} m")
        print(f"     • Velocity range: {min(downstream_velocities):.3f} - {max(downstream_velocities):.3f} m/s")
        
    except Exception as e:
        print(f"   ❌ Downstream analysis failed: {e}")
        return
    
    # Transition analysis
    print(f"\n   🔬 Transition Analysis:")
    
    # Energy analysis
    upstream_energy = transition_depth + (upstream_velocities[0]**2)/(2*9.81)
    downstream_energy = downstream_depths[0] + (downstream_velocities[0]**2)/(2*9.81)
    energy_loss = upstream_energy - downstream_energy
    
    print(f"     • Upstream energy: {upstream_energy:.3f} m")
    print(f"     • Downstream energy: {downstream_energy:.3f} m")
    print(f"     • Energy loss: {energy_loss:.3f} m")
    
    # Flow regime change
    upstream_froude = upstream_result.profile_points[0].froude_number
    downstream_froude = downstream_result.profile_points[0].froude_number
    
    print(f"     • Upstream Froude: {upstream_froude:.3f}")
    print(f"     • Downstream Froude: {downstream_froude:.3f}")
    
    if upstream_froude < 1 and downstream_froude > 1:
        print(f"     • Flow regime: Subcritical → Supercritical (acceleration)")
    elif upstream_froude > 1 and downstream_froude < 1:
        print(f"     • Flow regime: Supercritical → Subcritical (hydraulic jump)")
    else:
        print(f"     • Flow regime: {'Subcritical' if upstream_froude < 1 else 'Supercritical'} maintained")
    
    # Design recommendations
    print(f"\n   💡 Engineering Recommendations:")
    if energy_loss > 0.1:
        print(f"     • Significant energy loss ({energy_loss:.3f} m) - consider energy dissipation")
    if abs(upstream_froude - downstream_froude) > 0.2:
        print(f"     • Significant Froude number change - check for hydraulic jump")
    if max(upstream_depths) > upstream_normal * 1.2:
        print(f"     • Significant backwater - may affect upstream structures")


def main():
    """Demonstrate channel transition analysis."""
    
    print("PyOpenChannel - Channel Transition Analysis")
    print("=" * 60)
    print("Analysis of water surface profiles through channel transitions")
    
    poc.set_unit_system(poc.UnitSystem.SI)
    
    # Common parameters
    discharge = 20.0    # m³/s
    manning_n = 0.030   # Concrete channel
    
    # =================================================================
    # TRANSITION 1: CHANNEL CONTRACTION (BRIDGE)
    # =================================================================
    print(f"\n" + "=" * 60)
    print("TRANSITION 1: CHANNEL CONTRACTION (BRIDGE)")
    print("=" * 60)
    
    # Wide channel upstream, narrow channel at bridge
    upstream_channel = RectangularChannel(width=6.0)   # 6m wide
    bridge_channel = RectangularChannel(width=4.0)     # 4m wide (bridge opening)
    slope = 0.0008  # Mild slope
    
    # Bridge creates higher depth due to contraction
    bridge_critical = CriticalFlow(bridge_channel).calculate_critical_depth(discharge)
    bridge_depth = bridge_critical * 1.3  # Above critical to avoid choking
    
    analyze_transition(
        name="Bridge Contraction",
        upstream_channel=upstream_channel,
        downstream_channel=bridge_channel,
        discharge=discharge,
        upstream_slope=slope,
        downstream_slope=slope,
        manning_n=manning_n,
        transition_depth=bridge_depth
    )
    
    # =================================================================
    # TRANSITION 2: CHANNEL EXPANSION
    # =================================================================
    print(f"\n" + "=" * 60)
    print("TRANSITION 2: CHANNEL EXPANSION")
    print("=" * 60)
    
    # Narrow channel upstream, wide channel downstream
    narrow_channel = RectangularChannel(width=3.0)     # 3m wide
    wide_channel = RectangularChannel(width=7.0)       # 7m wide
    
    # Expansion typically causes depth increase upstream
    narrow_normal = NormalDepth.calculate(narrow_channel, discharge, slope, manning_n)
    expansion_depth = narrow_normal * 1.2  # Higher depth due to expansion
    
    analyze_transition(
        name="Channel Expansion",
        upstream_channel=narrow_channel,
        downstream_channel=wide_channel,
        discharge=discharge,
        upstream_slope=slope,
        downstream_slope=slope,
        manning_n=manning_n,
        transition_depth=expansion_depth
    )
    
    # =================================================================
    # TRANSITION 3: SLOPE CHANGE (MILD TO STEEP)
    # =================================================================
    print(f"\n" + "=" * 60)
    print("TRANSITION 3: SLOPE CHANGE (MILD TO STEEP)")
    print("=" * 60)
    
    # Same channel, different slopes
    channel = RectangularChannel(width=5.0)
    mild_slope = 0.0005   # 0.05% - mild
    steep_slope = 0.015   # 1.5% - steep
    
    # At slope break, use critical depth as transition
    critical_depth = CriticalFlow(channel).calculate_critical_depth(discharge)
    
    analyze_transition(
        name="Mild to Steep Slope",
        upstream_channel=channel,
        downstream_channel=channel,
        discharge=discharge,
        upstream_slope=mild_slope,
        downstream_slope=steep_slope,
        manning_n=manning_n,
        transition_depth=critical_depth * 1.1
    )
    
    # =================================================================
    # TRANSITION 4: GEOMETRY CHANGE (RECTANGULAR TO TRAPEZOIDAL)
    # =================================================================
    print(f"\n" + "=" * 60)
    print("TRANSITION 4: GEOMETRY CHANGE (RECTANGULAR TO TRAPEZOIDAL)")
    print("=" * 60)
    
    # Rectangular upstream, trapezoidal downstream
    rect_channel = RectangularChannel(width=4.0)
    trap_channel = TrapezoidalChannel(bottom_width=3.0, side_slope=1.5)  # 1.5:1 slopes
    slope = 0.001
    
    # Use normal depth of rectangular channel as transition depth
    rect_normal = NormalDepth.calculate(rect_channel, discharge, slope, manning_n)
    
    analyze_transition(
        name="Rectangular to Trapezoidal",
        upstream_channel=rect_channel,
        downstream_channel=trap_channel,
        discharge=discharge,
        upstream_slope=slope,
        downstream_slope=slope,
        manning_n=manning_n,
        transition_depth=rect_normal
    )
    
    # =================================================================
    # TRANSITION 5: CULVERT ANALYSIS
    # =================================================================
    print(f"\n" + "=" * 60)
    print("TRANSITION 5: CULVERT ANALYSIS")
    print("=" * 60)
    
    # Wide natural channel, narrow culvert, wide channel downstream
    natural_channel = TrapezoidalChannel(bottom_width=8.0, side_slope=2.0)  # Natural channel
    culvert_channel = RectangularChannel(width=2.5)  # Culvert box
    slope = 0.002
    
    # Culvert creates significant contraction
    culvert_critical = CriticalFlow(culvert_channel).calculate_critical_depth(discharge)
    culvert_depth = culvert_critical * 1.4  # Higher depth due to severe contraction
    
    analyze_transition(
        name="Culvert Contraction",
        upstream_channel=natural_channel,
        downstream_channel=culvert_channel,
        discharge=discharge,
        upstream_slope=slope,
        downstream_slope=slope,
        manning_n=0.035,  # Natural channel roughness
        transition_depth=culvert_depth
    )
    
    # =================================================================
    # SUMMARY AND DESIGN GUIDELINES
    # =================================================================
    print(f"\n" + "=" * 60)
    print("DESIGN GUIDELINES FOR CHANNEL TRANSITIONS")
    print("=" * 60)
    
    print(f"\n🏗️  Bridge Design:")
    print(f"   • Minimum opening width to avoid excessive backwater")
    print(f"   • Bridge deck elevation above maximum backwater level")
    print(f"   • Consider scour protection at abutments")
    
    print(f"\n🌊 Channel Expansions:")
    print(f"   • Gradual expansion (1:4 to 1:6 ratio) to minimize energy losses")
    print(f"   • Check for flow separation and recirculation zones")
    print(f"   • May require energy dissipation structures")
    
    print(f"\n⛰️  Slope Changes:")
    print(f"   • Critical depth typically occurs at slope breaks")
    print(f"   • Mild to steep: Check for supercritical flow downstream")
    print(f"   • Steep to mild: Design for hydraulic jump and energy dissipation")
    
    print(f"\n🔄 Geometry Changes:")
    print(f"   • Maintain similar hydraulic radius where possible")
    print(f"   • Gradual transitions reduce energy losses")
    print(f"   • Consider maintenance access and debris handling")
    
    print(f"\n🚇 Culvert Design:")
    print(f"   • Size to minimize upstream flooding")
    print(f"   • Consider inlet and outlet control conditions")
    print(f"   • Provide adequate freeboard and debris capacity")
    
    print(f"\n✅ Key Analysis Benefits:")
    print(f"   • Quantify backwater effects")
    print(f"   • Identify hydraulic jump locations")
    print(f"   • Calculate energy losses")
    print(f"   • Optimize transition geometry")
    print(f"   • Support regulatory compliance")
    
    print(f"\n" + "=" * 60)
    print("Channel Transition Analysis Completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
