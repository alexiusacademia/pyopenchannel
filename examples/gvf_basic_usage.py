#!/usr/bin/env python3
"""
PyOpenChannel - Basic GVF (Gradually Varied Flow) Usage Example
================================================================

This example demonstrates the basic usage of the GVF solver for computing
water surface profiles in open channels.

Features demonstrated:
- Basic GVF solver setup and usage
- Different boundary conditions
- Profile visualization and analysis
- Engineering interpretation

Author: Alexius Academia
Date: 2025-08-17
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pyopenchannel as poc
from pyopenchannel.gvf import GVFSolver, BoundaryType
from pyopenchannel.geometry import RectangularChannel

# Optional matplotlib import for plotting
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("📝 Note: matplotlib not available - plots will be skipped")


def main():
    """Demonstrate basic GVF solver usage."""
    
    print("\n" + "="*70)
    print("PyOpenChannel - Basic GVF Usage Example")
    print("="*70)
    print("Computing water surface profiles using Gradually Varied Flow analysis")
    
    # Set unit system
    poc.set_unit_system(poc.UnitSystem.SI)
    print(f"\n🔧 Unit System: {poc.get_unit_system()}")
    
    # =================================================================
    # EXAMPLE 1: M1 PROFILE (DAM BACKWATER)
    # =================================================================
    print("\n" + "="*70)
    print("           EXAMPLE 1: M1 PROFILE (DAM BACKWATER)")
    print("="*70)
    
    # Channel setup
    channel = RectangularChannel(width=1.0)  # 5m wide rectangular channel
    discharge = 1.0  # m³/s
    slope = 0.001    # 0.08% - mild slope
    manning_n = 0.015
    
    print(f"\n🏗️  Channel Configuration:")
    print(f"   • Type: Rectangular")
    print(f"   • Width: {channel.width} m")
    print(f"   • Discharge: {discharge} m³/s")
    print(f"   • Slope: {slope:.4f} ({slope*100:.2f}%)")
    print(f"   • Manning's n: {manning_n}")
    
    # Calculate reference depths
    from pyopenchannel.hydraulics import NormalDepth
    from pyopenchannel.flow_analysis import CriticalFlow
    
    critical_flow = CriticalFlow(channel)
    
    normal_depth = NormalDepth.calculate(channel, discharge, slope, manning_n)
    critical_depth = critical_flow.calculate_critical_depth(discharge)
    
    print(f"\n📊 Reference Depths:")
    print(f"   • Normal depth (yn): {normal_depth:.3f} m")
    print(f"   • Critical depth (yc): {critical_depth:.3f} m")
    print(f"   • Slope type: {'Mild' if normal_depth > critical_depth else 'Steep'}")
    
    # Setup GVF solver
    solver = GVFSolver()
    
    # Define boundary condition - dam creates backwater
    dam_depth = 1.2  # m - depth at dam (above normal depth)
    
    print(f"\n🔍 GVF Analysis Setup:")
    print(f"   • Boundary condition: Depth at dam (downstream control) = {dam_depth:.1f} m")
    print(f"   • Analysis type: Backwater curve (M1 profile)")
    print(f"   • Integration direction: Upstream from dam")
    
    # Solve GVF profile
    try:
        result = solver.solve_profile(
            channel=channel,
            discharge=discharge,
            slope=slope,
            manning_n=manning_n,
            x_start=-2000.0,  # 2 km upstream of the dam
            x_end=0.0,        # Dam (x increases in the flow direction)
            boundary_depth=dam_depth,
            boundary_type=BoundaryType.DOWNSTREAM_DEPTH,
            initial_step=10.0 # 10m initial step
        )
        
        print(f"\n✅ GVF Solution Successful!")
        print(f"   • Profile points computed: {len(result.profile_points)}")
        print(f"   • Total length: {result.length:.1f} m")
        print(f"   • Computation status: {result.message}")
        
        # Extract profile data (points run from the dam upstream)
        distances = [abs(p.x) for p in result.profile_points]  # distance upstream of the dam
        depths = [p.depth for p in result.profile_points]
        velocities = [p.velocity for p in result.profile_points]
        froude_numbers = [p.froude_number for p in result.profile_points]
        
        print(f"\n📈 Profile Characteristics:")
        print(f"   • Depth range: {min(depths):.3f} - {max(depths):.3f} m")
        print(f"   • Velocity range: {min(velocities):.3f} - {max(velocities):.3f} m/s")
        print(f"   • Froude number range: {min(froude_numbers):.3f} - {max(froude_numbers):.3f}")
        
        # Engineering analysis
        print(f"\n🔬 Engineering Analysis:")
        if max(depths) > normal_depth * 1.1:
            print(f"   • Significant backwater effect detected")
            print(f"   • Maximum backwater: {max(depths) - normal_depth:.3f} m above normal")
        
        backwater_length = max(distances) if depths[-1] > normal_depth * 1.05 else None
        if backwater_length:
            print(f"   • Backwater influence extends: {backwater_length:.0f} m upstream")
        
        # Plot results (if matplotlib available)
        if HAS_MATPLOTLIB:
            plt.figure(figsize=(12, 8))
            
            # Subplot 1: Water surface profile
            plt.subplot(2, 2, 1)
            plt.plot(distances, depths, 'b-', linewidth=2, label='Water surface')
            plt.axhline(y=normal_depth, color='g', linestyle='--', label=f'Normal depth ({normal_depth:.2f}m)')
            plt.axhline(y=critical_depth, color='r', linestyle='--', label=f'Critical depth ({critical_depth:.2f}m)')
            plt.xlabel('Distance upstream (m)')
            plt.ylabel('Depth (m)')
            plt.title('M1 Profile - Dam Backwater')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Subplot 2: Velocity profile
            plt.subplot(2, 2, 2)
            plt.plot(distances, velocities, 'r-', linewidth=2)
            plt.xlabel('Distance upstream (m)')
            plt.ylabel('Velocity (m/s)')
            plt.title('Velocity Distribution')
            plt.grid(True, alpha=0.3)
            
            # Subplot 3: Froude number
            plt.subplot(2, 2, 3)
            plt.plot(distances, froude_numbers, 'g-', linewidth=2)
            plt.axhline(y=1.0, color='r', linestyle='--', label='Critical flow (Fr=1)')
            plt.xlabel('Distance upstream (m)')
            plt.ylabel('Froude Number')
            plt.title('Flow Regime')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Subplot 4: Energy grade line
            plt.subplot(2, 2, 4)
            energy_heads = [d + v**2/(2*9.81) for d, v in zip(depths, velocities)]
            plt.plot(distances, energy_heads, 'm-', linewidth=2, label='Energy grade line')
            plt.plot(distances, depths, 'b-', linewidth=1, label='Water surface')
            plt.xlabel('Distance upstream (m)')
            plt.ylabel('Elevation (m)')
            plt.title('Energy Analysis')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('gvf_m1_profile.png', dpi=300, bbox_inches='tight')
            print(f"\n📊 Profile plot saved as 'gvf_m1_profile.png'")
        else:
            print(f"\n📊 Plotting skipped (matplotlib not available)")
        
    except Exception as e:
        print(f"\n❌ GVF Solution Failed: {e}")
        return
    
    # =================================================================
    # EXAMPLE 2: M2 PROFILE (FREE OVERFALL DRAWDOWN)
    # =================================================================
    print("\n" + "="*70)
    print("       EXAMPLE 2: M2 PROFILE (FREE OVERFALL DRAWDOWN)")
    print("="*70)
    
    # Same channel, now ending in a free overfall. Flow passes through
    # critical depth at the brink; start just above yc (the GVF equation is
    # singular at yc) and compute upstream toward normal depth.
    brink_depth = 1.02 * critical_depth  # m - just above critical, below normal
    
    print(f"\n🔍 M2 Profile Analysis:")
    print(f"   • Boundary condition: Depth near the brink = {brink_depth:.3f} m")
    print(f"   • Depth relationship: yc < y < yn")
    print(f"   • Expected profile: Drawdown curve (M2), computed upstream from the brink")
    
    try:
        result_m2 = solver.solve_profile(
            channel=channel,
            discharge=discharge,
            slope=slope,
            manning_n=manning_n,
            x_start=-1000.0,  # 1 km upstream of the brink
            x_end=0.0,        # Brink (downstream control)
            boundary_depth=brink_depth,
            boundary_type=BoundaryType.DOWNSTREAM_DEPTH,
            initial_step=5.0  # 5m initial step
        )
        
        print(f"\n✅ M2 Profile Solution Successful!")
        print(f"   • Profile points: {len(result_m2.profile_points)}")
        print(f"   • Length analyzed: {result_m2.length:.1f} m")
        
        depths_m2 = [p.depth for p in result_m2.profile_points]
        distances_m2 = [abs(p.x) for p in result_m2.profile_points]  # distance upstream of the brink
        
        print(f"   • Depth going upstream from the brink: {depths_m2[0]:.3f} → {depths_m2[-1]:.3f} m "
              f"(yn = {normal_depth:.3f} m)")
        print(f"   • Profile type: {'Drawdown toward the brink' if depths_m2[-1] > depths_m2[0] else 'Backwater'}")
        
    except Exception as e:
        print(f"\n❌ M2 Profile Failed: {e}")
    
    # =================================================================
    # EXAMPLE 3: STEEP CHANNEL (S1 PROFILE)
    # =================================================================
    print("\n" + "="*70)
    print("           EXAMPLE 3: S1 PROFILE (STEEP CHANNEL)")
    print("="*70)
    
    # Setup steep channel
    steep_channel = RectangularChannel(width=3.0)
    steep_discharge = 10.0  # m³/s
    steep_slope = 0.02      # 2% - steep slope
    
    # Calculate reference depths for steep channel
    critical_steep = CriticalFlow(steep_channel)
    
    normal_steep = NormalDepth.calculate(steep_channel, steep_discharge, steep_slope, manning_n)
    critical_steep_depth = critical_steep.calculate_critical_depth(steep_discharge)
    
    print(f"\n🏗️  Steep Channel Configuration:")
    print(f"   • Width: {steep_channel.width} m")
    print(f"   • Discharge: {steep_discharge} m³/s")
    print(f"   • Slope: {steep_slope:.3f} ({steep_slope*100:.1f}%)")
    print(f"   • Normal depth: {normal_steep:.3f} m")
    print(f"   • Critical depth: {critical_steep_depth:.3f} m")
    print(f"   • Slope type: {'Steep' if critical_steep_depth > normal_steep else 'Mild'}")
    
    # S1 profile - backwater in steep channel (e.g. behind a gate or dam).
    # The depth is above critical, so the flow is subcritical and controlled
    # from downstream: compute upstream from the control. Going upstream the
    # S1 depth falls quickly toward critical depth (the S1 curve begins at a
    # hydraulic jump from the supercritical approach flow), so only a short
    # reach is computed.
    s1_boundary_depth = 1.5  # m - depth at the downstream control, above critical depth
    s1_reach = 10.0          # m - stops short of where the curve reaches critical depth
    
    try:
        result_s1 = solver.solve_profile(
            channel=steep_channel,
            discharge=steep_discharge,
            slope=steep_slope,
            manning_n=manning_n,
            x_start=-s1_reach,  # Upstream end of the reach
            x_end=0.0,          # Downstream control
            boundary_depth=s1_boundary_depth,
            boundary_type=BoundaryType.DOWNSTREAM_DEPTH,
            initial_step=2.0  # 2m initial step
        )
        
        print(f"\n✅ S1 Profile Solution Successful!")
        print(f"   • Reach: {s1_reach:.0f} m upstream of the control (critical depth is reached just beyond)")
        depths_s1 = [p.depth for p in result_s1.profile_points]
        froude_s1 = [p.froude_number for p in result_s1.profile_points]
        
        print(f"   • Depth range: {min(depths_s1):.3f} - {max(depths_s1):.3f} m")
        print(f"   • Depth going upstream from the control: {depths_s1[0]:.3f} → {depths_s1[-1]:.3f} m")
        print(f"   • Froude range: {min(froude_s1):.3f} - {max(froude_s1):.3f}")
        print(f"   • Flow regime: {'Mixed' if any(f < 1 for f in froude_s1) and any(f > 1 for f in froude_s1) else 'Subcritical' if all(f < 1 for f in froude_s1) else 'Supercritical'}")
        
    except Exception as e:
        print(f"\n❌ S1 Profile Failed: {e}")
    
    # =================================================================
    # SUMMARY AND RECOMMENDATIONS
    # =================================================================
    print("\n" + "="*70)
    print("                    ANALYSIS SUMMARY")
    print("="*70)
    
    print(f"\n🎯 Key Findings:")
    print(f"   • M1 Profile: Effective for dam backwater analysis")
    print(f"   • M2 Profile: Useful for drawdown upstream of free overfalls and drops")
    print(f"   • S1 Profile: Critical for steep channel hydraulic analysis")
    
    print(f"\n🔧 Engineering Applications:")
    print(f"   • Flood mapping: Use M1 profiles for reservoir backwater")
    print(f"   • Bridge design: Analyze backwater effects upstream")
    print(f"   • Channel drops: Use M2 profiles for drawdown upstream of the brink")
    print(f"   • Energy dissipation: S1 profiles for steep channel design")
    
    print(f"\n📚 Next Steps:")
    print(f"   • Try different boundary conditions")
    print(f"   • Experiment with channel geometries")
    print(f"   • Use profile classification for automatic analysis")
    print(f"   • Combine with hydraulic structure analysis")
    
    print(f"\n" + "="*70)
    print("Basic GVF Usage Example Completed Successfully!")
    print("="*70)


if __name__ == "__main__":
    main()
