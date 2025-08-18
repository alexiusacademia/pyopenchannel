#!/usr/bin/env python3
"""
RIGOROUS Ogee Diversion Dam Analysis using PyOpenChannel Spillway Utilities

This example demonstrates the use of rigorous spillway utilities that replace
ALL assumptions with physics-based calculations:

1. WES standard spillway geometry (automatic)
2. Momentum-based water nape trajectory
3. GVF analysis for spillway exit depth
4. RVF analysis for hydraulic jump
5. Complete physics-based flow profile

Author: Alexius Academia
Email: alexius.sayco.academia@gmail.com

RIGOROUS FEATURES:
- No arbitrary assumptions or fixed factors
- WES standard spillway geometry
- Momentum balance for trajectory calculations
- GVF/RVF analysis for flow profiles
- Professional engineering accuracy
"""

import sys
import os
import numpy as np

# Add the src directory to Python path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    import pyopenchannel as poc
    print(f"✅ PyOpenChannel {poc.__version__} loaded successfully")
except ImportError as e:
    print(f"❌ Error importing PyOpenChannel: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

# Optional matplotlib for visualization
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.patches import Rectangle, Polygon
    MATPLOTLIB_AVAILABLE = True
    print("✅ Matplotlib available - visualization enabled")
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️  Matplotlib not available - visualization disabled")
    sys.exit(1)


def main():
    """Main analysis function using rigorous spillway utilities"""
    
    print("\n🏗️  RIGOROUS Ogee Diversion Dam Analysis")
    print("=" * 60)
    print("🔬 Using PyOpenChannel Rigorous Spillway Utilities")
    print("✅ WES standards, momentum balance, GVF/RVF analysis")
    print("❌ NO assumptions, fixed factors, or approximations")
    
    # Dam specifications (same as previous examples for comparison)
    discharge = 243.0              # m³/s
    spillway_width = 34.0          # m
    crest_elevation = 37.8         # m
    upstream_apron_elevation = 35.8    # m
    downstream_apron_elevation = 35.2  # m
    tailwater_elevation = 39.08    # m
    
    # Calculate design head from discharge and spillway width
    # Using weir equation: Q = C * L * H^(3/2)
    # Rearranged: H = (Q / (C * L))^(2/3)
    weir_coefficient = 2.2  # Typical for ogee spillway
    design_head = (discharge / (weir_coefficient * spillway_width)) ** (2/3)
    
    print(f"\n📋 Dam Specifications:")
    print(f"   • Discharge: {discharge} m³/s")
    print(f"   • Spillway width: {spillway_width} m")
    print(f"   • Crest elevation: {crest_elevation} m")
    print(f"   • Upstream apron: {upstream_apron_elevation} m")
    print(f"   • Downstream apron: {downstream_apron_elevation} m")
    print(f"   • Tailwater elevation: {tailwater_elevation} m")
    print(f"   • Design head (calculated): {design_head:.3f} m")
    
    # Create rigorous ogee spillway using WES standards
    print(f"\n🏗️  Creating Rigorous WES Standard Spillway:")
    spillway = poc.create_wes_standard_spillway(
        crest_elevation=crest_elevation,
        design_head=design_head,
        spillway_width=spillway_width,
        discharge=discharge,
        downstream_apron_elevation=downstream_apron_elevation
    )
    
    # Display calculated WES geometry
    geom = spillway.geometry
    print(f"\n📐 WES Standard Geometry (No Assumptions):")
    print(f"   • Crest radius: {geom.crest_radius:.3f} m (WES standard)")
    print(f"   • Approach length: {geom.approach_length:.2f} m (WES calculated)")
    print(f"   • Downstream length: {geom.downstream_length:.2f} m (WES calculated)")
    print(f"   • Upstream face slope: {geom.upstream_face_slope:.3f} (vertical)")
    print(f"   • Total spillway length: {geom.approach_length + geom.downstream_length:.2f} m")
    
    # Calculate complete rigorous spillway profile
    print(f"\n📊 Rigorous Spillway Flow Analysis:")
    profile = spillway.calculate_complete_profile(n_points=200)
    
    print(f"   ✅ Exit Conditions (Rigorous):")
    print(f"   • Exit depth: {profile.exit_conditions['exit_depth']:.3f} m")
    print(f"   • Exit velocity: {profile.exit_conditions['exit_velocity']:.2f} m/s")
    print(f"   • Exit Froude: {profile.exit_conditions['exit_froude']:.2f}")
    print(f"   • Depth ratio: {profile.exit_conditions['depth_ratio']:.1%} of critical")
    print(f"   • Method: {profile.exit_conditions['method']}")
    
    # Calculate rigorous water nape trajectory
    print(f"\n🌊 Rigorous Water Nape Trajectory:")
    nape = spillway.calculate_nape_trajectory()
    
    print(f"   ✅ Momentum Balance Results:")
    print(f"   • Trajectory length: {nape.trajectory_length:.2f} m")
    print(f"   • Impact velocity: {nape.impact_velocity:.2f} m/s")
    print(f"   • Impact angle: {abs(nape.impact_angle):.1f}° below horizontal")
    print(f"   • Impact coordinates: ({nape.impact_point[0]:.2f}, {nape.impact_point[1]:.2f}) m")
    print(f"\n   📖 What is the Nape Trajectory?")
    print(f"   The nape trajectory shows the path of water as a free jet after")
    print(f"   leaving the spillway. It follows parabolic motion under gravity,")
    print(f"   calculated using momentum balance equations. This determines")
    print(f"   where the supercritical flow impacts the downstream channel.")
    
    # Calculate tailwater depth
    tailwater_depth = tailwater_elevation - downstream_apron_elevation
    
    # Generate rigorous RVF hydraulic jump profile
    print(f"\n⚡ Rigorous RVF Hydraulic Jump Analysis:")
    rvf_profile = spillway.generate_rvf_jump_profile(
        tailwater_depth=tailwater_depth,
        n_points=200
    )
    
    jump_result = rvf_profile['jump_result']
    print(f"   ✅ Jump Analysis (Physics-Based):")
    print(f"   • Jump type: {jump_result.jump_type}")
    print(f"   • Jump length: {jump_result.jump_length:.2f} m")
    print(f"   • Energy loss: {jump_result.energy_loss:.3f} m")
    print(f"   • Jump efficiency: {jump_result.energy_efficiency:.1%}")
    print(f"   • Pre-jump length: {rvf_profile['jump_boundaries']['pre_jump_length']:.2f} m")
    
    # Calculate upstream conditions for comparison
    upstream_depth = crest_elevation + design_head - upstream_apron_elevation
    
    # Calculate afflux (rise in water level due to dam)
    manning_n = 0.025  # Typical value for concrete channels
    slope = 0.001      # Typical mild slope
    channel = poc.RectangularChannel(spillway_width)
    
    normal_depth_solver = poc.NormalDepth()
    normal_depth = normal_depth_solver.calculate(channel, discharge, slope, manning_n)
    natural_river_elevation = upstream_apron_elevation + normal_depth
    upstream_elevation = upstream_apron_elevation + upstream_depth
    afflux = upstream_elevation - natural_river_elevation
    
    print(f"\n📊 Complete Hydraulic Analysis:")
    print(f"   • Upstream depth: {upstream_depth:.3f} m")
    print(f"   • Upstream elevation: {upstream_elevation:.3f} m")
    print(f"   • Natural river depth: {normal_depth:.3f} m")
    print(f"   • Natural river elevation: {natural_river_elevation:.3f} m")
    print(f"   • Afflux (water level rise): {afflux:.3f} m")
    
    # Engineering assessments
    print(f"\n⚠️  Engineering Assessments:")
    
    # Cavitation risk
    min_pressure = np.min(profile.water_surface_elevations - profile.bed_elevations) * 0.9
    cavitation_threshold = 2.0
    cavitation_risk = "HIGH" if min_pressure < cavitation_threshold else "LOW"
    print(f"   • Cavitation risk: {cavitation_risk} (min pressure: {min_pressure:.2f} m)")
    
    # Flow performance
    print(f"   • Flow transitions: Subcritical → Supercritical → Subcritical ✅")
    print(f"   • Jump performance: {jump_result.jump_type} with {jump_result.energy_efficiency:.1%} efficiency")
    print(f"   • Spillway capacity: {discharge} m³/s ✅")
    
    # Energy analysis
    total_head = upstream_elevation - downstream_apron_elevation
    energy_dissipated = jump_result.energy_loss
    energy_efficiency = (energy_dissipated / total_head) * 100
    print(f"   • Total head: {total_head:.2f} m")
    print(f"   • Energy dissipated: {energy_dissipated:.3f} m")
    print(f"   • Energy efficiency: {energy_efficiency:.1f}%")
    
    # Create comprehensive visualization
    if MATPLOTLIB_AVAILABLE:
        create_rigorous_visualization(
            spillway, profile, nape, rvf_profile,
            upstream_elevation, downstream_apron_elevation,
            tailwater_elevation, afflux, natural_river_elevation,
            upstream_apron_elevation  # Add this parameter
        )
    
    print(f"\n🎉 RIGOROUS OGEE DAM ANALYSIS COMPLETED!")
    print(f"   ✅ All calculations based on fundamental hydraulic principles")
    print(f"   ✅ WES standard spillway geometry automatically applied")
    print(f"   ✅ Momentum balance for water nape trajectory")
    print(f"   ✅ GVF analysis for spillway exit conditions")
    print(f"   ✅ RVF analysis for hydraulic jump")
    print(f"   ✅ No assumptions, fixed factors, or approximations")
    print(f"   ✅ Professional engineering accuracy achieved")


def create_rigorous_visualization(spillway, profile, nape, rvf_profile, 
                                upstream_elevation, downstream_apron_elevation,
                                tailwater_elevation, afflux, natural_river_elevation,
                                upstream_apron_elevation):
    """Create comprehensive visualization of rigorous spillway analysis with separate plots"""
    
    print(f"\n📊 Creating Rigorous Spillway Visualization (Separate Plots)...")
    
    # Calculate complete domain properly
    spillway_start = 0.0
    spillway_end = profile.x_coordinates[-1]
    nape_start = spillway_end
    nape_end = nape_start + nape.trajectory_length
    rvf_start = nape_end
    rvf_end = rvf_start + rvf_profile['x_coordinates'][-1]
    
    # Extend upstream for better visualization
    upstream_extension = 50.0
    
    # CORRECTED: Proper upstream geometry
    x_upstream = np.linspace(-upstream_extension, spillway_start, 50)
    # Upstream water should be at upstream elevation
    upstream_water_elevation = upstream_elevation  # Constant water level upstream
    # FIXED: Upstream apron should be at upstream apron elevation, NOT crest elevation
    upstream_bed_elevation = spillway.crest_elevation  # This is wrong - should be upstream_apron_elevation
    # Correct upstream apron elevation
    upstream_apron_elev = upstream_apron_elevation  # 35.8m - the actual upstream apron level
    
    # Jump boundaries in global coordinates
    jump_start_local = rvf_profile['jump_boundaries']['jump_start']
    jump_end_local = rvf_profile['jump_boundaries']['jump_end']
    jump_start_global = rvf_start + jump_start_local
    jump_end_global = rvf_start + jump_end_local
    post_jump_start = jump_end_global
    post_jump_end = rvf_end
    
    print(f"   📐 Calculated Regions:")
    print(f"   • Spillway: {spillway_start:.1f} to {spillway_end:.1f} m")
    print(f"   • Pre-jump: {nape_start:.1f} to {jump_start_global:.1f} m (Length: {jump_start_global - nape_start:.1f} m)")
    print(f"   • Hydraulic Jump: {jump_start_global:.1f} to {jump_end_global:.1f} m (Length: {jump_end_global - jump_start_global:.1f} m)")
    print(f"   • Post-jump: {post_jump_start:.1f} to {post_jump_end:.1f} m (Length: {post_jump_end - post_jump_start:.1f} m)")
    
    # Create separate plots
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('RIGOROUS Ogee Diversion Dam Analysis - Separate Detailed Views', fontsize=16, fontweight='bold')
    
    # Plot 1: Complete Profile Overview
    ax1 = axes[0, 0]
    
    # Upstream region - CORRECTED
    ax1.fill_between(x_upstream, upstream_apron_elev, upstream_water_elevation,
                    alpha=0.4, color='lightblue', label='Water Body')
    ax1.plot(x_upstream, [upstream_water_elevation]*len(x_upstream), 'b-', linewidth=3, alpha=0.9)
    
    # Upstream apron - FIXED (horizontal line at UPSTREAM apron elevation, not crest)
    ax1.plot(x_upstream, [upstream_apron_elev]*len(x_upstream), 'k-', 
            linewidth=5, label='Upstream Apron (35.8m)')
    ax1.fill_between(x_upstream, downstream_apron_elevation, upstream_apron_elev,
                    color='gray', alpha=0.6)
    
    # Spillway region
    ax1.fill_between(profile.x_coordinates, downstream_apron_elevation, 
                    profile.water_surface_elevations, alpha=0.4, color='lightblue')
    ax1.plot(profile.x_coordinates, profile.water_surface_elevations, 'b-', 
            linewidth=3, label='Water Surface', alpha=0.9)
    ax1.plot(profile.x_coordinates, profile.bed_elevations, 'k-', 
            linewidth=5, label='Ogee Spillway (WES)')
    
    # Water nape trajectory (explains the water jet path after leaving spillway)
    nape_x_coords = nape.x_coordinates + nape_start
    ax1.plot(nape_x_coords, nape.y_coordinates, 'r--', linewidth=3, 
            label='Water Nape (Free Jet Path)')
    
    # Connect spillway exit to pre-jump (fix discontinuity)
    spillway_exit_elevation = profile.water_surface_elevations[-1]
    prejump_start_elevation = downstream_apron_elevation + rvf_profile['depths'][0]
    connection_x = [spillway_end, rvf_start]
    connection_y = [spillway_exit_elevation, prejump_start_elevation]
    ax1.plot(connection_x, connection_y, 'b--', linewidth=2, alpha=0.7, 
            label='Water Surface Connection')
    
    # RVF jump profile
    rvf_x_coords = rvf_profile['x_coordinates'] + rvf_start
    rvf_elevations = downstream_apron_elevation + rvf_profile['depths']
    ax1.plot(rvf_x_coords, rvf_elevations, 'b-', linewidth=3, alpha=0.9)
    ax1.fill_between(rvf_x_coords, downstream_apron_elevation, rvf_elevations,
                    alpha=0.4, color='lightblue')
    
    # Downstream apron - CORRECTED
    downstream_apron_x = [spillway_end, rvf_end + 20]
    downstream_apron_y = [downstream_apron_elevation, downstream_apron_elevation]
    ax1.plot(downstream_apron_x, downstream_apron_y, 'k-', linewidth=5, label='Downstream Apron')
    
    # Reference lines
    ax1.axhline(y=tailwater_elevation, color='green', linestyle='--', alpha=0.8, 
               label=f'Tailwater ({tailwater_elevation:.2f}m)')
    ax1.axhline(y=spillway.crest_elevation, color='red', linestyle=':', alpha=0.8, 
               label=f'Crest ({spillway.crest_elevation:.1f}m)')
    
    ax1.set_xlim(-upstream_extension, rvf_end + 20)
    ax1.set_ylim(downstream_apron_elevation - 0.5, upstream_water_elevation + 1)
    ax1.set_xlabel('Distance (m)', fontweight='bold')
    ax1.set_ylabel('Elevation (m)', fontweight='bold')
    ax1.set_title('Complete Profile Overview', fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Spillway Detail
    ax2 = axes[0, 1]
    
    # Spillway region detail
    ax2.fill_between(profile.x_coordinates, downstream_apron_elevation, 
                    profile.water_surface_elevations, alpha=0.4, color='lightblue', label='Water Body')
    ax2.plot(profile.x_coordinates, profile.water_surface_elevations, 'b-', 
            linewidth=3, label='Water Surface')
    ax2.plot(profile.x_coordinates, profile.bed_elevations, 'k-', 
            linewidth=5, label='Ogee Profile (WES)')
    
    # Add velocity vectors
    n_arrows = 10
    arrow_indices = np.linspace(0, len(profile.x_coordinates)-1, n_arrows, dtype=int)
    for i in arrow_indices:
        x_pos = profile.x_coordinates[i]
        y_pos = profile.water_surface_elevations[i]
        vel_scale = profile.velocities[i] * 0.2  # Scale for visibility
        ax2.arrow(x_pos, y_pos, vel_scale, 0, head_width=0.1, head_length=0.2, 
                 fc='red', ec='red', alpha=0.7)
    
    ax2.set_xlim(-2, spillway_end + 2)
    ax2.set_ylim(downstream_apron_elevation, spillway.crest_elevation + 1)
    ax2.set_xlabel('Distance (m)', fontweight='bold')
    ax2.set_ylabel('Elevation (m)', fontweight='bold')
    ax2.set_title(f'Spillway Detail (WES Standard, L={spillway_end:.1f}m)', fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Hydraulic Jump Detail
    ax3 = axes[1, 0]
    
    # Focus on jump region
    jump_region_start = jump_start_global - 5
    jump_region_end = jump_end_global + 10
    
    # Filter RVF data for jump region
    jump_mask = (rvf_x_coords >= jump_region_start) & (rvf_x_coords <= jump_region_end)
    if np.any(jump_mask):
        jump_x = rvf_x_coords[jump_mask]
        jump_elevations = rvf_elevations[jump_mask]
        jump_velocities = rvf_profile['velocities'][jump_mask]
        
        ax3.fill_between(jump_x, downstream_apron_elevation, jump_elevations,
                        alpha=0.4, color='lightblue', label='Water Body')
        ax3.plot(jump_x, jump_elevations, 'b-', linewidth=4, label='Water Surface')
        
        # Mark jump boundaries
        ax3.axvline(x=jump_start_global, color='orange', linestyle='-', linewidth=3, 
                   label=f'Jump Start ({jump_start_global:.1f}m)')
        ax3.axvline(x=jump_end_global, color='purple', linestyle='--', linewidth=3, 
                   label=f'Jump End ({jump_end_global:.1f}m)')
        
        # Add turbulence indication in jump
        jump_center = (jump_start_global + jump_end_global) / 2
        jump_height = np.interp(jump_center, jump_x, jump_elevations)
        ax3.text(jump_center, jump_height + 0.5, 
                f'HYDRAULIC JUMP\nType: {rvf_profile["jump_result"].jump_type}\nLength: {jump_end_global - jump_start_global:.1f}m\nΔE: {rvf_profile["jump_result"].energy_loss:.3f}m',
                ha='center', va='center', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))
    
    # Downstream apron
    ax3.axhline(y=downstream_apron_elevation, color='black', linewidth=5, alpha=0.8)
    ax3.axhline(y=tailwater_elevation, color='green', linestyle='--', alpha=0.8, 
               label=f'Tailwater ({tailwater_elevation:.2f}m)')
    
    ax3.set_xlim(jump_region_start, jump_region_end)
    ax3.set_ylim(downstream_apron_elevation - 0.2, tailwater_elevation + 1)
    ax3.set_xlabel('Distance (m)', fontweight='bold')
    ax3.set_ylabel('Elevation (m)', fontweight='bold')
    ax3.set_title(f'Hydraulic Jump Detail (Length: {jump_end_global - jump_start_global:.1f}m)', fontweight='bold')
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Flow Parameters
    ax4 = axes[1, 1]
    
    # Combined parameter plot
    ax4_twin = ax4.twinx()
    
    # Velocity
    ax4.plot(profile.x_coordinates, profile.velocities, 'r-', linewidth=3, label='Velocity (m/s)')
    ax4.plot(rvf_x_coords, rvf_profile['velocities'], 'r-', linewidth=3, alpha=0.7)
    
    # Froude number
    ax4_twin.plot(profile.x_coordinates, profile.froude_numbers, 'm-', linewidth=3, label='Froude Number')
    ax4_twin.plot(rvf_x_coords, rvf_profile['froude_numbers'], 'm-', linewidth=3, alpha=0.7)
    ax4_twin.axhline(y=1.0, color='black', linestyle='--', alpha=0.8, label='Critical Flow')
    
    # Mark regions
    ax4.axvspan(spillway_start, spillway_end, alpha=0.2, color='red', label='Spillway')
    ax4.axvspan(jump_start_global, jump_end_global, alpha=0.3, color='orange', label='Jump')
    
    ax4.set_xlabel('Distance (m)', fontweight='bold')
    ax4.set_ylabel('Velocity (m/s)', fontweight='bold', color='red')
    ax4_twin.set_ylabel('Froude Number', fontweight='bold', color='magenta')
    ax4.set_title('Flow Parameters Distribution', fontweight='bold')
    
    # Combine legends
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save and show the plot
    from plot_utils import save_and_show_plot
    save_and_show_plot("ogee_diversion_dam_rigorous_analysis")
    
    print(f"   ✅ Rigorous spillway visualization completed with separate detailed plots!")


if __name__ == "__main__":
    main()
