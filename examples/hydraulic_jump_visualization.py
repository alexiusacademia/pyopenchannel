#!/usr/bin/env python3
"""
Hydraulic Jump Visualization Example - PyOpenChannel

Author: Alexius Academia
Email: alexius.sayco.academia@gmail.com

This example demonstrates hydraulic jump analysis and visualization,
showing the difference between spillway flow (GVF) and hydraulic jump (RVF).

Features:
1. Hydraulic jump analysis with sequent depths
2. Energy dissipation in hydraulic jumps
3. Jump classification (weak, oscillating, steady, strong)
4. Comparison with spillway flow
5. Comprehensive matplotlib visualization
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pyopenchannel as poc
import math

# Optional matplotlib import for plotting
try:
    import matplotlib.pyplot as plt
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
    print("📊 Matplotlib available - plots will be generated")
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️  Matplotlib not available - analysis will run without plots")


def analyze_hydraulic_jump_scenario():
    """Analyze a hydraulic jump scenario with visualization."""
    
    print("🌊 HYDRAULIC JUMP ANALYSIS & VISUALIZATION")
    print("=" * 80)
    
    # Set up scenario: Flow from a sluice gate creating supercritical flow
    # that transitions to subcritical flow (hydraulic jump)
    
    poc.set_unit_system(poc.UnitSystem.SI)
    
    # Channel geometry
    channel_width = 8.0  # m
    channel = poc.RectangularChannel(width=channel_width)
    
    # Flow conditions
    discharge = 25.0  # m³/s
    upstream_depth = 0.8  # m (supercritical depth after gate)
    downstream_depth = 3.2  # m (subcritical depth - tailwater control)
    
    print(f"📊 FLOW CONDITIONS:")
    print(f"   Channel width: {channel_width} m")
    print(f"   Discharge: {discharge} m³/s")
    print(f"   Upstream depth (y1): {upstream_depth} m")
    print(f"   Downstream depth (y2): {downstream_depth} m")
    print(f"   Unit discharge: {discharge/channel_width:.2f} m³/s/m")
    
    # Calculate flow properties
    velocity_1 = discharge / (channel_width * upstream_depth)
    velocity_2 = discharge / (channel_width * downstream_depth)
    
    froude_1 = velocity_1 / math.sqrt(9.81 * upstream_depth)
    froude_2 = velocity_2 / math.sqrt(9.81 * downstream_depth)
    
    print(f"\n🌊 UPSTREAM CONDITIONS (Supercritical):")
    print(f"   Velocity: {velocity_1:.2f} m/s")
    print(f"   Froude number: {froude_1:.2f}")
    print(f"   Flow regime: {'Supercritical' if froude_1 > 1 else 'Subcritical'}")
    
    print(f"\n🌊 DOWNSTREAM CONDITIONS (Subcritical):")
    print(f"   Velocity: {velocity_2:.2f} m/s")
    print(f"   Froude number: {froude_2:.2f}")
    print(f"   Flow regime: {'Supercritical' if froude_2 > 1 else 'Subcritical'}")
    
    # RVF Analysis for hydraulic jump
    rvf_solver = poc.RVFSolver()
    
    # Analyze the hydraulic jump
    result = rvf_solver.analyze_hydraulic_jump(
        channel=channel,
        discharge=discharge,
        upstream_depth=upstream_depth,
        tailwater_depth=downstream_depth
    )
    
    if result.success:
        print(f"\n✅ HYDRAULIC JUMP ANALYSIS:")
        print(f"   Analysis Status: SUCCESS")
        print(f"   Jump type: {result.jump_type.value.upper()}")
        print(f"   Downstream depth: {result.downstream_depth:.3f} m")
        print(f"   Sequent depth ratio: {result.sequent_depth_ratio:.3f}")
        print(f"   Jump efficiency: {result.energy_efficiency:.1%}")
        
        print(f"\n⚡ ENERGY ANALYSIS:")
        print(f"   Upstream specific energy: {result.upstream_energy:.3f} m")
        print(f"   Downstream specific energy: {result.downstream_energy:.3f} m")
        print(f"   Energy dissipated: {result.energy_loss:.3f} m")
        print(f"   Energy loss: {(1-result.energy_efficiency)*100:.1f}%")
        
        # Power dissipation
        power_dissipated = 9810 * discharge * result.energy_loss / 1000  # kW
        print(f"   Power dissipated: {power_dissipated:.0f} kW")
        
        print(f"\n📏 JUMP CHARACTERISTICS:")
        print(f"   Jump height: {result.jump_height:.2f} m" if result.jump_height else "   Jump height: N/A")
        print(f"   Jump length: {result.jump_length:.1f} m" if result.jump_length else "   Jump length: N/A")
        
        if result.properties and 'stilling_basin' in result.properties:
            basin = result.properties['stilling_basin']
            print(f"\n🏗️  STILLING BASIN REQUIREMENTS:")
            print(f"   Recommended basin length: {basin.get('basin_length', 'N/A')}")
            print(f"   Basin depth: {basin.get('basin_depth', 'N/A')}")
            print(f"   End sill height: {basin.get('end_sill_height', 'N/A')}")
            print(f"   Baffle block spacing: {basin.get('baffle_spacing', 'N/A')}")
        
    else:
        print(f"❌ HYDRAULIC JUMP ANALYSIS FAILED: {result.message}")
        return None
    
    return result


def create_hydraulic_jump_visualization(result, channel_width, discharge, 
                                      upstream_depth, downstream_depth):
    """Create comprehensive hydraulic jump visualization."""
    
    if not MATPLOTLIB_AVAILABLE or not result:
        print("📊 Cannot create visualization")
        return
    
    print(f"\n📊 CREATING HYDRAULIC JUMP VISUALIZATION...")
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('Hydraulic Jump Analysis - RVF (Rapidly Varied Flow)\nPyOpenChannel Professional Analysis', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Jump Profile
    ax1 = plt.subplot(2, 3, 1)
    create_jump_profile_plot(ax1, result, upstream_depth, downstream_depth)
    
    # Plot 2: Energy Grade Line
    ax2 = plt.subplot(2, 3, 2)
    create_jump_energy_plot(ax2, result, upstream_depth, downstream_depth)
    
    # Plot 3: Velocity Distribution
    ax3 = plt.subplot(2, 3, 3)
    create_velocity_profile_plot(ax3, result, channel_width, discharge, 
                                upstream_depth, downstream_depth)
    
    # Plot 4: Froude Number Variation
    ax4 = plt.subplot(2, 3, 4)
    create_froude_variation_plot(ax4, result, upstream_depth, downstream_depth)
    
    # Plot 5: Jump Classification
    ax5 = plt.subplot(2, 3, 5)
    create_jump_classification_plot(ax5, result)
    
    # Plot 6: Stilling Basin Design
    ax6 = plt.subplot(2, 3, 6)
    create_stilling_basin_plot(ax6, result)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    # Save the plot
    plt.savefig('hydraulic_jump_analysis.png', dpi=300, bbox_inches='tight')
    print(f"   ✅ Hydraulic jump plots saved as 'hydraulic_jump_analysis.png'")
    
    plt.show()


def create_jump_profile_plot(ax, result, y1, y2):
    """Create hydraulic jump profile plot."""
    
    # Jump length (use default if not available)
    jump_length = result.jump_length if result.jump_length else 5.0
    
    # Create jump profile
    x = np.linspace(-10, jump_length + 10, 200)
    y = np.zeros_like(x)
    
    # Upstream supercritical flow
    upstream_region = x < 0
    y[upstream_region] = y1
    
    # Jump region (simplified S-curve)
    jump_region = (x >= 0) & (x <= jump_length)
    x_jump = x[jump_region]
    # Smooth transition using tanh function
    transition = 0.5 * (1 + np.tanh(4 * (x_jump / jump_length - 0.5)))
    y[jump_region] = y1 + (y2 - y1) * transition
    
    # Downstream subcritical flow
    downstream_region = x > jump_length
    y[downstream_region] = y2
    
    # Plot water surface
    ax.plot(x, y, 'b-', linewidth=3, label='Water Surface')
    ax.fill_between(x, 0, y, alpha=0.3, color='blue', label='Water')
    
    # Add channel bottom
    ax.axhline(y=0, color='brown', linewidth=3, label='Channel Bottom')
    
    # Mark jump boundaries
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Jump Start')
    ax.axvline(x=jump_length, color='red', linestyle='--', alpha=0.7, label='Jump End')
    
    # Add annotations
    ax.annotate(f'y₁ = {y1:.2f}m\nFr₁ = {result.upstream_froude:.2f}', 
                xy=(-5, y1/2), fontsize=10, ha='center',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    ax.annotate(f'y₂ = {y2:.2f}m\nFr₂ = {result.downstream_froude:.2f}', 
                xy=(jump_length + 5, y2/2), fontsize=10, ha='center',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    ax.annotate(f'Jump Length = {jump_length:.1f}m', 
                xy=(jump_length/2, max(y1, y2) + 0.2), fontsize=10, ha='center',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Depth (m)')
    ax.set_title('Hydraulic Jump Profile')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(y1, y2) * 1.3)


def create_jump_energy_plot(ax, result, y1, y2):
    """Create energy grade line for hydraulic jump."""
    
    # Jump length (use default if not available)
    jump_length = result.jump_length if result.jump_length else 5.0
    
    x = np.linspace(-10, jump_length + 10, 100)
    
    # Energy levels
    E1 = result.upstream_energy
    E2 = result.downstream_energy
    
    # Energy grade line (step function for jump)
    energy_line = np.where(x < jump_length/2, E1, E2)
    
    # Water surface (from previous plot)
    water_surface = np.where(x < 0, y1, 
                           np.where(x > jump_length, y2, 
                                  y1 + (y2-y1) * x/jump_length))
    
    ax.plot(x, energy_line, 'r-', linewidth=2, label='Energy Grade Line')
    ax.plot(x, water_surface, 'b-', linewidth=2, label='Water Surface')
    
    # Fill energy loss area
    jump_x = x[(x >= 0) & (x <= jump_length)]
    E1_line = np.full_like(jump_x, E1)
    E2_line = np.full_like(jump_x, E2)
    
    ax.fill_between(jump_x, E2_line, E1_line, alpha=0.3, color='red', 
                   label=f'Energy Loss ({result.energy_loss:.2f}m)')
    
    # Annotations
    ax.annotate(f'E₁ = {E1:.2f}m', xy=(-5, E1), xytext=(-5, E1 + 0.3),
                fontsize=10, ha='center', color='red', fontweight='bold')
    ax.annotate(f'E₂ = {E2:.2f}m', xy=(jump_length + 5, E2), 
                xytext=(jump_length + 5, E2 + 0.3),
                fontsize=10, ha='center', color='red', fontweight='bold')
    
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Energy (m)')
    ax.set_title('Energy Grade Line')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def create_velocity_profile_plot(ax, result, width, discharge, y1, y2):
    """Create velocity distribution plot."""
    
    # Jump length (use default if not available)
    jump_length = result.jump_length if result.jump_length else 5.0
    
    # Calculate velocities
    v1 = discharge / (width * y1)
    v2 = discharge / (width * y2)
    
    # Create velocity profile across jump
    x = np.linspace(-5, jump_length + 5, 50)
    velocities = np.where(x < jump_length/2, v1, v2)
    
    ax.plot(x, velocities, 'g-', linewidth=3, label='Average Velocity')
    
    # Add velocity vectors (simplified)
    x_arrows = np.linspace(-3, jump_length + 3, 8)
    for x_pos in x_arrows:
        v = v1 if x_pos < jump_length/2 else v2
        ax.arrow(x_pos, v, 0.5, 0, head_width=0.1, head_length=0.2, 
                fc='green', ec='green', alpha=0.7)
    
    # Mark critical velocity
    v_critical = math.sqrt(9.81 * (discharge/width)**(2/3))
    ax.axhline(y=v_critical, color='red', linestyle='--', alpha=0.7,
               label=f'Critical Velocity ({v_critical:.2f}m/s)')
    
    # Annotations
    ax.annotate(f'V₁ = {v1:.2f}m/s\n(Supercritical)', 
                xy=(-2, v1), fontsize=10, ha='center',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    ax.annotate(f'V₂ = {v2:.2f}m/s\n(Subcritical)', 
                xy=(jump_length + 2, v2), fontsize=10, ha='center',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title('Velocity Distribution')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def create_froude_variation_plot(ax, result, y1, y2):
    """Create Froude number variation plot."""
    
    # Jump length (use default if not available)
    jump_length = result.jump_length if result.jump_length else 5.0
    
    x = np.linspace(-5, jump_length + 5, 100)
    
    # Froude numbers
    Fr1 = result.upstream_froude
    Fr2 = result.downstream_froude
    
    # Froude variation (step function)
    froude_line = np.where(x < jump_length/2, Fr1, Fr2)
    
    ax.plot(x, froude_line, 'purple', linewidth=3, label='Froude Number')
    
    # Critical line
    ax.axhline(y=1.0, color='red', linestyle='-', alpha=0.7, linewidth=2,
               label='Critical Flow (Fr = 1)')
    
    # Shade flow regimes
    ax.axhspan(0, 1, alpha=0.2, color='blue', label='Subcritical (Fr < 1)')
    ax.axhspan(1, max(Fr1, 3), alpha=0.2, color='red', label='Supercritical (Fr > 1)')
    
    # Annotations
    ax.annotate(f'Fr₁ = {Fr1:.2f}\nSupercritical', 
                xy=(-2, Fr1), fontsize=10, ha='center',
                bbox=dict(boxstyle='round', facecolor='pink', alpha=0.8))
    
    ax.annotate(f'Fr₂ = {Fr2:.2f}\nSubcritical', 
                xy=(jump_length + 2, Fr2), fontsize=10, ha='center',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Froude Number')
    ax.set_title('Froude Number Variation')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(Fr1, 3))


def create_jump_classification_plot(ax, result):
    """Create jump classification plot."""
    
    # Jump classification ranges
    classifications = ['Undular\n(1.0-1.7)', 'Weak\n(1.7-2.5)', 'Oscillating\n(2.5-4.5)', 
                      'Steady\n(4.5-9.0)', 'Strong\n(>9.0)']
    ranges = [(1.0, 1.7), (1.7, 2.5), (2.5, 4.5), (4.5, 9.0), (9.0, 15.0)]
    colors = ['lightgreen', 'yellow', 'orange', 'red', 'darkred']
    
    # Create bars
    for i, (classification, (fr_min, fr_max), color) in enumerate(zip(classifications, ranges, colors)):
        ax.barh(i, fr_max - fr_min, left=fr_min, color=color, alpha=0.7, 
                edgecolor='black', linewidth=1)
        ax.text(fr_min + (fr_max - fr_min)/2, i, classification, 
                ha='center', va='center', fontweight='bold', fontsize=9)
    
    # Mark current jump
    current_fr = result.upstream_froude
    current_type = result.jump_type.value
    
    # Find which category
    for i, (fr_min, fr_max) in enumerate(ranges):
        if fr_min <= current_fr <= fr_max:
            ax.axvline(x=current_fr, color='blue', linewidth=3, 
                      label=f'Current Jump\nFr₁ = {current_fr:.2f}\n({current_type.title()})')
            break
    
    ax.set_xlabel('Upstream Froude Number (Fr₁)')
    ax.set_ylabel('Jump Classification')
    ax.set_title('Hydraulic Jump Classification')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlim(1, 12)
    ax.set_ylim(-0.5, 4.5)


def create_stilling_basin_plot(ax, result):
    """Create stilling basin design plot."""
    
    if not (result.properties and 'stilling_basin' in result.properties):
        ax.text(0.5, 0.5, 'Stilling Basin\nAnalysis\nNot Available', 
                transform=ax.transAxes, ha='center', va='center', 
                fontsize=14, bbox=dict(boxstyle='round', facecolor='lightgray'))
        ax.set_title('Stilling Basin Design')
        return
    
    basin = result.properties['stilling_basin']
    
    # Basin dimensions
    basin_length = basin['basin_length']
    basin_depth = basin['basin_depth']
    end_sill_height = basin['end_sill_height']
    
    # Create basin profile
    x_basin = [0, basin_length, basin_length, 0, 0]
    y_basin = [0, 0, -basin_depth, -basin_depth, 0]
    
    ax.plot(x_basin, y_basin, 'k-', linewidth=2, label='Stilling Basin')
    ax.fill(x_basin, y_basin, alpha=0.3, color='gray', label='Concrete Basin')
    
    # End sill
    sill_x = [basin_length, basin_length + 1, basin_length + 1, basin_length]
    sill_y = [0, 0, end_sill_height, end_sill_height]
    ax.fill(sill_x, sill_y, color='brown', alpha=0.7, label='End Sill')
    
    # Baffle blocks (simplified)
    if 'baffle_spacing' in basin:
        spacing = basin['baffle_spacing']
        for x_pos in np.arange(spacing, basin_length, spacing):
            ax.add_patch(plt.Rectangle((x_pos-0.2, -basin_depth), 0.4, basin_depth/2, 
                                     facecolor='darkgray', alpha=0.8))
    
    # Dimensions
    ax.annotate(f'L = {basin_length:.1f}m', 
                xy=(basin_length/2, -basin_depth/2), fontsize=10, ha='center',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    ax.annotate(f'D = {basin_depth:.1f}m', 
                xy=(-1, -basin_depth/2), fontsize=10, ha='center', rotation=90,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Elevation (m)')
    ax.set_title('Stilling Basin Design')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')


def compare_spillway_vs_jump():
    """Compare spillway flow vs hydraulic jump characteristics."""
    
    print(f"\n📊 SPILLWAY FLOW vs HYDRAULIC JUMP COMPARISON:")
    print("=" * 80)
    
    print(f"{'Characteristic':<25} | {'Spillway Flow':<20} | {'Hydraulic Jump':<20}")
    print(f"{'-'*25} | {'-'*20} | {'-'*20}")
    print(f"{'Flow Type':<25} | {'GVF (Gradual)':<20} | {'RVF (Rapid)':<20}")
    print(f"{'Energy Dissipation':<25} | {'Friction/Turbulence':<20} | {'Turbulent Mixing':<20}")
    print(f"{'Flow Transition':<25} | {'Smooth':<20} | {'Abrupt':<20}")
    print(f"{'Length Scale':<25} | {'Long (100s of m)':<20} | {'Short (10s of m)':<20}")
    print(f"{'Froude Change':<25} | {'Gradual':<20} | {'Sudden':<20}")
    print(f"{'Energy Loss':<25} | {'10-30%':<20} | {'20-70%':<20}")
    print(f"{'Application':<25} | {'Spillways, Weirs':<20} | {'Gates, Drops':<20}")
    
    print(f"\n💡 KEY DIFFERENCES:")
    print(f"   🌊 SPILLWAY: Water flows smoothly over curved surface")
    print(f"   ⚡ HYDRAULIC JUMP: Violent mixing with air entrainment")
    print(f"   📏 SPILLWAY: Energy loss through friction along length")
    print(f"   💥 HYDRAULIC JUMP: Energy loss through turbulent roller")


def main():
    """Run hydraulic jump analysis and visualization."""
    
    print("🌊 HYDRAULIC JUMP vs SPILLWAY FLOW ANALYSIS")
    print("=" * 90)
    print("Demonstrating RVF (Rapidly Varied Flow) vs GVF (Gradually Varied Flow)")
    print("Author: Alexius Academia")
    print("=" * 90)
    
    try:
        # Analyze hydraulic jump
        result = analyze_hydraulic_jump_scenario()
        
        if result and result.success:
            # Create visualization
            create_hydraulic_jump_visualization(
                result, 8.0, 25.0, 0.8, 3.2
            )
            
            # Compare with spillway flow
            compare_spillway_vs_jump()
            
            print("\n" + "=" * 90)
            print("🎉 HYDRAULIC JUMP ANALYSIS COMPLETED!")
            print("=" * 90)
            
            print(f"\n🎯 KEY FINDINGS:")
            print(f"   • Jump type: {result.jump_type.value.upper()}")
            print(f"   • Energy dissipation: {result.energy_loss:.1f}m ({(1-result.energy_efficiency)*100:.0f}%)")
            print(f"   • Jump length: {result.jump_length:.1f}m" if result.jump_length else "   • Jump length: N/A")
            print(f"   • Power dissipated: {9810 * 25.0 * result.energy_loss / 1000:.0f} kW")
            
        else:
            print("\n❌ Hydraulic jump analysis failed")
            
    except Exception as e:
        print(f"\n❌ Analysis failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
