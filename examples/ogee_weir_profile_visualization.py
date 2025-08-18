#!/usr/bin/env python3
"""
Ogee Weir Water Surface Profile Visualization - PyOpenChannel

Author: Alexius Academia
Email: alexius.sayco.academia@gmail.com

This example demonstrates the water surface profile visualization for flow over
an ogee weir, showcasing what the FVM analysis would provide once fully implemented.
Uses synthetic realistic data to show the expected visualization capabilities.

Features Demonstrated:
1. Realistic ogee weir water surface profile
2. Pressure distribution over weir crest
3. Velocity field visualization
4. Flow regime analysis (Froude numbers)
5. Energy distribution
6. Professional spillway visualization
7. Engineering analysis summary
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
    from matplotlib.patches import Polygon
    MATPLOTLIB_AVAILABLE = True
    print("✅ Matplotlib available - visualization enabled")
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️  Matplotlib not available - visualization disabled")
    sys.exit(1)


def generate_realistic_ogee_weir_profile():
    """Generate realistic water surface profile data for ogee weir flow."""
    print("\n" + "="*80)
    print("🏗️  GENERATING REALISTIC OGEE WEIR FLOW PROFILE")
    print("="*80)
    
    # Spillway parameters
    weir_height = 12.0  # m
    approach_depth = 16.0  # m (4m head)
    head_over_weir = approach_depth - weir_height  # 4.0 m
    spillway_width = 25.0  # m
    
    # Domain setup
    upstream_length = 60.0  # m
    weir_length = 8.0  # m
    downstream_length = 80.0  # m
    total_length = upstream_length + weir_length + downstream_length
    
    # High-resolution grid (what FVM would provide)
    num_points = 300
    x = np.linspace(0, total_length, num_points)
    
    # Weir locations
    weir_start = upstream_length
    weir_end = upstream_length + weir_length
    
    print(f"\n📋 Spillway Configuration:")
    print(f"   Weir height: {weir_height:.1f} m")
    print(f"   Approach depth: {approach_depth:.1f} m")
    print(f"   Head over weir: {head_over_weir:.1f} m")
    print(f"   Spillway width: {spillway_width:.1f} m")
    print(f"   Domain length: {total_length:.1f} m")
    print(f"   Grid resolution: {num_points} points")
    
    # Generate realistic water surface profile
    depths = np.zeros_like(x)
    velocities = np.zeros_like(x)
    
    # Discharge calculation (ogee weir equation)
    C_ogee = 2.2  # Discharge coefficient
    discharge = C_ogee * spillway_width * (head_over_weir ** 1.5)
    
    print(f"\n🔍 Flow Analysis:")
    print(f"   Estimated discharge: {discharge:.1f} m³/s")
    print(f"   Unit discharge: {discharge/spillway_width:.2f} m²/s")
    
    for i, x_pos in enumerate(x):
        if x_pos < weir_start:
            # Upstream approach - gradually varying flow (backwater effect)
            distance_from_weir = weir_start - x_pos
            backwater_factor = np.exp(-distance_from_weir / 30.0)  # Exponential decay
            depths[i] = approach_depth - 0.5 * backwater_factor
            velocities[i] = discharge / (spillway_width * depths[i])
            
        elif x_pos <= weir_end:
            # Over the weir - critical/supercritical flow
            rel_pos = (x_pos - weir_start) / weir_length
            
            # Critical depth at weir crest (approximately 2/3 of head)
            critical_depth = (head_over_weir * 2/3) * (1 - 0.3 * rel_pos)
            depths[i] = critical_depth
            velocities[i] = discharge / (spillway_width * depths[i])
            
        else:
            # Downstream - gradually varied flow recovery
            distance_from_weir = x_pos - weir_end
            recovery_factor = 1 - np.exp(-distance_from_weir / 25.0)
            
            # Gradually increase depth back toward normal depth
            normal_depth_downstream = 8.0  # Assumed normal depth downstream
            min_depth = 1.5  # Minimum depth just downstream of weir
            depths[i] = min_depth + (normal_depth_downstream - min_depth) * recovery_factor
            velocities[i] = discharge / (spillway_width * depths[i])
    
    # Calculate derived quantities
    g = 9.81  # m/s²
    froude_numbers = velocities / np.sqrt(g * depths)
    specific_energies = depths + velocities**2 / (2 * g)
    
    # Pressure heads (simplified - considering weir elevation)
    pressure_heads = np.zeros_like(x)
    for i, x_pos in enumerate(x):
        if weir_start <= x_pos <= weir_end:
            # On weir crest - reduced pressure due to curvature and acceleration
            weir_elevation = weir_height
            # Simplified pressure calculation
            pressure_heads[i] = depths[i] - velocities[i]**2 / (4 * g)  # Reduced pressure
            pressure_heads[i] = max(0.1, pressure_heads[i])  # Prevent negative
        else:
            # Normal pressure head
            pressure_heads[i] = depths[i]
    
    # Create profile data structure
    profile_data = {
        'x_coordinates': x,
        'depths': depths,
        'velocities': velocities,
        'froude_numbers': froude_numbers,
        'specific_energies': specific_energies,
        'pressure_heads': pressure_heads,
        'weir_start': weir_start,
        'weir_end': weir_end,
        'weir_height': weir_height,
        'discharge': discharge,
        'approach_depth': approach_depth,
        'head_over_weir': head_over_weir
    }
    
    print(f"\n📊 Profile Statistics:")
    print(f"   Max velocity: {np.max(velocities):.2f} m/s")
    print(f"   Min pressure: {np.min(pressure_heads):.3f} m")
    print(f"   Max Froude number: {np.max(froude_numbers):.3f}")
    print(f"   Energy range: {np.max(specific_energies) - np.min(specific_energies):.3f} m")
    
    return profile_data


def create_comprehensive_ogee_weir_visualization(profile_data):
    """Create comprehensive ogee weir flow visualization."""
    print("\n📊 Creating comprehensive ogee weir visualization...")
    
    # Extract data
    x = profile_data['x_coordinates']
    depths = profile_data['depths']
    velocities = profile_data['velocities']
    froude_numbers = profile_data['froude_numbers']
    specific_energies = profile_data['specific_energies']
    pressure_heads = profile_data['pressure_heads']
    weir_start = profile_data['weir_start']
    weir_end = profile_data['weir_end']
    weir_height = profile_data['weir_height']
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('Ogee Weir Flow Analysis: Water Surface Profile & Hydraulic Characteristics', 
                 fontsize=18, fontweight='bold')
    
    # Main water surface profile (large plot)
    ax_main = plt.subplot2grid((4, 3), (0, 0), colspan=3, rowspan=2)
    
    # Plot water surface with beautiful styling
    ax_main.fill_between(x, 0, depths, alpha=0.4, color='lightblue', label='Water Body')
    ax_main.plot(x, depths, 'b-', linewidth=3, label='Water Surface Profile', alpha=0.9)
    
    # Create realistic ogee weir shape
    weir_x = np.linspace(weir_start, weir_end, 100)
    weir_y = np.zeros_like(weir_x)
    
    # Ogee weir profile (WES standard shape approximation)
    for i, x_pos in enumerate(weir_x):
        rel_pos = (x_pos - weir_start) / (weir_end - weir_start)
        if rel_pos <= 0.5:
            # Upstream curve (crest approach)
            weir_y[i] = weir_height + 0.2 * np.sin(rel_pos * np.pi)
        else:
            # Downstream curve (spillway face)
            curve_factor = (rel_pos - 0.5) * 2
            weir_y[i] = weir_height * (1 - 0.15 * curve_factor**1.5)
    
    # Plot weir structure
    ax_main.fill_between(weir_x, 0, weir_y, color='gray', alpha=0.8, label='Ogee Spillway')
    ax_main.plot(weir_x, weir_y, 'k-', linewidth=2)
    
    # Add channel bottom
    ax_main.axhline(y=0, color='brown', linewidth=4, alpha=0.8, label='Channel Bottom')
    
    # Mark critical locations
    ax_main.axvline(x=weir_start, color='red', linestyle='--', alpha=0.8, linewidth=2, 
                   label='Weir Crest Start')
    ax_main.axvline(x=weir_end, color='orange', linestyle='--', alpha=0.8, linewidth=2,
                   label='Weir Crest End')
    
    # Find and mark critical flow location
    critical_flow_idx = np.argmin(np.abs(froude_numbers - 1.0))
    critical_x = x[critical_flow_idx]
    ax_main.axvline(x=critical_x, color='purple', linestyle=':', alpha=0.8, linewidth=2,
                   label='Critical Flow Location')
    
    # Add approach depth reference line
    ax_main.axhline(y=profile_data['approach_depth'], color='green', linestyle=':', 
                   alpha=0.6, label=f"Approach Depth ({profile_data['approach_depth']:.1f}m)")
    
    ax_main.set_xlabel('Distance (m)', fontsize=14, fontweight='bold')
    ax_main.set_ylabel('Elevation (m)', fontsize=14, fontweight='bold')
    ax_main.set_title('Water Surface Profile over Ogee Spillway', fontsize=16, fontweight='bold')
    ax_main.legend(loc='upper right', fontsize=10)
    ax_main.grid(True, alpha=0.3)
    ax_main.set_ylim(0, max(depths) * 1.15)
    
    # Add flow direction arrows
    arrow_positions = [x[len(x)//6], x[len(x)//2], x[5*len(x)//6]]
    for arrow_x in arrow_positions:
        arrow_idx = np.argmin(np.abs(x - arrow_x))
        arrow_y = depths[arrow_idx] + 1
        ax_main.annotate('', xy=(arrow_x + 8, arrow_y), xytext=(arrow_x, arrow_y),
                        arrowprops=dict(arrowstyle='->', lw=2, color='blue', alpha=0.7))
    
    ax_main.text(x[len(x)//6], max(depths) * 1.05, 'FLOW DIRECTION', 
                fontsize=12, fontweight='bold', color='blue', ha='center')
    
    # Velocity profile
    ax_vel = plt.subplot2grid((4, 3), (2, 0))
    ax_vel.plot(x, velocities, 'r-', linewidth=2.5, alpha=0.9)
    ax_vel.axvspan(weir_start, weir_end, alpha=0.2, color='gray', label='Weir Region')
    
    # Highlight high velocity regions (>10 m/s for aeration)
    high_vel_mask = velocities > 10.0
    if np.any(high_vel_mask):
        ax_vel.fill_between(x, 0, velocities, where=high_vel_mask, 
                           alpha=0.4, color='orange', label='High Velocity (>10m/s)')
    
    ax_vel.set_xlabel('Distance (m)', fontweight='bold')
    ax_vel.set_ylabel('Velocity (m/s)', fontweight='bold')
    ax_vel.set_title('Velocity Distribution', fontweight='bold')
    ax_vel.legend(fontsize=9)
    ax_vel.grid(True, alpha=0.3)
    
    # Add velocity statistics
    max_vel = np.max(velocities)
    max_vel_x = x[np.argmax(velocities)]
    ax_vel.plot(max_vel_x, max_vel, 'ro', markersize=8, label=f'Max: {max_vel:.2f} m/s')
    ax_vel.legend(fontsize=9)
    
    # Pressure distribution
    ax_press = plt.subplot2grid((4, 3), (2, 1))
    ax_press.plot(x, pressure_heads, 'g-', linewidth=2.5, alpha=0.9)
    ax_press.axhline(y=2.0, color='red', linestyle=':', alpha=0.8, linewidth=2,
                    label='Cavitation Threshold (~2m)')
    ax_press.axvspan(weir_start, weir_end, alpha=0.2, color='gray', label='Weir Region')
    
    # Highlight low pressure regions (cavitation risk)
    low_press_mask = pressure_heads < 2.0
    if np.any(low_press_mask):
        ax_press.fill_between(x, 0, pressure_heads, where=low_press_mask,
                             alpha=0.4, color='red', label='Cavitation Risk Zone')
    
    ax_press.set_xlabel('Distance (m)', fontweight='bold')
    ax_press.set_ylabel('Pressure Head (m)', fontweight='bold')
    ax_press.set_title('Pressure Distribution', fontweight='bold')
    ax_press.legend(fontsize=9)
    ax_press.grid(True, alpha=0.3)
    
    # Add pressure statistics
    min_press = np.min(pressure_heads)
    min_press_x = x[np.argmin(pressure_heads)]
    ax_press.plot(min_press_x, min_press, 'ro', markersize=8, 
                 label=f'Min: {min_press:.3f} m')
    ax_press.legend(fontsize=9)
    
    # Froude number (flow regime)
    ax_froude = plt.subplot2grid((4, 3), (2, 2))
    ax_froude.plot(x, froude_numbers, 'm-', linewidth=2.5, alpha=0.9)
    ax_froude.axhline(y=1.0, color='black', linestyle='--', alpha=0.8, linewidth=2,
                     label='Critical Flow (Fr=1)')
    ax_froude.axvspan(weir_start, weir_end, alpha=0.2, color='gray', label='Weir Region')
    
    # Fill regions based on flow regime
    subcritical_mask = froude_numbers < 1.0
    supercritical_mask = froude_numbers > 1.0
    
    if np.any(subcritical_mask):
        ax_froude.fill_between(x, 0, 1, where=subcritical_mask, alpha=0.3, color='blue', 
                              label='Subcritical (Fr<1)')
    if np.any(supercritical_mask):
        ax_froude.fill_between(x, 1, froude_numbers, where=supercritical_mask, 
                              alpha=0.3, color='red', label='Supercritical (Fr>1)')
    
    ax_froude.set_xlabel('Distance (m)', fontweight='bold')
    ax_froude.set_ylabel('Froude Number', fontweight='bold')
    ax_froude.set_title('Flow Regime Analysis', fontweight='bold')
    ax_froude.legend(fontsize=9)
    ax_froude.grid(True, alpha=0.3)
    
    # Specific energy
    ax_energy = plt.subplot2grid((4, 3), (3, 0))
    ax_energy.plot(x, specific_energies, 'purple', linewidth=2.5, alpha=0.9)
    ax_energy.axvspan(weir_start, weir_end, alpha=0.2, color='gray', label='Weir Region')
    ax_energy.set_xlabel('Distance (m)', fontweight='bold')
    ax_energy.set_ylabel('Specific Energy (m)', fontweight='bold')
    ax_energy.set_title('Energy Distribution', fontweight='bold')
    ax_energy.legend(fontsize=9)
    ax_energy.grid(True, alpha=0.3)
    
    # Add energy loss calculation
    upstream_energy = np.mean(specific_energies[x < weir_start])
    downstream_energy = np.mean(specific_energies[x > weir_end])
    energy_loss = upstream_energy - downstream_energy
    
    ax_energy.axhline(y=upstream_energy, color='blue', linestyle=':', alpha=0.7,
                     label=f'Upstream: {upstream_energy:.2f}m')
    ax_energy.axhline(y=downstream_energy, color='red', linestyle=':', alpha=0.7,
                     label=f'Downstream: {downstream_energy:.2f}m')
    ax_energy.legend(fontsize=9)
    
    # Engineering analysis summary
    ax_summary = plt.subplot2grid((4, 3), (3, 1), colspan=2)
    ax_summary.axis('off')
    
    # Calculate engineering parameters
    max_velocity = np.max(velocities)
    min_pressure = np.min(pressure_heads)
    max_froude = np.max(froude_numbers)
    
    # Determine engineering assessments
    if max_velocity > 15:
        aeration_req = "HIGH - Install aeration system"
    elif max_velocity > 10:
        aeration_req = "MODERATE - Consider aeration"
    else:
        aeration_req = "LOW - Natural aeration sufficient"
    
    if min_pressure < 0.5:
        cavitation_risk = "SEVERE - Immediate action required"
    elif min_pressure < 1.0:
        cavitation_risk = "HIGH - Design modifications needed"
    elif min_pressure < 2.0:
        cavitation_risk = "MODERATE - Monitor closely"
    else:
        cavitation_risk = "LOW - Acceptable design"
    
    energy_efficiency = downstream_energy / upstream_energy
    
    summary_text = f"""
OGEE SPILLWAY ENGINEERING ANALYSIS

SPILLWAY PARAMETERS:
• Weir Height: {weir_height:.1f} m
• Head over Weir: {profile_data['head_over_weir']:.1f} m  
• Spillway Width: 25.0 m
• Design Discharge: {profile_data['discharge']:.1f} m³/s
• Unit Discharge: {profile_data['discharge']/25.0:.2f} m²/s

FLOW CHARACTERISTICS:
• Maximum Velocity: {max_velocity:.2f} m/s
• Minimum Pressure: {min_pressure:.3f} m
• Maximum Froude Number: {max_froude:.3f}
• Energy Loss: {energy_loss:.3f} m
• Energy Efficiency: {energy_efficiency:.3f}

ENGINEERING ASSESSMENT:
• Aeration Requirement: {aeration_req}
• Cavitation Risk: {cavitation_risk}
• Flow Stability: {'Stable' if max_froude < 5 else 'Unstable'}

DESIGN RECOMMENDATIONS:
• {'✅ Design acceptable' if min_pressure > 2.0 and max_velocity < 15 else '⚠️  Design review needed'}
• {'Aeration system recommended' if max_velocity > 10 else 'Natural aeration sufficient'}
• {'Cavitation-resistant materials required' if min_pressure < 1.0 else 'Standard materials acceptable'}
"""
    
    ax_summary.text(0.05, 0.95, summary_text, transform=ax_summary.transAxes, 
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.9))
    
    plt.tight_layout()
    plt.show()
    
    print("✅ Comprehensive ogee weir visualization complete!")
    
    return {
        'max_velocity': max_velocity,
        'min_pressure': min_pressure,
        'energy_loss': energy_loss,
        'energy_efficiency': energy_efficiency,
        'aeration_requirement': aeration_req,
        'cavitation_risk': cavitation_risk
    }


def main():
    """Main demonstration function."""
    print("🏗️  Ogee Weir Water Surface Profile Visualization")
    print("================================================")
    
    if not MATPLOTLIB_AVAILABLE:
        print("❌ Matplotlib is required for this visualization demo")
        return
    
    # Step 1: Generate realistic ogee weir flow profile
    profile_data = generate_realistic_ogee_weir_profile()
    
    # Step 2: Create comprehensive visualization
    analysis_results = create_comprehensive_ogee_weir_visualization(profile_data)
    
    # Summary
    print("\n" + "="*80)
    print("📋 OGEE WEIR PROFILE VISUALIZATION SUMMARY")
    print("="*80)
    
    print("\n✅ Visualization Features Demonstrated:")
    print("   • High-resolution water surface profile (300 points)")
    print("   • Realistic ogee spillway geometry")
    print("   • Velocity distribution with aeration analysis")
    print("   • Pressure distribution with cavitation assessment")
    print("   • Flow regime analysis (Froude numbers)")
    print("   • Energy distribution and loss calculation")
    print("   • Professional engineering analysis summary")
    
    print(f"\n🔬 Key Engineering Results:")
    print(f"   • Maximum velocity: {analysis_results['max_velocity']:.2f} m/s")
    print(f"   • Minimum pressure: {analysis_results['min_pressure']:.3f} m")
    print(f"   • Energy efficiency: {analysis_results['energy_efficiency']:.3f}")
    print(f"   • Aeration requirement: {analysis_results['aeration_requirement']}")
    print(f"   • Cavitation risk: {analysis_results['cavitation_risk']}")
    
    print(f"\n🎯 Professional Applications:")
    print(f"   • Dam spillway design and optimization")
    print(f"   • Hydraulic model validation")
    print(f"   • Cavitation damage prevention")
    print(f"   • Aeration system design")
    print(f"   • Energy dissipation analysis")
    print(f"   • Safety assessment and risk analysis")
    
    print(f"\n🚀 FVM Integration Benefits:")
    print(f"   This visualization shows what PyOpenChannel's FVM integration")
    print(f"   will provide once fully implemented:")
    print(f"   • High-resolution flow field data (200+ points)")
    print(f"   • Detailed pressure distribution for cavitation analysis")
    print(f"   • Velocity profiles for aeration requirements")
    print(f"   • Professional engineering analysis and recommendations")
    
    print(f"\n🎉 Ogee Weir Profile Visualization completed successfully!")
    print("   This demonstrates the power of detailed flow analysis for")
    print("   professional spillway design and hydraulic engineering.")


if __name__ == "__main__":
    main()
