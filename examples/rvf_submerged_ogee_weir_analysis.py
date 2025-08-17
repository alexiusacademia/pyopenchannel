#!/usr/bin/env python3
"""
Submerged Crest Ogee Weir Analysis Example - PyOpenChannel

Author: Alexius Academia
Email: alexius.sayco.academia@gmail.com

This example demonstrates comprehensive analysis of a submerged crest ogee weir
using real engineering data. It covers:

1. Submerged flow conditions and submergence effects
2. Ogee spillway hydraulic analysis
3. Energy dissipation calculations
4. Aeration requirements for high-head spillways
5. Cavitation risk assessment
6. Professional design recommendations
7. Downstream energy dissipation analysis

GIVEN CONDITIONS:
- Discharge: 243 m³/s
- Spillway width: 34 m
- Crest elevation: 37.8 m
- Upstream apron elevation: 35.8 m
- Downstream apron elevation: 35.2 m
- Tailwater elevation: 39.08 m

This represents a typical dam spillway with significant submergence effects.
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


def analyze_submerged_ogee_weir():
    """Analyze the submerged crest ogee weir with given conditions."""
    
    print("🌊 SUBMERGED CREST OGEE WEIR ANALYSIS")
    print("=" * 80)
    
    # Given conditions
    discharge = 243.0  # m³/s
    spillway_width = 34.0  # m
    crest_elevation = 37.8  # m
    upstream_apron_elevation = 35.8  # m
    downstream_apron_elevation = 35.2  # m
    tailwater_elevation = 39.08  # m
    
    print(f"📊 GIVEN CONDITIONS:")
    print(f"   Discharge: {discharge} m³/s")
    print(f"   Spillway width: {spillway_width} m")
    print(f"   Crest elevation: {crest_elevation} m")
    print(f"   Upstream apron elevation: {upstream_apron_elevation} m")
    print(f"   Downstream apron elevation: {downstream_apron_elevation} m")
    print(f"   Tailwater elevation: {tailwater_elevation} m")
    
    # Calculate approach conditions
    # Assume approach channel is 1.5 times spillway width
    approach_width = spillway_width * 1.0
    
    # Calculate upstream water level from energy considerations
    # Assume approach velocity head is small initially
    unit_discharge = discharge / spillway_width
    print(f"   Unit discharge: {unit_discharge:.2f} m³/s/m")
    
    # Estimate upstream water level (iterative approach)
    # Start with assumption of critical flow over spillway
    critical_depth_spillway = (unit_discharge**2 / 9.81)**(1/3)
    estimated_upstream_level = crest_elevation + critical_depth_spillway * 1.5
    
    print(f"   Estimated critical depth on spillway: {critical_depth_spillway:.2f} m")
    print(f"   Estimated upstream water level: {estimated_upstream_level:.2f} m")
    
    # Calculate actual depths
    approach_depth = estimated_upstream_level - upstream_apron_elevation
    downstream_depth = tailwater_elevation - downstream_apron_elevation
    
    print(f"   Approach depth: {approach_depth:.2f} m")
    print(f"   Downstream depth: {downstream_depth:.2f} m")
    
    # Set up channel and weir geometry
    poc.set_unit_system(poc.UnitSystem.SI)
    
    # Create approach channel (wider than spillway)
    approach_channel = poc.RectangularChannel(width=approach_width)
    
    # Create ogee weir geometry
    weir_height = crest_elevation - upstream_apron_elevation
    
    ogee_weir = poc.WeirGeometry(
        weir_type=poc.WeirType.OGEE_SPILLWAY,
        weir_height=weir_height,
        crest_length=spillway_width,
        weir_width=spillway_width,
        approach_length=approach_width * 2,  # Adequate approach length
        spillway_shape="WES",
        upstream_slope=0,  # Typical ogee upstream slope
        downstream_slope=0.75  # Typical ogee downstream slope
    )
    
    print(f"\n🏗️  SPILLWAY GEOMETRY:")
    print(f"   Weir height above approach: {weir_height:.2f} m")
    print(f"   Spillway crest length: {spillway_width} m")
    print(f"   Approach channel width: {approach_width} m")
    print(f"   Spillway shape: {ogee_weir.spillway_shape}")
    
    # Initialize weir flow solver
    weir_solver = poc.WeirFlowSolver()
    
    # Analyze weir flow with submergence
    result = weir_solver.analyze_weir_flow(
        channel=approach_channel,
        weir=ogee_weir,
        approach_depth=approach_depth,
        discharge=discharge,
        downstream_depth=downstream_depth
    )
    
    if result.success:
        print(f"\n✅ HYDRAULIC ANALYSIS RESULTS:")
        print(f"   Analysis Status: SUCCESS")
        print(f"   Head over weir: {result.head_over_weir:.3f} m")
        print(f"   Flow condition: {result.weir_condition.value.upper()}")
        print(f"   Submergence ratio: {result.submergence_ratio:.3f}")
        print(f"   Modular limit: {result.modular_limit:.3f}")
        
        # Check if submerged
        if result.submergence_ratio > result.modular_limit:
            submergence_effect = (1 - result.submergence_ratio / result.modular_limit) * 100
            print(f"   🌊 SUBMERGED FLOW DETECTED!")
            print(f"   Submergence reduction: {-submergence_effect:.1f}%")
        else:
            print(f"   ✅ Free flow conditions")
        
        print(f"\n⚡ DISCHARGE ANALYSIS:")
        print(f"   Calculated discharge: {result.discharge:.1f} m³/s")
        print(f"   Target discharge: {discharge:.1f} m³/s")
        discharge_error = abs(result.discharge - discharge) / discharge * 100
        print(f"   Discharge accuracy: {100-discharge_error:.1f}%")
        print(f"   Discharge coefficient: {result.discharge_coefficient:.3f}")
        print(f"   Submergence factor: {result.submergence_factor:.3f}")
        
        print(f"\n🌊 FLOW CHARACTERISTICS:")
        print(f"   Approach velocity: {result.approach_velocity:.2f} m/s")
        print(f"   Approach Froude number: {result.froude_approach:.2f}")
        print(f"   Effective spillway length: {result.effective_length:.1f} m")
        
        print(f"\n⚡ ENERGY ANALYSIS:")
        print(f"   Approach energy: {result.approach_energy:.2f} m")
        print(f"   Crest energy: {result.crest_energy:.2f} m")
        if result.downstream_energy:
            print(f"   Downstream energy: {result.downstream_energy:.2f} m")
        print(f"   Energy dissipated: {result.energy_dissipated:.2f} m")
        print(f"   Energy efficiency: {result.energy_efficiency:.1%}")
        
        # Power calculations
        power_dissipated = 9810 * discharge * result.energy_dissipated / 1000  # kW
        print(f"   Power dissipated: {power_dissipated:.0f} kW")
        
        print(f"\n🌬️  AERATION ANALYSIS:")
        print(f"   Aeration requirement: {result.aeration_requirement.value.upper()}")
        if result.aeration_analysis:
            aeration = result.aeration_analysis
            print(f"   Required aeration capacity: {aeration['aeration_capacity']:.2f} m³/s")
            print(f"   Aeration ratio: {aeration['aeration_ratio']*100:.1f}%")
            if aeration['aeration_slots'] > 0:
                print(f"   Recommended aeration slots: {aeration['aeration_slots']}")
                print(f"   Slot width: {aeration['slot_width']:.1f} m")
                print(f"   Total slot area: {aeration['slot_area']:.1f} m²")
        
        print(f"\n⚠️  CAVITATION RISK ASSESSMENT:")
        print(f"   Cavitation risk level: {result.cavitation_risk.value.upper()}")
        print(f"   Cavitation index: {result.cavitation_index:.3f}")
        
        if result.cavitation_risk.value in ['moderate', 'high', 'critical']:
            print(f"   🚨 ATTENTION: Cavitation risk requires design consideration")
        else:
            print(f"   ✅ Acceptable cavitation risk")
        
        print(f"\n🏗️  DOWNSTREAM CONDITIONS:")
        print(f"   Energy dissipation length: {result.energy_dissipation_length:.1f} m")
        print(f"   Scour potential: {result.scour_potential}")
        
        if result.nappe_trajectory:
            print(f"   Nappe trajectory calculated: {len(result.nappe_trajectory.get('trajectory_points', []))} points")
        
        # Spillway profile
        if result.spillway_profile:
            print(f"\n📐 SPILLWAY PROFILE:")
            print(f"   Standard profile points: {len(result.spillway_profile)}")
            print(f"   Profile type: WES Standard")
            
            # Show first few profile points
            print(f"   Sample coordinates (x, y):")
            for i, (x, y) in enumerate(result.spillway_profile[:5]):
                x_scaled = x * result.head_over_weir  # Scale to actual head
                y_scaled = y * result.head_over_weir
                actual_elevation = crest_elevation + y_scaled
                print(f"     Point {i+1}: x={x_scaled:.2f}m, elevation={actual_elevation:.2f}m")
        
    else:
        print(f"❌ ANALYSIS FAILED: {result.message}")
        return None
    
    return result


def professional_design_recommendations(result):
    """Generate professional design recommendations."""
    
    print(f"\n💡 PROFESSIONAL DESIGN RECOMMENDATIONS:")
    print("=" * 80)
    
    if not result or not result.success:
        print("❌ Cannot provide recommendations - analysis failed")
        return
    
    # Get standard recommendations
    analyzer = poc.WeirFlowAnalyzer()
    recommendations = analyzer.recommend_weir_design(result)
    
    print("🔧 STANDARD RECOMMENDATIONS:")
    for i, rec in enumerate(recommendations, 1):
        print(f"   {i}. {rec}")
    
    # Additional specific recommendations for this case
    print(f"\n🎯 SPECIFIC RECOMMENDATIONS FOR THIS SPILLWAY:")
    
    # Submergence effects
    if result.submergence_ratio > result.modular_limit:
        reduction = (1 - result.submergence_factor) * 100
        print(f"   🌊 Submergence reduces discharge capacity by {reduction:.1f}%")
        print(f"   📊 Consider spillway modifications if higher capacity needed")
        print(f"   ⚠️  Monitor tailwater levels during flood events")
    
    # Energy dissipation
    if result.energy_dissipated > 2.0:
        print(f"   ⚡ High energy dissipation ({result.energy_dissipated:.1f}m) detected")
        print(f"   🛡️  Provide robust downstream energy dissipation structure")
        print(f"   📏 Stilling basin length should be at least {result.energy_dissipation_length:.0f}m")
    
    # Aeration requirements
    if result.aeration_requirement.value in ['required', 'critical']:
        print(f"   🌬️  Aeration system is essential for this spillway")
        print(f"   🔧 Install aeration slots at regular intervals")
        print(f"   📊 Monitor air entrainment during operation")
    
    # Cavitation protection
    if result.cavitation_risk.value in ['moderate', 'high', 'critical']:
        print(f"   ⚠️  Cavitation protection measures recommended")
        print(f"   🛡️  Use cavitation-resistant concrete or coatings")
        print(f"   🔍 Regular inspection of spillway surface required")
    
    # Operational recommendations
    print(f"\n🎛️  OPERATIONAL RECOMMENDATIONS:")
    print(f"   📊 Design discharge capacity: {result.discharge:.0f} m³/s")
    print(f"   🌊 Optimal operating head: {result.head_over_weir:.1f}m")
    print(f"   ⚡ Power dissipation: {9810 * result.discharge * result.energy_dissipated / 1000:.0f} kW")
    
    # Safety factors
    print(f"\n🛡️  SAFETY CONSIDERATIONS:")
    print(f"   📏 Freeboard above maximum water level: minimum 1.0m recommended")
    print(f"   🌊 Emergency spillway capacity: consider 150% of design flood")
    print(f"   🔧 Regular maintenance of spillway surface critical")


def compare_free_vs_submerged_conditions(result):
    """Compare free flow vs submerged flow conditions."""
    
    print(f"\n📊 FREE FLOW vs SUBMERGED FLOW COMPARISON:")
    print("=" * 80)
    
    if not result or not result.success:
        return
    
    # Calculate what discharge would be under free flow conditions
    weir_solver = poc.WeirFlowSolver()
    
    # Create same weir but analyze without downstream depth (free flow)
    approach_channel = poc.RectangularChannel(width=34.0 * 1.5)
    
    ogee_weir = poc.WeirGeometry(
        weir_type=poc.WeirType.OGEE_SPILLWAY,
        weir_height=result.weir_height,
        crest_length=34.0,
        spillway_shape="WES"
    )
    
    free_flow_result = weir_solver.analyze_weir_flow(
        channel=approach_channel,
        weir=ogee_weir,
        approach_depth=result.approach_depth,
        discharge=result.discharge,
        downstream_depth=None  # No downstream influence
    )
    
    if free_flow_result.success:
        print(f"                        | Free Flow    | Submerged    | Difference")
        print(f"                        | Condition    | Condition    | (Impact)")
        print(f"------------------------|--------------|--------------|-------------")
        print(f"Discharge Coefficient   | {free_flow_result.discharge_coefficient:.3f}        | {result.discharge_coefficient:.3f}        | {((result.discharge_coefficient/free_flow_result.discharge_coefficient-1)*100):+.1f}%")
        print(f"Flow Condition          | {free_flow_result.weir_condition.value:12s} | {result.weir_condition.value:12s} | ---")
        print(f"Energy Efficiency       | {free_flow_result.energy_efficiency:.1%}        | {result.energy_efficiency:.1%}        | {((result.energy_efficiency/free_flow_result.energy_efficiency-1)*100):+.1f}%")
        print(f"Energy Dissipated (m)   | {free_flow_result.energy_dissipated:.2f}         | {result.energy_dissipated:.2f}         | {(result.energy_dissipated-free_flow_result.energy_dissipated):+.2f}m")
        
        # Capacity comparison
        capacity_reduction = (1 - result.submergence_factor) * 100
        print(f"\n📉 SUBMERGENCE EFFECTS:")
        print(f"   Submergence ratio: {result.submergence_ratio:.3f}")
        print(f"   Capacity reduction: {capacity_reduction:.1f}%")
        print(f"   Submergence factor: {result.submergence_factor:.3f}")
        
        if capacity_reduction > 10:
            print(f"   ⚠️  SIGNIFICANT submergence effects detected")
        elif capacity_reduction > 5:
            print(f"   📊 MODERATE submergence effects")
        else:
            print(f"   ✅ MINIMAL submergence effects")


def create_spillway_visualization(result, discharge, spillway_width, crest_elevation, 
                                upstream_apron_elevation, downstream_apron_elevation, 
                                tailwater_elevation):
    """Create comprehensive spillway visualization plots."""
    
    if not MATPLOTLIB_AVAILABLE:
        print("📊 Matplotlib not available - skipping plots")
        return
    
    if not result or not result.success:
        print("❌ Cannot create plots - analysis failed")
        return
    
    print(f"\n📊 CREATING SPILLWAY VISUALIZATION PLOTS...")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('Ogee Spillway Hydraulic Analysis\nPyOpenChannel Professional Analysis', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Spillway Profile and Water Surface
    ax1 = plt.subplot(2, 3, 1)
    create_spillway_profile_plot(ax1, result, crest_elevation, upstream_apron_elevation, 
                                downstream_apron_elevation, tailwater_elevation)
    
    # Plot 2: Energy Grade Line
    ax2 = plt.subplot(2, 3, 2)
    create_energy_grade_line_plot(ax2, result, crest_elevation, upstream_apron_elevation, 
                                 downstream_apron_elevation)
    
    # Plot 3: Discharge Characteristics
    ax3 = plt.subplot(2, 3, 3)
    create_discharge_characteristics_plot(ax3, result, discharge, spillway_width)
    
    # Plot 4: Flow Velocity Distribution
    ax4 = plt.subplot(2, 3, 4)
    create_velocity_distribution_plot(ax4, result)
    
    # Plot 5: Submergence Analysis
    ax5 = plt.subplot(2, 3, 5)
    create_submergence_analysis_plot(ax5, result)
    
    # Plot 6: Energy Dissipation
    ax6 = plt.subplot(2, 3, 6)
    create_energy_dissipation_plot(ax6, result)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    # Save the plot
    plt.savefig('spillway_analysis.png', dpi=300, bbox_inches='tight')
    print(f"   ✅ Spillway analysis plots saved as 'spillway_analysis.png'")
    
    # Show the plot
    plt.show()


def create_spillway_profile_plot(ax, result, crest_elev, upstream_elev, downstream_elev, tailwater_elev):
    """Create spillway profile and water surface plot."""
    
    # Generate spillway profile coordinates
    if result.spillway_profile:
        # Scale profile to actual dimensions
        x_profile = []
        y_profile = []
        
        for x_norm, y_norm in result.spillway_profile:
            x_actual = x_norm * result.head_over_weir
            y_actual = crest_elev + y_norm * result.head_over_weir
            x_profile.append(x_actual)
            y_profile.append(y_actual)
        
        # Extend profile upstream and downstream
        x_extended = [-5, 0] + x_profile + [x_profile[-1] + 5, x_profile[-1] + 10]
        y_extended = [crest_elev, crest_elev] + y_profile + [y_profile[-1], downstream_elev]
        
        ax.plot(x_extended, y_extended, 'k-', linewidth=2, label='Spillway Profile')
    
    # Water surface elevations
    upstream_water_level = upstream_elev + result.approach_depth
    downstream_water_level = tailwater_elev
    
    # Plot water surfaces
    ax.axhline(y=upstream_water_level, color='blue', linestyle='--', alpha=0.7, 
               label=f'Upstream Water Level ({upstream_water_level:.1f}m)')
    ax.axhline(y=downstream_water_level, color='cyan', linestyle='--', alpha=0.7,
               label=f'Tailwater Level ({downstream_water_level:.1f}m)')
    ax.axhline(y=crest_elev, color='red', linestyle='-', alpha=0.5,
               label=f'Crest Elevation ({crest_elev:.1f}m)')
    
    # Add approach and tailwater areas
    x_range = ax.get_xlim()
    ax.fill_between(x_range, upstream_elev, upstream_water_level, 
                   alpha=0.3, color='blue', label='Approach Flow')
    ax.fill_between(x_range, downstream_elev, downstream_water_level,
                   alpha=0.3, color='cyan', label='Tailwater')
    
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Elevation (m)')
    ax.set_title('Spillway Profile & Water Surface')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def create_energy_grade_line_plot(ax, result, crest_elev, upstream_elev, downstream_elev):
    """Create energy grade line plot."""
    
    # Calculate energy elevations
    upstream_energy_elev = upstream_elev + result.approach_energy
    crest_energy_elev = crest_elev + result.crest_energy - crest_elev + upstream_elev
    
    if result.downstream_energy:
        downstream_energy_elev = downstream_elev + result.downstream_energy
    else:
        downstream_energy_elev = downstream_elev + result.crest_energy
    
    # Energy grade line
    x_energy = [-10, -2, 0, 5, 15]
    y_energy = [upstream_energy_elev, upstream_energy_elev, crest_energy_elev, 
                downstream_energy_elev, downstream_energy_elev]
    
    ax.plot(x_energy, y_energy, 'r-', linewidth=2, label='Energy Grade Line')
    
    # Water surface
    upstream_water = upstream_elev + result.approach_depth
    downstream_water = downstream_elev + (result.downstream_depth if result.downstream_depth else 2.0)
    
    x_water = [-10, -2, 0, 5, 15]
    y_water = [upstream_water, upstream_water, crest_elev + result.head_over_weir,
               downstream_water, downstream_water]
    
    ax.plot(x_water, y_water, 'b-', linewidth=2, label='Water Surface')
    
    # Fill energy loss area
    ax.fill_between([0, 15], [crest_energy_elev, downstream_energy_elev], 
                   [downstream_water, downstream_water], 
                   alpha=0.3, color='red', label=f'Energy Loss ({result.energy_dissipated:.1f}m)')
    
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Elevation (m)')
    ax.set_title('Energy Grade Line')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def create_discharge_characteristics_plot(ax, result, discharge, spillway_width):
    """Create discharge characteristics plot."""
    
    # Create head vs discharge curve
    heads = np.linspace(0.5, result.head_over_weir * 1.5, 50)
    discharges = []
    
    for h in heads:
        # Simplified ogee spillway equation: Q = Cd * L * H^1.5 * sqrt(2g)
        Q = result.discharge_coefficient * spillway_width * h**1.5 * math.sqrt(2 * 9.81)
        discharges.append(Q)
    
    ax.plot(heads, discharges, 'b-', linewidth=2, label='Rating Curve')
    
    # Mark current operating point
    ax.plot(result.head_over_weir, discharge, 'ro', markersize=8, 
            label=f'Operating Point\n(H={result.head_over_weir:.1f}m, Q={discharge:.0f}m³/s)')
    
    # Add design discharge line
    ax.axhline(y=discharge, color='red', linestyle='--', alpha=0.5, 
               label=f'Design Discharge ({discharge:.0f}m³/s)')
    
    ax.set_xlabel('Head over Weir (m)')
    ax.set_ylabel('Discharge (m³/s)')
    ax.set_title('Spillway Rating Curve')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def create_velocity_distribution_plot(ax, result):
    """Create velocity distribution plot."""
    
    # Theoretical velocity distribution across spillway
    positions = np.linspace(0, 1, 20)  # Normalized positions across spillway
    
    # Assume parabolic velocity distribution (typical for spillways)
    velocities = []
    max_velocity = result.approach_velocity * 1.5  # Peak velocity higher than average
    
    for pos in positions:
        # Parabolic distribution: v = v_max * (1 - 4*(pos-0.5)^2)
        v = max_velocity * (1 - 4 * (pos - 0.5)**2)
        velocities.append(v)
    
    ax.plot(positions, velocities, 'g-', linewidth=2, label='Velocity Distribution')
    ax.axhline(y=result.approach_velocity, color='blue', linestyle='--', 
               label=f'Average Velocity ({result.approach_velocity:.2f}m/s)')
    
    # Fill velocity area
    ax.fill_between(positions, 0, velocities, alpha=0.3, color='green')
    
    ax.set_xlabel('Normalized Position Across Spillway')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title('Velocity Distribution')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def create_submergence_analysis_plot(ax, result):
    """Create submergence analysis plot."""
    
    # Submergence ratio vs discharge reduction
    submergence_ratios = np.linspace(0, 1.2, 50)
    discharge_factors = []
    
    for sr in submergence_ratios:
        if sr < result.modular_limit:
            factor = 1.0  # No reduction
        else:
            # Villemonte equation
            factor = (1 - (sr / result.modular_limit)**1.5)**0.385
        discharge_factors.append(factor)
    
    ax.plot(submergence_ratios, discharge_factors, 'purple', linewidth=2, 
            label='Discharge Reduction Factor')
    
    # Mark current condition
    ax.plot(result.submergence_ratio, result.submergence_factor, 'ro', markersize=8,
            label=f'Current Condition\n(S={result.submergence_ratio:.2f}, F={result.submergence_factor:.2f})')
    
    # Mark modular limit
    ax.axvline(x=result.modular_limit, color='red', linestyle='--', alpha=0.7,
               label=f'Modular Limit ({result.modular_limit:.2f})')
    
    # Shade free flow region
    ax.axvspan(0, result.modular_limit, alpha=0.2, color='green', label='Free Flow Region')
    ax.axvspan(result.modular_limit, 1.2, alpha=0.2, color='red', label='Submerged Flow Region')
    
    ax.set_xlabel('Submergence Ratio')
    ax.set_ylabel('Discharge Factor')
    ax.set_title('Submergence Effects')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)


def create_energy_dissipation_plot(ax, result):
    """Create energy dissipation analysis plot."""
    
    # Energy components
    components = ['Approach\nEnergy', 'Crest\nEnergy', 'Downstream\nEnergy', 'Energy\nDissipated']
    values = [result.approach_energy, result.crest_energy, 
              result.downstream_energy or 0, result.energy_dissipated]
    colors = ['blue', 'green', 'cyan', 'red']
    
    bars = ax.bar(components, values, color=colors, alpha=0.7)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{value:.2f}m', ha='center', va='bottom', fontweight='bold')
    
    # Add efficiency text
    ax.text(0.5, 0.95, f'Energy Efficiency: {result.energy_efficiency:.1%}', 
            transform=ax.transAxes, ha='center', va='top', 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=10, fontweight='bold')
    
    ax.set_ylabel('Energy (m)')
    ax.set_title('Energy Analysis')
    ax.grid(True, alpha=0.3, axis='y')


def create_detailed_spillway_plot(result, discharge, spillway_width, crest_elevation,
                                upstream_apron_elevation, downstream_apron_elevation, 
                                tailwater_elevation):
    """Create a detailed spillway cross-section plot."""
    
    if not MATPLOTLIB_AVAILABLE:
        return
    
    print(f"\n📊 CREATING DETAILED SPILLWAY CROSS-SECTION...")
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Extended spillway profile
    x_upstream = np.linspace(-20, 0, 50)
    y_upstream = np.full_like(x_upstream, crest_elevation)
    
    # Spillway profile (scaled)
    if result.spillway_profile:
        x_spillway = []
        y_spillway = []
        
        for x_norm, y_norm in result.spillway_profile:
            x_actual = x_norm * result.head_over_weir
            y_actual = crest_elevation + y_norm * result.head_over_weir
            x_spillway.append(x_actual)
            y_spillway.append(y_actual)
        
        # Extend downstream
        x_downstream = np.linspace(x_spillway[-1], x_spillway[-1] + 30, 20)
        y_downstream = np.linspace(y_spillway[-1], downstream_apron_elevation, 20)
        
        # Complete profile
        x_complete = list(x_upstream) + [0] + x_spillway + list(x_downstream)
        y_complete = list(y_upstream) + [crest_elevation] + y_spillway + list(y_downstream)
        
        ax.plot(x_complete, y_complete, 'k-', linewidth=3, label='Spillway Structure')
    
    # Water surface profile
    upstream_water_level = upstream_apron_elevation + result.approach_depth
    downstream_water_level = tailwater_elevation
    
    # Approach water surface
    x_approach = np.linspace(-20, 0, 20)
    y_approach = np.full_like(x_approach, upstream_water_level)
    
    # Spillway water surface (simplified)
    x_spillway_water = np.linspace(0, 10, 20)
    y_spillway_water = np.linspace(upstream_water_level, downstream_water_level, 20)
    
    # Tailwater
    x_tailwater = np.linspace(10, 40, 20)
    y_tailwater = np.full_like(x_tailwater, downstream_water_level)
    
    # Plot water surface
    x_water_complete = list(x_approach) + list(x_spillway_water) + list(x_tailwater)
    y_water_complete = list(y_approach) + list(y_spillway_water) + list(y_tailwater)
    
    ax.plot(x_water_complete, y_water_complete, 'b-', linewidth=2, label='Water Surface')
    
    # Fill water areas
    ax.fill_between(x_approach, upstream_apron_elevation, y_approach, 
                   alpha=0.4, color='blue', label='Reservoir')
    ax.fill_between(x_tailwater, downstream_apron_elevation, y_tailwater,
                   alpha=0.4, color='cyan', label='Tailwater')
    
    # Add dimensions and annotations
    ax.annotate(f'H = {result.head_over_weir:.1f}m', 
                xy=(0, crest_elevation), xytext=(-5, upstream_water_level),
                arrowprops=dict(arrowstyle='<->', color='red', lw=2),
                fontsize=12, fontweight='bold', color='red')
    
    ax.annotate(f'Q = {discharge:.0f} m³/s', 
                xy=(5, (upstream_water_level + downstream_water_level)/2),
                fontsize=12, fontweight='bold', color='blue',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    ax.annotate(f'Cd = {result.discharge_coefficient:.3f}', 
                xy=(0, crest_elevation - 1), 
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    # Energy dissipation annotation
    if result.energy_dissipated > 0.5:
        ax.annotate(f'Energy Loss = {result.energy_dissipated:.1f}m\n({(1-result.energy_efficiency)*100:.0f}%)', 
                    xy=(15, downstream_water_level + 1),
                    fontsize=10, fontweight='bold', color='red',
                    bbox=dict(boxstyle='round', facecolor='pink', alpha=0.8))
    
    # Set labels and title
    ax.set_xlabel('Distance (m)', fontsize=12)
    ax.set_ylabel('Elevation (m)', fontsize=12)
    ax.set_title('Ogee Spillway - Detailed Cross Section\nPyOpenChannel Professional Analysis', 
                 fontsize=14, fontweight='bold')
    
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Save detailed plot
    plt.tight_layout()
    plt.savefig('spillway_cross_section.png', dpi=300, bbox_inches='tight')
    print(f"   ✅ Detailed cross-section saved as 'spillway_cross_section.png'")
    plt.show()


def main():
    """Run the complete submerged ogee weir analysis."""
    
    print("🌊 SUBMERGED CREST OGEE WEIR - COMPREHENSIVE ANALYSIS")
    print("=" * 90)
    print("Professional spillway analysis for dam engineering")
    print("Author: Alexius Academia")
    print("=" * 90)
    
    try:
        # Main analysis
        result = analyze_submerged_ogee_weir()
        
        if result and result.success:
            # Professional recommendations
            professional_design_recommendations(result)
            
            # Comparison analysis
            compare_free_vs_submerged_conditions(result)
            
            # Create visualization plots
            create_spillway_visualization(
                result, 243.0, 34.0, 37.8, 35.8, 35.2, 39.08
            )
            
            # Create detailed spillway cross-section
            create_detailed_spillway_plot(
                result, 243.0, 34.0, 37.8, 35.8, 35.2, 39.08
            )
            
            print("\n" + "=" * 90)
            print("🎉 SUBMERGED OGEE WEIR ANALYSIS COMPLETED SUCCESSFULLY!")
            print("=" * 90)
            
            print(f"\n🎯 KEY FINDINGS:")
            print(f"   • Spillway operates under {'SUBMERGED' if result.submergence_ratio > result.modular_limit else 'FREE FLOW'} conditions")
            print(f"   • Discharge coefficient: {result.discharge_coefficient:.3f}")
            print(f"   • Energy dissipation: {result.energy_dissipated:.1f}m ({(1-result.energy_efficiency)*100:.0f}%)")
            print(f"   • Aeration requirement: {result.aeration_requirement.value.upper()}")
            print(f"   • Cavitation risk: {result.cavitation_risk.value.upper()}")
            
            if result.submergence_ratio > result.modular_limit:
                capacity_reduction = (1 - result.submergence_factor) * 100
                print(f"   • Submergence reduces capacity by {capacity_reduction:.1f}%")
            
        else:
            print("\n❌ Analysis could not be completed successfully")
            
    except Exception as e:
        print(f"\n❌ Analysis failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
