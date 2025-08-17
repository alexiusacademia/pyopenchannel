#!/usr/bin/env python3
"""
Complete Diversion Dam Flow Profile - PyOpenChannel

Author: Alexius Academia
Email: alexius.sayco.academia@gmail.com

This example demonstrates the complete flow profile for a diversion dam,
combining both GVF (Gradually Varied Flow) and RVF (Rapidly Varied Flow):

1. GVF: Backwater curve upstream of the weir (300m upstream)
2. GVF: Flow over the ogee spillway/weir
3. RVF: Hydraulic jump downstream of the weir
4. GVF: Flow to tailwater conditions

This represents a typical diversion dam scenario where:
- Upstream: M1 backwater curve due to weir obstruction
- Over weir: Controlled critical/supercritical flow
- Downstream: Hydraulic jump for energy dissipation
- Tailwater: Return to normal subcritical flow

GIVEN CONDITIONS (Updated from your spillway):
- Discharge: 243 m³/s
- Spillway width: 34 m
- Crest elevation: 37.8 m
- Upstream channel: mild slope, normal depth < critical depth
- Downstream: hydraulic jump to dissipate energy
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pyopenchannel as poc
import math
import numpy as np

# Optional matplotlib import for plotting
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
    print("📊 Matplotlib available - complete profile will be generated")
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️  Matplotlib not available - analysis will run without plots")


def analyze_complete_diversion_dam_profile():
    """Analyze the complete flow profile for a diversion dam."""
    
    print("🏗️  COMPLETE DIVERSION DAM FLOW PROFILE ANALYSIS")
    print("=" * 80)
    
    # Given conditions (from your spillway example)
    discharge = 243.0  # m³/s
    spillway_width = 34.0  # m
    crest_elevation = 37.8  # m
    upstream_apron_elevation = 35.8  # m
    downstream_apron_elevation = 35.2  # m
    tailwater_elevation = 39.08  # m
    
    # Channel properties
    channel_width = spillway_width * 1.2  # Approach channel slightly wider
    channel_slope = 0.001  # Mild slope (0.1%)
    manning_n = 0.030  # Concrete-lined channel
    
    print(f"📊 DIVERSION DAM CONDITIONS:")
    print(f"   Discharge: {discharge} m³/s")
    print(f"   Spillway width: {spillway_width} m")
    print(f"   Channel width: {channel_width} m")
    print(f"   Crest elevation: {crest_elevation} m")
    print(f"   Channel slope: {channel_slope:.3f}")
    print(f"   Manning's n: {manning_n}")
    print(f"   Tailwater elevation: {tailwater_elevation} m")
    
    poc.set_unit_system(poc.UnitSystem.SI)
    
    # Create channel geometry
    channel = poc.RectangularChannel(width=channel_width)
    
    # Calculate normal depth in approach channel
    normal_depth = poc.NormalDepth.calculate(
        channel=channel,
        discharge=discharge,
        slope=channel_slope,
        manning_n=manning_n
    )
    
    # Calculate critical depth
    critical_flow = poc.CriticalFlow(channel)
    critical_depth = critical_flow.calculate_critical_depth(discharge)
    
    print(f"\n🌊 CHANNEL FLOW CHARACTERISTICS:")
    print(f"   Normal depth: {normal_depth:.2f} m")
    print(f"   Critical depth: {critical_depth:.2f} m")
    print(f"   Channel type: {'Mild slope (yn > yc)' if normal_depth > critical_depth else 'Steep slope (yn < yc)'}")
    
    # Upstream water level (affected by weir)
    upstream_water_level = upstream_apron_elevation + 4.60  # From your spillway analysis
    upstream_depth = upstream_water_level - upstream_apron_elevation
    
    print(f"   Upstream depth (at weir): {upstream_depth:.2f} m")
    print(f"   Upstream water level: {upstream_water_level:.2f} m")
    
    # === PART 1: GVF UPSTREAM BACKWATER ANALYSIS ===
    print(f"\n🔄 PART 1: UPSTREAM BACKWATER CURVE (GVF)")
    print("-" * 60)
    
    # GVF analysis for backwater curve
    gvf_solver = poc.GVFSolver()
    
    # Solve backwater profile from weir upstream
    upstream_profile = gvf_solver.solve_profile(
        channel=channel,
        discharge=discharge,
        slope=channel_slope,
        manning_n=manning_n,
        boundary_depth=upstream_depth,
        boundary_type=poc.BoundaryType.DOWNSTREAM_DEPTH,  # Known depth at weir
        x_start=0,  # At weir
        x_end=-100  # 300m upstream
    )
    
    if upstream_profile.success:
        print(f"   ✅ Backwater analysis: SUCCESS")
        print(f"   Profile length: {abs(upstream_profile.length):.1f} m")
        print(f"   Number of points: {len(upstream_profile.profile_points)}")
        print(f"   Depth at 300m upstream: {upstream_profile.profile_points[-1].depth:.2f} m")
        print(f"   Water level at 300m upstream: {upstream_apron_elevation + upstream_profile.profile_points[-1].depth:.2f} m")
        
        # Classify the profile
        classifier = poc.ProfileClassifier()
        profile_analysis = classifier.classify_profile(
            gvf_result=upstream_profile,
            channel=channel,
            discharge=discharge,
            slope=channel_slope,
            manning_n=manning_n
        )
        print(f"   Profile type: {profile_analysis.profile_type.value}")
        print(f"   Slope type: {profile_analysis.slope_type.value}")
        print(f"   Flow regime: {profile_analysis.flow_regime.value}")
        print(f"   Curvature: {profile_analysis.curvature}")
        print(f"   Engineering significance: {profile_analysis.engineering_significance}")
    else:
        print(f"   ❌ Backwater analysis failed: {upstream_profile.message}")
        return None
    
    # === PART 2: WEIR FLOW ANALYSIS ===
    print(f"\n🌊 PART 2: WEIR FLOW ANALYSIS")
    print("-" * 60)
    
    # Create weir geometry
    weir_height = crest_elevation - upstream_apron_elevation
    ogee_weir = poc.WeirGeometry(
        weir_type=poc.WeirType.OGEE_SPILLWAY,
        weir_height=weir_height,
        crest_length=spillway_width,
        weir_width=spillway_width,
        spillway_shape="WES",
        upstream_slope=0,
        downstream_slope=0.75
    )
    
    # Analyze weir flow
    weir_solver = poc.WeirFlowSolver()
    weir_result = weir_solver.analyze_weir_flow(
        channel=channel,
        weir=ogee_weir,
        approach_depth=upstream_depth,
        discharge=discharge,
        downstream_depth=tailwater_elevation - downstream_apron_elevation
    )
    
    if weir_result.success:
        print(f"   ✅ Weir analysis: SUCCESS")
        print(f"   Head over weir: {weir_result.head_over_weir:.2f} m")
        print(f"   Discharge coefficient: {weir_result.discharge_coefficient:.3f}")
        print(f"   Flow condition: {weir_result.weir_condition.value}")
        print(f"   Energy dissipated: {weir_result.energy_dissipated:.2f} m")
        
        # Calculate depth at toe of spillway (supercritical)
        spillway_toe_depth = critical_depth * 0.7  # Typical for ogee spillway
        spillway_toe_velocity = discharge / (spillway_width * spillway_toe_depth)
        spillway_toe_froude = spillway_toe_velocity / math.sqrt(9.81 * spillway_toe_depth)
        
        print(f"   Depth at spillway toe: {spillway_toe_depth:.2f} m (supercritical)")
        print(f"   Velocity at toe: {spillway_toe_velocity:.2f} m/s")
        print(f"   Froude at toe: {spillway_toe_froude:.2f}")
    else:
        print(f"   ❌ Weir analysis failed: {weir_result.message}")
        return None
    
    # === PART 3: HYDRAULIC JUMP ANALYSIS ===
    print(f"\n⚡ PART 3: HYDRAULIC JUMP ANALYSIS (RVF)")
    print("-" * 60)
    
    # RVF analysis for hydraulic jump
    rvf_solver = poc.RVFSolver()
    
    # Downstream depth (tailwater controlled)
    downstream_depth = tailwater_elevation - downstream_apron_elevation
    
    jump_result = rvf_solver.analyze_hydraulic_jump(
        channel=poc.RectangularChannel(width=spillway_width),  # Jump occurs in spillway width
        discharge=discharge,
        upstream_depth=spillway_toe_depth,
        tailwater_depth=downstream_depth
    )
    
    if jump_result.success:
        print(f"   ✅ Hydraulic jump analysis: SUCCESS")
        print(f"   Jump type: {jump_result.jump_type.value.upper()}")
        print(f"   Upstream depth (y1): {jump_result.upstream_depth:.2f} m")
        print(f"   Downstream depth (y2): {jump_result.downstream_depth:.2f} m")
        print(f"   Sequent depth ratio: {jump_result.sequent_depth_ratio:.2f}")
        print(f"   Energy dissipated: {jump_result.energy_loss:.2f} m")
        print(f"   Jump length: {jump_result.jump_length:.1f} m" if jump_result.jump_length else "   Jump length: N/A")
        print(f"   Jump efficiency: {jump_result.energy_efficiency:.1%}")
    else:
        print(f"   ❌ Hydraulic jump analysis failed: {jump_result.message}")
        return None
    
    # === PART 4: DOWNSTREAM FLOW TO TAILWATER ===
    print(f"\n🔄 PART 4: DOWNSTREAM FLOW TO TAILWATER")
    print("-" * 60)
    
    # Calculate flow from jump end to tailwater
    jump_end_depth = jump_result.downstream_depth
    jump_length = jump_result.jump_length if jump_result.jump_length else 10.0
    
    # Expand back to full channel width gradually
    transition_length = 50.0  # Length for width transition
    
    print(f"   Jump end depth: {jump_end_depth:.2f} m")
    print(f"   Jump length: {jump_length:.1f} m")
    print(f"   Transition length: {transition_length:.1f} m")
    print(f"   Final tailwater depth: {downstream_depth:.2f} m")
    
    # Return all results for plotting
    return {
        'upstream_profile': upstream_profile,
        'weir_result': weir_result,
        'jump_result': jump_result,
        'parameters': {
            'discharge': discharge,
            'spillway_width': spillway_width,
            'channel_width': channel_width,
            'crest_elevation': crest_elevation,
            'upstream_apron_elevation': upstream_apron_elevation,
            'downstream_apron_elevation': downstream_apron_elevation,
            'tailwater_elevation': tailwater_elevation,
            'normal_depth': normal_depth,
            'critical_depth': critical_depth,
            'spillway_toe_depth': spillway_toe_depth,
            'jump_length': jump_length,
            'transition_length': transition_length
        }
    }


def create_complete_profile_visualization(results):
    """Create comprehensive visualization of the complete diversion dam profile."""
    
    if not MATPLOTLIB_AVAILABLE or not results:
        print("📊 Cannot create visualization")
        return
    
    print(f"\n📊 CREATING COMPLETE DIVERSION DAM PROFILE...")
    
    # Extract results
    upstream_profile = results['upstream_profile']
    weir_result = results['weir_result']
    jump_result = results['jump_result']
    params = results['parameters']
    
    # Create main profile plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12))
    fig.suptitle('Complete Diversion Dam Flow Profile\nGVF + RVF Analysis - PyOpenChannel', 
                 fontsize=16, fontweight='bold')
    
    # === MAIN PROFILE PLOT ===
    create_main_profile_plot(ax1, results)
    
    # === DETAILED HYDRAULIC JUMP PLOT ===
    create_detailed_jump_plot(ax2, results)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    # Save the plot
    plt.savefig('diversion_dam_complete_profile.png', dpi=300, bbox_inches='tight')
    print(f"   ✅ Complete profile saved as 'diversion_dam_complete_profile.png'")
    
    # Create additional detailed plots
    create_energy_analysis_plots(results)
    
    plt.show()


def create_main_profile_plot(ax, results):
    """Create the main longitudinal profile plot."""
    
    upstream_profile = results['upstream_profile']
    weir_result = results['weir_result']
    jump_result = results['jump_result']
    params = results['parameters']
    
    # === UPSTREAM BACKWATER CURVE ===
    if upstream_profile.success:
        # Extract upstream profile data
        x_upstream = [-p.x for p in upstream_profile.profile_points]  # Make positive going upstream
        y_upstream = [params['upstream_apron_elevation'] + p.depth for p in upstream_profile.profile_points]
        
        ax.plot(x_upstream, y_upstream, 'b-', linewidth=3, label='Upstream Backwater (M1 Profile)')
        ax.fill_between(x_upstream, params['upstream_apron_elevation'], y_upstream, 
                       alpha=0.3, color='blue', label='Upstream Water')
    
    # === CHANNEL BOTTOM PROFILE ===
    x_total = np.linspace(-100, 200, 1000)
    
    # Channel bottom elevations
    channel_bottom = np.where(
        x_total < 0, 
        params['upstream_apron_elevation'],  # Upstream channel
        np.where(
            x_total < 50,
            params['downstream_apron_elevation'],  # Downstream channel
            params['downstream_apron_elevation']   # Far downstream
        )
    )
    
    ax.plot(x_total, channel_bottom, 'brown', linewidth=3, label='Channel Bottom')
    
    # === WEIR STRUCTURE ===
    # Weir crest
    weir_x = np.linspace(-2, 10, 50)
    if weir_result.spillway_profile:
        # Scale spillway profile
        weir_y = []
        for x_norm, y_norm in weir_result.spillway_profile:
            x_actual = x_norm * weir_result.head_over_weir
            y_actual = params['crest_elevation'] + y_norm * weir_result.head_over_weir
            if 0 <= x_actual <= 10:  # Only plot reasonable extent
                weir_y.append(y_actual)
            else:
                weir_y.append(params['crest_elevation'])
        
        weir_x_profile = np.linspace(0, 10, len(weir_y))
        ax.plot(weir_x_profile, weir_y, 'k-', linewidth=4, label='Ogee Spillway')
    else:
        # Simple weir representation
        ax.axhline(y=params['crest_elevation'], xmin=0.4, xmax=0.42, 
                  color='black', linewidth=6, label='Weir Crest')
    
    # === WATER SURFACE OVER WEIR ===
    weir_water_x = np.linspace(0, 15, 30)
    weir_water_y = []
    
    for x in weir_water_x:
        if x <= 2:
            # Approach to crest
            y = params['upstream_apron_elevation'] + upstream_profile.profile_points[0].depth
        elif x <= 10:
            # Over spillway - transition to critical/supercritical
            progress = (x - 2) / 8
            start_depth = upstream_profile.profile_points[0].depth
            end_depth = params['spillway_toe_depth']
            depth = start_depth + (end_depth - start_depth) * progress
            y = params['downstream_apron_elevation'] + depth
        else:
            # Supercritical flow before jump
            y = params['downstream_apron_elevation'] + params['spillway_toe_depth']
        
        weir_water_y.append(y)
    
    ax.plot(weir_water_x, weir_water_y, 'g-', linewidth=3, label='Water Surface Over Weir')
    
    # === HYDRAULIC JUMP ===
    if jump_result.success and jump_result.jump_length:
        jump_start = 15
        jump_end = jump_start + jump_result.jump_length
        
        # Jump profile (simplified)
        x_jump = np.linspace(jump_start, jump_end, 20)
        y_jump = []
        
        for x in x_jump:
            progress = (x - jump_start) / jump_result.jump_length
            # Smooth transition using tanh
            transition = 0.5 * (1 + np.tanh(4 * (progress - 0.5)))
            depth = params['spillway_toe_depth'] + (jump_result.downstream_depth - params['spillway_toe_depth']) * transition
            y_jump.append(params['downstream_apron_elevation'] + depth)
        
        ax.plot(x_jump, y_jump, 'r-', linewidth=4, label=f'Hydraulic Jump ({jump_result.jump_type.value.title()})')
        
        # Mark jump boundaries
        ax.axvline(x=jump_start, color='red', linestyle='--', alpha=0.7, label='Jump Start')
        ax.axvline(x=jump_end, color='red', linestyle='--', alpha=0.7, label='Jump End')
        
        # Jump roller (simplified)
        roller_x = np.linspace(jump_start, jump_end, 10)
        roller_y_top = [params['downstream_apron_elevation'] + jump_result.downstream_depth * 1.2] * len(roller_x)
        roller_y_bottom = [params['downstream_apron_elevation'] + params['spillway_toe_depth']] * len(roller_x)
        ax.fill_between(roller_x, roller_y_bottom, roller_y_top, 
                       alpha=0.2, color='red', label='Turbulent Roller')
    
    # === DOWNSTREAM FLOW TO TAILWATER ===
    tailwater_start = jump_start + jump_result.jump_length if jump_result.jump_length else 25
    x_downstream = np.linspace(tailwater_start, 200, 50)
    
    # Gradual transition to tailwater depth
    y_downstream = []
    for x in x_downstream:
        progress = min(1.0, (x - tailwater_start) / params['transition_length'])
        start_depth = jump_result.downstream_depth
        end_depth = params['tailwater_elevation'] - params['downstream_apron_elevation']
        depth = start_depth + (end_depth - start_depth) * progress
        y_downstream.append(params['downstream_apron_elevation'] + depth)
    
    ax.plot(x_downstream, y_downstream, 'c-', linewidth=3, label='Flow to Tailwater')
    
    # === TAILWATER LEVEL ===
    ax.axhline(y=params['tailwater_elevation'], color='cyan', linestyle='-', alpha=0.8,
               linewidth=2, label=f'Tailwater Level ({params["tailwater_elevation"]:.1f}m)')
    
    # === REFERENCE LINES ===
    # Normal depth line
    normal_water_level = params['upstream_apron_elevation'] + params['normal_depth']
    ax.axhline(y=normal_water_level, color='green', linestyle=':', alpha=0.7,
               label=f'Normal Depth ({params["normal_depth"]:.1f}m)')
    
    # Critical depth line
    critical_water_level = params['upstream_apron_elevation'] + params['critical_depth']
    ax.axhline(y=critical_water_level, color='orange', linestyle=':', alpha=0.7,
               label=f'Critical Depth ({params["critical_depth"]:.1f}m)')
    
    # === ANNOTATIONS ===
    # Flow direction arrow
    ax.annotate('', xy=(50, params['crest_elevation'] + 2), xytext=(-50, params['crest_elevation'] + 2),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.text(0, params['crest_elevation'] + 2.5, 'FLOW DIRECTION', ha='center', fontweight='bold')
    
    # Discharge annotation
    ax.text(-150, params['tailwater_elevation'] + 1, 
            f'Q = {params["discharge"]:.0f} m³/s\nWidth = {params["spillway_width"]:.0f}m',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
            fontsize=12, fontweight='bold')
    
    # Zone labels
    ax.text(-200, params['crest_elevation'] + 3, 'BACKWATER\nZONE\n(GVF)', 
            ha='center', va='center', fontweight='bold', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    ax.text(7, params['crest_elevation'] + 2, 'SPILLWAY\nZONE\n(GVF)', 
            ha='center', va='center', fontweight='bold', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    if jump_result.jump_length:
        jump_center = 15 + jump_result.jump_length/2
        ax.text(jump_center, params['crest_elevation'] + 2, 'HYDRAULIC\nJUMP\n(RVF)', 
                ha='center', va='center', fontweight='bold', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='pink', alpha=0.8))
    
    ax.text(100, params['crest_elevation'] + 2, 'TAILWATER\nZONE\n(GVF)', 
            ha='center', va='center', fontweight='bold', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
    
    # Formatting
    ax.set_xlabel('Distance from Weir (m)', fontsize=12)
    ax.set_ylabel('Elevation (m)', fontsize=12)
    ax.set_title('Complete Longitudinal Profile - 300m Upstream to Tailwater', fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-100, 200)


def create_detailed_jump_plot(ax, results):
    """Create detailed hydraulic jump plot."""
    
    jump_result = results['jump_result']
    params = results['parameters']
    
    if not jump_result.success:
        ax.text(0.5, 0.5, 'Hydraulic Jump Analysis\nNot Available', 
                transform=ax.transAxes, ha='center', va='center', 
                fontsize=14, bbox=dict(boxstyle='round', facecolor='lightgray'))
        ax.set_title('Hydraulic Jump Detail')
        return
    
    # Jump length
    jump_length = jump_result.jump_length if jump_result.jump_length else 10.0
    
    # Create detailed jump profile
    x = np.linspace(-5, jump_length + 10, 200)
    y = np.zeros_like(x)
    
    # Upstream supercritical flow
    upstream_region = x < 0
    y[upstream_region] = params['spillway_toe_depth']
    
    # Jump region
    jump_region = (x >= 0) & (x <= jump_length)
    x_jump = x[jump_region]
    transition = 0.5 * (1 + np.tanh(4 * (x_jump / jump_length - 0.5)))
    y[jump_region] = params['spillway_toe_depth'] + (jump_result.downstream_depth - params['spillway_toe_depth']) * transition
    
    # Downstream subcritical flow
    downstream_region = x > jump_length
    y[downstream_region] = jump_result.downstream_depth
    
    # Plot water surface
    ax.plot(x, y, 'b-', linewidth=3, label='Water Surface')
    ax.fill_between(x, 0, y, alpha=0.3, color='blue')
    
    # Channel bottom
    ax.axhline(y=0, color='brown', linewidth=3, label='Channel Bottom')
    
    # Jump boundaries
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Jump Start')
    ax.axvline(x=jump_length, color='red', linestyle='--', alpha=0.7, label='Jump End')
    
    # Energy grade line
    E1 = jump_result.upstream_energy
    E2 = jump_result.downstream_energy
    energy_line = np.where(x < jump_length/2, E1, E2)
    ax.plot(x, energy_line, 'r-', linewidth=2, label='Energy Grade Line')
    
    # Annotations
    ax.annotate(f'y₁ = {params["spillway_toe_depth"]:.2f}m\nFr₁ = {jump_result.upstream_froude:.2f}\nSupercritical', 
                xy=(-3, params['spillway_toe_depth']/2), fontsize=10, ha='center',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    ax.annotate(f'y₂ = {jump_result.downstream_depth:.2f}m\nFr₂ = {jump_result.downstream_froude:.2f}\nSubcritical', 
                xy=(jump_length + 3, jump_result.downstream_depth/2), fontsize=10, ha='center',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    ax.annotate(f'Jump Length = {jump_length:.1f}m\nEnergy Loss = {jump_result.energy_loss:.2f}m\nType: {jump_result.jump_type.value.title()}', 
                xy=(jump_length/2, max(params['spillway_toe_depth'], jump_result.downstream_depth) + 0.5), 
                fontsize=10, ha='center',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Depth (m)')
    ax.set_title('Hydraulic Jump Detail - RVF Analysis')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(params['spillway_toe_depth'], jump_result.downstream_depth) * 1.5)


def create_energy_analysis_plots(results):
    """Create additional energy analysis plots."""
    
    if not MATPLOTLIB_AVAILABLE:
        return
    
    print(f"   📊 Creating energy analysis plots...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Diversion Dam - Energy Analysis\nPyOpenChannel Professional Analysis', 
                 fontsize=16, fontweight='bold')
    
    # Energy grade line along entire profile
    create_complete_energy_plot(ax1, results)
    
    # Froude number variation
    create_froude_variation_plot(ax2, results)
    
    # Velocity distribution
    create_velocity_analysis_plot(ax3, results)
    
    # Flow regime classification
    create_flow_regime_plot(ax4, results)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    plt.savefig('diversion_dam_energy_analysis.png', dpi=300, bbox_inches='tight')
    print(f"   ✅ Energy analysis saved as 'diversion_dam_energy_analysis.png'")
    
    plt.show()


def create_complete_energy_plot(ax, results):
    """Create complete energy grade line plot."""
    
    params = results['parameters']
    upstream_profile = results['upstream_profile']
    weir_result = results['weir_result']
    jump_result = results['jump_result']
    
    # Distance points
    x_points = [-150, -150, 0, 5, 15, 25, 100, 200]
    energy_points = []
    water_points = []
    
    # Calculate energy at key points
    for x in x_points:
        if x <= -10:  # Upstream region
            if upstream_profile.success:
                # Find closest profile point
                profile_x = [-p.x for p in upstream_profile.profile_points]
                profile_depths = [p.depth for p in upstream_profile.profile_points]
                closest_idx = min(range(len(profile_x)), key=lambda i: abs(profile_x[i] - abs(x)))
                depth = profile_depths[closest_idx]
                velocity = params['discharge'] / (params['channel_width'] * depth)
                energy = depth + velocity**2 / (2 * 9.81)
                water_level = params['upstream_apron_elevation'] + depth
            else:
                energy = params['normal_depth'] + (params['discharge'] / (params['channel_width'] * params['normal_depth']))**2 / (2 * 9.81)
                water_level = params['upstream_apron_elevation'] + params['normal_depth']
        elif x <= 10:  # Weir region
            energy = weir_result.approach_energy if weir_result.success else 5.0
            water_level = params['crest_elevation'] + weir_result.head_over_weir if weir_result.success else params['crest_elevation'] + 2
        elif x <= 30:  # Jump region
            if jump_result.success:
                if x <= 15:
                    energy = jump_result.upstream_energy
                    water_level = params['downstream_apron_elevation'] + params['spillway_toe_depth']
                else:
                    energy = jump_result.downstream_energy
                    water_level = params['downstream_apron_elevation'] + jump_result.downstream_depth
            else:
                energy = 3.0
                water_level = params['downstream_apron_elevation'] + 2.0
        else:  # Downstream region
            depth = params['tailwater_elevation'] - params['downstream_apron_elevation']
            velocity = params['discharge'] / (params['channel_width'] * depth)
            energy = depth + velocity**2 / (2 * 9.81)
            water_level = params['tailwater_elevation']
        
        energy_points.append(energy)
        water_points.append(water_level)
    
    # Plot energy grade line
    ax.plot(x_points, energy_points, 'r-', linewidth=3, label='Energy Grade Line', marker='o')
    
    # Plot water surface
    ax.plot(x_points, water_points, 'b-', linewidth=2, label='Water Surface', marker='s')
    
    # Mark major energy losses
    if weir_result.success:
        ax.annotate(f'Weir Loss\n{weir_result.energy_dissipated:.1f}m', 
                    xy=(5, weir_result.approach_energy - weir_result.energy_dissipated/2),
                    fontsize=10, ha='center',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    if jump_result.success:
        ax.annotate(f'Jump Loss\n{jump_result.energy_loss:.1f}m', 
                    xy=(20, jump_result.upstream_energy - jump_result.energy_loss/2),
                    fontsize=10, ha='center',
                    bbox=dict(boxstyle='round', facecolor='pink', alpha=0.8))
    
    ax.set_xlabel('Distance from Weir (m)')
    ax.set_ylabel('Energy (m)')
    ax.set_title('Complete Energy Grade Line')
    ax.legend()
    ax.grid(True, alpha=0.3)


def create_froude_variation_plot(ax, results):
    """Create Froude number variation plot."""
    
    params = results['parameters']
    jump_result = results['jump_result']
    
    x_points = [-200, -150, 0, 5, 15, 25, 100, 200]
    froude_points = []
    
    for x in x_points:
        if x <= -10:  # Upstream - subcritical
            depth = params['normal_depth']
            velocity = params['discharge'] / (params['channel_width'] * depth)
            froude = velocity / math.sqrt(9.81 * depth)
        elif x <= 10:  # Over weir - critical/supercritical
            froude = 1.2  # Approximate
        elif x <= 15:  # Before jump - supercritical
            froude = jump_result.upstream_froude if jump_result.success else 2.0
        elif x <= 30:  # After jump - subcritical
            froude = jump_result.downstream_froude if jump_result.success else 0.3
        else:  # Downstream - subcritical
            depth = params['tailwater_elevation'] - params['downstream_apron_elevation']
            velocity = params['discharge'] / (params['channel_width'] * depth)
            froude = velocity / math.sqrt(9.81 * depth)
        
        froude_points.append(froude)
    
    ax.plot(x_points, froude_points, 'purple', linewidth=3, marker='o', label='Froude Number')
    
    # Critical line
    ax.axhline(y=1.0, color='red', linestyle='-', alpha=0.7, linewidth=2, label='Critical Flow (Fr = 1)')
    
    # Shade flow regimes
    ax.axhspan(0, 1, alpha=0.2, color='blue', label='Subcritical (Fr < 1)')
    ax.axhspan(1, max(froude_points), alpha=0.2, color='red', label='Supercritical (Fr > 1)')
    
    ax.set_xlabel('Distance from Weir (m)')
    ax.set_ylabel('Froude Number')
    ax.set_title('Froude Number Variation')
    ax.legend()
    ax.grid(True, alpha=0.3)


def create_velocity_analysis_plot(ax, results):
    """Create velocity analysis plot."""
    
    params = results['parameters']
    jump_result = results['jump_result']
    
    x_points = [-200, -150, 0, 5, 15, 25, 100, 200]
    velocity_points = []
    
    for x in x_points:
        if x <= -10:  # Upstream
            velocity = params['discharge'] / (params['channel_width'] * params['normal_depth'])
        elif x <= 10:  # Over weir
            velocity = 4.0  # Approximate high velocity over weir
        elif x <= 15:  # Before jump
            velocity = params['discharge'] / (params['spillway_width'] * params['spillway_toe_depth'])
        elif x <= 30:  # After jump
            velocity = params['discharge'] / (params['spillway_width'] * jump_result.downstream_depth) if jump_result.success else 1.5
        else:  # Downstream
            depth = params['tailwater_elevation'] - params['downstream_apron_elevation']
            velocity = params['discharge'] / (params['channel_width'] * depth)
        
        velocity_points.append(velocity)
    
    ax.plot(x_points, velocity_points, 'g-', linewidth=3, marker='o', label='Average Velocity')
    
    # Critical velocity
    v_critical = math.sqrt(9.81 * params['critical_depth'])
    ax.axhline(y=v_critical, color='orange', linestyle='--', alpha=0.7, 
               label=f'Critical Velocity ({v_critical:.1f}m/s)')
    
    ax.set_xlabel('Distance from Weir (m)')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title('Velocity Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)


def create_flow_regime_plot(ax, results):
    """Create flow regime classification plot."""
    
    # Flow regimes along the profile
    regimes = ['Subcritical\n(Backwater)', 'Critical\n(Over Weir)', 'Supercritical\n(Spillway Toe)', 
               'Jump\n(RVF)', 'Subcritical\n(Tailwater)']
    positions = [-200, 5, 12, 20, 100]
    colors = ['blue', 'orange', 'red', 'purple', 'cyan']
    
    for i, (regime, pos, color) in enumerate(zip(regimes, positions, colors)):
        ax.barh(i, 50, left=pos-25, color=color, alpha=0.7, height=0.6)
        ax.text(pos, i, regime, ha='center', va='center', fontweight='bold', fontsize=10)
    
    ax.set_xlabel('Distance from Weir (m)')
    ax.set_ylabel('Flow Regime')
    ax.set_title('Flow Regime Classification')
    ax.set_ylim(-0.5, len(regimes)-0.5)
    ax.set_xlim(-200, 200)
    ax.grid(True, alpha=0.3, axis='x')


def main():
    """Run complete diversion dam analysis."""
    
    print("🏗️  COMPLETE DIVERSION DAM FLOW PROFILE")
    print("=" * 90)
    print("Comprehensive GVF + RVF Analysis")
    print("Author: Alexius Academia")
    print("=" * 90)
    
    try:
        # Analyze complete profile
        results = analyze_complete_diversion_dam_profile()
        
        if results:
            # Create visualization
            create_complete_profile_visualization(results)
            
            print("\n" + "=" * 90)
            print("🎉 COMPLETE DIVERSION DAM ANALYSIS COMPLETED!")
            print("=" * 90)
            
            # Summary
            params = results['parameters']
            weir_result = results['weir_result']
            jump_result = results['jump_result']
            
            print(f"\n🎯 SUMMARY:")
            print(f"   • Total profile length: 500m (300m upstream + 200m downstream)")
            print(f"   • Upstream: M1 backwater curve (GVF)")
            print(f"   • Weir: Ogee spillway with Cd = {weir_result.discharge_coefficient:.3f}")
            print(f"   • Jump: {jump_result.jump_type.value.title()} hydraulic jump (RVF)")
            print(f"   • Total energy loss: {weir_result.energy_dissipated + jump_result.energy_loss:.1f}m")
            print(f"   • Power dissipated: {9810 * params['discharge'] * (weir_result.energy_dissipated + jump_result.energy_loss) / 1000:.0f} kW")
            
        else:
            print("\n❌ Complete analysis could not be completed")
            
    except Exception as e:
        print(f"\n❌ Analysis failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
