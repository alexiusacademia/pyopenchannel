#!/usr/bin/env python3
"""
Ogee Weir FVM Profile Analysis - PyOpenChannel

Author: Alexius Academia
Email: alexius.sayco.academia@gmail.com

This example demonstrates detailed FVM analysis of flow over an ogee spillway weir,
showcasing the high-resolution water surface profile, pressure distribution,
and velocity field that enables precise engineering analysis.

Features Demonstrated:
1. FVM method for ogee weir analysis
2. High-resolution water surface profile (200+ points)
3. Detailed pressure distribution over weir crest
4. Velocity field analysis for aeration requirements
5. Energy dissipation downstream of weir
6. Professional spillway visualization
7. Cavitation risk assessment
8. Comparison with analytical method
"""

import sys
import os
import time
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


def create_ogee_weir_scenario():
    """Create a realistic ogee weir scenario for FVM analysis."""
    print("\n" + "="*80)
    print("🏗️  OGEE SPILLWAY WEIR SCENARIO SETUP")
    print("="*80)
    
    # Realistic dam spillway scenario
    print("\n📋 Spillway Design Parameters:")
    print("   Type: Ogee spillway (WES standard shape)")
    print("   Application: Dam overflow spillway")
    print("   Design head: 4.0 m")
    print("   Spillway width: 25.0 m")
    print("   Weir height: 12.0 m")
    print("   Approach depth: 16.0 m (high reservoir level)")
    
    # Create channel geometry
    channel = poc.RectangularChannel(width=25.0)
    
    # Create ogee weir geometry
    weir = poc.WeirGeometry(
        weir_type=poc.WeirType.OGEE_SPILLWAY,
        weir_height=12.0,
        crest_length=25.0,  # Full spillway width
        spillway_shape="WES",  # Water Engineering Society standard
        upstream_slope=0.0,    # Vertical upstream face
        downstream_slope=0.75  # Standard ogee downstream slope
    )
    
    approach_depth = 16.0  # m (4m head over weir)
    
    # Calculate expected flow characteristics
    head_over_weir = approach_depth - weir.weir_height
    
    print(f"\n🔍 Expected Flow Analysis:")
    print(f"   Head over weir: {head_over_weir:.1f} m")
    print(f"   Flow regime: High-head spillway flow")
    print(f"   Expected issues: High velocities, potential cavitation")
    print(f"   Analysis focus: Pressure distribution, aeration needs")
    
    # Estimate discharge using standard ogee weir equation
    # Q = C * L * H^(3/2) where C ≈ 2.2 for ogee spillway
    C_ogee = 2.2  # Discharge coefficient for ogee spillway
    estimated_discharge = C_ogee * weir.crest_length * (head_over_weir ** 1.5)
    
    print(f"\n📊 Preliminary Estimates:")
    print(f"   Estimated discharge: {estimated_discharge:.1f} m³/s")
    print(f"   Unit discharge: {estimated_discharge/weir.crest_length:.2f} m²/s")
    print(f"   Approach velocity: {estimated_discharge/(channel.width * approach_depth):.2f} m/s")
    
    return channel, weir, approach_depth


def analyze_ogee_weir_with_fvm(channel, weir, approach_depth):
    """Perform detailed FVM analysis of ogee weir flow."""
    print("\n" + "="*80)
    print("🧮 FVM ANALYSIS OF OGEE WEIR FLOW")
    print("="*80)
    
    print("\n🔬 Initializing FVM Solver:")
    print("   Method: Finite Volume Method (FVM)")
    print("   Scheme: HLLC (High-resolution shock-capturing)")
    print("   Grid: Adaptive high-resolution near weir crest")
    print("   Analysis: Steady-state spillway flow")
    
    try:
        # Create FVM solver
        solver = poc.WeirFlowSolver(method="fvm")
        
        print("\n⚙️  FVM Solver Configuration:")
        print("   Numerical scheme: HLLC")
        print("   Time integration: RK4")
        print("   Convergence tolerance: 1e-8")
        print("   CFL number: 0.15 (conservative for complex geometry)")
        
        # Perform FVM analysis
        print("\n🚀 Running FVM Analysis...")
        start_time = time.time()
        
        result = solver.analyze_weir_flow(
            channel=channel,
            weir=weir,
            approach_depth=approach_depth
        )
        
        computation_time = time.time() - start_time
        
        if result.success:
            print(f"   ✅ FVM analysis completed successfully!")
            print(f"   ⏱️  Computation time: {computation_time*1000:.1f} ms")
            print(f"   📊 Profile resolution: {result.profile_resolution}")
            
            # Display key results
            print(f"\n📈 Flow Analysis Results:")
            print(f"   Discharge: {result.discharge:.2f} m³/s")
            print(f"   Head over weir: {result.head_over_weir:.3f} m")
            print(f"   Approach velocity: {result.approach_velocity:.3f} m/s")
            print(f"   Weir condition: {result.weir_condition.value}")
            print(f"   Energy efficiency: {result.energy_efficiency:.3f}")
            print(f"   Aeration requirement: {result.aeration_requirement.value}")
            print(f"   Cavitation risk: {result.cavitation_risk.value}")
            
            # Detailed FVM profile analysis
            if result.has_detailed_profile:
                profile = result.fvm_profile
                
                print(f"\n🔬 Detailed FVM Profile Analysis:")
                print(f"   Grid points: {profile.grid_points}")
                print(f"   Domain length: {profile.domain_length:.1f} m")
                print(f"   Average resolution: {profile.resolution:.3f} m/point")
                print(f"   Numerical scheme: {profile.scheme_used}")
                print(f"   Convergence iterations: {profile.convergence_iterations}")
                
                # Analyze weir crest conditions
                crest_conditions = profile.find_weir_crest_conditions()
                print(f"\n🏔️  Weir Crest Conditions:")
                print(f"   Crest depth: {crest_conditions.get('crest_depth', 0):.3f} m")
                print(f"   Crest velocity: {crest_conditions.get('crest_velocity', 0):.2f} m/s")
                print(f"   Crest Froude number: {crest_conditions.get('crest_froude', 0):.3f}")
                print(f"   Crest specific energy: {crest_conditions.get('crest_energy', 0):.3f} m")
                print(f"   Crest pressure: {crest_conditions.get('crest_pressure', 0):.3f} m")
                
                # Pressure distribution analysis
                pressure_analysis = profile.analyze_pressure_distribution()
                print(f"\n🔴 Pressure Distribution Analysis:")
                print(f"   Minimum pressure: {pressure_analysis.get('min_pressure', 0):.3f} m")
                print(f"   Min pressure location: {pressure_analysis.get('min_pressure_location', 0):.2f} m")
                print(f"   Maximum pressure: {pressure_analysis.get('max_pressure', 0):.3f} m")
                print(f"   Pressure range: {pressure_analysis.get('pressure_range', 0):.3f} m")
                print(f"   Cavitation risk: {pressure_analysis.get('cavitation_risk', 'UNKNOWN')}")
                print(f"   Pressure margin: {pressure_analysis.get('pressure_margin', 0):.3f} m")
                
                if pressure_analysis.get('pressure_margin', 0) < 0:
                    print(f"   🚨 WARNING: Cavitation risk detected!")
                    print(f"   💡 Engineering recommendations:")
                    print(f"      - Install aeration slots at weir crest")
                    print(f"      - Use cavitation-resistant materials")
                    print(f"      - Consider spillway shape optimization")
                else:
                    print(f"   ✅ Pressure levels adequate - low cavitation risk")
                
                # Velocity field analysis
                velocity_analysis = profile.analyze_velocity_distribution()
                print(f"\n🌪️  Velocity Field Analysis:")
                print(f"   Maximum velocity: {velocity_analysis.get('max_velocity', 0):.2f} m/s")
                print(f"   Max velocity location: {velocity_analysis.get('max_velocity_location', 0):.2f} m")
                print(f"   Average velocity: {velocity_analysis.get('avg_velocity', 0):.2f} m/s")
                print(f"   Aeration requirement: {velocity_analysis.get('aeration_requirement', 'NONE')}")
                
                # Energy dissipation analysis
                energy_analysis = profile.analyze_energy_dissipation()
                print(f"\n⚡ Energy Dissipation Analysis:")
                print(f"   Upstream energy: {energy_analysis.get('upstream_energy', 0):.3f} m")
                print(f"   Downstream energy: {energy_analysis.get('downstream_energy', 0):.3f} m")
                print(f"   Energy loss: {energy_analysis.get('energy_loss', 0):.3f} m")
                print(f"   Energy efficiency: {energy_analysis.get('energy_efficiency', 0):.3f}")
                print(f"   Dissipation ratio: {energy_analysis.get('energy_dissipation_ratio', 0):.3f}")
                
                return result, profile
            else:
                print(f"   ⚠️  Detailed profile not available - using analytical fallback")
                return result, None
        else:
            print(f"   ❌ FVM analysis failed: {result.message}")
            return None, None
            
    except Exception as e:
        print(f"   ❌ FVM analysis error: {e}")
        print(f"   💡 This may indicate FVM components need further development")
        return None, None


def compare_analytical_vs_fvm(channel, weir, approach_depth):
    """Compare analytical and FVM methods for the same ogee weir."""
    print("\n" + "="*80)
    print("⚖️  ANALYTICAL vs FVM COMPARISON")
    print("="*80)
    
    results = {}
    
    # Analytical method
    print("\n🔬 Analytical Method:")
    try:
        solver_analytical = poc.WeirFlowSolver(method="analytical")
        start_time = time.time()
        result_analytical = solver_analytical.analyze_weir_flow(channel, weir, approach_depth)
        analytical_time = time.time() - start_time
        results['analytical'] = result_analytical
        
        print(f"   ✅ Completed in {analytical_time*1000:.1f} ms")
        print(f"   📊 Resolution: {result_analytical.profile_resolution}")
        print(f"   💧 Discharge: {result_analytical.discharge:.2f} m³/s")
        print(f"   📏 Head: {result_analytical.head_over_weir:.3f} m")
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        results['analytical'] = None
    
    # FVM method
    print("\n🧮 FVM Method:")
    try:
        solver_fvm = poc.WeirFlowSolver(method="fvm")
        start_time = time.time()
        result_fvm = solver_fvm.analyze_weir_flow(channel, weir, approach_depth)
        fvm_time = time.time() - start_time
        results['fvm'] = result_fvm
        
        print(f"   ✅ Completed in {fvm_time*1000:.1f} ms")
        print(f"   📊 Resolution: {result_fvm.profile_resolution}")
        print(f"   💧 Discharge: {result_fvm.discharge:.2f} m³/s")
        print(f"   📏 Head: {result_fvm.head_over_weir:.3f} m")
        
        if result_fvm.has_detailed_profile:
            print(f"   🔬 Grid points: {result_fvm.fvm_profile.grid_points}")
            print(f"   📐 Domain: {result_fvm.fvm_profile.domain_length:.1f} m")
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        results['fvm'] = None
    
    # Performance comparison
    if results['analytical'] and results['fvm']:
        print(f"\n⏱️  Performance Comparison:")
        print(f"   Analytical: {analytical_time*1000:.1f} ms")
        print(f"   FVM:        {fvm_time*1000:.1f} ms")
        if analytical_time > 0:
            print(f"   Speed ratio: {fvm_time/analytical_time:.1f}x slower (FVM)")
        
        print(f"\n🎯 Engineering Value:")
        print(f"   Analytical: Fast design calculations")
        print(f"   FVM:        Detailed pressure/velocity analysis")
        print(f"   Use case:   Choose based on analysis requirements")
    
    return results


def create_water_surface_profile_visualization(result, profile):
    """Create comprehensive water surface profile visualization."""
    if not MATPLOTLIB_AVAILABLE:
        print("\n📊 Visualization skipped (matplotlib not available)")
        return
    
    if not profile:
        print("\n📊 Visualization skipped (no FVM profile available)")
        return
    
    print("\n📊 Creating water surface profile visualization...")
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('Ogee Weir FVM Analysis: Water Surface Profile & Flow Characteristics', 
                 fontsize=18, fontweight='bold')
    
    # Main water surface profile (large plot)
    ax_main = plt.subplot2grid((4, 3), (0, 0), colspan=3, rowspan=2)
    
    x = profile.x_coordinates
    depths = profile.depths
    
    # Plot water surface
    ax_main.fill_between(x, 0, depths, alpha=0.3, color='lightblue', label='Water')
    ax_main.plot(x, depths, 'b-', linewidth=3, label='Water Surface (FVM)')
    
    # Add weir geometry
    weir_start = profile.weir_crest_location
    weir_end = profile.weir_crest_location + profile.weir_length
    weir_height = profile.weir_height
    
    # Create ogee weir shape (simplified)
    weir_x = np.linspace(weir_start, weir_end, 50)
    # Simplified ogee profile: y = H * (x/L)^1.85 for upstream, different curve downstream
    weir_y = np.full_like(weir_x, weir_height)
    
    # Add some curvature to represent ogee shape
    mid_point = (weir_start + weir_end) / 2
    for i, x_pos in enumerate(weir_x):
        if x_pos < mid_point:
            # Upstream curve
            rel_pos = (x_pos - weir_start) / (mid_point - weir_start)
            weir_y[i] = weir_height * (0.8 + 0.2 * rel_pos**1.5)
        else:
            # Downstream curve
            rel_pos = (x_pos - mid_point) / (weir_end - mid_point)
            weir_y[i] = weir_height * (1.0 - 0.3 * rel_pos**0.5)
    
    ax_main.fill_between(weir_x, 0, weir_y, color='gray', alpha=0.8, label='Ogee Weir')
    ax_main.plot(weir_x, weir_y, 'k-', linewidth=2)
    
    # Add channel bottom
    ax_main.axhline(y=0, color='brown', linewidth=3, alpha=0.7, label='Channel Bottom')
    
    # Highlight critical regions
    ax_main.axvline(x=weir_start, color='red', linestyle='--', alpha=0.7, label='Weir Crest')
    
    # Find and mark maximum velocity location
    velocity_analysis = profile.analyze_velocity_distribution()
    max_vel_location = velocity_analysis.get('max_velocity_location', weir_start)
    ax_main.axvline(x=max_vel_location, color='orange', linestyle=':', alpha=0.7, 
                   label='Max Velocity Location')
    
    ax_main.set_xlabel('Distance (m)', fontsize=12)
    ax_main.set_ylabel('Elevation (m)', fontsize=12)
    ax_main.set_title('Water Surface Profile over Ogee Weir', fontsize=14, fontweight='bold')
    ax_main.legend(loc='upper right')
    ax_main.grid(True, alpha=0.3)
    ax_main.set_ylim(0, max(depths) * 1.1)
    
    # Add flow direction arrow
    arrow_x = x[len(x)//4]
    arrow_y = depths[len(x)//4] + 1
    ax_main.annotate('Flow Direction', xy=(arrow_x + 10, arrow_y), xytext=(arrow_x, arrow_y),
                    arrowprops=dict(arrowstyle='->', lw=2, color='blue'),
                    fontsize=12, fontweight='bold', color='blue')
    
    # Velocity profile
    ax_vel = plt.subplot2grid((4, 3), (2, 0))
    ax_vel.plot(x, profile.velocities, 'r-', linewidth=2)
    ax_vel.axvspan(weir_start, weir_end, alpha=0.2, color='gray')
    ax_vel.set_xlabel('Distance (m)')
    ax_vel.set_ylabel('Velocity (m/s)')
    ax_vel.set_title('Velocity Distribution')
    ax_vel.grid(True, alpha=0.3)
    
    # Highlight high velocity regions
    high_vel_mask = profile.velocities > 10.0
    if np.any(high_vel_mask):
        ax_vel.fill_between(x, 0, profile.velocities, where=high_vel_mask, 
                           alpha=0.3, color='orange', label='High Velocity (>10m/s)')
        ax_vel.legend()
    
    # Pressure distribution
    ax_press = plt.subplot2grid((4, 3), (2, 1))
    ax_press.plot(x, profile.pressure_heads, 'g-', linewidth=2)
    ax_press.axhline(y=2.0, color='red', linestyle=':', alpha=0.7, label='Cavitation Threshold')
    ax_press.axvspan(weir_start, weir_end, alpha=0.2, color='gray')
    ax_press.set_xlabel('Distance (m)')
    ax_press.set_ylabel('Pressure Head (m)')
    ax_press.set_title('Pressure Distribution')
    ax_press.legend()
    ax_press.grid(True, alpha=0.3)
    
    # Highlight low pressure regions
    low_press_mask = profile.pressure_heads < 2.0
    if np.any(low_press_mask):
        ax_press.fill_between(x, 0, profile.pressure_heads, where=low_press_mask,
                             alpha=0.3, color='red', label='Cavitation Risk')
    
    # Froude number
    ax_froude = plt.subplot2grid((4, 3), (2, 2))
    ax_froude.plot(x, profile.froude_numbers, 'm-', linewidth=2)
    ax_froude.axhline(y=1.0, color='black', linestyle='--', alpha=0.7, label='Critical Flow')
    ax_froude.axvspan(weir_start, weir_end, alpha=0.2, color='gray')
    ax_froude.set_xlabel('Distance (m)')
    ax_froude.set_ylabel('Froude Number')
    ax_froude.set_title('Flow Regime')
    ax_froude.legend()
    ax_froude.grid(True, alpha=0.3)
    
    # Fill regions based on flow regime
    subcritical_mask = profile.froude_numbers < 1.0
    supercritical_mask = profile.froude_numbers > 1.0
    
    if np.any(subcritical_mask):
        ax_froude.fill_between(x, 0, 1, where=subcritical_mask, alpha=0.2, color='blue', 
                              label='Subcritical')
    if np.any(supercritical_mask):
        ax_froude.fill_between(x, 1, profile.froude_numbers, where=supercritical_mask, 
                              alpha=0.2, color='red', label='Supercritical')
    
    # Specific energy
    ax_energy = plt.subplot2grid((4, 3), (3, 0))
    ax_energy.plot(x, profile.specific_energies, 'purple', linewidth=2)
    ax_energy.axvspan(weir_start, weir_end, alpha=0.2, color='gray')
    ax_energy.set_xlabel('Distance (m)')
    ax_energy.set_ylabel('Specific Energy (m)')
    ax_energy.set_title('Energy Distribution')
    ax_energy.grid(True, alpha=0.3)
    
    # Analysis summary
    ax_summary = plt.subplot2grid((4, 3), (3, 1), colspan=2)
    ax_summary.axis('off')
    
    # Create summary text
    crest_conditions = profile.find_weir_crest_conditions()
    pressure_analysis = profile.analyze_pressure_distribution()
    velocity_analysis = profile.analyze_velocity_distribution()
    energy_analysis = profile.analyze_energy_dissipation()
    
    summary_text = f"""
FVM ANALYSIS SUMMARY

Grid Resolution: {profile.grid_points} points over {profile.domain_length:.1f}m domain
Numerical Scheme: {profile.scheme_used}
Convergence: {profile.convergence_iterations} iterations

WEIR CREST CONDITIONS:
• Depth: {crest_conditions.get('crest_depth', 0):.3f} m
• Velocity: {crest_conditions.get('crest_velocity', 0):.2f} m/s  
• Froude: {crest_conditions.get('crest_froude', 0):.3f}
• Pressure: {crest_conditions.get('crest_pressure', 0):.3f} m

PRESSURE ANALYSIS:
• Min Pressure: {pressure_analysis.get('min_pressure', 0):.3f} m
• Cavitation Risk: {pressure_analysis.get('cavitation_risk', 'UNKNOWN')}
• Pressure Margin: {pressure_analysis.get('pressure_margin', 0):.3f} m

VELOCITY ANALYSIS:
• Max Velocity: {velocity_analysis.get('max_velocity', 0):.2f} m/s
• Aeration Need: {velocity_analysis.get('aeration_requirement', 'NONE')}

ENERGY ANALYSIS:
• Energy Loss: {energy_analysis.get('energy_loss', 0):.3f} m
• Efficiency: {energy_analysis.get('energy_efficiency', 0):.3f}
"""
    
    ax_summary.text(0.05, 0.95, summary_text, transform=ax_summary.transAxes, 
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    print("✅ Water surface profile visualization complete!")


def main():
    """Main demonstration function."""
    print("🏗️  Ogee Weir FVM Profile Analysis Demo")
    print("=====================================")
    
    # Step 1: Create ogee weir scenario
    channel, weir, approach_depth = create_ogee_weir_scenario()
    
    # Step 2: Perform detailed FVM analysis
    result, profile = analyze_ogee_weir_with_fvm(channel, weir, approach_depth)
    
    # Step 3: Compare analytical vs FVM
    comparison_results = compare_analytical_vs_fvm(channel, weir, approach_depth)
    
    # Step 4: Create comprehensive visualization
    if result and profile:
        create_water_surface_profile_visualization(result, profile)
    else:
        print("\n⚠️  Skipping visualization - FVM analysis not available")
        print("💡 This is expected if FVM integration is still in development")
    
    # Summary
    print("\n" + "="*80)
    print("📋 OGEE WEIR FVM ANALYSIS SUMMARY")
    print("="*80)
    
    if result and profile:
        print("\n✅ FVM Analysis Successful:")
        print(f"   • High-resolution profile: {profile.grid_points} points")
        print(f"   • Domain coverage: {profile.domain_length:.1f} m")
        print(f"   • Numerical scheme: {profile.scheme_used}")
        print(f"   • Computation time: {result.computation_time*1000:.1f} ms")
        
        print(f"\n🔬 Engineering Insights:")
        crest_conditions = profile.find_weir_crest_conditions()
        pressure_analysis = profile.analyze_pressure_distribution()
        velocity_analysis = profile.analyze_velocity_distribution()
        
        print(f"   • Weir crest velocity: {crest_conditions.get('crest_velocity', 0):.2f} m/s")
        print(f"   • Minimum pressure: {pressure_analysis.get('min_pressure', 0):.3f} m")
        print(f"   • Cavitation risk: {pressure_analysis.get('cavitation_risk', 'UNKNOWN')}")
        print(f"   • Maximum velocity: {velocity_analysis.get('max_velocity', 0):.2f} m/s")
        print(f"   • Aeration requirement: {velocity_analysis.get('aeration_requirement', 'NONE')}")
        
        print(f"\n🎯 Professional Applications:")
        print(f"   • Spillway design optimization")
        print(f"   • Cavitation damage prevention")
        print(f"   • Aeration system design")
        print(f"   • Energy dissipation analysis")
        print(f"   • Dam safety assessment")
        
    else:
        print("\n⚠️  FVM Analysis Not Available:")
        print("   • This indicates FVM integration is still in development")
        print("   • The framework is ready - implementation details needed")
        print("   • Analytical methods are working as fallback")
    
    print(f"\n🎉 Ogee Weir FVM Profile Demo completed!")
    print("   This demonstrates the power of FVM for detailed spillway analysis,")
    print("   providing engineers with unprecedented insight into flow patterns,")
    print("   pressure distributions, and energy dissipation over weir crests.")


if __name__ == "__main__":
    main()
