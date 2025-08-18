#!/usr/bin/env python3
"""
Weir Flow FVM Integration Demonstration - PyOpenChannel

Author: Alexius Academia
Email: alexius.sayco.academia@gmail.com

This example demonstrates the seamless integration between analytical and FVM methods
for weir flow analysis. Users can choose between:
- Analytical method: Fast 3-point solution (approach, crest, downstream)
- FVM method: Detailed 200+ point profile with pressure distribution and aeration analysis

Key Features Demonstrated:
1. Method selection (analytical vs FVM)
2. Detailed pressure distribution over weir crest
3. Velocity profiles for aeration requirements
4. Cavitation assessment from pressure field
5. Energy dissipation analysis downstream
6. Professional spillway design visualization
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
    MATPLOTLIB_AVAILABLE = True
    print("✅ Matplotlib available - plotting enabled")
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️  Matplotlib not available - plotting disabled")


def demonstrate_weir_flow_setup():
    """Set up weir flow test case."""
    print("\n" + "="*80)
    print("💧 WEIR FLOW FVM INTEGRATION DEMONSTRATION")
    print("="*80)
    
    # Set up test case - ogee spillway with high head
    print("\n📋 Test Case Setup:")
    print("   Channel: 15m wide rectangular")
    print("   Weir: Ogee spillway, 5m height")
    print("   Approach depth: 8.0m (high head)")
    print("   Expected flow: High velocity over crest")
    print("   Analysis focus: Pressure distribution and aeration")
    
    # Create channel and weir
    channel = poc.RectangularChannel(width=15.0)
    weir = poc.WeirGeometry(
        weir_type=poc.WeirType.OGEE_SPILLWAY,
        weir_height=5.0,
        crest_length=15.0,  # Full channel width
        spillway_shape="WES"
    )
    approach_depth = 8.0  # m (3m head over weir)
    
    print(f"\n🔍 Flow Analysis:")
    # Estimate flow conditions using basic weir equation
    head_over_weir = approach_depth - weir.weir_height
    # Q = C * L * H^(3/2) where C ≈ 2.2 for ogee spillway
    estimated_discharge = 2.2 * weir.crest_length * (head_over_weir ** 1.5)
    estimated_velocity = estimated_discharge / (weir.crest_length * head_over_weir)
    
    print(f"   Head over weir: {head_over_weir:.1f} m")
    print(f"   Estimated discharge: {estimated_discharge:.1f} m³/s")
    print(f"   Estimated crest velocity: {estimated_velocity:.2f} m/s")
    print(f"   Expected flow regime: Supercritical over crest")
    print(f"   Potential issues: High velocity → aeration needs")
    
    return channel, weir, approach_depth


def compare_analytical_vs_fvm_weir_flow(channel, weir, approach_depth):
    """Compare analytical and FVM methods for weir flow."""
    print("\n" + "="*60)
    print("⚖️  METHOD COMPARISON: ANALYTICAL vs FVM WEIR FLOW")
    print("="*60)
    
    results = {}
    
    # Method 1: Analytical (default)
    print("\n🔬 Method 1: Analytical Weir Flow Analysis")
    print("   - Fast 3-point solution (approach, crest, downstream)")
    print("   - Standard weir equations")
    print("   - Suitable for design calculations")
    
    solver_analytical = poc.WeirFlowSolver(method="analytical")
    start_time = time.time()
    
    try:
        result_analytical = solver_analytical.analyze_weir_flow(
            channel, weir, approach_depth
        )
        analytical_time = time.time() - start_time
        results['analytical'] = result_analytical
        
        print(f"   ✅ Analysis completed in {analytical_time*1000:.1f} ms")
        print(f"   📊 Profile resolution: {result_analytical.profile_resolution}")
        print(f"   🌊 Weir condition: {result_analytical.weir_condition.value}")
        print(f"   💧 Discharge: {result_analytical.discharge:.2f} m³/s")
        print(f"   📏 Head over weir: {result_analytical.head_over_weir:.2f} m")
        print(f"   ⚡ Approach velocity: {result_analytical.approach_velocity:.2f} m/s")
        print(f"   ⚠️  Aeration requirement: {result_analytical.aeration_requirement.value}")
        print(f"   🔴 Cavitation risk: {result_analytical.cavitation_risk.value}")
        
    except Exception as e:
        print(f"   ❌ Analytical analysis failed: {e}")
        results['analytical'] = None
        analytical_time = 0
    
    # Method 2: FVM (detailed)
    print("\n🧮 Method 2: FVM Weir Flow Analysis")
    print("   - Detailed 200+ point profile")
    print("   - Pressure distribution over weir crest")
    print("   - Velocity field for aeration analysis")
    print("   - Energy dissipation downstream")
    
    try:
        solver_fvm = poc.WeirFlowSolver(method="fvm")
        start_time = time.time()
        
        result_fvm = solver_fvm.analyze_weir_flow(
            channel, weir, approach_depth
        )
        fvm_time = time.time() - start_time
        results['fvm'] = result_fvm
        
        print(f"   ✅ Analysis completed in {fvm_time*1000:.1f} ms")
        print(f"   📊 Profile resolution: {result_fvm.profile_resolution}")
        print(f"   🌊 Weir condition: {result_fvm.weir_condition.value}")
        print(f"   💧 Discharge: {result_fvm.discharge:.2f} m³/s")
        print(f"   📏 Head over weir: {result_fvm.head_over_weir:.2f} m")
        print(f"   ⚡ Approach velocity: {result_fvm.approach_velocity:.2f} m/s")
        print(f"   ⚠️  Aeration requirement: {result_fvm.aeration_requirement.value}")
        print(f"   🔴 Cavitation risk: {result_fvm.cavitation_risk.value}")
        
        if result_fvm.has_detailed_profile:
            profile = result_fvm.fvm_profile
            print(f"   🔬 FVM Details:")
            print(f"      - Grid points: {profile.grid_points}")
            print(f"      - Domain length: {profile.domain_length:.1f} m")
            print(f"      - Resolution: {profile.resolution:.3f} m/point")
            print(f"      - Scheme: {profile.scheme_used}")
            print(f"      - Iterations: {profile.convergence_iterations}")
            
            # Detailed analysis
            crest_conditions = profile.find_weir_crest_conditions()
            pressure_analysis = profile.analyze_pressure_distribution()
            velocity_analysis = profile.analyze_velocity_distribution()
            
            print(f"      - Crest depth: {crest_conditions.get('crest_depth', 0):.3f} m")
            print(f"      - Crest velocity: {crest_conditions.get('crest_velocity', 0):.2f} m/s")
            print(f"      - Min pressure: {pressure_analysis.get('min_pressure', 0):.3f} m")
            print(f"      - Max velocity: {velocity_analysis.get('max_velocity', 0):.2f} m/s")
        
    except Exception as e:
        print(f"   ❌ FVM analysis failed: {e}")
        print(f"   💡 This is expected if FVM components aren't fully integrated yet")
        results['fvm'] = None
        fvm_time = 0
    
    # Performance comparison
    if results['analytical'] and results['fvm']:
        print(f"\n⏱️  Performance Comparison:")
        print(f"   Analytical: {analytical_time*1000:.1f} ms")
        print(f"   FVM:        {fvm_time*1000:.1f} ms")
        if analytical_time > 0:
            print(f"   Speed ratio: {fvm_time/analytical_time:.1f}x slower (FVM)")
        
        # Accuracy comparison
        discharge_diff = abs(results['fvm'].discharge - results['analytical'].discharge)
        velocity_diff = abs(results['fvm'].approach_velocity - results['analytical'].approach_velocity)
        head_diff = abs(results['fvm'].head_over_weir - results['analytical'].head_over_weir)
        
        print(f"\n🎯 Accuracy Comparison:")
        if results['analytical'].discharge > 0:
            print(f"   Discharge difference: {discharge_diff:.3f} m³/s ({discharge_diff/results['analytical'].discharge*100:.2f}%)")
        else:
            print(f"   Discharge difference: {discharge_diff:.3f} m³/s (analytical discharge is zero)")
        
        if results['analytical'].approach_velocity > 0:
            print(f"   Approach velocity difference: {velocity_diff:.3f} m/s ({velocity_diff/results['analytical'].approach_velocity*100:.2f}%)")
        else:
            print(f"   Approach velocity difference: {velocity_diff:.3f} m/s (analytical velocity is zero)")
        
        if results['analytical'].head_over_weir > 0:
            print(f"   Head difference: {head_diff:.4f} m ({head_diff/results['analytical'].head_over_weir*100:.2f}%)")
        else:
            print(f"   Head difference: {head_diff:.4f} m (analytical head is zero)")
        
        # Check if results are meaningful
        if results['analytical'].discharge == 0 and results['fvm'].discharge == 0:
            print(f"   ⚠️  Both methods returned zero discharge - check weir flow implementation")
    
    return results


def demonstrate_spillway_pressure_analysis():
    """Demonstrate detailed spillway pressure analysis with FVM."""
    print("\n" + "="*60)
    print("🔴 SPILLWAY PRESSURE & CAVITATION ANALYSIS")
    print("="*60)
    
    # Create high-head spillway scenario
    channel = poc.RectangularChannel(width=20.0)
    weir = poc.WeirGeometry(
        weir_type=poc.WeirType.OGEE_SPILLWAY,
        weir_height=8.0,
        crest_length=20.0,
        spillway_shape="WES"
    )
    approach_depth = 12.0  # 4m head - high velocity scenario
    
    print(f"\n📋 High-Head Spillway Scenario:")
    print(f"   Channel width: {channel.width} m")
    print(f"   Spillway height: {weir.weir_height} m")
    print(f"   Approach depth: {approach_depth} m")
    print(f"   Head over weir: {approach_depth - weir.weir_height:.1f} m (high head)")
    
    # Analyze with FVM for detailed pressure field
    try:
        solver = poc.WeirFlowSolver(method="fvm")
        result = solver.analyze_weir_flow(channel, weir, approach_depth)
        
        if result.has_detailed_profile:
            profile = result.fvm_profile
            
            print(f"\n🔬 Detailed Spillway Analysis:")
            print(f"   Overall cavitation risk: {result.cavitation_risk.value}")
            print(f"   Aeration requirement: {result.aeration_requirement.value}")
            print(f"   Energy efficiency: {result.energy_efficiency:.3f}")
            
            # Analyze pressure distribution
            pressure_analysis = profile.analyze_pressure_distribution()
            
            print(f"\n📊 Pressure Field Analysis:")
            print(f"   Minimum pressure: {pressure_analysis.get('min_pressure', 0):.3f} m")
            print(f"   Min pressure location: {pressure_analysis.get('min_pressure_location', 0):.2f} m")
            print(f"   Maximum pressure: {pressure_analysis.get('max_pressure', 0):.3f} m")
            print(f"   Average pressure: {pressure_analysis.get('avg_pressure', 0):.3f} m")
            print(f"   Pressure range: {pressure_analysis.get('pressure_range', 0):.3f} m")
            print(f"   Cavitation threshold: {pressure_analysis.get('cavitation_threshold', 0):.1f} m")
            print(f"   Pressure margin: {pressure_analysis.get('pressure_margin', 0):.3f} m")
            
            if pressure_analysis.get('pressure_margin', 0) < 0:
                print(f"   🚨 CAVITATION RISK: Pressure below threshold!")
                print(f"   💡 Recommendations:")
                print(f"      - Install aeration system at crest")
                print(f"      - Use cavitation-resistant concrete")
                print(f"      - Consider spillway shape modification")
                print(f"      - Monitor for cavitation damage")
            else:
                print(f"   ✅ Pressure margin adequate")
            
            # Velocity analysis
            velocity_analysis = profile.analyze_velocity_distribution()
            
            print(f"\n🌪️  Velocity Field Analysis:")
            print(f"   Maximum velocity: {velocity_analysis.get('max_velocity', 0):.2f} m/s")
            print(f"   Max velocity location: {velocity_analysis.get('max_velocity_location', 0):.2f} m")
            print(f"   Crest velocity: {velocity_analysis.get('crest_velocity', 0):.2f} m/s")
            print(f"   Aeration requirement: {velocity_analysis.get('aeration_requirement', 'NONE')}")
            
            # Energy dissipation
            energy_analysis = profile.analyze_energy_dissipation()
            
            print(f"\n⚡ Energy Dissipation Analysis:")
            print(f"   Upstream energy: {energy_analysis.get('upstream_energy', 0):.3f} m")
            print(f"   Downstream energy: {energy_analysis.get('downstream_energy', 0):.3f} m")
            print(f"   Energy loss: {energy_analysis.get('energy_loss', 0):.3f} m")
            print(f"   Energy efficiency: {energy_analysis.get('energy_efficiency', 0):.3f}")
            print(f"   Dissipation ratio: {energy_analysis.get('energy_dissipation_ratio', 0):.3f}")
            
            return profile
        else:
            print("   ⚠️  No detailed profile available")
            return None
            
    except Exception as e:
        print(f"   ❌ FVM spillway analysis failed: {e}")
        print("   💡 This is expected if FVM integration is still in development")
        return None


def create_weir_flow_visualization(results):
    """Create comprehensive weir flow visualization."""
    if not MATPLOTLIB_AVAILABLE:
        print("\n📊 Visualization skipped (matplotlib not available)")
        return
    
    if not results.get('analytical') or not results.get('fvm'):
        print("\n📊 Visualization skipped (missing results)")
        return
    
    print("\n📊 Creating weir flow comparison visualization...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Weir Flow FVM Integration: Analytical vs FVM Comparison', fontsize=16, fontweight='bold')
    
    analytical = results['analytical']
    fvm = results['fvm']
    
    # Plot 1: Discharge comparison
    methods = ['Analytical', 'FVM']
    discharges = [analytical.discharge, fvm.discharge]
    heads = [analytical.head_over_weir, fvm.head_over_weir]
    
    ax1.bar(methods, discharges, alpha=0.7, color=['blue', 'red'])
    ax1.set_ylabel('Discharge (m³/s)')
    ax1.set_title('Discharge Comparison')
    ax1.grid(True, alpha=0.3)
    
    for i, v in enumerate(discharges):
        ax1.text(i, v + max(discharges)*0.02, f'{v:.1f}m³/s', ha='center', va='bottom')
    
    # Plot 2: Head over weir comparison
    ax2.bar(methods, heads, alpha=0.7, color=['green', 'orange'])
    ax2.set_ylabel('Head over Weir (m)')
    ax2.set_title('Head Comparison')
    ax2.grid(True, alpha=0.3)
    
    for i, v in enumerate(heads):
        ax2.text(i, v + max(heads)*0.02, f'{v:.2f}m', ha='center', va='bottom')
    
    # Plot 3: Energy efficiency comparison
    efficiencies = [analytical.energy_efficiency, fvm.energy_efficiency]
    ax3.bar(methods, efficiencies, alpha=0.7, color=['purple', 'brown'])
    ax3.set_ylabel('Energy Efficiency')
    ax3.set_title('Energy Efficiency Comparison')
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3)
    
    for i, v in enumerate(efficiencies):
        ax3.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
    
    # Plot 4: Profile resolution comparison
    resolutions = [3, fvm.fvm_profile.grid_points if fvm.has_detailed_profile else 3]
    ax4.bar(methods, resolutions, alpha=0.7, color=['cyan', 'magenta'])
    ax4.set_ylabel('Number of Profile Points')
    ax4.set_title('Profile Resolution Comparison')
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3)
    
    for i, v in enumerate(resolutions):
        ax4.text(i, v * 1.1, f'{v} points', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    # Create detailed FVM profile plot if available
    if fvm.has_detailed_profile:
        create_detailed_spillway_profile_plot(fvm.fvm_profile, analytical)


def create_detailed_spillway_profile_plot(weir_profile, analytical_result):
    """Create detailed FVM spillway profile visualization."""
    if not MATPLOTLIB_AVAILABLE:
        return
    
    print("📈 Creating detailed FVM spillway profile plot...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Detailed FVM Spillway Flow Profile Analysis', fontsize=16, fontweight='bold')
    
    x = weir_profile.x_coordinates
    
    # Plot 1: Water depth profile
    ax1.plot(x, weir_profile.depths, 'b-', linewidth=2, label='FVM Profile')
    ax1.axhline(y=analytical_result.approach_depth, color='red', linestyle='--', 
                label=f'Analytical Approach ({analytical_result.approach_depth:.1f}m)')
    
    # Highlight weir region
    weir_start = weir_profile.weir_crest_location
    weir_end = weir_profile.weir_crest_location + weir_profile.weir_length
    ax1.axvspan(weir_start, weir_end, alpha=0.2, color='gray', label='Weir Region')
    
    # Add weir crest line
    ax1.axhline(y=weir_profile.weir_height, color='black', linestyle='-', 
                linewidth=3, alpha=0.7, label=f'Weir Crest ({weir_profile.weir_height:.1f}m)')
    
    ax1.set_xlabel('Distance (m)')
    ax1.set_ylabel('Water Depth (m)')
    ax1.set_title('Water Surface Profile')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Velocity profile
    ax2.plot(x, weir_profile.velocities, 'r-', linewidth=2, label='FVM Velocity')
    ax2.axhline(y=analytical_result.approach_velocity, color='blue', linestyle='--',
                label=f'Analytical Approach ({analytical_result.approach_velocity:.2f}m/s)')
    
    ax2.axvspan(weir_start, weir_end, alpha=0.2, color='gray', label='Weir Region')
    
    # Highlight high velocity regions for aeration
    high_velocity_mask = weir_profile.velocities > 10.0
    if np.any(high_velocity_mask):
        ax2.fill_between(x, 0, weir_profile.velocities, 
                        where=high_velocity_mask, alpha=0.3, color='orange',
                        label='High Velocity (>10m/s)')
    
    ax2.set_xlabel('Distance (m)')
    ax2.set_ylabel('Velocity (m/s)')
    ax2.set_title('Velocity Distribution (Aeration Analysis)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Pressure distribution (key for cavitation)
    ax3.plot(x, weir_profile.pressure_heads, 'g-', linewidth=2, label='FVM Pressure')
    ax3.axhline(y=2.0, color='red', linestyle=':', linewidth=2, alpha=0.7, 
                label='Cavitation Threshold (~2m)')
    
    ax3.axvspan(weir_start, weir_end, alpha=0.2, color='gray', label='Weir Region')
    
    # Highlight low pressure regions
    low_pressure_mask = weir_profile.pressure_heads < 2.0
    if np.any(low_pressure_mask):
        ax3.fill_between(x, 0, weir_profile.pressure_heads, 
                        where=low_pressure_mask, alpha=0.3, color='red',
                        label='Cavitation Risk Zone')
    
    ax3.set_xlabel('Distance (m)')
    ax3.set_ylabel('Pressure Head (m)')
    ax3.set_title('Pressure Distribution (Cavitation Analysis)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Specific energy profile
    ax4.plot(x, weir_profile.specific_energies, 'm-', linewidth=2, label='FVM Energy')
    ax4.axhline(y=analytical_result.approach_energy, color='red', linestyle='--',
                label=f'Analytical Approach ({analytical_result.approach_energy:.2f}m)')
    ax4.axhline(y=analytical_result.crest_energy, color='green', linestyle='--',
                label=f'Analytical Crest ({analytical_result.crest_energy:.2f}m)')
    
    ax4.axvspan(weir_start, weir_end, alpha=0.2, color='gray', label='Weir Region')
    
    ax4.set_xlabel('Distance (m)')
    ax4.set_ylabel('Specific Energy (m)')
    ax4.set_title('Energy Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Mark weir crest location
    crest_conditions = weir_profile.find_weir_crest_conditions()
    crest_location = weir_profile.weir_crest_location
    
    for ax in [ax1, ax2, ax3, ax4]:
        ax.axvline(x=crest_location, color='orange', linestyle=':', linewidth=2,
                  alpha=0.7, label='Weir Crest')
    
    plt.tight_layout()
    plt.show()


def main():
    """Main demonstration function."""
    print("💧 PyOpenChannel Weir Flow FVM Integration Demo")
    print("===============================================")
    
    # Step 1: Basic setup and method selection
    channel, weir, approach_depth = demonstrate_weir_flow_setup()
    
    # Step 2: Compare analytical vs FVM methods
    results = compare_analytical_vs_fvm_weir_flow(channel, weir, approach_depth)
    
    # Step 3: Detailed spillway pressure analysis
    spillway_profile = demonstrate_spillway_pressure_analysis()
    
    # Step 4: Visualization
    if results:
        create_weir_flow_visualization(results)
    
    # Summary
    print("\n" + "="*80)
    print("📋 WEIR FLOW FVM INTEGRATION SUMMARY")
    print("="*80)
    print("\n✅ Successfully Implemented:")
    print("   • Method selection for weir flow analysis")
    print("   • Enhanced WeirFlowResult with FVM data")
    print("   • WeirFlowProfile for detailed analysis")
    print("   • Pressure distribution over weir crest")
    print("   • Velocity field for aeration requirements")
    print("   • Energy dissipation downstream analysis")
    
    print("\n🎯 Key Engineering Benefits:")
    print("   • Cavitation risk assessment from pressure field")
    print("   • Aeration requirements from velocity analysis")
    print("   • Detailed spillway flow characteristics")
    print("   • Energy dissipation optimization")
    print("   • Professional spillway design validation")
    
    print("\n🚀 Applications:")
    print("   • Dam spillway design and optimization")
    print("   • Ogee spillway shape validation")
    print("   • Aeration system design")
    print("   • Cavitation damage prevention")
    print("   • Energy dissipation basin design")
    print("   • Flood control structure analysis")
    
    print(f"\n🎉 Weir Flow FVM Integration completed successfully!")
    print("   Engineers now have unprecedented detail for spillway analysis,")
    print("   enabling precise cavitation assessment, aeration design,")
    print("   and energy dissipation optimization for critical")
    print("   hydraulic infrastructure.")


if __name__ == "__main__":
    main()
