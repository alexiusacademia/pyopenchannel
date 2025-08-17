#!/usr/bin/env python3
"""
PyOpenChannel - GVF Profile Classification Example
===================================================

This example demonstrates the automatic classification of water surface profiles
using the GVF profile classification system.

Features demonstrated:
- Automatic profile type identification (M1, M2, S1, etc.)
- Slope classification (mild, steep, critical, horizontal, adverse)
- Flow regime analysis (subcritical, supercritical, critical)
- Engineering significance interpretation
- Profile comparison and analysis
- Professional reporting

Author: Alexius Academia
Date: 2025-08-17
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pyopenchannel as poc
from pyopenchannel.gvf import GVFSolver, BoundaryType, ProfileClassifier, ProfileAnalyzer
from pyopenchannel.geometry import RectangularChannel, TrapezoidalChannel
import matplotlib.pyplot as plt


def main():
    """Demonstrate profile classification capabilities."""
    
    print("\n" + "="*75)
    print("PyOpenChannel - GVF Profile Classification Example")
    print("="*75)
    print("Automatic identification and analysis of water surface profiles")
    
    # Set unit system
    poc.set_unit_system(poc.UnitSystem.SI)
    print(f"\n🔧 Unit System: {poc.get_current_unit_system()}")
    
    # Initialize components
    solver = GVFSolver()
    classifier = ProfileClassifier(tolerance=0.15)  # 15% tolerance for classification
    analyzer = ProfileAnalyzer()
    
    print(f"\n🧠 Classification System Initialized:")
    print(f"   • Tolerance: {classifier.tolerance*100:.1f}%")
    print(f"   • Profile types supported: M1, M2, M3, S1, S2, S3, C1, C3, H2, H3, A2, A3")
    print(f"   • Slope types: Mild, Steep, Critical, Horizontal, Adverse")
    
    # =================================================================
    # EXAMPLE 1: COMPREHENSIVE PROFILE ANALYSIS
    # =================================================================
    print("\n" + "="*75)
    print("        EXAMPLE 1: COMPREHENSIVE PROFILE ANALYSIS")
    print("="*75)
    
    profiles_data = []
    
    # Profile 1: M1 (Dam Backwater)
    print(f"\n🔍 Analyzing Profile 1: M1 (Dam Backwater)")
    print(f"   {'─'*50}")
    
    channel1 = RectangularChannel(width=6.0)
    discharge1 = 25.0
    slope1 = 0.0006  # Mild slope
    manning_n = 0.025
    boundary_depth1 = 4.5  # High depth for backwater
    
    try:
        result1 = solver.solve_profile(
            channel=channel1,
            discharge=discharge1,
            slope=slope1,
            manning_n=manning_n,
            boundary_depth=boundary_depth1,
            boundary_type=BoundaryType.UPSTREAM,
            distance=3000.0,
            step_size=15.0
        )
        
        # Classify the profile
        profile1 = classifier.classify_profile(
            gvf_result=result1,
            channel=channel1,
            discharge=discharge1,
            slope=slope1,
            manning_n=manning_n
        )
        
        profiles_data.append(("M1 Dam Backwater", profile1, result1))
        
        print(f"   ✅ Profile computed and classified")
        print(f"   📊 Classification: {profile1.profile_type.value}")
        print(f"   🏔️  Slope type: {profile1.slope_type.value}")
        print(f"   🌊 Flow regime: {profile1.flow_regime.value}")
        print(f"   📏 Length: {profile1.length:.0f} m")
        print(f"   📈 Depth range: {profile1.min_depth:.3f} - {profile1.max_depth:.3f} m")
        
    except Exception as e:
        print(f"   ❌ Profile 1 failed: {e}")
    
    # Profile 2: M2 (Channel Entrance)
    print(f"\n🔍 Analyzing Profile 2: M2 (Channel Entrance)")
    print(f"   {'─'*50}")
    
    # Same channel, different boundary
    boundary_depth2 = 2.8  # Between critical and normal
    
    try:
        result2 = solver.solve_profile(
            channel=channel1,
            discharge=discharge1,
            slope=slope1,
            manning_n=manning_n,
            boundary_depth=boundary_depth2,
            boundary_type=BoundaryType.UPSTREAM,
            distance=1500.0,
            step_size=10.0
        )
        
        profile2 = classifier.classify_profile(
            gvf_result=result2,
            channel=channel1,
            discharge=discharge1,
            slope=slope1,
            manning_n=manning_n
        )
        
        profiles_data.append(("M2 Channel Entrance", profile2, result2))
        
        print(f"   ✅ Profile computed and classified")
        print(f"   📊 Classification: {profile2.profile_type.value}")
        print(f"   🏔️  Slope type: {profile2.slope_type.value}")
        print(f"   🌊 Flow regime: {profile2.flow_regime.value}")
        print(f"   📏 Length: {profile2.length:.0f} m")
        print(f"   📈 Depth range: {profile2.min_depth:.3f} - {profile2.max_depth:.3f} m")
        
    except Exception as e:
        print(f"   ❌ Profile 2 failed: {e}")
    
    # Profile 3: S1 (Steep Channel Backwater)
    print(f"\n🔍 Analyzing Profile 3: S1 (Steep Channel Backwater)")
    print(f"   {'─'*50}")
    
    channel3 = RectangularChannel(width=4.0)
    discharge3 = 12.0
    slope3 = 0.015  # Steep slope
    boundary_depth3 = 1.8  # Above critical depth
    
    try:
        result3 = solver.solve_profile(
            channel=channel3,
            discharge=discharge3,
            slope=slope3,
            manning_n=manning_n,
            boundary_depth=boundary_depth3,
            boundary_type=BoundaryType.UPSTREAM,
            distance=800.0,
            step_size=5.0
        )
        
        profile3 = classifier.classify_profile(
            gvf_result=result3,
            channel=channel3,
            discharge=discharge3,
            slope=slope3,
            manning_n=manning_n
        )
        
        profiles_data.append(("S1 Steep Backwater", profile3, result3))
        
        print(f"   ✅ Profile computed and classified")
        print(f"   📊 Classification: {profile3.profile_type.value}")
        print(f"   🏔️  Slope type: {profile3.slope_type.value}")
        print(f"   🌊 Flow regime: {profile3.flow_regime.value}")
        print(f"   📏 Length: {profile3.length:.0f} m")
        print(f"   📈 Depth range: {profile3.min_depth:.3f} - {profile3.max_depth:.3f} m")
        
    except Exception as e:
        print(f"   ❌ Profile 3 failed: {e}")
    
    # Profile 4: Trapezoidal Channel (Different Geometry)
    print(f"\n🔍 Analyzing Profile 4: Trapezoidal Channel")
    print(f"   {'─'*50}")
    
    channel4 = TrapezoidalChannel(bottom_width=3.0, side_slope=1.5)  # 1.5:1 side slopes
    discharge4 = 18.0
    slope4 = 0.001  # Mild slope
    boundary_depth4 = 3.2  # Moderate backwater
    
    try:
        result4 = solver.solve_profile(
            channel=channel4,
            discharge=discharge4,
            slope=slope4,
            manning_n=manning_n,
            boundary_depth=boundary_depth4,
            boundary_type=BoundaryType.UPSTREAM,
            distance=2000.0,
            step_size=12.0
        )
        
        profile4 = classifier.classify_profile(
            gvf_result=result4,
            channel=channel4,
            discharge=discharge4,
            slope=slope4,
            manning_n=manning_n
        )
        
        profiles_data.append(("Trapezoidal Channel", profile4, result4))
        
        print(f"   ✅ Profile computed and classified")
        print(f"   📊 Classification: {profile4.profile_type.value}")
        print(f"   🏔️  Slope type: {profile4.slope_type.value}")
        print(f"   🌊 Flow regime: {profile4.flow_regime.value}")
        print(f"   📏 Length: {profile4.length:.0f} m")
        print(f"   📈 Depth range: {profile4.min_depth:.3f} - {profile4.max_depth:.3f} m")
        
    except Exception as e:
        print(f"   ❌ Profile 4 failed: {e}")
    
    # =================================================================
    # EXAMPLE 2: DETAILED PROFILE ANALYSIS
    # =================================================================
    print("\n" + "="*75)
    print("           EXAMPLE 2: DETAILED PROFILE ANALYSIS")
    print("="*75)
    
    if profiles_data:
        # Analyze the first profile in detail
        name, profile, result = profiles_data[0]
        
        print(f"\n🔬 Detailed Analysis: {name}")
        print(f"   {'─'*60}")
        
        print(f"\n📊 Profile Classification:")
        print(f"   • Type: {profile.profile_type.value}")
        print(f"   • Slope: {profile.slope_type.value}")
        print(f"   • Flow regime: {profile.flow_regime.value}")
        print(f"   • Normal depth: {profile.normal_depth:.3f} m")
        print(f"   • Critical depth: {profile.critical_depth:.3f} m")
        
        print(f"\n📏 Geometric Properties:")
        print(f"   • Profile length: {profile.length:.1f} m")
        print(f"   • Minimum depth: {profile.min_depth:.3f} m")
        print(f"   • Maximum depth: {profile.max_depth:.3f} m")
        print(f"   • Depth variation: {profile.max_depth - profile.min_depth:.3f} m")
        
        print(f"\n🌊 Hydraulic Characteristics:")
        print(f"   • Curvature: {profile.curvature}")
        print(f"   • Asymptotic behavior: {profile.asymptotic_behavior}")
        
        print(f"\n💡 Engineering Significance:")
        significance_lines = profile.engineering_significance.split('. ')
        for line in significance_lines:
            if line.strip():
                print(f"   • {line.strip()}")
        
        # Plot detailed analysis
        distances = [p.distance for p in result.profile_points]
        depths = [p.depth for p in result.profile_points]
        velocities = [p.velocity for p in result.profile_points]
        froude_numbers = [p.froude_number for p in result.profile_points]
        
        plt.figure(figsize=(15, 10))
        
        # Main profile plot
        plt.subplot(2, 3, 1)
        plt.plot(distances, depths, 'b-', linewidth=2.5, label='Water surface')
        plt.axhline(y=profile.normal_depth, color='g', linestyle='--', 
                   linewidth=2, label=f'Normal depth ({profile.normal_depth:.2f}m)')
        plt.axhline(y=profile.critical_depth, color='r', linestyle='--', 
                   linewidth=2, label=f'Critical depth ({profile.critical_depth:.2f}m)')
        plt.xlabel('Distance (m)')
        plt.ylabel('Depth (m)')
        plt.title(f'{name}\nProfile Type: {profile.profile_type.value}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Velocity distribution
        plt.subplot(2, 3, 2)
        plt.plot(distances, velocities, 'r-', linewidth=2)
        plt.xlabel('Distance (m)')
        plt.ylabel('Velocity (m/s)')
        plt.title('Velocity Distribution')
        plt.grid(True, alpha=0.3)
        
        # Froude number
        plt.subplot(2, 3, 3)
        plt.plot(distances, froude_numbers, 'g-', linewidth=2)
        plt.axhline(y=1.0, color='r', linestyle='--', linewidth=2, label='Critical (Fr=1)')
        plt.xlabel('Distance (m)')
        plt.ylabel('Froude Number')
        plt.title(f'Flow Regime: {profile.flow_regime.value}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Energy analysis
        plt.subplot(2, 3, 4)
        energy_heads = [d + v**2/(2*9.81) for d, v in zip(depths, velocities)]
        plt.plot(distances, energy_heads, 'm-', linewidth=2, label='Energy grade line')
        plt.plot(distances, depths, 'b-', linewidth=1.5, label='Water surface')
        plt.xlabel('Distance (m)')
        plt.ylabel('Head (m)')
        plt.title('Energy Analysis')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Depth vs Distance (zoomed)
        plt.subplot(2, 3, 5)
        plt.plot(distances, depths, 'b-', linewidth=2)
        plt.fill_between(distances, depths, alpha=0.3)
        plt.xlabel('Distance (m)')
        plt.ylabel('Depth (m)')
        plt.title('Water Surface Profile (Detail)')
        plt.grid(True, alpha=0.3)
        
        # Classification summary
        plt.subplot(2, 3, 6)
        plt.axis('off')
        summary_text = f"""
Profile Classification Summary

Type: {profile.profile_type.value}
Slope: {profile.slope_type.value}
Flow: {profile.flow_regime.value}

Depths:
• Normal: {profile.normal_depth:.3f} m
• Critical: {profile.critical_depth:.3f} m
• Range: {profile.min_depth:.3f} - {profile.max_depth:.3f} m

Length: {profile.length:.0f} m
Curvature: {profile.curvature}
        """
        plt.text(0.1, 0.9, summary_text, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(f'profile_classification_{name.lower().replace(" ", "_")}.png', 
                   dpi=300, bbox_inches='tight')
        print(f"\n📊 Detailed analysis plot saved")
    
    # =================================================================
    # EXAMPLE 3: MULTI-PROFILE COMPARISON
    # =================================================================
    print("\n" + "="*75)
    print("          EXAMPLE 3: MULTI-PROFILE COMPARISON")
    print("="*75)
    
    if len(profiles_data) >= 2:
        print(f"\n🔍 Comparing {len(profiles_data)} profiles:")
        
        # Create comparison analysis
        water_surface_profiles = [profile for _, profile, _ in profiles_data]
        comparison = analyzer.compare_profiles(water_surface_profiles)
        
        print(f"\n📊 Comparison Results:")
        print(f"   • Total profiles: {comparison['total_profiles']}")
        print(f"   • Profile types found: {', '.join(comparison['profile_types'])}")
        print(f"   • Slope types: {', '.join(comparison['slope_types'])}")
        print(f"   • Flow regimes: {', '.join(comparison['flow_regimes'])}")
        print(f"   • Length range: {comparison['length_range'][0]:.0f} - {comparison['length_range'][1]:.0f} m")
        print(f"   • Depth range: {comparison['depth_range'][0]:.3f} - {comparison['depth_range'][1]:.3f} m")
        
        print(f"\n📋 Profile Summary Table:")
        print(f"   {'Profile':<20} {'Type':<12} {'Slope':<8} {'Length (m)':<10} {'Max Depth (m)':<12}")
        print(f"   {'-'*70}")
        
        for name, profile, _ in profiles_data:
            print(f"   {name:<20} {profile.profile_type.value:<12} {profile.slope_type.value:<8} "
                  f"{profile.length:<10.0f} {profile.max_depth:<12.3f}")
        
        # Create comparison plot
        plt.figure(figsize=(14, 8))
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i, (name, profile, result) in enumerate(profiles_data):
            distances = [p.distance for p in result.profile_points]
            depths = [p.depth for p in result.profile_points]
            
            plt.plot(distances, depths, color=colors[i % len(colors)], 
                    linewidth=2, label=f'{name} ({profile.profile_type.value})')
        
        plt.xlabel('Distance (m)')
        plt.ylabel('Depth (m)')
        plt.title('Multi-Profile Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('profile_comparison.png', dpi=300, bbox_inches='tight')
        print(f"\n📊 Comparison plot saved as 'profile_comparison.png'")
    
    # =================================================================
    # EXAMPLE 4: ENGINEERING APPLICATIONS
    # =================================================================
    print("\n" + "="*75)
    print("           EXAMPLE 4: ENGINEERING APPLICATIONS")
    print("="*75)
    
    print(f"\n🏗️  Engineering Design Applications:")
    
    for name, profile, result in profiles_data:
        print(f"\n   📋 {name}:")
        print(f"      • Profile type: {profile.profile_type.value}")
        
        if profile.profile_type.value.startswith('M1'):
            print(f"      • Application: Dam backwater analysis")
            print(f"      • Design consideration: Bridge clearance, flood mapping")
            print(f"      • Backwater extent: {profile.length:.0f} m")
            
        elif profile.profile_type.value.startswith('M2'):
            print(f"      • Application: Channel entrance/transition design")
            print(f"      • Design consideration: Energy losses, flow acceleration")
            print(f"      • Transition length: {profile.length:.0f} m")
            
        elif profile.profile_type.value.startswith('S1'):
            print(f"      • Application: Steep channel backwater")
            print(f"      • Design consideration: Hydraulic jump location")
            print(f"      • Backwater influence: {profile.length:.0f} m")
            
        else:
            print(f"      • Application: General hydraulic analysis")
            print(f"      • Design consideration: Flow regime, energy dissipation")
        
        print(f"      • Maximum depth: {profile.max_depth:.3f} m")
        print(f"      • Flow regime: {profile.flow_regime.value}")
    
    print(f"\n🎯 Classification System Benefits:")
    print(f"   • Automatic profile identification saves engineering time")
    print(f"   • Consistent classification reduces human error")
    print(f"   • Engineering significance provides design insights")
    print(f"   • Multi-profile comparison enables optimization")
    print(f"   • Professional reporting supports documentation")
    
    # =================================================================
    # SUMMARY
    # =================================================================
    print("\n" + "="*75)
    print("                     ANALYSIS SUMMARY")
    print("="*75)
    
    print(f"\n✅ Profile Classification Results:")
    successful_profiles = len(profiles_data)
    print(f"   • Profiles analyzed: {successful_profiles}")
    print(f"   • Classification accuracy: Professional grade")
    print(f"   • Engineering insights: Comprehensive")
    
    if successful_profiles > 0:
        profile_types = list(set(profile.profile_type.value for _, profile, _ in profiles_data))
        slope_types = list(set(profile.slope_type.value for _, profile, _ in profiles_data))
        flow_regimes = list(set(profile.flow_regime.value for _, profile, _ in profiles_data))
        
        print(f"   • Profile types identified: {', '.join(profile_types)}")
        print(f"   • Slope types: {', '.join(slope_types)}")
        print(f"   • Flow regimes: {', '.join(flow_regimes)}")
    
    print(f"\n🚀 System Capabilities Demonstrated:")
    print(f"   ✅ Automatic profile type classification")
    print(f"   ✅ Slope and flow regime identification")
    print(f"   ✅ Engineering significance interpretation")
    print(f"   ✅ Multi-profile comparison analysis")
    print(f"   ✅ Professional visualization and reporting")
    print(f"   ✅ Different channel geometries support")
    
    print(f"\n📚 Recommended Next Steps:")
    print(f"   • Experiment with different boundary conditions")
    print(f"   • Try various channel geometries (circular, parabolic)")
    print(f"   • Combine with hydraulic structure analysis")
    print(f"   • Use for real-world engineering projects")
    
    print(f"\n" + "="*75)
    print("Profile Classification Example Completed Successfully!")
    print("="*75)


if __name__ == "__main__":
    main()
