#!/usr/bin/env python3
"""
PyOpenChannel - Dam Backwater Analysis Example
===============================================

This example demonstrates comprehensive dam backwater analysis using GVF methods.
This is a critical application for flood studies, bridge design, and reservoir operations.

Features demonstrated:
- Dam backwater curve computation (M1 profiles)
- Multiple dam scenarios and heights
- Flood elevation mapping
- Bridge clearance analysis
- Sensitivity analysis for different conditions
- Professional engineering reporting

Author: Alexius Academia
Date: 2025-08-17
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pyopenchannel as poc
from pyopenchannel.gvf import GVFSolver, BoundaryType, ProfileClassifier
from pyopenchannel.geometry import RectangularChannel, TrapezoidalChannel
from pyopenchannel.hydraulics import ManningEquation
from pyopenchannel.flow_analysis import CriticalFlow
import matplotlib.pyplot as plt
import numpy as np


def main():
    """Demonstrate comprehensive dam backwater analysis."""
    
    print("\n" + "="*80)
    print("PyOpenChannel - Dam Backwater Analysis Example")
    print("="*80)
    print("Comprehensive analysis of dam backwater effects for engineering design")
    
    # Set unit system
    poc.set_unit_system(poc.UnitSystem.SI)
    print(f"\n🔧 Unit System: {poc.get_current_unit_system()}")
    
    # Initialize components
    solver = GVFSolver()
    classifier = ProfileClassifier(tolerance=0.12)
    
    # =================================================================
    # PROJECT SETUP: RIVER CHANNEL AND DAM CONFIGURATION
    # =================================================================
    print("\n" + "="*80)
    print("                    PROJECT SETUP")
    print("="*80)
    
    # River channel configuration
    channel = TrapezoidalChannel(bottom_width=15.0, side_slope=2.0)  # 2:1 side slopes
    discharge = 150.0  # m³/s - design flood discharge
    slope = 0.0004     # 0.04% - mild slope typical of rivers
    manning_n = 0.035  # Natural channel with some vegetation
    
    print(f"\n🏞️  River Channel Configuration:")
    print(f"   • Type: Trapezoidal")
    print(f"   • Bottom width: {channel.bottom_width} m")
    print(f"   • Side slopes: {channel.side_slope}:1 (H:V)")
    print(f"   • Design discharge: {discharge} m³/s")
    print(f"   • Channel slope: {slope:.4f} ({slope*100:.2f}%)")
    print(f"   • Manning's n: {manning_n}")
    
    # Calculate reference conditions
    manning = ManningEquation(channel)
    critical_flow = CriticalFlow(channel)
    
    normal_depth = manning.calculate_normal_depth(discharge, slope, manning_n)
    critical_depth = critical_flow.calculate_critical_depth(discharge)
    normal_velocity = discharge / channel.area(normal_depth)
    
    print(f"\n📊 Reference Hydraulic Conditions:")
    print(f"   • Normal depth (yn): {normal_depth:.3f} m")
    print(f"   • Critical depth (yc): {critical_depth:.3f} m")
    print(f"   • Normal velocity: {normal_velocity:.3f} m/s")
    print(f"   • Channel type: {'Mild slope' if normal_depth > critical_depth else 'Steep slope'}")
    print(f"   • Normal flow area: {channel.area(normal_depth):.2f} m²")
    print(f"   • Normal top width: {channel.top_width(normal_depth):.2f} m")
    
    # Dam configurations to analyze
    dam_scenarios = [
        {"name": "Low Dam", "height": 2.0, "description": "Small irrigation dam"},
        {"name": "Medium Dam", "height": 4.0, "description": "Flood control structure"},
        {"name": "High Dam", "height": 6.0, "description": "Major reservoir dam"},
        {"name": "Extreme Dam", "height": 8.0, "description": "Large hydroelectric dam"}
    ]
    
    print(f"\n🏗️  Dam Scenarios for Analysis:")
    for i, scenario in enumerate(dam_scenarios, 1):
        print(f"   {i}. {scenario['name']}: {scenario['height']} m - {scenario['description']}")
    
    # =================================================================
    # BACKWATER ANALYSIS FOR EACH DAM SCENARIO
    # =================================================================
    print("\n" + "="*80)
    print("                 BACKWATER ANALYSIS")
    print("="*80)
    
    backwater_results = []
    
    for scenario in dam_scenarios:
        dam_name = scenario['name']
        dam_height = scenario['height']
        
        print(f"\n🔍 Analyzing {dam_name} (Height: {dam_height} m)")
        print(f"   {'─'*60}")
        
        # Dam creates a boundary condition - water depth at dam
        dam_depth = normal_depth + dam_height  # Simplified: normal depth + dam height
        
        print(f"   • Dam height: {dam_height} m")
        print(f"   • Water depth at dam: {dam_depth:.3f} m")
        print(f"   • Backwater above normal: {dam_depth - normal_depth:.3f} m")
        
        try:
            # Solve backwater profile
            result = solver.solve_profile(
                channel=channel,
                discharge=discharge,
                slope=slope,
                manning_n=manning_n,
                boundary_depth=dam_depth,
                boundary_type=BoundaryType.UPSTREAM,
                distance=10000.0,  # Analyze 10 km upstream
                step_size=50.0     # 50 m steps for river analysis
            )
            
            # Classify the profile
            profile = classifier.classify_profile(
                gvf_result=result,
                channel=channel,
                discharge=discharge,
                slope=slope,
                manning_n=manning_n
            )
            
            # Extract profile data
            distances = [p.distance for p in result.profile_points]
            depths = [p.depth for p in result.profile_points]
            velocities = [p.velocity for p in result.profile_points]
            water_levels = [d for d in depths]  # Assuming channel bottom at elevation 0
            
            # Find backwater extent (where depth approaches normal depth)
            backwater_extent = 0
            for i, depth in enumerate(depths):
                if abs(depth - normal_depth) / normal_depth < 0.05:  # Within 5% of normal
                    backwater_extent = distances[i]
                    break
            if backwater_extent == 0:
                backwater_extent = max(distances)  # Backwater extends beyond analysis distance
            
            # Store results
            backwater_results.append({
                'name': dam_name,
                'height': dam_height,
                'result': result,
                'profile': profile,
                'distances': distances,
                'depths': depths,
                'velocities': velocities,
                'water_levels': water_levels,
                'backwater_extent': backwater_extent,
                'max_depth': max(depths),
                'dam_depth': dam_depth
            })
            
            print(f"   ✅ Backwater analysis successful")
            print(f"   📊 Profile type: {profile.profile_type.value}")
            print(f"   📏 Backwater extent: {backwater_extent/1000:.1f} km")
            print(f"   📈 Maximum depth: {max(depths):.3f} m")
            print(f"   🌊 Depth increase at dam: {max(depths) - normal_depth:.3f} m")
            print(f"   ⚡ Velocity range: {min(velocities):.3f} - {max(velocities):.3f} m/s")
            
        except Exception as e:
            print(f"   ❌ Analysis failed: {e}")
    
    # =================================================================
    # FLOOD ELEVATION MAPPING
    # =================================================================
    print("\n" + "="*80)
    print("                FLOOD ELEVATION MAPPING")
    print("="*80)
    
    if backwater_results:
        print(f"\n🗺️  Flood Elevation Analysis:")
        
        # Create comprehensive flood elevation plot
        plt.figure(figsize=(16, 12))
        
        # Main backwater profiles
        plt.subplot(2, 2, 1)
        colors = ['blue', 'green', 'orange', 'red']
        
        for i, result in enumerate(backwater_results):
            plt.plot(np.array(result['distances'])/1000, result['depths'], 
                    color=colors[i], linewidth=2.5, 
                    label=f"{result['name']} ({result['height']}m)")
        
        plt.axhline(y=normal_depth, color='black', linestyle='--', linewidth=2, 
                   label=f'Normal depth ({normal_depth:.2f}m)')
        plt.axhline(y=critical_depth, color='red', linestyle=':', linewidth=2, 
                   label=f'Critical depth ({critical_depth:.2f}m)')
        
        plt.xlabel('Distance upstream (km)')
        plt.ylabel('Water depth (m)')
        plt.title('Dam Backwater Profiles - Flood Elevation Mapping')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Backwater extent comparison
        plt.subplot(2, 2, 2)
        dam_heights = [r['height'] for r in backwater_results]
        extents = [r['backwater_extent']/1000 for r in backwater_results]
        max_depths = [r['max_depth'] for r in backwater_results]
        
        plt.bar(range(len(dam_heights)), extents, color='lightblue', alpha=0.7, 
               label='Backwater extent')
        plt.xlabel('Dam Scenario')
        plt.ylabel('Backwater extent (km)')
        plt.title('Backwater Extent vs Dam Height')
        plt.xticks(range(len(dam_heights)), [r['name'] for r in backwater_results], rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Maximum depth vs dam height
        plt.subplot(2, 2, 3)
        plt.plot(dam_heights, max_depths, 'ro-', linewidth=2, markersize=8)
        plt.axhline(y=normal_depth, color='black', linestyle='--', alpha=0.7, 
                   label='Normal depth')
        plt.xlabel('Dam height (m)')
        plt.ylabel('Maximum water depth (m)')
        plt.title('Maximum Flood Depth vs Dam Height')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Velocity analysis
        plt.subplot(2, 2, 4)
        for i, result in enumerate(backwater_results):
            plt.plot(np.array(result['distances'])/1000, result['velocities'], 
                    color=colors[i], linewidth=2, alpha=0.8,
                    label=f"{result['name']}")
        
        plt.xlabel('Distance upstream (km)')
        plt.ylabel('Velocity (m/s)')
        plt.title('Velocity Distribution in Backwater')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('dam_backwater_analysis.png', dpi=300, bbox_inches='tight')
        print(f"   📊 Flood elevation maps saved as 'dam_backwater_analysis.png'")
    
    # =================================================================
    # BRIDGE CLEARANCE ANALYSIS
    # =================================================================
    print("\n" + "="*80)
    print("               BRIDGE CLEARANCE ANALYSIS")
    print("="*80)
    
    # Bridge locations to analyze (distances upstream from dam)
    bridge_locations = [500, 1000, 2000, 5000]  # meters
    bridge_clearance_required = 2.0  # meters minimum clearance
    
    print(f"\n🌉 Bridge Clearance Requirements:")
    print(f"   • Minimum clearance required: {bridge_clearance_required} m")
    print(f"   • Bridge locations analyzed: {', '.join([f'{loc/1000:.1f}km' for loc in bridge_locations])}")
    
    print(f"\n📋 Bridge Clearance Analysis Results:")
    print(f"   {'Location (km)':<12} {'Low Dam':<10} {'Medium Dam':<12} {'High Dam':<10} {'Extreme Dam':<12}")
    print(f"   {'-'*70}")
    
    for location in bridge_locations:
        location_km = location / 1000
        clearances = []
        
        for result in backwater_results:
            # Find water depth at bridge location
            bridge_depth = normal_depth  # Default to normal depth
            for i, dist in enumerate(result['distances']):
                if dist >= location:
                    bridge_depth = result['depths'][i]
                    break
            
            # Calculate required bridge elevation
            bridge_elevation = bridge_depth + bridge_clearance_required
            clearances.append(bridge_elevation)
        
        print(f"   {location_km:<12.1f} {clearances[0]:<10.2f} {clearances[1]:<12.2f} "
              f"{clearances[2]:<10.2f} {clearances[3]:<12.2f}")
    
    # =================================================================
    # SENSITIVITY ANALYSIS
    # =================================================================
    print("\n" + "="*80)
    print("                SENSITIVITY ANALYSIS")
    print("="*80)
    
    print(f"\n🔬 Sensitivity Analysis - Effect of Channel Parameters:")
    
    # Test different Manning's n values
    manning_values = [0.025, 0.030, 0.035, 0.040]  # Different roughness conditions
    base_dam_height = 4.0  # Use medium dam for sensitivity
    
    print(f"\n📊 Manning's n Sensitivity (Dam height: {base_dam_height} m):")
    print(f"   {'Manning n':<10} {'Normal Depth (m)':<15} {'Backwater Extent (km)':<20} {'Max Depth (m)':<12}")
    print(f"   {'-'*65}")
    
    for n_value in manning_values:
        try:
            # Recalculate normal depth for this Manning's n
            n_normal_depth = manning.calculate_normal_depth(discharge, slope, n_value)
            n_dam_depth = n_normal_depth + base_dam_height
            
            # Solve backwater profile
            n_result = solver.solve_profile(
                channel=channel,
                discharge=discharge,
                slope=slope,
                manning_n=n_value,
                boundary_depth=n_dam_depth,
                boundary_type=BoundaryType.UPSTREAM,
                distance=8000.0,
                step_size=100.0
            )
            
            n_depths = [p.depth for p in n_result.profile_points]
            n_distances = [p.distance for p in n_result.profile_points]
            
            # Find backwater extent
            n_backwater_extent = 0
            for i, depth in enumerate(n_depths):
                if abs(depth - n_normal_depth) / n_normal_depth < 0.05:
                    n_backwater_extent = n_distances[i]
                    break
            if n_backwater_extent == 0:
                n_backwater_extent = max(n_distances)
            
            print(f"   {n_value:<10.3f} {n_normal_depth:<15.3f} {n_backwater_extent/1000:<20.1f} {max(n_depths):<12.3f}")
            
        except Exception as e:
            print(f"   {n_value:<10.3f} {'Failed':<15} {'Failed':<20} {'Failed':<12}")
    
    # =================================================================
    # ENGINEERING RECOMMENDATIONS
    # =================================================================
    print("\n" + "="*80)
    print("             ENGINEERING RECOMMENDATIONS")
    print("="*80)
    
    if backwater_results:
        print(f"\n🎯 Design Recommendations:")
        
        # Find the most critical scenario
        max_extent_result = max(backwater_results, key=lambda x: x['backwater_extent'])
        max_depth_result = max(backwater_results, key=lambda x: x['max_depth'])
        
        print(f"\n   🔴 Critical Scenarios:")
        print(f"      • Greatest backwater extent: {max_extent_result['name']} "
              f"({max_extent_result['backwater_extent']/1000:.1f} km)")
        print(f"      • Highest flood elevation: {max_depth_result['name']} "
              f"({max_depth_result['max_depth']:.2f} m)")
        
        print(f"\n   📏 Flood Zone Recommendations:")
        for result in backwater_results:
            flood_zone = result['backwater_extent'] / 1000
            print(f"      • {result['name']}: Flood zone extends {flood_zone:.1f} km upstream")
            if flood_zone > 5:
                print(f"        ⚠️  Significant impact - detailed environmental assessment required")
            elif flood_zone > 2:
                print(f"        ⚡ Moderate impact - consider mitigation measures")
            else:
                print(f"        ✅ Limited impact - standard design acceptable")
        
        print(f"\n   🌉 Bridge Design Guidelines:")
        print(f"      • Minimum bridge elevation should account for highest dam scenario")
        print(f"      • Consider {bridge_clearance_required} m clearance above design flood level")
        print(f"      • Bridge locations within {max_extent_result['backwater_extent']/1000:.1f} km "
              f"require backwater analysis")
        
        print(f"\n   🏞️  Environmental Considerations:")
        print(f"      • Wetland impacts may extend {max_extent_result['backwater_extent']/1000:.1f} km upstream")
        print(f"      • Fish passage requirements for dams > 2 m height")
        print(f"      • Sediment deposition patterns will change in backwater zone")
        
        print(f"\n   ⚖️  Regulatory Compliance:")
        print(f"      • Floodplain mapping required for all scenarios")
        print(f"      • Environmental impact assessment for backwater > 2 km")
        print(f"      • Dam safety analysis for structures > 4 m height")
    
    # =================================================================
    # SUMMARY REPORT
    # =================================================================
    print("\n" + "="*80)
    print("                    SUMMARY REPORT")
    print("="*80)
    
    print(f"\n📊 Dam Backwater Analysis Summary:")
    print(f"   • Channel analyzed: {channel.bottom_width} m trapezoidal, {channel.side_slope}:1 slopes")
    print(f"   • Design discharge: {discharge} m³/s")
    print(f"   • Channel slope: {slope*100:.2f}% (mild slope)")
    print(f"   • Normal depth: {normal_depth:.3f} m")
    
    if backwater_results:
        print(f"\n   🏗️  Dam Scenarios Analyzed: {len(backwater_results)}")
        print(f"   📏 Backwater extent range: {min(r['backwater_extent'] for r in backwater_results)/1000:.1f} - "
              f"{max(r['backwater_extent'] for r in backwater_results)/1000:.1f} km")
        print(f"   📈 Flood elevation range: {min(r['max_depth'] for r in backwater_results):.2f} - "
              f"{max(r['max_depth'] for r in backwater_results):.2f} m")
        
        print(f"\n   ✅ Analysis Capabilities Demonstrated:")
        print(f"      • M1 profile computation and classification")
        print(f"      • Flood elevation mapping")
        print(f"      • Bridge clearance analysis")
        print(f"      • Sensitivity analysis for design parameters")
        print(f"      • Professional engineering recommendations")
    
    print(f"\n🚀 Professional Applications:")
    print(f"   • Flood risk assessment and mapping")
    print(f"   • Dam design and safety analysis")
    print(f"   • Bridge hydraulics and clearance design")
    print(f"   • Environmental impact assessment")
    print(f"   • Regulatory compliance documentation")
    
    print(f"\n" + "="*80)
    print("Dam Backwater Analysis Example Completed Successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
