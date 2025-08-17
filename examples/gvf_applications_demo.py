#!/usr/bin/env python3
"""
PyOpenChannel - GVF Applications Demo
=====================================

This example demonstrates the high-level GVF applications module, which provides
pre-built solutions for common hydraulic engineering scenarios.

Applications demonstrated:
- DamAnalysis: Comprehensive backwater analysis
- BridgeAnalysis: Bridge hydraulics and clearance design
- ChuteAnalysis: Steep channel energy dissipation
- ChannelTransition: Geometry and slope transitions

Author: Alexius Academia
Date: 2025-08-17
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pyopenchannel as poc
from pyopenchannel.gvf import (
    DamAnalysis, BridgeAnalysis, ChuteAnalysis, ChannelTransition,
    DesignCriteria, AnalysisType
)
from pyopenchannel.geometry import RectangularChannel, TrapezoidalChannel


def main():
    """Demonstrate GVF applications module."""
    
    print("PyOpenChannel - GVF Applications Demo")
    print("=" * 60)
    print("High-level engineering applications for hydraulic design")
    
    # Set unit system
    poc.set_unit_system(poc.UnitSystem.SI)
    
    # =================================================================
    # APPLICATION 1: DAM BACKWATER ANALYSIS
    # =================================================================
    print(f"\n" + "=" * 60)
    print("APPLICATION 1: DAM BACKWATER ANALYSIS")
    print("=" * 60)
    
    # Initialize dam analysis with standard design criteria
    dam_analyzer = DamAnalysis(design_criteria=DesignCriteria.STANDARD)
    
    # Define river channel and conditions
    river_channel = TrapezoidalChannel(bottom_width=12.0, side_slope=2.0)
    discharge = 100.0    # m³/s - design flood
    slope = 0.0005       # 0.05% - mild slope
    manning_n = 0.035    # Natural channel
    dam_height = 3.5     # m - proposed dam height
    
    # Bridge locations for clearance analysis
    bridge_locations = [1000.0, 2500.0, 5000.0]  # meters upstream
    
    print(f"🏞️  River Configuration:")
    print(f"   • Channel: Trapezoidal, {river_channel.bottom_width}m bottom, {river_channel.side_slope}:1 slopes")
    print(f"   • Design discharge: {discharge} m³/s")
    print(f"   • Channel slope: {slope:.4f} ({slope*100:.2f}%)")
    print(f"   • Manning's n: {manning_n}")
    print(f"   • Proposed dam height: {dam_height} m")
    
    try:
        # Perform comprehensive dam analysis
        dam_result = dam_analyzer.analyze_backwater(
            channel=river_channel,
            discharge=discharge,
            slope=slope,
            manning_n=manning_n,
            dam_height=dam_height,
            analysis_distance=8000.0,  # 8km upstream analysis
            bridge_locations=bridge_locations
        )
        
        if dam_result.success:
            print(f"\n✅ Dam Analysis Results:")
            print(f"   • Analysis type: {dam_result.analysis_type.value}")
            print(f"   • Profile type: {dam_result.profile.profile_type.value}")
            print(f"   • Flow regime: {dam_result.profile.flow_regime.value}")
            
            params = dam_result.design_parameters
            print(f"\n📊 Key Design Parameters:")
            print(f"   • Normal depth: {params['normal_depth']:.3f} m")
            print(f"   • Dam depth: {params['dam_depth']:.3f} m")
            print(f"   • Maximum backwater: {params['max_backwater']:.3f} m")
            print(f"   • Backwater extent: {params['backwater_extent']/1000:.1f} km")
            print(f"   • Flood elevation: {params['flood_elevation']:.3f} m")
            
            # Bridge clearances
            if 'bridge_clearances' in params:
                print(f"\n🌉 Bridge Clearance Requirements:")
                for loc, data in params['bridge_clearances'].items():
                    print(f"   • Bridge at {loc/1000:.1f}km: {data['required_elevation']:.2f} m elevation")
            
            print(f"\n💡 Engineering Recommendations:")
            for rec in dam_result.recommendations[:3]:  # Show first 3
                print(f"   • {rec}")
            
            if dam_result.warnings:
                print(f"\n⚠️  Design Warnings:")
                for warning in dam_result.warnings:
                    print(f"   • {warning}")
        
        else:
            print(f"❌ Dam analysis failed: {dam_result.message}")
    
    except Exception as e:
        print(f"❌ Dam analysis error: {e}")
    
    # =================================================================
    # APPLICATION 2: BRIDGE HYDRAULIC ANALYSIS
    # =================================================================
    print(f"\n" + "=" * 60)
    print("APPLICATION 2: BRIDGE HYDRAULIC ANALYSIS")
    print("=" * 60)
    
    # Initialize bridge analysis
    bridge_analyzer = BridgeAnalysis(design_criteria=DesignCriteria.STANDARD)
    
    # Define bridge configuration
    approach_channel = RectangularChannel(width=8.0)   # 8m wide approach
    bridge_opening = RectangularChannel(width=6.0)     # 6m wide bridge opening
    bridge_discharge = 50.0  # m³/s
    bridge_slope = 0.001     # 0.1%
    bridge_manning = 0.030   # Concrete channel
    
    print(f"🌉 Bridge Configuration:")
    print(f"   • Approach channel: {approach_channel.width}m wide")
    print(f"   • Bridge opening: {bridge_opening.width}m wide")
    print(f"   • Design discharge: {bridge_discharge} m³/s")
    print(f"   • Channel slope: {bridge_slope:.3f} ({bridge_slope*100:.1f}%)")
    print(f"   • Contraction ratio: {bridge_opening.width/approach_channel.width:.2f}")
    
    try:
        # Perform bridge hydraulic analysis
        bridge_result = bridge_analyzer.analyze_bridge_hydraulics(
            approach_channel=approach_channel,
            bridge_opening=bridge_opening,
            discharge=bridge_discharge,
            slope=bridge_slope,
            manning_n=bridge_manning,
            analysis_distance=1500.0
        )
        
        if bridge_result.success:
            print(f"\n✅ Bridge Analysis Results:")
            print(f"   • Profile type: {bridge_result.profile.profile_type.value}")
            
            params = bridge_result.design_parameters
            print(f"\n📊 Hydraulic Parameters:")
            print(f"   • Approach normal depth: {params['approach_normal_depth']:.3f} m")
            print(f"   • Bridge depth: {params['bridge_depth']:.3f} m")
            print(f"   • Maximum upstream depth: {params['max_upstream_depth']:.3f} m")
            print(f"   • Backwater rise: {params['backwater_rise']:.3f} m")
            print(f"   • Bridge Froude number: {params['froude_number_bridge']:.3f}")
            
            print(f"\n🔧 Design Requirements:")
            print(f"   • Required clearance: {params['required_clearance']:.2f} m")
            print(f"   • Estimated scour depth: {params['estimated_scour_depth']:.2f} m")
            
            print(f"\n💡 Design Recommendations:")
            for rec in bridge_result.recommendations[:3]:
                print(f"   • {rec}")
        
        else:
            print(f"❌ Bridge analysis failed: {bridge_result.message}")
    
    except Exception as e:
        print(f"❌ Bridge analysis error: {e}")
    
    # =================================================================
    # APPLICATION 3: CHUTE ENERGY DISSIPATION ANALYSIS
    # =================================================================
    print(f"\n" + "=" * 60)
    print("APPLICATION 3: CHUTE ENERGY DISSIPATION")
    print("=" * 60)
    
    # Initialize chute analysis
    chute_analyzer = ChuteAnalysis(design_criteria=DesignCriteria.STANDARD)
    
    # Define chute configuration
    chute_channel = RectangularChannel(width=4.0)      # 4m wide chute
    tailwater_channel = RectangularChannel(width=5.0)  # 5m wide tailwater
    chute_discharge = 25.0   # m³/s
    chute_slope = 0.08       # 8% - steep chute
    tailwater_slope = 0.002  # 0.2% - mild tailwater
    chute_manning = 0.025    # Smooth concrete
    
    print(f"🏔️  Chute Configuration:")
    print(f"   • Chute channel: {chute_channel.width}m wide")
    print(f"   • Chute slope: {chute_slope:.3f} ({chute_slope*100:.1f}%)")
    print(f"   • Tailwater channel: {tailwater_channel.width}m wide")
    print(f"   • Tailwater slope: {tailwater_slope:.3f} ({tailwater_slope*100:.1f}%)")
    print(f"   • Design discharge: {chute_discharge} m³/s")
    
    try:
        # Perform chute analysis
        chute_result = chute_analyzer.analyze_chute(
            chute_channel=chute_channel,
            tailwater_channel=tailwater_channel,
            discharge=chute_discharge,
            chute_slope=chute_slope,
            tailwater_slope=tailwater_slope,
            manning_n=chute_manning,
            chute_length=200.0
        )
        
        if chute_result.success:
            print(f"\n✅ Chute Analysis Results:")
            print(f"   • Profile type: {chute_result.profile.profile_type.value}")
            
            params = chute_result.design_parameters
            print(f"\n📊 Energy Dissipation Parameters:")
            print(f"   • Exit depth: {params['exit_depth']:.3f} m")
            print(f"   • Exit velocity: {params['exit_velocity']:.2f} m/s")
            print(f"   • Exit Froude number: {params['exit_froude']:.2f}")
            print(f"   • Energy dissipated: {params['energy_dissipated']:.2f} m")
            
            if params['jump_required']:
                print(f"\n🌊 Hydraulic Jump Design:")
                print(f"   • Jump required: Yes")
                print(f"   • Sequent depth: {params['sequent_depth']:.3f} m")
                print(f"   • Jump length: {params['jump_length']:.1f} m")
            else:
                print(f"\n🌊 Hydraulic Jump: Not required")
            
            print(f"\n💡 Design Recommendations:")
            for rec in chute_result.recommendations[:3]:
                print(f"   • {rec}")
        
        else:
            print(f"❌ Chute analysis failed: {chute_result.message}")
    
    except Exception as e:
        print(f"❌ Chute analysis error: {e}")
    
    # =================================================================
    # APPLICATION 4: CHANNEL TRANSITION ANALYSIS
    # =================================================================
    print(f"\n" + "=" * 60)
    print("APPLICATION 4: CHANNEL TRANSITION ANALYSIS")
    print("=" * 60)
    
    # Initialize transition analysis
    transition_analyzer = ChannelTransition(design_criteria=DesignCriteria.STANDARD)
    
    # Define transition configuration
    upstream_channel = TrapezoidalChannel(bottom_width=6.0, side_slope=2.0)
    downstream_channel = RectangularChannel(width=5.0)
    transition_discharge = 30.0  # m³/s
    upstream_slope = 0.0008      # 0.08%
    downstream_slope = 0.0012    # 0.12%
    transition_manning = 0.030
    
    print(f"🔄 Transition Configuration:")
    print(f"   • Upstream: Trapezoidal ({upstream_channel.bottom_width}m bottom, {upstream_channel.side_slope}:1)")
    print(f"   • Downstream: Rectangular ({downstream_channel.width}m wide)")
    print(f"   • Upstream slope: {upstream_slope:.4f} ({upstream_slope*100:.2f}%)")
    print(f"   • Downstream slope: {downstream_slope:.4f} ({downstream_slope*100:.2f}%)")
    print(f"   • Discharge: {transition_discharge} m³/s")
    
    try:
        # Perform transition analysis
        transition_result = transition_analyzer.analyze_transition(
            upstream_channel=upstream_channel,
            downstream_channel=downstream_channel,
            discharge=transition_discharge,
            upstream_slope=upstream_slope,
            downstream_slope=downstream_slope,
            manning_n=transition_manning,
            transition_length=50.0
        )
        
        if transition_result.success:
            print(f"\n✅ Transition Analysis Results:")
            print(f"   • Profile type: {transition_result.profile.profile_type.value}")
            
            params = transition_result.design_parameters
            print(f"\n📊 Transition Parameters:")
            print(f"   • Upstream normal: {params['upstream_normal']:.3f} m")
            print(f"   • Downstream normal: {params['downstream_normal']:.3f} m")
            print(f"   • Transition depth: {params['transition_depth']:.3f} m")
            print(f"   • Energy loss: {params['energy_loss']:.3f} m")
            print(f"   • Area ratio: {params['contraction_ratio']:.3f}")
            print(f"   • Slope change: {params['slope_change']:.4f}")
            
            print(f"\n💡 Design Recommendations:")
            for rec in transition_result.recommendations:
                print(f"   • {rec}")
        
        else:
            print(f"❌ Transition analysis failed: {transition_result.message}")
    
    except Exception as e:
        print(f"❌ Transition analysis error: {e}")
    
    # =================================================================
    # SUMMARY OF APPLICATIONS MODULE
    # =================================================================
    print(f"\n" + "=" * 60)
    print("APPLICATIONS MODULE SUMMARY")
    print("=" * 60)
    
    print(f"\n🚀 GVF Applications Demonstrated:")
    print(f"   ✅ DamAnalysis - Comprehensive backwater studies")
    print(f"   ✅ BridgeAnalysis - Hydraulic design and clearance")
    print(f"   ✅ ChuteAnalysis - Energy dissipation and hydraulic jumps")
    print(f"   ✅ ChannelTransition - Geometry and slope changes")
    
    print(f"\n🎯 Key Benefits:")
    print(f"   • High-level engineering interfaces")
    print(f"   • Domain-specific design knowledge")
    print(f"   • Comprehensive analysis results")
    print(f"   • Professional recommendations")
    print(f"   • Regulatory compliance support")
    print(f"   • Multiple design criteria levels")
    
    print(f"\n🔧 Design Criteria Options:")
    print(f"   • Conservative: Maximum safety factors")
    print(f"   • Standard: Typical engineering practice")
    print(f"   • Optimized: Efficient design with adequate safety")
    
    print(f"\n📊 Analysis Results Include:")
    print(f"   • Detailed hydraulic parameters")
    print(f"   • Engineering recommendations")
    print(f"   • Design warnings and considerations")
    print(f"   • Regulatory compliance notes")
    print(f"   • Professional documentation support")
    
    print(f"\n🏗️  Professional Applications:")
    print(f"   • Flood risk assessment and dam design")
    print(f"   • Bridge hydraulics and infrastructure")
    print(f"   • Spillway and energy dissipation design")
    print(f"   • Channel modification and optimization")
    print(f"   • Environmental impact assessment")
    print(f"   • Regulatory compliance documentation")
    
    print(f"\n" + "=" * 60)
    print("GVF Applications Demo Completed Successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
