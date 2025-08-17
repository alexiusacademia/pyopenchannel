#!/usr/bin/env python3
"""
RVF Hydraulic Jump Analysis Example - PyOpenChannel

Author: Alexius Academia
Email: alexius.sayco.academia@gmail.com

This example demonstrates comprehensive hydraulic jump analysis using the
RVF (Rapidly Varied Flow) system. It covers:

1. Basic hydraulic jump analysis
2. Jump type classification (undular, weak, oscillating, steady, strong)
3. Energy dissipation calculations
4. Tailwater adequacy analysis
5. Stilling basin design recommendations
6. Comparison of different upstream conditions
7. Professional engineering insights

Key RVF Features Demonstrated:
- RVFSolver for hydraulic jump analysis
- Jump classification based on Froude numbers
- Energy-momentum balance calculations
- Professional design recommendations
- Multi-scenario comparison
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pyopenchannel as poc
import math


def basic_hydraulic_jump_analysis():
    """Demonstrate basic hydraulic jump analysis."""
    
    print("🌊 BASIC HYDRAULIC JUMP ANALYSIS")
    print("=" * 70)
    
    # Set up conditions
    poc.set_unit_system(poc.UnitSystem.SI)
    channel = poc.RectangularChannel(width=8.0)
    discharge = 40.0
    upstream_depth = 1.2  # Supercritical depth
    tailwater_depth = 3.5  # Available tailwater
    
    print(f"📏 Channel: {channel.width}m wide rectangular")
    print(f"💧 Discharge: {discharge} m³/s")
    print(f"⬆️  Upstream Depth: {upstream_depth} m (supercritical)")
    print(f"⬇️  Available Tailwater: {tailwater_depth} m")
    
    # Initialize RVF solver
    rvf_solver = poc.RVFSolver()
    
    # Analyze hydraulic jump
    result = rvf_solver.analyze_hydraulic_jump(
        channel=channel,
        discharge=discharge,
        upstream_depth=upstream_depth,
        tailwater_depth=tailwater_depth
    )
    
    if result.success:
        print(f"\n✅ HYDRAULIC JUMP ANALYSIS RESULTS:")
        print(f"   Jump Type: {result.jump_type.value.upper()}")
        print(f"   Upstream Froude Number: {result.upstream_froude:.2f}")
        print(f"   Downstream Froude Number: {result.downstream_froude:.2f}")
        print(f"   Sequent Depth: {result.downstream_depth:.3f} m")
        print(f"   Jump Height: {result.jump_height:.3f} m")
        print(f"   Jump Length: {result.jump_length:.1f} m")
        
        print(f"\n⚡ ENERGY ANALYSIS:")
        print(f"   Upstream Energy: {result.upstream_energy:.3f} m")
        print(f"   Downstream Energy: {result.downstream_energy:.3f} m")
        print(f"   Energy Loss: {result.energy_loss:.3f} m")
        print(f"   Energy Efficiency: {result.energy_efficiency:.1%}")
        print(f"   Energy Dissipated: {(1-result.energy_efficiency)*100:.1f}%")
        
        print(f"\n🔧 ENGINEERING PARAMETERS:")
        print(f"   Sequent Depth Ratio: {result.sequent_depth_ratio:.2f}")
        print(f"   Momentum Change: {result.momentum_change:.2f} m³")
        
        # Tailwater adequacy
        tailwater_adequate = result.properties.get("tailwater_adequate", False)
        print(f"   Tailwater Adequate: {'✅ YES' if tailwater_adequate else '❌ NO'}")
        
        if not tailwater_adequate:
            print(f"   ⚠️  Required Tailwater: {result.downstream_depth:.3f} m")
            print(f"   📊 Tailwater Deficit: {result.downstream_depth - tailwater_depth:.3f} m")
        
    else:
        print(f"❌ Analysis Failed: {result.message}")
    
    return result


def jump_type_classification_demo():
    """Demonstrate different hydraulic jump types."""
    
    print("\n\n🏷️  HYDRAULIC JUMP TYPE CLASSIFICATION")
    print("=" * 70)
    
    channel = poc.RectangularChannel(width=6.0)
    discharge = 25.0
    rvf_solver = poc.RVFSolver()
    
    # Different upstream depths to create different Froude numbers
    test_conditions = [
        {"depth": 2.0, "description": "Mild supercritical"},
        {"depth": 1.5, "description": "Moderate supercritical"},
        {"depth": 1.0, "description": "Strong supercritical"},
        {"depth": 0.7, "description": "Very strong supercritical"},
        {"depth": 0.5, "description": "Extreme supercritical"}
    ]
    
    print("Upstream | Froude | Jump    | Sequent | Energy  | Jump")
    print("Depth    | Number | Type    | Depth   | Loss    | Length")
    print("---------|--------|---------|---------|---------|--------")
    
    for condition in test_conditions:
        result = rvf_solver.analyze_hydraulic_jump(
            channel=channel,
            discharge=discharge,
            upstream_depth=condition["depth"]
        )
        
        if result.success:
            print(f"{result.upstream_depth:6.2f}   | "
                  f"{result.upstream_froude:6.2f} | "
                  f"{result.jump_type.value:7s} | "
                  f"{result.downstream_depth:7.3f} | "
                  f"{result.energy_loss:7.3f} | "
                  f"{result.jump_length:6.1f}")
        else:
            print(f"{condition['depth']:6.2f}   |   FAILED TO ANALYZE")
    
    print("\n💡 JUMP TYPE CHARACTERISTICS:")
    print("   • Undular (Fr₁ ≤ 1.7): Smooth, wave-like, minimal energy loss")
    print("   • Weak (1.7 < Fr₁ ≤ 2.5): Stable, moderate energy dissipation")
    print("   • Oscillating (2.5 < Fr₁ ≤ 4.5): Unstable, avoid if possible")
    print("   • Steady (4.5 < Fr₁ ≤ 9.0): Excellent energy dissipation")
    print("   • Strong (Fr₁ > 9.0): Very rough, maximum energy dissipation")


def stilling_basin_design_analysis():
    """Demonstrate stilling basin design considerations."""
    
    print("\n\n🏗️  STILLING BASIN DESIGN ANALYSIS")
    print("=" * 70)
    
    channel = poc.RectangularChannel(width=10.0)
    discharge = 60.0
    upstream_depth = 0.8
    
    rvf_solver = poc.RVFSolver()
    analyzer = poc.RVFAnalyzer()
    
    result = rvf_solver.analyze_hydraulic_jump(
        channel=channel,
        discharge=discharge,
        upstream_depth=upstream_depth
    )
    
    if result.success:
        print(f"📊 DESIGN CONDITIONS:")
        print(f"   Channel Width: {channel.width} m")
        print(f"   Discharge: {discharge} m³/s")
        print(f"   Upstream Depth: {upstream_depth} m")
        print(f"   Jump Type: {result.jump_type.value}")
        
        print(f"\n🏗️  STILLING BASIN DIMENSIONS:")
        print(f"   Minimum Basin Length: {result.jump_length:.1f} m")
        print(f"   Recommended Basin Length: {result.jump_length * 1.2:.1f} m (20% safety factor)")
        print(f"   Basin Depth (from upstream invert): {result.downstream_depth:.2f} m")
        print(f"   Side Wall Height: {result.downstream_depth * 1.3:.2f} m (30% freeboard)")
        
        print(f"\n⚡ ENERGY DISSIPATION:")
        print(f"   Total Energy Dissipated: {result.energy_loss:.2f} m")
        print(f"   Power Dissipated: {9810 * discharge * result.energy_loss / 1000:.0f} kW")
        print(f"   Energy Dissipation Rate: {result.energy_loss / result.jump_length:.3f} m/m")
        
        # Get professional recommendations
        recommendations = analyzer.recommend_jump_design(result)
        print(f"\n💡 DESIGN RECOMMENDATIONS:")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
        
        # Additional design considerations
        print(f"\n🔧 ADDITIONAL DESIGN CONSIDERATIONS:")
        
        if result.jump_type in [poc.JumpType.OSCILLATING]:
            print("   ⚠️  Consider forced jump with baffle blocks or sill")
            print("   📊 Monitor tailwater levels to prevent jump movement")
        
        if result.jump_type in [poc.JumpType.STEADY, poc.JumpType.STRONG]:
            print("   🛡️  Heavy erosion protection required downstream")
            print("   🔧 Consider chute blocks and baffle blocks")
            print("   📏 Extend apron length for scour protection")
        
        # Scour analysis
        velocity_downstream = discharge / (channel.width * result.downstream_depth)
        print(f"\n🌊 SCOUR ANALYSIS:")
        print(f"   Downstream Velocity: {velocity_downstream:.2f} m/s")
        if velocity_downstream > 3.0:
            print("   ⚠️  High velocity - significant scour potential")
            print("   🛡️  Rock riprap or concrete apron recommended")
        else:
            print("   ✅ Moderate velocity - standard protection adequate")


def tailwater_sensitivity_analysis():
    """Analyze sensitivity to tailwater variations."""
    
    print("\n\n📊 TAILWATER SENSITIVITY ANALYSIS")
    print("=" * 70)
    
    channel = poc.RectangularChannel(width=7.0)
    discharge = 35.0
    upstream_depth = 1.0
    
    rvf_solver = poc.RVFSolver()
    
    # Calculate required sequent depth
    base_result = rvf_solver.analyze_hydraulic_jump(
        channel=channel,
        discharge=discharge,
        upstream_depth=upstream_depth
    )
    
    if not base_result.success:
        print("❌ Base analysis failed")
        return
    
    required_depth = base_result.downstream_depth
    print(f"📏 Required Sequent Depth: {required_depth:.3f} m")
    print(f"🌊 Jump Type: {base_result.jump_type.value}")
    
    # Test different tailwater depths
    tailwater_ratios = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
    
    print(f"\nTailwater | Ratio to | Jump      | Energy    | Stability")
    print(f"Depth     | Required | Condition | Efficiency| Assessment")
    print(f"----------|----------|-----------|-----------|------------")
    
    for ratio in tailwater_ratios:
        tailwater_depth = required_depth * ratio
        
        result = rvf_solver.analyze_hydraulic_jump(
            channel=channel,
            discharge=discharge,
            upstream_depth=upstream_depth,
            tailwater_depth=tailwater_depth
        )
        
        if result.success:
            adequate = result.properties.get("tailwater_adequate", False)
            
            if ratio < 0.9:
                condition = "Swept Out"
                stability = "❌ Unstable"
            elif ratio < 1.1:
                condition = "Perfect"
                stability = "✅ Stable"
            else:
                condition = "Submerged"
                stability = "⚠️  Submerged"
            
            print(f"{tailwater_depth:7.3f}   | "
                  f"{ratio:6.2f}   | "
                  f"{condition:9s} | "
                  f"{result.energy_efficiency:8.1%}  | "
                  f"{stability}")
    
    print(f"\n💡 TAILWATER DESIGN GUIDELINES:")
    print(f"   • Optimal Range: {required_depth*0.95:.2f} - {required_depth*1.05:.2f} m")
    print(f"   • Minimum Acceptable: {required_depth*0.9:.2f} m (90% of required)")
    print(f"   • Maximum Efficient: {required_depth*1.2:.2f} m (120% of required)")


def multi_channel_comparison():
    """Compare hydraulic jumps in different channel geometries."""
    
    print("\n\n🔄 MULTI-CHANNEL GEOMETRY COMPARISON")
    print("=" * 70)
    
    discharge = 30.0
    upstream_depth = 1.2
    
    # Different channel geometries
    channels = [
        ("Rectangular 6m", poc.RectangularChannel(width=6.0)),
        ("Rectangular 8m", poc.RectangularChannel(width=8.0)),
        ("Trapezoidal 4m+1:1", poc.TrapezoidalChannel(bottom_width=4.0, side_slope=1.0)),
        ("Trapezoidal 5m+1.5:1", poc.TrapezoidalChannel(bottom_width=5.0, side_slope=1.5))
    ]
    
    rvf_solver = poc.RVFSolver()
    
    print("Channel Type        | Upstream | Jump    | Sequent | Energy  | Jump")
    print("                    | Froude   | Type    | Depth   | Loss    | Length")
    print("--------------------|----------|---------|---------|---------|--------")
    
    for name, channel in channels:
        result = rvf_solver.analyze_hydraulic_jump(
            channel=channel,
            discharge=discharge,
            upstream_depth=upstream_depth
        )
        
        if result.success:
            print(f"{name:18s}  | "
                  f"{result.upstream_froude:6.2f}   | "
                  f"{result.jump_type.value:7s} | "
                  f"{result.downstream_depth:7.3f} | "
                  f"{result.energy_loss:7.3f} | "
                  f"{result.jump_length:6.1f}")
        else:
            print(f"{name:18s}  |   ANALYSIS FAILED")
    
    print(f"\n💡 GEOMETRY EFFECTS:")
    print(f"   • Wider channels: Lower upstream Froude, longer jumps")
    print(f"   • Trapezoidal channels: More stable jumps, better energy dissipation")
    print(f"   • Side slopes: Provide natural jump containment")


def professional_design_example():
    """Complete professional design example."""
    
    print("\n\n🎯 PROFESSIONAL DESIGN EXAMPLE")
    print("=" * 70)
    print("Design a stilling basin for a spillway with the following conditions:")
    
    # Design conditions
    channel = poc.RectangularChannel(width=12.0)
    discharge = 100.0  # Design flood discharge
    upstream_depth = 0.9  # From spillway analysis
    design_tailwater = 4.2  # Normal tailwater level
    
    print(f"   • Spillway width: {channel.width} m")
    print(f"   • Design discharge: {discharge} m³/s")
    print(f"   • Upstream depth: {upstream_depth} m")
    print(f"   • Normal tailwater: {design_tailwater} m")
    
    rvf_solver = poc.RVFSolver()
    analyzer = poc.RVFAnalyzer()
    
    # Analyze hydraulic jump
    result = rvf_solver.analyze_hydraulic_jump(
        channel=channel,
        discharge=discharge,
        upstream_depth=upstream_depth,
        tailwater_depth=design_tailwater
    )
    
    if result.success:
        print(f"\n📊 HYDRAULIC ANALYSIS:")
        print(f"   Jump Type: {result.jump_type.value.upper()}")
        print(f"   Upstream Froude: {result.upstream_froude:.2f}")
        print(f"   Required Sequent Depth: {result.downstream_depth:.2f} m")
        print(f"   Jump Length: {result.jump_length:.1f} m")
        print(f"   Energy Dissipation: {result.energy_loss:.2f} m ({(1-result.energy_efficiency)*100:.0f}%)")
        
        print(f"\n🏗️  STILLING BASIN DESIGN:")
        basin_length = result.jump_length * 1.3  # 30% safety factor
        basin_width = channel.width + 2.0  # 1m on each side
        basin_depth = result.downstream_depth + 0.5  # 0.5m below invert
        wall_height = result.downstream_depth * 1.4  # 40% freeboard
        
        print(f"   Basin Length: {basin_length:.1f} m")
        print(f"   Basin Width: {basin_width:.1f} m")
        print(f"   Basin Depth: {basin_depth:.1f} m")
        print(f"   Wall Height: {wall_height:.1f} m")
        
        # Volume calculations
        concrete_volume = basin_length * basin_width * 0.5  # Assume 0.5m thick slab
        excavation_volume = basin_length * basin_width * basin_depth
        
        print(f"\n📏 CONSTRUCTION QUANTITIES:")
        print(f"   Concrete Volume: {concrete_volume:.0f} m³")
        print(f"   Excavation Volume: {excavation_volume:.0f} m³")
        
        # Professional recommendations
        recommendations = analyzer.recommend_jump_design(result)
        print(f"\n💡 PROFESSIONAL RECOMMENDATIONS:")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
        
        # Safety factors and checks
        print(f"\n🛡️  SAFETY CHECKS:")
        tailwater_adequate = result.properties.get("tailwater_adequate", False)
        print(f"   Tailwater Adequacy: {'✅ PASS' if tailwater_adequate else '❌ FAIL'}")
        
        # Cavitation is not typically a concern for hydraulic jumps
        print(f"   Cavitation Risk: ✅ NOT APPLICABLE (hydraulic jumps are low-pressure phenomena)")
        
        print(f"   Energy Dissipation: {'✅ EXCELLENT' if result.energy_loss > 2.0 else '✅ ADEQUATE'}")
        
    else:
        print(f"❌ Design Analysis Failed: {result.message}")


def main():
    """Run all hydraulic jump analysis examples."""
    
    print("🌊 RVF HYDRAULIC JUMP ANALYSIS - COMPREHENSIVE EXAMPLES")
    print("=" * 80)
    print("Demonstrating PyOpenChannel's RVF capabilities for hydraulic jump analysis")
    print("Author: Alexius Academia")
    print("=" * 80)
    
    try:
        # Run all examples
        basic_hydraulic_jump_analysis()
        jump_type_classification_demo()
        stilling_basin_design_analysis()
        tailwater_sensitivity_analysis()
        multi_channel_comparison()
        professional_design_example()
        
        print("\n" + "=" * 80)
        print("🎉 ALL RVF HYDRAULIC JUMP EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\n💡 Key Takeaways:")
        print("   • RVF system provides comprehensive hydraulic jump analysis")
        print("   • Automatic jump classification based on Froude numbers")
        print("   • Professional design recommendations for stilling basins")
        print("   • Energy dissipation calculations for engineering design")
        print("   • Tailwater adequacy analysis for operational stability")
        print("   • Multi-geometry support for various channel types")
        
    except Exception as e:
        print(f"\n❌ Example execution failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
