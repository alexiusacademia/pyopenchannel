#!/usr/bin/env python3
"""
Channel design examples using PyOpenChannel.

This example demonstrates:
- Optimal channel design for hydraulic efficiency
- Economic channel design considering costs
- Channel sizing for specific requirements
"""

import sys
import os

# Add the src directory to the path so we can import pyopenchannel
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pyopenchannel as poc


def optimal_rectangular_design():
    """Design optimal rectangular channel for hydraulic efficiency."""
    print("=" * 60)
    print("OPTIMAL RECTANGULAR CHANNEL DESIGN")
    print("=" * 60)
    
    # Design requirements
    discharge = 15.0  # m³/s
    slope = 0.002  # 0.2%
    manning_n = 0.025  # concrete lining
    
    print(f"Design requirements:")
    print(f"  Discharge: {discharge} m³/s")
    print(f"  Slope: {slope} ({slope*100:.1f}%)")
    print(f"  Manning's n: {manning_n}")
    print()
    
    try:
        # Design optimal rectangular section
        result = poc.OptimalSections.rectangular(discharge, slope, manning_n)
        
        print("Optimal Design Results:")
        print(f"  Channel width: {result.channel.width:.3f} m")
        print(f"  Flow depth: {result.depth:.3f} m")
        print(f"  Flow velocity: {result.velocity:.3f} m/s")
        print(f"  Hydraulic radius: {result.hydraulic_radius:.3f} m")
        print(f"  Froude number: {result.froude_number:.3f}")
        print(f"  Excavation area: {result.excavation_area:.3f} m²")
        print(f"  Recommended freeboard: {result.freeboard:.3f} m")
        print(f"  Total channel depth: {result.total_depth:.3f} m")
        print()
        
        # Verify the optimal condition (depth = width/2)
        width_to_depth_ratio = result.channel.width / result.depth
        print(f"Width to depth ratio: {width_to_depth_ratio:.3f}")
        print("(Optimal rectangular section has width/depth = 2.0)")
        
        # Check velocity limits
        velocity_check = poc.ChannelDesigner.check_velocity_limits(
            result.velocity, "concrete"
        )
        print(f"Velocity check: {'PASS' if velocity_check['is_acceptable'] else 'FAIL'}")
        
    except Exception as e:
        print(f"Error in optimal design: {e}")
    
    print()


def optimal_trapezoidal_design():
    """Design optimal trapezoidal channel for hydraulic efficiency."""
    print("=" * 60)
    print("OPTIMAL TRAPEZOIDAL CHANNEL DESIGN")
    print("=" * 60)
    
    # Design requirements
    discharge = 25.0  # m³/s
    slope = 0.0015  # 0.15%
    manning_n = 0.030  # earth channel with grass lining
    side_slope = 2.0  # 2:1 (H:V) for stable earth slopes
    
    print(f"Design requirements:")
    print(f"  Discharge: {discharge} m³/s")
    print(f"  Slope: {slope} ({slope*100:.2f}%)")
    print(f"  Manning's n: {manning_n}")
    print(f"  Side slope: {side_slope}:1 (H:V)")
    print()
    
    try:
        # Design optimal trapezoidal section
        result = poc.OptimalSections.trapezoidal(discharge, slope, manning_n, side_slope)
        
        print("Optimal Design Results:")
        print(f"  Bottom width: {result.channel.bottom_width:.3f} m")
        print(f"  Flow depth: {result.depth:.3f} m")
        print(f"  Top width: {result.channel.top_width(result.depth):.3f} m")
        print(f"  Flow velocity: {result.velocity:.3f} m/s")
        print(f"  Hydraulic radius: {result.hydraulic_radius:.3f} m")
        print(f"  Froude number: {result.froude_number:.3f}")
        print(f"  Excavation area: {result.excavation_area:.3f} m²")
        print(f"  Recommended freeboard: {result.freeboard:.3f} m")
        print(f"  Total channel depth: {result.total_depth:.3f} m")
        print()
        
        # Check velocity limits for grass-lined earth channel
        velocity_check = poc.ChannelDesigner.check_velocity_limits(
            result.velocity, "grass"
        )
        print(f"Velocity check: {'PASS' if velocity_check['is_acceptable'] else 'FAIL'}")
        if velocity_check['warnings']:
            for warning in velocity_check['warnings']:
                print(f"  Warning: {warning}")
        
    except Exception as e:
        print(f"Error in optimal design: {e}")
    
    print()


def economic_channel_design():
    """Design economical channel considering construction costs."""
    print("=" * 60)
    print("ECONOMIC CHANNEL DESIGN")
    print("=" * 60)
    
    # Design requirements
    discharge = 20.0  # m³/s
    slope = 0.001  # 0.1%
    manning_n = 0.015  # concrete lining
    
    # Cost parameters (example values)
    excavation_cost = 25.0  # $/m³
    lining_cost = 50.0  # $/m²
    land_cost = 100.0  # $/m²
    
    print(f"Design requirements:")
    print(f"  Discharge: {discharge} m³/s")
    print(f"  Slope: {slope} ({slope*100:.1f}%)")
    print(f"  Manning's n: {manning_n}")
    print()
    print(f"Cost parameters:")
    print(f"  Excavation: ${excavation_cost}/m³")
    print(f"  Lining: ${lining_cost}/m²")
    print(f"  Land: ${land_cost}/m²")
    print()
    
    try:
        # Create economic designer
        economic_designer = poc.EconomicSections(
            excavation_cost_per_m3=excavation_cost,
            lining_cost_per_m2=lining_cost,
            land_cost_per_m2=land_cost
        )
        
        # Design economic rectangular section
        result = economic_designer.design_rectangular(
            discharge, slope, manning_n, width_range=(2.0, 10.0), num_trials=50
        )
        
        print("Economic Design Results:")
        print(f"  Channel width: {result.channel.width:.3f} m")
        print(f"  Flow depth: {result.depth:.3f} m")
        print(f"  Flow velocity: {result.velocity:.3f} m/s")
        print(f"  Hydraulic radius: {result.hydraulic_radius:.3f} m")
        print(f"  Froude number: {result.froude_number:.3f}")
        print(f"  Cost per meter: ${result.cost_per_meter:.2f}/m")
        print(f"  Recommended freeboard: {result.freeboard:.3f} m")
        print(f"  Total channel depth: {result.total_depth:.3f} m")
        print()
        
        # Compare with optimal hydraulic design
        optimal_result = poc.OptimalSections.rectangular(discharge, slope, manning_n)
        
        print("Comparison with Optimal Hydraulic Design:")
        print(f"  Economic width: {result.channel.width:.3f} m vs Optimal: {optimal_result.channel.width:.3f} m")
        print(f"  Economic depth: {result.depth:.3f} m vs Optimal: {optimal_result.depth:.3f} m")
        print(f"  Economic excavation: {result.excavation_area:.3f} m² vs Optimal: {optimal_result.excavation_area:.3f} m²")
        
        area_difference = ((result.excavation_area - optimal_result.excavation_area) / 
                          optimal_result.excavation_area * 100)
        print(f"  Excavation difference: {area_difference:+.1f}%")
        
    except Exception as e:
        print(f"Error in economic design: {e}")
    
    print()


def channel_sizing_example():
    """Example of sizing channels for specific requirements."""
    print("=" * 60)
    print("CHANNEL SIZING FOR CAPACITY")
    print("=" * 60)
    
    # Multiple design scenarios
    scenarios = [
        {
            "name": "Urban Storm Drain",
            "discharge": 5.0,
            "slope": 0.005,
            "manning_n": 0.013,
            "channel_type": "rectangular"
        },
        {
            "name": "Agricultural Irrigation Canal",
            "discharge": 30.0,
            "slope": 0.0008,
            "manning_n": 0.025,
            "channel_type": "trapezoidal",
            "side_slope": 1.5
        },
        {
            "name": "Highway Drainage Ditch",
            "discharge": 8.0,
            "slope": 0.003,
            "manning_n": 0.035,
            "channel_type": "triangular"
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"Scenario {i}: {scenario['name']}")
        print(f"  Required discharge: {scenario['discharge']} m³/s")
        print(f"  Available slope: {scenario['slope']} ({scenario['slope']*100:.2f}%)")
        print(f"  Manning's n: {scenario['manning_n']}")
        print(f"  Channel type: {scenario['channel_type']}")
        
        try:
            # Size channel for capacity
            result = poc.ChannelDesigner.size_channel_for_capacity(
                target_discharge=scenario['discharge'],
                slope=scenario['slope'],
                manning_n=scenario['manning_n'],
                channel_type=scenario['channel_type'],
                **{k: v for k, v in scenario.items() if k not in 
                   ['name', 'discharge', 'slope', 'manning_n', 'channel_type']}
            )
            
            print(f"  Design results:")
            if scenario['channel_type'] == 'rectangular':
                print(f"    Width: {result.channel.width:.3f} m")
            elif scenario['channel_type'] == 'trapezoidal':
                print(f"    Bottom width: {result.channel.bottom_width:.3f} m")
                print(f"    Side slope: {result.channel.side_slope}:1")
            elif scenario['channel_type'] == 'triangular':
                print(f"    Side slope: {result.channel.side_slope:.3f}:1")
            
            print(f"    Flow depth: {result.depth:.3f} m")
            print(f"    Flow velocity: {result.velocity:.3f} m/s")
            print(f"    Froude number: {result.froude_number:.3f}")
            print(f"    Freeboard: {result.freeboard:.3f} m")
            print(f"    Total depth: {result.total_depth:.3f} m")
            
            # Flow classification
            if result.froude_number < 1.0:
                flow_type = "Subcritical"
            elif result.froude_number > 1.0:
                flow_type = "Supercritical"
            else:
                flow_type = "Critical"
            print(f"    Flow regime: {flow_type}")
            
        except Exception as e:
            print(f"    Error: {e}")
        
        print()


def side_slope_recommendations():
    """Example of side slope recommendations for different soil types."""
    print("=" * 60)
    print("SIDE SLOPE RECOMMENDATIONS")
    print("=" * 60)
    
    soil_types = [
        "vertical",
        "steep_rock", 
        "ordinary_rock",
        "earth_firm",
        "earth_ordinary",
        "earth_sandy",
        "earth_loose"
    ]
    
    print("Recommended side slopes by soil type:")
    print("(horizontal:vertical ratio)")
    print()
    
    for soil_type in soil_types:
        side_slope = poc.ChannelDesigner.recommend_side_slope(soil_type)
        print(f"  {soil_type.replace('_', ' ').title()}: {side_slope}:1")
    
    print()
    
    # Design example with recommended side slope
    print("Design example using recommended side slope:")
    discharge = 15.0
    slope = 0.002
    manning_n = 0.030
    soil_type = "earth_ordinary"
    
    recommended_side_slope = poc.ChannelDesigner.recommend_side_slope(soil_type)
    
    print(f"  Soil type: {soil_type.replace('_', ' ')}")
    print(f"  Recommended side slope: {recommended_side_slope}:1")
    print(f"  Discharge: {discharge} m³/s")
    print(f"  Slope: {slope} ({slope*100:.1f}%)")
    print(f"  Manning's n: {manning_n}")
    print()
    
    try:
        result = poc.OptimalSections.trapezoidal(
            discharge, slope, manning_n, recommended_side_slope
        )
        
        print("Design results:")
        print(f"  Bottom width: {result.channel.bottom_width:.3f} m")
        print(f"  Flow depth: {result.depth:.3f} m")
        print(f"  Top width: {result.channel.top_width(result.depth):.3f} m")
        print(f"  Flow velocity: {result.velocity:.3f} m/s")
        print(f"  Total depth with freeboard: {result.total_depth:.3f} m")
        
    except Exception as e:
        print(f"Error in design: {e}")
    
    print()


def main():
    """Run all design examples."""
    print("PyOpenChannel - Channel Design Examples")
    print("=" * 60)
    print()
    
    try:
        optimal_rectangular_design()
        optimal_trapezoidal_design()
        economic_channel_design()
        channel_sizing_example()
        side_slope_recommendations()
        
        print("=" * 60)
        print("All design examples completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
