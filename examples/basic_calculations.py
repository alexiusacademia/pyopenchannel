#!/usr/bin/env python3
"""
Basic open channel flow calculations using PyOpenChannel.

This example demonstrates fundamental calculations including:
- Channel geometry properties
- Normal depth calculations
- Critical depth calculations
- Uniform flow analysis
"""

import sys
import os

# Add the src directory to the path so we can import pyopenchannel
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pyopenchannel as poc


def rectangular_channel_example():
    """Example calculations for a rectangular channel."""
    print("=" * 60)
    print("RECTANGULAR CHANNEL EXAMPLE")
    print("=" * 60)
    
    # Channel parameters
    width = 1.0  # m
    slope = 0.001  # dimensionless (0.1%)
    manning_n = 0.015  # concrete channel
    discharge = 1.0  # m³/s
    
    print(f"Channel width: {width} m")
    print(f"Channel slope: {slope} ({slope*100:.1f}%)")
    print(f"Manning's n: {manning_n}")
    print(f"Design discharge: {discharge} m³/s")
    print()
    
    # Create rectangular channel
    channel = poc.RectangularChannel(width)
    
    # Calculate normal depth
    try:
        normal_depth = poc.NormalDepth.calculate(channel, discharge, slope, manning_n)
        print(f"Normal depth: {normal_depth:.3f} m")
        
        # Calculate flow properties at normal depth
        area = channel.area(normal_depth)
        velocity = discharge / area
        hydraulic_radius = channel.hydraulic_radius(normal_depth)
        froude_number = poc.CriticalDepth.froude_number(
            velocity, area / channel.top_width(normal_depth)
        )
        
        print(f"Flow area: {area:.3f} m²")
        print(f"Flow velocity: {velocity:.3f} m/s")
        print(f"Hydraulic radius: {hydraulic_radius:.3f} m")
        print(f"Froude number: {froude_number:.3f}")
        
        if froude_number < 1.0:
            print("Flow regime: Subcritical")
        elif froude_number > 1.0:
            print("Flow regime: Supercritical")
        else:
            print("Flow regime: Critical")
            
    except Exception as e:
        print(f"Error calculating normal depth: {e}")
    
    print()
    
    # Calculate critical depth
    try:
        critical_depth = poc.CriticalDepth.calculate(channel, discharge)
        print(f"Critical depth: {critical_depth:.3f} m")
        
        # Compare normal and critical depths
        if normal_depth > critical_depth:
            print("Channel slope: Mild (normal depth > critical depth)")
        elif normal_depth < critical_depth:
            print("Channel slope: Steep (normal depth < critical depth)")
        else:
            print("Channel slope: Critical (normal depth = critical depth)")
            
    except Exception as e:
        print(f"Error calculating critical depth: {e}")
    
    print()


def trapezoidal_channel_example():
    """Example calculations for a trapezoidal channel."""
    print("=" * 60)
    print("TRAPEZOIDAL CHANNEL EXAMPLE")
    print("=" * 60)
    
    # Channel parameters
    bottom_width = 2.0  # m
    side_slope = 1.5  # horizontal:vertical
    slope = 0.002  # dimensionless (0.2%)
    manning_n = 0.030  # earth channel
    discharge = 8.0  # m³/s
    
    print(f"Bottom width: {bottom_width} m")
    print(f"Side slope: {side_slope}:1 (H:V)")
    print(f"Channel slope: {slope} ({slope*100:.1f}%)")
    print(f"Manning's n: {manning_n}")
    print(f"Design discharge: {discharge} m³/s")
    print()
    
    # Create trapezoidal channel
    channel = poc.TrapezoidalChannel(bottom_width, side_slope)
    
    # Calculate normal depth
    try:
        normal_depth = poc.NormalDepth.calculate(channel, discharge, slope, manning_n)
        print(f"Normal depth: {normal_depth:.3f} m")
        
        # Calculate flow properties
        area = channel.area(normal_depth)
        top_width = channel.top_width(normal_depth)
        wetted_perimeter = channel.wetted_perimeter(normal_depth)
        hydraulic_radius = channel.hydraulic_radius(normal_depth)
        velocity = discharge / area
        
        print(f"Flow area: {area:.3f} m²")
        print(f"Top width: {top_width:.3f} m")
        print(f"Wetted perimeter: {wetted_perimeter:.3f} m")
        print(f"Hydraulic radius: {hydraulic_radius:.3f} m")
        print(f"Flow velocity: {velocity:.3f} m/s")
        
    except Exception as e:
        print(f"Error calculating normal depth: {e}")
    
    print()


def circular_channel_example():
    """Example calculations for a circular channel (pipe)."""
    print("=" * 60)
    print("CIRCULAR CHANNEL EXAMPLE")
    print("=" * 60)
    
    # Channel parameters
    diameter = 1.2  # m
    slope = 0.005  # dimensionless (0.5%)
    manning_n = 0.013  # smooth concrete pipe
    discharge = 0.8  # m³/s
    
    print(f"Pipe diameter: {diameter} m")
    print(f"Channel slope: {slope} ({slope*100:.1f}%)")
    print(f"Manning's n: {manning_n}")
    print(f"Design discharge: {discharge} m³/s")
    print()
    
    # Create circular channel
    channel = poc.CircularChannel(diameter)
    
    # Calculate normal depth
    try:
        normal_depth = poc.NormalDepth.calculate(channel, discharge, slope, manning_n)
        print(f"Normal depth: {normal_depth:.3f} m")
        print(f"Depth ratio (y/D): {normal_depth/diameter:.3f}")
        
        # Calculate flow properties
        area = channel.area(normal_depth)
        velocity = discharge / area
        hydraulic_radius = channel.hydraulic_radius(normal_depth)
        
        print(f"Flow area: {area:.3f} m²")
        print(f"Flow velocity: {velocity:.3f} m/s")
        print(f"Hydraulic radius: {hydraulic_radius:.3f} m")
        
        # Check if pipe is flowing full
        if normal_depth >= diameter:
            print("Pipe is flowing full (pressurized flow)")
        else:
            print(f"Pipe is partially full ({normal_depth/diameter*100:.1f}% full)")
            
    except Exception as e:
        print(f"Error calculating normal depth: {e}")
    
    print()


def energy_calculations_example():
    """Example of energy calculations and alternate depths."""
    print("=" * 60)
    print("ENERGY CALCULATIONS EXAMPLE")
    print("=" * 60)
    
    # Channel parameters
    width = 4.0  # m
    discharge = 10.0  # m³/s
    
    print(f"Rectangular channel width: {width} m")
    print(f"Discharge: {discharge} m³/s")
    print()
    
    # Create channel
    channel = poc.RectangularChannel(width)
    
    # Calculate critical depth and minimum specific energy
    try:
        critical_depth = poc.CriticalDepth.calculate(channel, discharge)
        min_energy = poc.EnergyEquation.minimum_specific_energy(channel, discharge)
        
        print(f"Critical depth: {critical_depth:.3f} m")
        print(f"Minimum specific energy: {min_energy:.3f} m")
        print()
        
        # Calculate alternate depths for a given specific energy
        specific_energy = min_energy + 0.5  # Add 0.5 m to minimum energy
        print(f"Given specific energy: {specific_energy:.3f} m")
        
        subcritical_depth, supercritical_depth = poc.EnergyEquation.alternate_depths(
            channel, discharge, specific_energy
        )
        
        print(f"Subcritical depth: {subcritical_depth:.3f} m")
        print(f"Supercritical depth: {supercritical_depth:.3f} m")
        
        # Calculate velocities and Froude numbers
        area_sub = channel.area(subcritical_depth)
        area_super = channel.area(supercritical_depth)
        velocity_sub = discharge / area_sub
        velocity_super = discharge / area_super
        
        fr_sub = poc.CriticalDepth.froude_number(velocity_sub, subcritical_depth)
        fr_super = poc.CriticalDepth.froude_number(velocity_super, supercritical_depth)
        
        print(f"Subcritical velocity: {velocity_sub:.3f} m/s (Fr = {fr_sub:.3f})")
        print(f"Supercritical velocity: {velocity_super:.3f} m/s (Fr = {fr_super:.3f})")
        
    except Exception as e:
        print(f"Error in energy calculations: {e}")
    
    print()


def uniform_flow_analysis_example():
    """Example of uniform flow analysis."""
    print("=" * 60)
    print("UNIFORM FLOW ANALYSIS EXAMPLE")
    print("=" * 60)
    
    # Channel parameters
    bottom_width = 3.0  # m
    side_slope = 2.0  # horizontal:vertical
    slope = 0.0015  # dimensionless
    manning_n = 0.035  # natural earth channel
    discharge = 12.0  # m³/s
    
    print(f"Trapezoidal channel:")
    print(f"  Bottom width: {bottom_width} m")
    print(f"  Side slope: {side_slope}:1 (H:V)")
    print(f"  Channel slope: {slope} ({slope*100:.2f}%)")
    print(f"  Manning's n: {manning_n}")
    print(f"  Discharge: {discharge} m³/s")
    print()
    
    # Create channel and uniform flow analyzer
    channel = poc.TrapezoidalChannel(bottom_width, side_slope)
    uniform_flow = poc.UniformFlow(channel, slope, manning_n)
    
    try:
        # Calculate flow state
        flow_state = uniform_flow.calculate_flow_state(discharge)
        
        print("Flow State Results:")
        print(f"  Depth: {flow_state.depth:.3f} m")
        print(f"  Velocity: {flow_state.velocity:.3f} m/s")
        print(f"  Area: {flow_state.area:.3f} m²")
        print(f"  Top width: {flow_state.top_width:.3f} m")
        print(f"  Hydraulic radius: {flow_state.hydraulic_radius:.3f} m")
        print(f"  Froude number: {flow_state.froude_number:.3f}")
        print(f"  Specific energy: {flow_state.specific_energy:.3f} m")
        print()
        
        # Classify flow
        if flow_state.is_subcritical:
            print("Flow classification: Subcritical")
        elif flow_state.is_supercritical:
            print("Flow classification: Supercritical")
        else:
            print("Flow classification: Critical")
        
        # Check velocity limits
        velocity_check = poc.ChannelDesigner.check_velocity_limits(
            flow_state.velocity, "earth"
        )
        
        print(f"Velocity check: {'PASS' if velocity_check['is_acceptable'] else 'FAIL'}")
        if velocity_check['warnings']:
            for warning in velocity_check['warnings']:
                print(f"  Warning: {warning}")
        
    except Exception as e:
        print(f"Error in uniform flow analysis: {e}")
    
    print()


def main():
    """Run all examples."""
    print("PyOpenChannel - Basic Calculations Examples")
    print("=" * 60)
    print()
    
    try:
        rectangular_channel_example()
        trapezoidal_channel_example()
        circular_channel_example()
        energy_calculations_example()
        uniform_flow_analysis_example()
        
        print("=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
