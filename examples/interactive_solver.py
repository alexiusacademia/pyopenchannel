#!/usr/bin/env python3
"""
Interactive solver demonstration showing how to switch unknowns easily.

This script shows the same channel with the same basic parameters,
but solving for different unknowns by simply changing which function we call.
"""

import sys
import os

# Add the src directory to the path so we can import pyopenchannel
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pyopenchannel as poc


def main():
    """Demonstrate solving for different unknowns with the same channel."""
    print("PyOpenChannel - Interactive Solver Demonstration")
    print("=" * 60)
    print("Same channel, different unknowns - just change the function call!")
    print()
    
    # Define a standard channel
    channel = poc.RectangularChannel(width=3.0)
    print(f"Channel: Rectangular, width = {channel.width} m")
    print()
    
    # Scenario 1: Known Q, S, n → Find depth
    print("SCENARIO 1: Find normal depth")
    print("-" * 30)
    discharge = 8.0
    slope = 0.002
    manning_n = 0.025
    
    print(f"Given: Q = {discharge} m³/s, S = {slope}, n = {manning_n}")
    
    # Just call the normal depth function
    depth = poc.NormalDepth.calculate(channel, discharge, slope, manning_n)
    print(f"Result: Normal depth = {depth:.3f} m")
    print()
    
    # Scenario 2: Known depth, S, n → Find Q
    print("SCENARIO 2: Find discharge")
    print("-" * 30)
    depth = 2.0
    slope = 0.002
    manning_n = 0.025
    
    print(f"Given: depth = {depth} m, S = {slope}, n = {manning_n}")
    
    # Just call Manning's equation directly
    area = channel.area(depth)
    hydraulic_radius = channel.hydraulic_radius(depth)
    discharge = poc.ManningEquation.discharge(area, hydraulic_radius, slope, manning_n)
    print(f"Result: Discharge = {discharge:.3f} m³/s")
    print()
    
    # Scenario 3: Known Q, depth, n → Find required slope
    print("SCENARIO 3: Find required slope")
    print("-" * 30)
    discharge = 8.0
    depth = 2.0
    manning_n = 0.025
    
    print(f"Given: Q = {discharge} m³/s, depth = {depth} m, n = {manning_n}")
    
    # Just call the required slope function
    area = channel.area(depth)
    hydraulic_radius = channel.hydraulic_radius(depth)
    slope = poc.ManningEquation.required_slope(discharge, area, hydraulic_radius, manning_n)
    print(f"Result: Required slope = {slope:.6f} ({slope*100:.4f}%)")
    print()
    
    # Scenario 4: Known Q → Find critical depth
    print("SCENARIO 4: Find critical depth")
    print("-" * 30)
    discharge = 8.0
    
    print(f"Given: Q = {discharge} m³/s")
    
    # Just call the critical depth function
    critical_depth = poc.CriticalDepth.calculate(channel, discharge)
    print(f"Result: Critical depth = {critical_depth:.3f} m")
    print()
    
    # Scenario 5: Design optimal channel for given Q
    print("SCENARIO 5: Design optimal channel dimensions")
    print("-" * 30)
    discharge = 8.0
    slope = 0.002
    manning_n = 0.025
    
    print(f"Given: Q = {discharge} m³/s, S = {slope}, n = {manning_n}")
    
    # Just call the optimal design function
    result = poc.OptimalSections.rectangular(discharge, slope, manning_n)
    print(f"Result: Optimal width = {result.channel.width:.3f} m")
    print(f"        Flow depth = {result.depth:.3f} m")
    print()
    
    print("=" * 60)
    print("KEY INSIGHT: Same underlying physics, different entry points!")
    print("=" * 60)
    print("The library automatically handles:")
    print("✓ Iterative solutions (Newton-Raphson)")
    print("✓ Convergence checking")
    print("✓ Input validation")
    print("✓ Error handling")
    print("✓ Unit consistency")
    print()
    print("You just focus on WHAT you want to solve for!")


if __name__ == "__main__":
    main()
