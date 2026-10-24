# Channels and uniform flow

## Channel sections

| Section | Constructor |
|---|---|
| Rectangular | `poc.RectangularChannel(width)` |
| Trapezoidal | `poc.TrapezoidalChannel(bottom_width, side_slope)` (side slope as horizontal per vertical) |
| Triangular | `poc.TriangularChannel(side_slope)` |
| Circular (partly full) | `poc.CircularChannel(diameter)` |
| Parabolic | `poc.ParabolicChannel(shape_parameter)` |

Each section gives `area(depth)`, `wetted_perimeter(depth)`, `top_width(depth)` and `hydraulic_radius(depth)`.

## Normal depth, critical depth and uniform flow

Normal depth solves Manning's equation, Q = (k/n) A R^2/3 S^1/2, with k = 1.0 in SI and 1.486 in US units. Critical depth solves Q²T / (g A³) = 1.

```python
import pyopenchannel as poc

poc.set_unit_system(poc.UnitSystem.SI)

channel = poc.TrapezoidalChannel(bottom_width=3.0, side_slope=1.5)
discharge = 12.0    # m³/s
slope = 0.0008
manning_n = 0.025

normal_depth = poc.NormalDepth.calculate(channel, discharge, slope, manning_n)
critical_depth = poc.CriticalDepth.calculate(channel, discharge)
critical_slope = poc.CriticalFlow(channel).calculate_critical_slope(discharge, manning_n)

state = poc.UniformFlow(channel, slope, manning_n).calculate_flow_state(discharge)

print(f"Normal depth:   {normal_depth:.3f} m")    # 1.783 m
print(f"Critical depth: {critical_depth:.3f} m")  # 0.990 m
print(f"Critical slope: {critical_slope:.5f}")    # 0.00769
print(f"Velocity:       {state.velocity:.3f} m/s")  # 1.186 m/s
print(f"Froude number:  {state.froude_number:.3f}")  # 0.344
```

The channel slope (0.0008) is below the critical slope (0.00769), so it is a mild slope.

## Specific energy and momentum

```python
import pyopenchannel as poc

channel = poc.RectangularChannel(width=4.0)
discharge = 12.0

energy = poc.EnergyEquation.specific_energy_from_discharge(channel, 2.0, discharge)
subcritical, supercritical = poc.EnergyEquation.alternate_depths(channel, discharge, energy)
sequent = poc.MomentumEquation.conjugate_depths(channel, discharge, 0.4)

print(f"Specific energy at 2.0 m: {energy:.3f} m")                            # 2.115 m
print(f"Alternate depths: {subcritical:.3f} m and {supercritical:.3f} m")     # 2.000 m and 0.540 m
print(f"Sequent depth of a jump from 0.4 m: {sequent:.3f} m")                # 1.951 m
```

## Channel design

`OptimalSections` gives the best hydraulic section (least wetted perimeter for the area) for a discharge, slope and roughness. `EconomicSections` weighs excavation, lining and land costs.

```python
import pyopenchannel as poc

result = poc.OptimalSections.trapezoidal(discharge=15.0, slope=0.001, manning_n=0.025, side_slope=1.5)

print(f"Bottom width: {result.channel.bottom_width:.3f} m")  # 1.377 m
print(f"Flow depth:   {result.depth:.3f} m")                 # 2.274 m
print(f"Freeboard:    {result.freeboard:.3f} m")             # 0.500 m
```

The best trapezoidal section has b = 2y(√(1+z²) − z), here 0.606 times the depth.
