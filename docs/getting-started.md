# Getting started

## Install

PyOpenChannel needs Python 3.9 or later.

```bash
pip install pyopenchannel
```

With [uv](https://docs.astral.sh/uv/):

```bash
uv add pyopenchannel
```

## Units

Calculations use SI units (metres, m³/s) unless you switch to US customary units (feet, ft³/s). The unit system changes gravity and the Manning factor (1.0 or 1.486) everywhere.

```python
import pyopenchannel as poc

poc.set_unit_system(poc.UnitSystem.SI)            # the default
poc.set_unit_system(poc.UnitSystem.US_CUSTOMARY)  # feet and ft³/s
```

## A first calculation

```python
import pyopenchannel as poc

poc.set_unit_system(poc.UnitSystem.SI)

channel = poc.TrapezoidalChannel(bottom_width=3.0, side_slope=1.5)
discharge = 12.0    # m³/s
slope = 0.0008
manning_n = 0.025

normal_depth = poc.NormalDepth.calculate(channel, discharge, slope, manning_n)
critical_depth = poc.CriticalDepth.calculate(channel, discharge)

print(f"Normal depth:   {normal_depth:.3f} m")    # 1.783 m
print(f"Critical depth: {critical_depth:.3f} m")  # 0.990 m
```

Normal depth is above critical depth, so the slope is mild and the uniform flow is subcritical.

## Errors

Invalid input (a negative discharge, a zero roughness) raises an exception derived from `poc.PyOpenChannelError`. Analyses that can fail for physical reasons, such as a weir whose approach depth is below the crest, return a result with `success = False` and a `message` explaining why.

## Next

- [Channels and uniform flow](uniform-flow.md) for sections, energy and design.
- [Water surface profiles](GVF_User_Guide.md) for backwater and drawdown.
- [Weirs, gates and jumps](structures.md) for hydraulic structures.
