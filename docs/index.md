---
title: Open channel flow in Python
hide:
  - navigation
---

# PyOpenChannel

<p class="hero-lead">Open channel flow analysis and design in Python: normal and critical depth, water surface profiles, weirs, gates and hydraulic jumps, in SI or US customary units.</p>

```bash
pip install pyopenchannel
```

```python
import pyopenchannel as poc

channel = poc.TrapezoidalChannel(bottom_width=3.0, side_slope=1.5)

normal_depth = poc.NormalDepth.calculate(channel, discharge=12.0, slope=0.0008, manning_n=0.025)
critical_depth = poc.CriticalDepth.calculate(channel, discharge=12.0)

print(f"Normal depth: {normal_depth:.3f} m")      # 1.783 m
print(f"Critical depth: {critical_depth:.3f} m")  # 0.990 m
```

## What it covers

<div class="grid-cards">
<a href="uniform-flow/"><strong>Channels and uniform flow</strong>Rectangular, trapezoidal, triangular, circular and parabolic sections. Manning's equation, normal and critical depth, specific energy, channel design.</a>
<a href="GVF_User_Guide/"><strong>Water surface profiles</strong>Gradually varied flow: M, S, C, H and A profiles, dam backwater, bridges by the energy method, chutes and transitions.</a>
<a href="structures/"><strong>Weirs, gates and jumps</strong>Sharp-crested, broad-crested and ogee weirs, sluice and radial gates in free or submerged flow, and hydraulic jumps.</a>
<a href="examples/"><strong>Examples</strong>About forty ready-to-run scripts, from first calculations to spillway profiles.</a>
</div>

## Checked against the textbook

Results are tested against independent calculations, not just run:

- Water surface profiles against an independent Runge-Kutta integration (within 0.3 mm).
- Bridge backwater against the energy balance, with choking at critical depth.
- Weirs against the Rehbock, V-notch, USBR ogee and Villemonte equations.
- Sluice gates against the energy and momentum equations (Henderson) and Henry's discharge coefficients.
- Hydraulic jumps against the Belanger equation.

PyOpenChannel is open source under the MIT License. Source, issues and releases are on [GitHub](https://github.com/alexiusacademia/pyopenchannel); the package is on [PyPI](https://pypi.org/project/pyopenchannel/).
