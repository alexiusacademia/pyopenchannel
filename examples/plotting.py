"""
Plotting with PyOpenChannel - SVG charts with no plotting library

Author: Alexius Academia
Date: 2026-10-24

Writes SVG files into examples/output/: a profile through a hydraulic jump
below a sluice gate, a cross-section, a specific energy diagram and a
rating curve. Open them in any browser.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pyopenchannel as poc
from pyopenchannel import plot

OUT = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT, exist_ok=True)
poc.set_unit_system(poc.UnitSystem.SI)

channel = poc.RectangularChannel(width=5.0)
discharge, slope, manning_n = 20.0, 0.0004, 0.015

# A sluice gate releases a 0.35 m jet at x = 0; a weir 500 m downstream
# holds the water at 2.5 m. The jet rises along an M3 curve, jumps, and
# the flow continues on an M1 curve to the weir.
reach = poc.solve_profile_with_jump(
    channel, discharge, slope, manning_n, x_start=0.0, x_end=500.0,
    upstream_depth=0.35, downstream_depth=2.5)
print(reach.message)
jump = reach.jump
print(f"Jump at x = {jump.x:.1f} m, Froude {jump.froude_number:.2f}, "
      f"energy loss {jump.energy_loss:.2f} m, length about {jump.length:.1f} m")

files = [
    plot.profile(reach, title="Sluice gate, hydraulic jump and weir").save(os.path.join(OUT, "profile.svg")),
    plot.cross_section(channel, jump.sequent_depth).save(os.path.join(OUT, "section.svg")),
    plot.specific_energy(channel, discharge, depth=2.5).save(os.path.join(OUT, "specific_energy.svg")),
    plot.rating_curve(channel, slope, manning_n, 3.0).save(os.path.join(OUT, "rating_curve.svg")),
]
for path in files:
    print("Wrote", os.path.relpath(path))
