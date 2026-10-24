"""
Make the SVG figures used on the documentation website.

Author: Alexius Academia
Date: 2026-10-24

Run from the repository root:  python tools/make_doc_plots.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pyopenchannel as poc
from pyopenchannel import plot

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "images")


def main():
    os.makedirs(OUT, exist_ok=True)
    poc.set_unit_system(poc.UnitSystem.SI)

    gate_reach = poc.solve_profile_with_jump(
        poc.RectangularChannel(width=5.0), discharge=20.0, slope=0.0004, manning_n=0.015,
        x_start=0.0, x_end=500.0, upstream_depth=0.35, downstream_depth=2.5)
    plot.profile(gate_reach, title="Below a sluice gate: M3 curve, jump, M1 curve").save(
        os.path.join(OUT, "profile-jump.svg"))

    dam = poc.DamAnalysis().analyze_backwater(
        poc.RectangularChannel(width=10.0), 30.0, 0.0008, 0.025, dam_height=2.0, analysis_distance=8000.0)
    plot.profile(dam, title="Dam backwater (M1 curve)").save(os.path.join(OUT, "profile-dam.svg"))

    steep = poc.GVFSolver().solve_profile(
        poc.RectangularChannel(width=3.0), 10.0, 0.02, 0.015, x_start=-200.0, x_end=0.0,
        boundary_depth=1.5, boundary_type=poc.BoundaryType.DOWNSTREAM_DEPTH)
    plot.profile(steep, title="Steep channel: S1 curve up to critical depth", show_energy_line=True).save(
        os.path.join(OUT, "profile-s1.svg"))

    plot.cross_section(poc.TrapezoidalChannel(bottom_width=3.0, side_slope=1.5), 1.783).save(
        os.path.join(OUT, "section-trapezoid.svg"))
    plot.cross_section(poc.CircularChannel(diameter=1.2), 0.57).save(
        os.path.join(OUT, "section-pipe.svg"))
    plot.specific_energy(poc.RectangularChannel(width=5.0), 20.0, depth=2.5).save(
        os.path.join(OUT, "specific-energy.svg"))
    plot.rating_curve(poc.TrapezoidalChannel(bottom_width=3.0, side_slope=1.5), 0.0008, 0.025, 3.0).save(
        os.path.join(OUT, "rating-curve.svg"))
    print("Wrote figures to", os.path.abspath(OUT))


if __name__ == "__main__":
    main()
