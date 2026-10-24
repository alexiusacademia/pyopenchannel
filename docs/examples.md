# Examples

The [`examples/`](https://github.com/alexiusacademia/pyopenchannel/tree/main/examples) folder in the repository has ready-to-run scripts. Clone the repository and run one from its root:

```bash
git clone https://github.com/alexiusacademia/pyopenchannel.git
cd pyopenchannel
pip install -e .
python examples/basic_calculations.py
```

Some scripts draw plots with matplotlib and save them as PNG files.

## Basics

| Script | What it shows |
|---|---|
| [`plotting.py`](https://github.com/alexiusacademia/pyopenchannel/blob/main/examples/plotting.py) | A sluice gate, jump and weir; SVG profile, section and curves |
| [`basic_calculations.py`](https://github.com/alexiusacademia/pyopenchannel/blob/main/examples/basic_calculations.py) | Normal depth, critical depth, flow state |
| [`complete_rectangular_example.py`](https://github.com/alexiusacademia/pyopenchannel/blob/main/examples/complete_rectangular_example.py) | A full rectangular channel workflow |
| [`channel_design.py`](https://github.com/alexiusacademia/pyopenchannel/blob/main/examples/channel_design.py) | Optimal and economic sections |
| [`solve_for_unknowns.py`](https://github.com/alexiusacademia/pyopenchannel/blob/main/examples/solve_for_unknowns.py) | Normal depth from the discharge, discharge from the depth, required slope |
| [`unit_systems.py`](https://github.com/alexiusacademia/pyopenchannel/blob/main/examples/unit_systems.py), [`unit_aware_hydraulics.py`](https://github.com/alexiusacademia/pyopenchannel/blob/main/examples/unit_aware_hydraulics.py) | SI and US customary units |

## Water surface profiles

| Script | What it shows |
|---|---|
| [`gvf_simple_example.py`](https://github.com/alexiusacademia/pyopenchannel/blob/main/examples/gvf_simple_example.py) | M1 backwater, M2 drawdown to a free overfall |
| [`gvf_basic_usage.py`](https://github.com/alexiusacademia/pyopenchannel/blob/main/examples/gvf_basic_usage.py) | M1, M2 and S1 profiles; a profile ending at critical depth |
| [`backwater_analysis_demo.py`](https://github.com/alexiusacademia/pyopenchannel/blob/main/examples/backwater_analysis_demo.py) | Backwater from a downstream control |
| [`gvf_applications_demo.py`](https://github.com/alexiusacademia/pyopenchannel/blob/main/examples/gvf_applications_demo.py) | Dam, bridge, chute and transition analyses |
| [`gvf_channel_transitions.py`](https://github.com/alexiusacademia/pyopenchannel/blob/main/examples/gvf_channel_transitions.py) | Changes of section and slope |

## Weirs, gates and jumps

| Script | What it shows |
|---|---|
| [`rvf_hydraulic_jump_analysis.py`](https://github.com/alexiusacademia/pyopenchannel/blob/main/examples/rvf_hydraulic_jump_analysis.py) | Jump types, sequent depths, energy loss |
| [`hydraulic_jump_visualization.py`](https://github.com/alexiusacademia/pyopenchannel/blob/main/examples/hydraulic_jump_visualization.py) | Plots of a jump |
| [`gate_to_hydraulic_jump_simulation.py`](https://github.com/alexiusacademia/pyopenchannel/blob/main/examples/gate_to_hydraulic_jump_simulation.py) | Gate flow followed by a jump |
| [`rvf_submerged_ogee_weir_analysis.py`](https://github.com/alexiusacademia/pyopenchannel/blob/main/examples/rvf_submerged_ogee_weir_analysis.py) | A submerged ogee crest |
| [`diversion_dam_complete_profile.py`](https://github.com/alexiusacademia/pyopenchannel/blob/main/examples/diversion_dam_complete_profile.py) | A diversion dam from approach to tailwater |
| [`rigorous_spillway_demo.py`](https://github.com/alexiusacademia/pyopenchannel/blob/main/examples/rigorous_spillway_demo.py) | Nappe trajectory, WES ogee crest, chute exit depth and the jump below |

## Finite volume solver (experimental)

The finite volume solver resolves flows with shocks, such as jumps, in more detail. It has not yet passed its own validation suite, so treat its results as experimental.

| Script | What it shows |
|---|---|
| [`fvm_solver_simple_demo.py`](https://github.com/alexiusacademia/pyopenchannel/blob/main/examples/fvm_solver_simple_demo.py) | A first finite volume run |
| [`weir_flow_fvm_comparison_demo.py`](https://github.com/alexiusacademia/pyopenchannel/blob/main/examples/weir_flow_fvm_comparison_demo.py) | Weir flow: finite volume against the analytical result |
| [`gate_flow_fvm_comparison_demo.py`](https://github.com/alexiusacademia/pyopenchannel/blob/main/examples/gate_flow_fvm_comparison_demo.py) | Gate flow: finite volume against the analytical result |
| [`complete_gate_flow_analysis.py`](https://github.com/alexiusacademia/pyopenchannel/blob/main/examples/complete_gate_flow_analysis.py) | Gate flow profiles with tables |
| [`ogee_weir_fvm_profile_demo.py`](https://github.com/alexiusacademia/pyopenchannel/blob/main/examples/ogee_weir_fvm_profile_demo.py) | Profile over an ogee weir |
