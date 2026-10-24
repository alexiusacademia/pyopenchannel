# Weirs, gates and jumps

```python
import pyopenchannel as poc
from pyopenchannel import (
    WeirFlowSolver, WeirGeometry, WeirType,
    GateFlowSolver, GateGeometry, GateType,
    RVFSolver,
)

channel = poc.RectangularChannel(width=10.0)

# Ogee spillway: 5 m high, 10 m crest, 1 m of head
spillway = WeirGeometry(WeirType.OGEE_SPILLWAY, weir_height=5.0, crest_length=10.0)
weir = WeirFlowSolver().analyze_weir_flow(channel, spillway, approach_depth=6.0)
print(f"Spillway discharge: {weir.discharge:.2f} m³/s ({weir.weir_condition.value})")
# Spillway discharge: 22.02 m³/s (free_flow)

# Sluice gate: 0.5 m opening, 3 m upstream depth, 2.2 m tailwater
gate = GateGeometry(GateType.SLUICE, gate_width=10.0, gate_opening=0.5)
flow = GateFlowSolver().analyze_gate_flow(channel, gate, upstream_depth=3.0, downstream_depth=2.2)
print(f"Gate discharge: {flow.discharge:.2f} m³/s ({flow.flow_condition.value})")
# Gate discharge: 14.10 m³/s (submerged)

# Hydraulic jump from 0.4 m
jump = RVFSolver().analyze_hydraulic_jump(channel, discharge=30.0, upstream_depth=0.4)
print(f"Jump: {jump.upstream_depth:.2f} m -> {jump.downstream_depth:.3f} m, "
      f"energy loss {jump.energy_loss:.3f} m, {jump.jump_type.value}")
# Jump: 0.40 m -> 1.951 m, energy loss 1.195 m, oscillating
```

## Weirs

`WeirFlowSolver().analyze_weir_flow(channel, weir, approach_depth, discharge=None, downstream_depth=None)` takes the approach depth measured from the channel bottom; the head over the crest is the approach depth minus `weir_height`. Without a discharge it computes one.

| Weir type | Equation | Notes |
|---|---|---|
| `SHARP_CRESTED_RECTANGULAR` | Q = ⅔ Cd √(2g) Le H^1.5, Rehbock Cd = 0.602 + 0.083 H/P | Le = L − 0.2H (Francis) when the weir is narrower than the channel |
| `SHARP_CRESTED_TRIANGULAR` | Q = Cd (8/15) √(2g) tan(θ/2) H^2.5 | `vertex_angle` in degrees, default 90 |
| `SHARP_CRESTED_TRAPEZOIDAL` | rectangular part plus one V-notch with tan(θ/2) = z | `side_slope` z = 0.25 is the Cipolletti weir |
| `BROAD_CRESTED` | Q = Cd L √(2g) He^1.5, Cd 0.32 to 0.385 | He includes the approach velocity head |
| `OGEE_SPILLWAY` | Q = ⅔ Cd √(2g) L He^1.5, Cd = 0.738 (C = 2.18 SI) | high crest at its design head (USBR) |

**Submergence.** With a `downstream_depth` above the crest, the submergence ratio is S = H₂/H₁. Sharp-crested weirs follow Villemonte, Qs/Qf = (1 − S^n)^0.385, with n = 1.5 (2.5 for a V-notch). Broad-crested weirs and ogee spillways are unaffected up to their modular limit (0.8 and 0.75). The result's `weir_condition` is free flow above 99% of the free discharge, transitional above 90%, and submerged below.

`design_weir_dimensions(weir_type, target_discharge, available_head, channel)` sizes a weir: the crest length of rectangular, broad-crested and ogee weirs, the crest width of a trapezoidal weir, and the notch angle of a V-notch.

## Gates

`GateFlowSolver().analyze_gate_flow(channel, gate, upstream_depth, downstream_depth=None)` analyzes flow under a sluice (`GateType.SLUICE`) or radial gate. The jet contracts to Cc × opening (Cc about 0.61 for a sluice gate).

- **Free flow** comes from the energy equation between the upstream section and the vena contracta. The result's `discharge_coefficient` is Cd in Q = Cd b a √(2g y₁), about 0.58 when the upstream depth is six times the opening (Henry).
- **Submerged flow** applies when the tailwater is deeper than the sequent depth of the jet. The discharge and the depth behind the gate come from the energy equation upstream of the gate and the momentum equation downstream of it (Henderson). The flow is reported as `DROWNED` when it falls below half of the free flow.
- In free flow with a given tailwater, `jump_analysis` holds the hydraulic jump below the gate.

`design_gate_opening(...)` finds the opening that passes a target discharge.

## Hydraulic jumps

`RVFSolver().analyze_hydraulic_jump(channel, discharge, upstream_depth)` needs supercritical flow upstream. For rectangular channels the sequent depth is the Belanger equation, y₂/y₁ = ½(√(1 + 8F₁²) − 1); other sections solve the momentum equation. The result gives the sequent depth, energy loss, jump length and the USBR jump type (undular, weak, oscillating, steady, strong).

## Water surface profiles near structures

Dams, bridges, chutes and transitions are covered under [Water surface profiles](GVF_User_Guide.md): `DamAnalysis`, `BridgeAnalysis` (energy method with contraction and expansion losses, and choking), `ChuteAnalysis` and `ChannelTransition`.
