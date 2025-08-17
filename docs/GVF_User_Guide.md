# PyOpenChannel - Gradually Varied Flow (GVF) User Guide

## Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Core Concepts](#core-concepts)
4. [Basic Usage](#basic-usage)
5. [Profile Classification](#profile-classification)
6. [Applications Module](#applications-module)
7. [Advanced Features](#advanced-features)
8. [Examples and Tutorials](#examples-and-tutorials)
9. [API Reference](#api-reference)
10. [Best Practices](#best-practices)
11. [Troubleshooting](#troubleshooting)

---

## Introduction

The Gradually Varied Flow (GVF) module in PyOpenChannel provides professional-grade analysis of water surface profiles in open channels. This comprehensive system combines advanced numerical methods with engineering expertise to deliver accurate, reliable results for hydraulic design and analysis.

### Key Features

- **High-accuracy numerical integration** using Runge-Kutta methods (RK4, RKF45, Dormand-Prince)
- **Automatic profile classification** (M1, M2, M3, S1, S2, S3, C1, C3, H2, H3, A2, A3)
- **Professional applications** for dams, bridges, chutes, and transitions
- **Event detection** for critical depth transitions and hydraulic jumps
- **Analytical validation** against known solutions
- **Multiple channel geometries** (rectangular, trapezoidal, circular, etc.)
- **Unit system support** (SI and US Customary)
- **Engineering-grade documentation** and reporting

### Applications

- **Flood studies** and dam backwater analysis
- **Bridge hydraulics** and clearance design
- **Spillway** and energy dissipation design
- **Channel transitions** and modifications
- **Environmental impact** assessment
- **Regulatory compliance** documentation

---

## Getting Started

### Installation

```bash
pip install pyopenchannel
```

### Basic Import

```python
import pyopenchannel as poc
from pyopenchannel.gvf import GVFSolver, BoundaryType, ProfileClassifier
```

### Quick Example

```python
# Set up the analysis
poc.set_unit_system(poc.UnitSystem.SI)
solver = GVFSolver()

# Define channel and conditions
channel = poc.RectangularChannel(width=5.0)  # 5m wide
discharge = 20.0    # m³/s
slope = 0.001       # 0.1%
manning_n = 0.030

# Solve GVF profile
result = solver.solve_profile(
    channel=channel,
    discharge=discharge,
    slope=slope,
    manning_n=manning_n,
    x_start=0.0,
    x_end=1000.0,
    boundary_depth=3.0,
    boundary_type=BoundaryType.UPSTREAM_DEPTH
)

print(f"Profile computed with {len(result.profile_points)} points")
```

---

## Core Concepts

### Water Surface Profiles

Water surface profiles describe how the depth of flow varies along a channel. The GVF equation governs this variation:

```
dy/dx = (S₀ - Sf) / (1 - Fr²)
```

Where:
- `dy/dx` = water surface slope
- `S₀` = channel slope
- `Sf` = friction slope
- `Fr` = Froude number

### Profile Types

#### M Profiles (Mild Slope: yn > yc)
- **M1**: y > yn > yc (backwater curve)
- **M2**: yn > y > yc (drawdown curve)
- **M3**: yn > yc > y (backwater curve)

#### S Profiles (Steep Slope: yc > yn)
- **S1**: y > yc > yn (backwater curve)
- **S2**: yc > y > yn (drawdown curve)
- **S3**: yc > yn > y (backwater curve)

#### Other Profiles
- **C1, C3**: Critical slope profiles
- **H2, H3**: Horizontal channel profiles
- **A2, A3**: Adverse slope profiles

### Boundary Conditions

- **UPSTREAM_DEPTH**: Specify depth at upstream end
- **DOWNSTREAM_DEPTH**: Specify depth at downstream end
- **CRITICAL_DEPTH**: Use critical depth as boundary
- **NORMAL_DEPTH**: Use normal depth as boundary
- **CONTROL_STRUCTURE**: Specify control structure effects

---

## Basic Usage

### 1. Channel Definition

```python
# Rectangular channel
channel = poc.RectangularChannel(width=6.0)

# Trapezoidal channel
channel = poc.TrapezoidalChannel(bottom_width=4.0, side_slope=1.5)

# Circular channel
channel = poc.CircularChannel(diameter=2.0)
```

### 2. GVF Solver Setup

```python
solver = GVFSolver()

# Optional: Configure solver parameters
solver = GVFSolver(
    integration_method="dormand_prince",  # RK4, RKF45, or dormand_prince
    tolerance=1e-6,
    max_iterations=1000
)
```

### 3. Profile Computation

```python
result = solver.solve_profile(
    channel=channel,
    discharge=25.0,           # m³/s or ft³/s
    slope=0.0008,            # Channel slope
    manning_n=0.025,         # Roughness coefficient
    x_start=0.0,             # Starting position
    x_end=2000.0,            # Ending position
    boundary_depth=4.0,      # Boundary depth
    boundary_type=BoundaryType.UPSTREAM_DEPTH,
    initial_step=10.0        # Initial step size (optional)
)
```

### 4. Results Analysis

```python
if result.success:
    print(f"Profile length: {result.length:.1f} m")
    print(f"Points computed: {len(result.profile_points)}")
    
    # Extract data
    distances = [p.x for p in result.profile_points]
    depths = [p.depth for p in result.profile_points]
    velocities = [p.velocity for p in result.profile_points]
    froude_numbers = [p.froude_number for p in result.profile_points]
    
    print(f"Depth range: {min(depths):.3f} - {max(depths):.3f} m")
    print(f"Velocity range: {min(velocities):.3f} - {max(velocities):.3f} m/s")
else:
    print(f"Computation failed: {result.message}")
```

---

## Profile Classification

The profile classification system automatically identifies water surface profile types and provides engineering interpretation.

### Basic Classification

```python
classifier = ProfileClassifier(tolerance=0.15)  # 15% tolerance

profile = classifier.classify_profile(
    gvf_result=result,
    channel=channel,
    discharge=discharge,
    slope=slope,
    manning_n=manning_n
)

print(f"Profile type: {profile.profile_type.value}")
print(f"Slope type: {profile.slope_type.value}")
print(f"Flow regime: {profile.flow_regime.value}")
print(f"Engineering significance: {profile.engineering_significance}")
```

### Profile Properties

```python
# Geometric properties
print(f"Profile length: {profile.length:.1f} m")
print(f"Depth range: {profile.min_depth:.3f} - {profile.max_depth:.3f} m")

# Hydraulic properties
print(f"Normal depth: {profile.normal_depth:.3f} m")
print(f"Critical depth: {profile.critical_depth:.3f} m")

# Analysis results
print(f"Curvature: {profile.curvature}")
print(f"Asymptotic behavior: {profile.asymptotic_behavior}")
```

### Multi-Profile Analysis

```python
analyzer = ProfileAnalyzer()

# Compare multiple profiles
profiles = [profile1, profile2, profile3]
comparison = analyzer.compare_profiles(profiles)

print(f"Total profiles: {comparison['total_profiles']}")
print(f"Profile types: {', '.join(comparison['profile_types'])}")
print(f"Length range: {comparison['length_range'][0]:.0f} - {comparison['length_range'][1]:.0f} m")
```

---

## Applications Module

The applications module provides high-level interfaces for common engineering scenarios.

### Dam Backwater Analysis

```python
from pyopenchannel.gvf import DamAnalysis, DesignCriteria

# Initialize with design criteria
dam_analyzer = DamAnalysis(design_criteria=DesignCriteria.STANDARD)

# Perform analysis
result = dam_analyzer.analyze_backwater(
    channel=channel,
    discharge=100.0,
    slope=0.0005,
    manning_n=0.035,
    dam_height=4.0,
    analysis_distance=8000.0,
    bridge_locations=[1000.0, 2500.0, 5000.0]
)

if result.success:
    params = result.design_parameters
    print(f"Backwater extent: {params['backwater_extent']/1000:.1f} km")
    print(f"Maximum flood elevation: {params['flood_elevation']:.2f} m")
    
    # Bridge clearances
    for loc, data in params['bridge_clearances'].items():
        print(f"Bridge at {loc/1000:.1f}km: {data['required_elevation']:.2f} m")
    
    # Recommendations
    for rec in result.recommendations:
        print(f"• {rec}")
```

### Bridge Hydraulic Analysis

```python
from pyopenchannel.gvf import BridgeAnalysis

bridge_analyzer = BridgeAnalysis(design_criteria=DesignCriteria.STANDARD)

result = bridge_analyzer.analyze_bridge_hydraulics(
    approach_channel=poc.RectangularChannel(width=8.0),
    bridge_opening=poc.RectangularChannel(width=6.0),
    discharge=50.0,
    slope=0.001,
    manning_n=0.030
)

if result.success:
    params = result.design_parameters
    print(f"Required clearance: {params['required_clearance']:.2f} m")
    print(f"Estimated scour depth: {params['estimated_scour_depth']:.2f} m")
    print(f"Backwater rise: {params['backwater_rise']:.3f} m")
```

### Chute Energy Dissipation

```python
from pyopenchannel.gvf import ChuteAnalysis

chute_analyzer = ChuteAnalysis(design_criteria=DesignCriteria.STANDARD)

result = chute_analyzer.analyze_chute(
    chute_channel=poc.RectangularChannel(width=4.0),
    tailwater_channel=poc.RectangularChannel(width=5.0),
    discharge=25.0,
    chute_slope=0.08,      # 8% steep slope
    tailwater_slope=0.002, # 0.2% mild slope
    manning_n=0.025
)

if result.success:
    params = result.design_parameters
    print(f"Exit Froude number: {params['exit_froude']:.2f}")
    print(f"Energy dissipated: {params['energy_dissipated']:.2f} m")
    
    if params['jump_required']:
        print(f"Hydraulic jump length: {params['jump_length']:.1f} m")
```

### Channel Transitions

```python
from pyopenchannel.gvf import ChannelTransition

transition_analyzer = ChannelTransition(design_criteria=DesignCriteria.STANDARD)

result = transition_analyzer.analyze_transition(
    upstream_channel=poc.TrapezoidalChannel(bottom_width=6.0, side_slope=2.0),
    downstream_channel=poc.RectangularChannel(width=5.0),
    discharge=30.0,
    upstream_slope=0.0008,
    downstream_slope=0.0012,
    manning_n=0.030
)

if result.success:
    params = result.design_parameters
    print(f"Energy loss: {params['energy_loss']:.3f} m")
    print(f"Contraction ratio: {params['contraction_ratio']:.3f}")
```

---

## Advanced Features

### Custom Integration Methods

```python
# Use specific integration method
solver = GVFSolver(integration_method="runge_kutta_4")

# Configure adaptive stepping
solver = GVFSolver(
    integration_method="dormand_prince",
    tolerance=1e-8,
    min_step=0.1,
    max_step=50.0
)
```

### Event Detection

```python
# Events are automatically detected during integration
if result.events_detected:
    for event in result.events_detected:
        print(f"Event: {event.type} at x = {event.location:.1f} m")
```

### Analytical Validation

```python
# Validation results are included in GVF results
if result.validation_results:
    for validation in result.validation_results:
        print(f"Validation: {validation['method']}")
        print(f"Error: {validation['error']:.2e}")
```

### Custom Boundary Conditions

```python
# Use control structure boundary
result = solver.solve_profile(
    # ... other parameters ...
    boundary_depth=critical_depth * 1.1,
    boundary_type=BoundaryType.CONTROL_STRUCTURE
)
```

---

## Examples and Tutorials

### Complete Examples Available

1. **`gvf_simple_example.py`** - Basic introduction, no dependencies
2. **`gvf_basic_usage.py`** - Comprehensive tutorial with visualization
3. **`gvf_profile_classification.py`** - Advanced classification features
4. **`gvf_dam_backwater_analysis.py`** - Professional flood analysis
5. **`gvf_channel_transitions.py`** - Transition analysis
6. **`gvf_applications_demo.py`** - Applications module demonstration

### Running Examples

```bash
# Basic introduction
python3 examples/gvf_simple_example.py

# Comprehensive tutorial
python3 examples/gvf_basic_usage.py

# Applications demo
python3 examples/gvf_applications_demo.py
```

### Tutorial Progression

1. **Start with simple example** - Learn basic API
2. **Try profile classification** - Understand automatic analysis
3. **Explore applications** - Use high-level interfaces
4. **Advanced features** - Custom configurations and validation

---

## API Reference

### Core Classes

#### GVFSolver

```python
class GVFSolver:
    def __init__(self, integration_method="dormand_prince", tolerance=1e-6, max_iterations=1000)
    def solve_profile(self, channel, discharge, slope, manning_n, x_start, x_end, 
                     boundary_depth, boundary_type, initial_step=None) -> GVFResult
```

#### ProfileClassifier

```python
class ProfileClassifier:
    def __init__(self, tolerance=0.15)
    def classify_profile(self, gvf_result, channel, discharge, slope, manning_n) -> WaterSurfaceProfile
```

#### Applications

```python
class DamAnalysis:
    def __init__(self, design_criteria=DesignCriteria.STANDARD)
    def analyze_backwater(self, channel, discharge, slope, manning_n, dam_height, 
                         analysis_distance, bridge_locations=None) -> AnalysisResult

class BridgeAnalysis:
    def analyze_bridge_hydraulics(self, approach_channel, bridge_opening, discharge, 
                                 slope, manning_n, analysis_distance=2000.0) -> AnalysisResult

class ChuteAnalysis:
    def analyze_chute(self, chute_channel, tailwater_channel, discharge, chute_slope, 
                     tailwater_slope, manning_n, chute_length=500.0) -> AnalysisResult

class ChannelTransition:
    def analyze_transition(self, upstream_channel, downstream_channel, discharge, 
                          upstream_slope, downstream_slope, manning_n, 
                          transition_length=100.0) -> AnalysisResult
```

### Data Structures

#### GVFResult

```python
@dataclass
class GVFResult:
    profile_points: List[ProfilePoint]
    integration_result: IntegrationResult
    boundary_conditions: Dict[str, Any]
    channel_properties: Dict[str, Any]
    computation_summary: Dict[str, Any]
    events_detected: List[Any]
    validation_results: List[Dict[str, Any]]
    success: bool
    message: str
```

#### ProfilePoint

```python
@dataclass
class ProfilePoint:
    x: float                    # Distance along channel
    depth: float               # Flow depth
    velocity: float            # Flow velocity
    discharge: float           # Discharge
    area: float               # Cross-sectional area
    top_width: float          # Top width
    hydraulic_radius: float   # Hydraulic radius
    froude_number: float      # Froude number
    specific_energy: float    # Specific energy
    slope_friction: float     # Friction slope
    slope_energy: float       # Energy slope
```

#### WaterSurfaceProfile

```python
@dataclass
class WaterSurfaceProfile:
    profile_type: ProfileType
    slope_type: SlopeType
    flow_regime: FlowRegime
    normal_depth: float
    critical_depth: float
    length: float
    min_depth: float
    max_depth: float
    curvature: str
    asymptotic_behavior: str
    engineering_significance: str
```

---

## Best Practices

### 1. Channel Definition

```python
# Good: Use appropriate geometry
channel = poc.TrapezoidalChannel(bottom_width=5.0, side_slope=2.0)  # Natural channel

# Avoid: Unrealistic geometries
channel = poc.RectangularChannel(width=0.1)  # Too narrow
```

### 2. Boundary Conditions

```python
# Good: Use physically reasonable boundary depths
boundary_depth = max(critical_depth * 1.1, normal_depth * 0.9)

# Avoid: Extreme boundary conditions
boundary_depth = critical_depth * 10  # Unrealistically high
```

### 3. Analysis Distance

```python
# Good: Appropriate analysis distance
analysis_distance = 10 * normal_depth * discharge**0.5  # Rule of thumb

# Avoid: Excessive distances without justification
analysis_distance = 100000.0  # 100km may be excessive
```

### 4. Step Size Selection

```python
# Good: Let adaptive stepping handle it
result = solver.solve_profile(..., initial_step=None)  # Auto-select

# Or provide reasonable initial step
initial_step = min(50.0, analysis_distance / 100)  # 1% of distance
```

### 5. Error Handling

```python
try:
    result = solver.solve_profile(...)
    if result.success:
        # Process results
        pass
    else:
        print(f"Analysis failed: {result.message}")
except Exception as e:
    print(f"Computation error: {e}")
```

### 6. Unit Consistency

```python
# Always set unit system first
poc.set_unit_system(poc.UnitSystem.SI)

# Verify units are consistent
current_units = poc.get_unit_system()
print(f"Using {current_units.length_unit} for length")
```

---

## Troubleshooting

### Common Issues

#### 1. Convergence Problems

**Problem**: Solver fails to converge
```
GVF solver error: Maximum iterations exceeded
```

**Solutions**:
- Reduce tolerance: `solver = GVFSolver(tolerance=1e-4)`
- Increase max iterations: `solver = GVFSolver(max_iterations=2000)`
- Check boundary conditions for physical reasonableness
- Try different integration method: `integration_method="runge_kutta_4"`

#### 2. Unrealistic Results

**Problem**: Computed depths are unrealistic

**Solutions**:
- Verify channel geometry parameters
- Check discharge and slope values
- Ensure Manning's n is appropriate for channel type
- Validate boundary depth against normal and critical depths

#### 3. Profile Classification Issues

**Problem**: Profile classified as UNKNOWN

**Solutions**:
- Increase classifier tolerance: `ProfileClassifier(tolerance=0.2)`
- Check if boundary conditions create transitional profiles
- Verify slope type (mild vs steep) is correct
- Ensure sufficient profile length for classification

#### 4. Import Errors

**Problem**: Module import failures

**Solutions**:
- Ensure PyOpenChannel is properly installed: `pip install pyopenchannel`
- Check Python version compatibility (3.9+)
- Verify all dependencies are installed

#### 5. Unit System Issues

**Problem**: Results don't match expected values

**Solutions**:
- Verify unit system: `poc.get_unit_system()`
- Ensure all inputs use consistent units
- Check Manning's n values for unit system
- Validate gravity constant: `poc.get_gravity()`

### Performance Optimization

#### 1. Large Analysis Distances

```python
# Use larger initial step for long profiles
initial_step = analysis_distance / 200  # Start with 0.5% of distance

# Consider breaking into segments
segment_length = 5000.0  # 5km segments
```

#### 2. Multiple Analyses

```python
# Reuse solver instance
solver = GVFSolver()
for scenario in scenarios:
    result = solver.solve_profile(...)  # Reuse solver
```

#### 3. Memory Management

```python
# For many profiles, process results immediately
for case in cases:
    result = solver.solve_profile(...)
    process_results(result)  # Process immediately
    del result  # Free memory
```

### Getting Help

1. **Check examples** - Most issues are covered in examples
2. **Review documentation** - This guide covers common scenarios
3. **Validate inputs** - Ensure physical reasonableness
4. **Start simple** - Use basic examples first
5. **Report issues** - Use GitHub issue tracker for bugs

---

## Conclusion

The PyOpenChannel GVF module provides professional-grade gradually varied flow analysis with:

- **Accurate numerical methods** for reliable results
- **Automatic profile classification** for engineering insight
- **High-level applications** for common design scenarios
- **Comprehensive documentation** for effective use
- **Professional output** suitable for engineering reports

This system enables hydraulic engineers to perform sophisticated water surface profile analysis with confidence, supporting everything from academic research to professional consulting and regulatory compliance.

For additional support, examples, and updates, visit the PyOpenChannel documentation and GitHub repository.
