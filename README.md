# PyOpenChannel

A comprehensive Python library for open channel flow analysis and design. PyOpenChannel provides tools for hydraulic engineers, water resources professionals, and students to analyze and design open channel systems with ease and accuracy.

## Features

### 🏗️ Channel Geometries
- **Rectangular channels**: Standard rectangular cross-sections
- **Trapezoidal channels**: Most common natural and constructed channels  
- **Triangular channels**: V-shaped channels and ditches
- **Circular channels**: Pipes and culverts with free surface flow
- **Parabolic channels**: Efficient hydraulic sections

### 💧 Hydraulic Calculations
- **Manning's equation**: Uniform flow calculations
- **Critical depth**: Critical flow conditions and Froude number analysis
- **Normal depth**: Uniform flow depth calculations
- **Energy equation**: Specific energy and alternate depths
- **Momentum equation**: Hydraulic jump analysis

### 📐 Flow Analysis
- **Uniform flow**: Complete flow state analysis
- **Critical flow**: Critical conditions and flow regime classification
- **Gradually varied flow**: Water surface profile calculations (basic implementation)
- **Flow classification**: Subcritical, critical, and supercritical flow identification

### 🎯 Channel Design
- **Optimal sections**: Hydraulically efficient channel design
- **Economic sections**: Cost-optimized channel design
- **Channel sizing**: Capacity-based channel dimensioning
- **Design recommendations**: Freeboard, velocity limits, and side slopes

## Installation

### From Source (Development)

```bash
git clone https://github.com/yourusername/pyopenchannel.git
cd pyopenchannel
pip install -e .
```

### Requirements

- Python 3.12+
- No external dependencies for core functionality

## Quick Start

```python
import pyopenchannel as poc

# Create a rectangular channel
channel = poc.RectangularChannel(width=3.0)

# Calculate normal depth for given flow conditions
discharge = 5.0  # m³/s
slope = 0.001    # dimensionless
manning_n = 0.025

normal_depth = poc.NormalDepth.calculate(channel, discharge, slope, manning_n)
print(f"Normal depth: {normal_depth:.3f} m")

# Calculate critical depth
critical_depth = poc.CriticalDepth.calculate(channel, discharge)
print(f"Critical depth: {critical_depth:.3f} m")

# Analyze flow regime
if normal_depth > critical_depth:
    print("Flow regime: Subcritical (mild slope)")
else:
    print("Flow regime: Supercritical (steep slope)")
```

## Examples

### Basic Flow Analysis

```python
import pyopenchannel as poc

# Create a trapezoidal channel
channel = poc.TrapezoidalChannel(bottom_width=2.0, side_slope=1.5)

# Flow conditions
discharge = 10.0  # m³/s
slope = 0.002     # 0.2%
manning_n = 0.030 # earth channel

# Create uniform flow analyzer
uniform_flow = poc.UniformFlow(channel, slope, manning_n)
flow_state = uniform_flow.calculate_flow_state(discharge)

print(f"Flow depth: {flow_state.depth:.3f} m")
print(f"Flow velocity: {flow_state.velocity:.3f} m/s")
print(f"Froude number: {flow_state.froude_number:.3f}")
print(f"Flow type: {'Subcritical' if flow_state.is_subcritical else 'Supercritical'}")
```

### Channel Design

```python
import pyopenchannel as poc

# Design optimal rectangular channel
discharge = 15.0  # m³/s
slope = 0.001     # 0.1%
manning_n = 0.025 # concrete lining

result = poc.OptimalSections.rectangular(discharge, slope, manning_n)

print(f"Optimal width: {result.channel.width:.3f} m")
print(f"Flow depth: {result.depth:.3f} m")
print(f"Flow velocity: {result.velocity:.3f} m/s")
print(f"Recommended freeboard: {result.freeboard:.3f} m")
print(f"Total channel depth: {result.total_depth:.3f} m")
```

### Economic Design

```python
import pyopenchannel as poc

# Economic channel design considering costs
economic_designer = poc.EconomicSections(
    excavation_cost_per_m3=25.0,  # $/m³
    lining_cost_per_m2=50.0,      # $/m²
    land_cost_per_m2=100.0        # $/m²
)

result = economic_designer.design_rectangular(
    discharge=20.0, slope=0.001, manning_n=0.015
)

print(f"Economic width: {result.channel.width:.3f} m")
print(f"Flow depth: {result.depth:.3f} m")
print(f"Cost per meter: ${result.cost_per_meter:.2f}/m")
```

## API Reference

### Channel Geometries

#### RectangularChannel(width)
Rectangular channel cross-section.

**Parameters:**
- `width` (float): Channel bottom width in meters

**Methods:**
- `area(depth)`: Cross-sectional area
- `wetted_perimeter(depth)`: Wetted perimeter
- `top_width(depth)`: Top width (constant for rectangular)
- `hydraulic_radius(depth)`: Hydraulic radius (A/P)

#### TrapezoidalChannel(bottom_width, side_slope)
Trapezoidal channel cross-section.

**Parameters:**
- `bottom_width` (float): Channel bottom width in meters
- `side_slope` (float): Side slope ratio (horizontal:vertical)

#### TriangularChannel(side_slope)
Triangular channel cross-section.

**Parameters:**
- `side_slope` (float): Side slope ratio (horizontal:vertical)

#### CircularChannel(diameter)
Circular channel cross-section (pipe with free surface).

**Parameters:**
- `diameter` (float): Pipe diameter in meters

#### ParabolicChannel(shape_parameter)
Parabolic channel cross-section.

**Parameters:**
- `shape_parameter` (float): Shape parameter 'a' in y = ax²

### Hydraulic Calculations

#### ManningEquation
Static methods for Manning's equation calculations.

**Methods:**
- `discharge(area, hydraulic_radius, slope, manning_n)`: Calculate discharge
- `velocity(hydraulic_radius, slope, manning_n)`: Calculate velocity
- `required_slope(discharge, area, hydraulic_radius, manning_n)`: Calculate required slope

#### CriticalDepth
Critical depth and critical flow calculations.

**Methods:**
- `calculate(channel, discharge)`: Calculate critical depth
- `froude_number(velocity, hydraulic_depth)`: Calculate Froude number

#### NormalDepth
Normal depth calculations for uniform flow.

**Methods:**
- `calculate(channel, discharge, slope, manning_n)`: Calculate normal depth

### Flow Analysis

#### UniformFlow(channel, slope, manning_n)
Uniform flow analysis.

**Methods:**
- `calculate_flow_state(discharge)`: Get complete flow state
- `calculate_discharge(depth)`: Calculate discharge for given depth

#### EnergyEquation
Energy equation applications.

**Methods:**
- `specific_energy(depth, velocity)`: Calculate specific energy
- `minimum_specific_energy(channel, discharge)`: Minimum specific energy
- `alternate_depths(channel, discharge, specific_energy)`: Calculate alternate depths

### Design Tools

#### OptimalSections
Hydraulically optimal channel design.

**Methods:**
- `rectangular(discharge, slope, manning_n)`: Optimal rectangular section
- `trapezoidal(discharge, slope, manning_n, side_slope)`: Optimal trapezoidal section
- `triangular(discharge, slope, manning_n)`: Optimal triangular section

#### EconomicSections(excavation_cost, lining_cost, land_cost)
Economic channel design.

**Methods:**
- `design_rectangular(discharge, slope, manning_n)`: Economic rectangular design
- `design_trapezoidal(discharge, slope, manning_n, side_slope)`: Economic trapezoidal design

#### ChannelDesigner
General design utilities.

**Methods:**
- `calculate_freeboard(discharge, depth, velocity)`: Recommended freeboard
- `check_velocity_limits(velocity, channel_material)`: Velocity limit checks
- `recommend_side_slope(soil_type)`: Side slope recommendations
- `size_channel_for_capacity(discharge, slope, manning_n, channel_type)`: Size channel for capacity

## Constants and Utilities

### Physical Constants
- `GRAVITY`: 9.81 m/s²
- `WATER_DENSITY`: 1000 kg/m³
- `KINEMATIC_VISCOSITY`: 1.004×10⁻⁶ m²/s

### Manning's Roughness Coefficients
Pre-defined roughness coefficients for common channel materials:
- Concrete (smooth): 0.012
- Concrete (rough): 0.017
- Earth channels: 0.025-0.040
- Natural channels: 0.030-0.040

### Side Slope Recommendations
Recommended side slopes for different soil types:
- Rock: 0.25:1 to 0.5:1
- Firm earth: 1:1
- Ordinary earth: 1.5:1
- Sandy earth: 2:1
- Loose earth: 3:1

## Error Handling

PyOpenChannel includes comprehensive error handling with custom exceptions:

- `PyOpenChannelError`: Base exception class
- `InvalidGeometryError`: Invalid channel geometry parameters
- `ConvergenceError`: Iterative calculations failed to converge
- `InvalidFlowConditionError`: Physically impossible flow conditions
- `InvalidRoughnessError`: Invalid Manning's roughness coefficient
- `InvalidSlopeError`: Invalid channel slope

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=pyopenchannel

# Run specific test categories
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m "not slow"    # Skip slow tests
```

## Examples Directory

The `examples/` directory contains comprehensive examples:

- `basic_calculations.py`: Fundamental hydraulic calculations
- `channel_design.py`: Channel design and optimization examples

Run examples:

```bash
python examples/basic_calculations.py
python examples/channel_design.py
```

## Applications

PyOpenChannel is suitable for:

### Water Resources Engineering
- River and stream analysis
- Flood control channel design
- Irrigation and drainage systems
- Culvert and bridge hydraulics

### Civil Engineering
- Storm water management
- Highway drainage design
- Urban channel systems
- Infrastructure hydraulics

### Environmental Engineering
- Constructed wetlands
- Stream restoration
- Ecological channel design
- Natural channel analysis

### Education and Research
- Hydraulic engineering coursework
- Research applications
- Design verification
- Parametric studies

## Theory and Background

PyOpenChannel implements fundamental open channel flow theory:

### Manning's Equation
```
Q = (1/n) × A × R^(2/3) × S^(1/2)
```
Where:
- Q = discharge (m³/s)
- n = Manning's roughness coefficient
- A = cross-sectional area (m²)
- R = hydraulic radius (m)
- S = channel slope (dimensionless)

### Critical Flow Condition
```
Q² = g × A³ / T
```
Where:
- g = gravitational acceleration (m/s²)
- T = top width (m)

### Froude Number
```
Fr = V / √(g × D)
```
Where:
- V = average velocity (m/s)
- D = hydraulic depth (m)

Flow classification:
- Fr < 1: Subcritical flow
- Fr = 1: Critical flow  
- Fr > 1: Supercritical flow

## Contributing

Contributions are welcome! Please see CONTRIBUTING.md for guidelines.

### Development Setup

```bash
git clone https://github.com/yourusername/pyopenchannel.git
cd pyopenchannel
pip install -e ".[dev]"
```

### Running Tests

```bash
pytest
```

If using `uv`
```bash
uv run --with pytest pytest
```

### Code Style

This project uses:
- Black for code formatting
- isort for import sorting
- flake8 for linting
- mypy for type checking

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Based on established open channel flow theory from hydraulic engineering literature
- Inspired by classic texts: "Open Channel Hydraulics" by Ven Te Chow
- Developed for educational and professional use in hydraulic engineering

## Support

For questions, issues, or contributions:

1. Check the documentation and examples
2. Search existing issues on GitHub
3. Create a new issue with detailed information
4. Consider contributing improvements

## Roadmap

Future enhancements planned:

- [ ] Complete gradually varied flow implementation
- [ ] Hydraulic jump calculations
- [ ] Compound channel analysis
- [ ] Sediment transport basics
- [ ] Unsteady flow analysis
- [ ] Advanced optimization algorithms
- [ ] Integration with GIS data
- [ ] Visualization tools
- [ ] Additional channel shapes
- [ ] Performance optimizations

---

**PyOpenChannel** - Making open channel flow analysis accessible, accurate, and efficient.
