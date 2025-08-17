# PyOpenChannel - GVF API Reference

## Table of Contents

1. [Core Classes](#core-classes)
2. [Data Structures](#data-structures)
3. [Enumerations](#enumerations)
4. [Applications Module](#applications-module)
5. [Numerical Methods](#numerical-methods)
6. [Exceptions](#exceptions)
7. [Utility Functions](#utility-functions)

---

## Core Classes

### GVFSolver

Main class for solving gradually varied flow profiles.

```python
class GVFSolver:
    """
    Gradually Varied Flow solver using advanced numerical methods.
    
    This class provides high-accuracy computation of water surface profiles
    in open channels using adaptive Runge-Kutta integration methods.
    """
```

#### Constructor

```python
def __init__(
    self,
    integration_method: str = "dormand_prince",
    tolerance: float = 1e-6,
    max_iterations: int = 1000,
    min_step: float = 0.01,
    max_step: float = 100.0
) -> None
```

**Parameters:**
- `integration_method` (str): Integration method ("runge_kutta_4", "rkf45", "dormand_prince")
- `tolerance` (float): Convergence tolerance for adaptive stepping
- `max_iterations` (int): Maximum number of integration steps
- `min_step` (float): Minimum step size (m or ft)
- `max_step` (float): Maximum step size (m or ft)

#### Methods

##### solve_profile

```python
def solve_profile(
    self,
    channel: ChannelGeometry,
    discharge: float,
    slope: float,
    manning_n: float,
    x_start: float,
    x_end: float,
    boundary_depth: float,
    boundary_type: BoundaryType = BoundaryType.UPSTREAM_DEPTH,
    initial_step: Optional[float] = None
) -> GVFResult
```

Solve water surface profile using GVF equation.

**Parameters:**
- `channel` (ChannelGeometry): Channel geometry object
- `discharge` (float): Discharge (m³/s or ft³/s)
- `slope` (float): Channel slope (dimensionless)
- `manning_n` (float): Manning's roughness coefficient
- `x_start` (float): Starting x-coordinate (m or ft)
- `x_end` (float): Ending x-coordinate (m or ft)
- `boundary_depth` (float): Boundary condition depth (m or ft)
- `boundary_type` (BoundaryType): Type of boundary condition
- `initial_step` (float, optional): Initial step size (m or ft)

**Returns:**
- `GVFResult`: Complete analysis results

**Raises:**
- `PyOpenChannelError`: If computation fails
- `ConvergenceError`: If solver fails to converge
- `InvalidFlowConditionError`: If flow conditions are invalid

**Example:**
```python
solver = GVFSolver()
result = solver.solve_profile(
    channel=RectangularChannel(width=5.0),
    discharge=20.0,
    slope=0.001,
    manning_n=0.030,
    x_start=0.0,
    x_end=1000.0,
    boundary_depth=3.0,
    boundary_type=BoundaryType.UPSTREAM_DEPTH
)
```

---

### ProfileClassifier

Automatic classification of water surface profiles.

```python
class ProfileClassifier:
    """
    Automatic classification of water surface profiles.
    
    Identifies profile types (M1, M2, S1, etc.) based on hydraulic
    characteristics and provides engineering interpretation.
    """
```

#### Constructor

```python
def __init__(self, tolerance: float = 0.15) -> None
```

**Parameters:**
- `tolerance` (float): Tolerance for depth comparisons (fraction of depth)

#### Methods

##### classify_profile

```python
def classify_profile(
    self,
    gvf_result: GVFResult,
    channel: ChannelGeometry,
    discharge: float,
    slope: float,
    manning_n: float
) -> WaterSurfaceProfile
```

Classify water surface profile and provide engineering analysis.

**Parameters:**
- `gvf_result` (GVFResult): GVF computation results
- `channel` (ChannelGeometry): Channel geometry
- `discharge` (float): Discharge (m³/s or ft³/s)
- `slope` (float): Channel slope
- `manning_n` (float): Manning's roughness coefficient

**Returns:**
- `WaterSurfaceProfile`: Classified profile with engineering analysis

**Example:**
```python
classifier = ProfileClassifier(tolerance=0.15)
profile = classifier.classify_profile(
    gvf_result=result,
    channel=channel,
    discharge=discharge,
    slope=slope,
    manning_n=manning_n
)
```

---

### ProfileAnalyzer

Analysis and comparison of multiple water surface profiles.

```python
class ProfileAnalyzer:
    """
    Analysis and comparison of water surface profiles.
    
    Provides statistical analysis and comparison capabilities
    for multiple profiles.
    """
```

#### Methods

##### compare_profiles

```python
def compare_profiles(
    self,
    profiles: List[WaterSurfaceProfile]
) -> Dict[str, Any]
```

Compare multiple water surface profiles.

**Parameters:**
- `profiles` (List[WaterSurfaceProfile]): List of classified profiles

**Returns:**
- `Dict[str, Any]`: Comparison statistics and analysis

**Example:**
```python
analyzer = ProfileAnalyzer()
comparison = analyzer.compare_profiles([profile1, profile2, profile3])
```

---

## Data Structures

### GVFResult

Complete results from GVF analysis.

```python
@dataclass
class GVFResult:
    """
    Results from GVF analysis.
    
    Contains computed profile points, integration results,
    boundary conditions, and analysis summary.
    """
    profile_points: List[ProfilePoint]
    integration_result: IntegrationResult
    boundary_conditions: Dict[str, Any]
    channel_properties: Dict[str, Any]
    computation_summary: Dict[str, Any]
    events_detected: List[Any] = field(default_factory=list)
    validation_results: List[Dict[str, Any]] = field(default_factory=list)
    success: bool = True
    message: str = "GVF computation completed successfully"
```

#### Properties

##### length

```python
@property
def length(self) -> float
```

Total profile length (m or ft).

##### max_depth

```python
@property
def max_depth(self) -> float
```

Maximum depth in profile (m or ft).

##### min_depth

```python
@property
def min_depth(self) -> float
```

Minimum depth in profile (m or ft).

---

### ProfilePoint

Individual point on water surface profile.

```python
@dataclass
class ProfilePoint:
    """
    Represents a point on the water surface profile.
    
    Contains all hydraulic properties at a specific location.
    """
    x: float                    # Distance along channel (m or ft)
    depth: float               # Flow depth (m or ft)
    velocity: float            # Flow velocity (m/s or ft/s)
    discharge: float           # Discharge (m³/s or ft³/s)
    area: float               # Cross-sectional area (m² or ft²)
    top_width: float          # Top width (m or ft)
    hydraulic_radius: float   # Hydraulic radius (m or ft)
    froude_number: float      # Froude number (dimensionless)
    specific_energy: float    # Specific energy (m or ft)
    slope_friction: float     # Friction slope (dimensionless)
    slope_energy: float       # Energy slope (dimensionless)
```

---

### WaterSurfaceProfile

Classified water surface profile with engineering analysis.

```python
@dataclass
class WaterSurfaceProfile:
    """
    Classified water surface profile with engineering interpretation.
    
    Contains profile classification, hydraulic characteristics,
    and engineering significance.
    """
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

### AnalysisResult

Results from applications module analysis.

```python
@dataclass
class AnalysisResult:
    """
    Results from application analysis.
    
    Contains analysis results, design parameters,
    recommendations, and compliance notes.
    """
    analysis_type: AnalysisType
    success: bool
    message: str
    gvf_result: Optional[GVFResult] = None
    profile: Optional[WaterSurfaceProfile] = None
    design_parameters: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    compliance: List[str] = field(default_factory=list)
```

---

## Enumerations

### BoundaryType

Types of boundary conditions for GVF analysis.

```python
class BoundaryType(Enum):
    """Types of boundary conditions for GVF analysis."""
    UPSTREAM_DEPTH = "upstream_depth"
    DOWNSTREAM_DEPTH = "downstream_depth"
    CRITICAL_DEPTH = "critical_depth"
    NORMAL_DEPTH = "normal_depth"
    CONTROL_STRUCTURE = "control_structure"
```

### ProfileType

Water surface profile classifications.

```python
class ProfileType(Enum):
    """Water surface profile types."""
    M1 = "M1"  # Mild slope, backwater curve
    M2 = "M2"  # Mild slope, drawdown curve
    M3 = "M3"  # Mild slope, backwater curve (supercritical)
    S1 = "S1"  # Steep slope, backwater curve
    S2 = "S2"  # Steep slope, drawdown curve
    S3 = "S3"  # Steep slope, backwater curve (supercritical)
    C1 = "C1"  # Critical slope, backwater curve
    C3 = "C3"  # Critical slope, backwater curve (supercritical)
    H2 = "H2"  # Horizontal channel, subcritical
    H3 = "H3"  # Horizontal channel, supercritical
    A2 = "A2"  # Adverse slope, subcritical
    A3 = "A3"  # Adverse slope, supercritical
    UNIFORM = "UNIFORM"           # Uniform flow
    CRITICAL_FLOW = "CRITICAL"    # Critical flow
    UNKNOWN = "UNKNOWN"           # Cannot classify
```

### SlopeType

Channel slope classifications.

```python
class SlopeType(Enum):
    """Channel slope types."""
    MILD = "mild"           # yn > yc
    STEEP = "steep"         # yc > yn
    CRITICAL = "critical"   # yn ≈ yc
    HORIZONTAL = "horizontal"  # S₀ = 0
    ADVERSE = "adverse"     # S₀ < 0
```

### FlowRegime

Flow regime classifications.

```python
class FlowRegime(Enum):
    """Flow regime classification."""
    SUBCRITICAL = "subcritical"  # Fr < 1
    SUPERCRITICAL = "supercritical"  # Fr > 1
    CRITICAL = "critical"        # Fr ≈ 1
    MIXED = "mixed"             # Transitions between regimes
```

### AnalysisType

Types of hydraulic analysis.

```python
class AnalysisType(Enum):
    """Types of hydraulic analysis."""
    BACKWATER = "backwater"
    DRAWDOWN = "drawdown"
    TRANSITION = "transition"
    ENERGY_DISSIPATION = "energy_dissipation"
    FLOOD_ROUTING = "flood_routing"
```

### DesignCriteria

Design criteria standards.

```python
class DesignCriteria(Enum):
    """Design criteria standards."""
    CONSERVATIVE = "conservative"  # Maximum safety factors
    STANDARD = "standard"         # Typical engineering practice
    OPTIMIZED = "optimized"       # Efficient design with adequate safety
```

---

## Applications Module

### DamAnalysis

Comprehensive dam backwater analysis.

```python
class DamAnalysis:
    """
    Dam backwater analysis for flood studies and reservoir operations.
    
    Provides comprehensive analysis of dam-induced backwater effects,
    including flood elevation mapping and bridge clearance analysis.
    """
```

#### Constructor

```python
def __init__(self, design_criteria: DesignCriteria = DesignCriteria.STANDARD) -> None
```

#### Methods

##### analyze_backwater

```python
def analyze_backwater(
    self,
    channel: ChannelGeometry,
    discharge: float,
    slope: float,
    manning_n: float,
    dam_height: float,
    analysis_distance: float = 10000.0,
    bridge_locations: Optional[List[float]] = None
) -> AnalysisResult
```

Analyze dam backwater effects.

**Parameters:**
- `channel` (ChannelGeometry): Channel geometry
- `discharge` (float): Design discharge (m³/s or ft³/s)
- `slope` (float): Channel slope
- `manning_n` (float): Manning's roughness coefficient
- `dam_height` (float): Dam height above channel bottom (m or ft)
- `analysis_distance` (float): Distance upstream to analyze (m or ft)
- `bridge_locations` (List[float], optional): Bridge distances upstream (m or ft)

**Returns:**
- `AnalysisResult`: Comprehensive dam analysis results

---

### BridgeAnalysis

Bridge hydraulics analysis for clearance design.

```python
class BridgeAnalysis:
    """
    Bridge hydraulics analysis for clearance design and scour assessment.
    
    Provides comprehensive bridge hydraulic analysis including
    backwater effects and scour potential.
    """
```

#### Methods

##### analyze_bridge_hydraulics

```python
def analyze_bridge_hydraulics(
    self,
    approach_channel: ChannelGeometry,
    bridge_opening: ChannelGeometry,
    discharge: float,
    slope: float,
    manning_n: float,
    analysis_distance: float = 2000.0
) -> AnalysisResult
```

Analyze bridge hydraulics and design requirements.

**Parameters:**
- `approach_channel` (ChannelGeometry): Upstream channel geometry
- `bridge_opening` (ChannelGeometry): Bridge opening geometry
- `discharge` (float): Design discharge
- `slope` (float): Channel slope
- `manning_n` (float): Manning's roughness coefficient
- `analysis_distance` (float): Analysis distance upstream

**Returns:**
- `AnalysisResult`: Bridge analysis results

---

### ChuteAnalysis

Steep channel and chute analysis for energy dissipation.

```python
class ChuteAnalysis:
    """
    Steep channel and chute analysis for energy dissipation design.
    
    Provides analysis of steep channels, chutes, and spillways
    including hydraulic jump design.
    """
```

#### Methods

##### analyze_chute

```python
def analyze_chute(
    self,
    chute_channel: ChannelGeometry,
    tailwater_channel: ChannelGeometry,
    discharge: float,
    chute_slope: float,
    tailwater_slope: float,
    manning_n: float,
    chute_length: float = 500.0
) -> AnalysisResult
```

Analyze steep chute and energy dissipation.

**Parameters:**
- `chute_channel` (ChannelGeometry): Steep chute geometry
- `tailwater_channel` (ChannelGeometry): Downstream channel geometry
- `discharge` (float): Design discharge
- `chute_slope` (float): Steep chute slope
- `tailwater_slope` (float): Downstream channel slope
- `manning_n` (float): Manning's roughness coefficient
- `chute_length` (float): Length of steep chute

**Returns:**
- `AnalysisResult`: Chute analysis results

---

### ChannelTransition

Channel transition analysis for geometry changes.

```python
class ChannelTransition:
    """
    Channel transition analysis for geometry and slope changes.
    
    Provides analysis of channel transitions including
    contractions, expansions, and slope changes.
    """
```

#### Methods

##### analyze_transition

```python
def analyze_transition(
    self,
    upstream_channel: ChannelGeometry,
    downstream_channel: ChannelGeometry,
    discharge: float,
    upstream_slope: float,
    downstream_slope: float,
    manning_n: float,
    transition_length: float = 100.0
) -> AnalysisResult
```

Analyze channel transition effects.

**Parameters:**
- `upstream_channel` (ChannelGeometry): Upstream channel geometry
- `downstream_channel` (ChannelGeometry): Downstream channel geometry
- `discharge` (float): Design discharge
- `upstream_slope` (float): Upstream channel slope
- `downstream_slope` (float): Downstream channel slope
- `manning_n` (float): Manning's roughness coefficient
- `transition_length` (float): Length of transition zone

**Returns:**
- `AnalysisResult`: Transition analysis results

---

## Numerical Methods

### Integration Methods

Available integration methods for GVF solver:

#### Runge-Kutta 4th Order ("runge_kutta_4")
- **Accuracy**: 4th order
- **Stability**: Good
- **Performance**: Fast
- **Use case**: General purpose, stable solutions

#### Runge-Kutta-Fehlberg 4(5) ("rkf45")
- **Accuracy**: 4th/5th order adaptive
- **Stability**: Excellent
- **Performance**: Moderate
- **Use case**: High accuracy requirements

#### Dormand-Prince ("dormand_prince")
- **Accuracy**: 5th order adaptive
- **Stability**: Excellent
- **Performance**: Moderate
- **Use case**: Default method, best overall performance

### Adaptive Stepping

All integration methods support adaptive step sizing:

```python
solver = GVFSolver(
    integration_method="dormand_prince",
    tolerance=1e-6,      # Convergence tolerance
    min_step=0.01,       # Minimum step size
    max_step=100.0       # Maximum step size
)
```

### Event Detection

Automatic detection of hydraulic events:

- **Critical depth transitions**
- **Hydraulic jumps**
- **Flow regime changes**
- **Shock waves**

Events are reported in `GVFResult.events_detected`.

### Analytical Validation

Cross-checking against analytical solutions where available:

- **Uniform flow validation**
- **Critical flow validation**
- **Energy conservation checks**
- **Momentum conservation checks**

Validation results are reported in `GVFResult.validation_results`.

---

## Exceptions

### PyOpenChannelError

Base exception for PyOpenChannel errors.

```python
class PyOpenChannelError(Exception):
    """Base exception for PyOpenChannel errors."""
    pass
```

### ConvergenceError

Raised when numerical methods fail to converge.

```python
class ConvergenceError(PyOpenChannelError):
    """Raised when numerical methods fail to converge."""
    pass
```

### InvalidFlowConditionError

Raised when flow conditions are physically invalid.

```python
class InvalidFlowConditionError(PyOpenChannelError):
    """Raised when flow conditions are invalid."""
    pass
```

### InvalidGeometryError

Raised when channel geometry is invalid.

```python
class InvalidGeometryError(PyOpenChannelError):
    """Raised when channel geometry is invalid."""
    pass
```

---

## Utility Functions

### Unit System Functions

```python
# Set unit system
poc.set_unit_system(poc.UnitSystem.SI)
poc.set_unit_system(poc.UnitSystem.US_CUSTOMARY)

# Get current unit system
units = poc.get_unit_system()

# Get unit-aware constants
gravity = poc.get_gravity()          # 9.81 m/s² or 32.2 ft/s²
manning_factor = poc.get_manning_factor()  # 1.0 or 1.49
```

### Conversion Functions

```python
# Length conversions
meters = poc.ft_to_m(feet)
feet = poc.m_to_ft(meters)

# Discharge conversions
cms = poc.cfs_to_cms(cfs)
cfs = poc.cms_to_cfs(cms)
cms = poc.gpm_to_cms(gpm)
gpm = poc.cms_to_gpm(cms)
```

### Validation Functions

```python
from pyopenchannel.validators import (
    validate_positive,
    validate_depth,
    validate_discharge,
    validate_slope,
    validate_manning_n
)

# Validate inputs
discharge = validate_discharge(discharge)
depth = validate_depth(depth)
slope = validate_slope(slope)
manning_n = validate_manning_n(manning_n)
```

---

## Usage Examples

### Basic GVF Analysis

```python
import pyopenchannel as poc
from pyopenchannel.gvf import GVFSolver, BoundaryType

# Setup
poc.set_unit_system(poc.UnitSystem.SI)
solver = GVFSolver()

# Define problem
channel = poc.RectangularChannel(width=5.0)
result = solver.solve_profile(
    channel=channel,
    discharge=20.0,
    slope=0.001,
    manning_n=0.030,
    x_start=0.0,
    x_end=1000.0,
    boundary_depth=3.0,
    boundary_type=BoundaryType.UPSTREAM_DEPTH
)

# Process results
if result.success:
    depths = [p.depth for p in result.profile_points]
    print(f"Depth range: {min(depths):.3f} - {max(depths):.3f} m")
```

### Profile Classification

```python
from pyopenchannel.gvf import ProfileClassifier

classifier = ProfileClassifier()
profile = classifier.classify_profile(
    gvf_result=result,
    channel=channel,
    discharge=20.0,
    slope=0.001,
    manning_n=0.030
)

print(f"Profile type: {profile.profile_type.value}")
print(f"Engineering significance: {profile.engineering_significance}")
```

### Applications Module

```python
from pyopenchannel.gvf import DamAnalysis, DesignCriteria

dam_analyzer = DamAnalysis(design_criteria=DesignCriteria.STANDARD)
result = dam_analyzer.analyze_backwater(
    channel=channel,
    discharge=100.0,
    slope=0.0005,
    manning_n=0.035,
    dam_height=4.0
)

if result.success:
    params = result.design_parameters
    print(f"Backwater extent: {params['backwater_extent']/1000:.1f} km")
    for rec in result.recommendations:
        print(f"• {rec}")
```

---

This API reference provides complete documentation for all public interfaces in the PyOpenChannel GVF module. For usage examples and tutorials, see the User Guide and example files.
