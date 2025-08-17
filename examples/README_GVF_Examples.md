# PyOpenChannel - GVF Examples Guide

This directory contains comprehensive examples demonstrating the Gradually Varied Flow (GVF) capabilities of PyOpenChannel. These examples showcase professional-grade hydraulic analysis for real-world engineering applications.

## 📚 Available Examples

### 1. **Simple GVF Example** (`gvf_simple_example.py`)
**Perfect for beginners and quick testing**

- **Purpose**: Clean, straightforward demonstration of core GVF functionality
- **Dependencies**: None (no matplotlib required)
- **Features**:
  - M1 profiles (dam backwater)
  - M2 profiles (channel entrance)
  - Multiple channel geometries
  - Automatic profile classification
  - Easy-to-understand output

**Use Cases**: Learning the API, quick validation, integration testing

```bash
python3 examples/gvf_simple_example.py
```

### 2. **Basic GVF Usage** (`gvf_basic_usage.py`)
**Comprehensive introduction with visualization**

- **Purpose**: Detailed demonstration of GVF solver capabilities
- **Dependencies**: matplotlib (optional for plotting)
- **Features**:
  - M1, M2, and S1 profile analysis
  - Detailed hydraulic calculations
  - Engineering analysis and interpretation
  - Professional visualization (if matplotlib available)
  - Multiple channel types and boundary conditions

**Use Cases**: Learning GVF theory, detailed analysis, report generation

```bash
python3 examples/gvf_basic_usage.py
```

### 3. **Profile Classification** (`gvf_profile_classification.py`)
**Advanced profile analysis and classification**

- **Purpose**: Demonstrate automatic water surface profile classification
- **Dependencies**: matplotlib (optional for plotting)
- **Features**:
  - Automatic profile type identification (M1, M2, S1, etc.)
  - Slope classification (mild, steep, critical, horizontal, adverse)
  - Flow regime analysis (subcritical, supercritical, critical)
  - Engineering significance interpretation
  - Multi-profile comparison
  - Professional reporting

**Use Cases**: Automated analysis, engineering documentation, research

```bash
python3 examples/gvf_profile_classification.py
```

### 4. **Dam Backwater Analysis** (`gvf_dam_backwater_analysis.py`)
**Professional flood and dam analysis**

- **Purpose**: Comprehensive dam backwater analysis for engineering design
- **Dependencies**: matplotlib, numpy (optional for advanced plotting)
- **Features**:
  - Multiple dam scenarios and heights
  - Flood elevation mapping
  - Bridge clearance analysis
  - Sensitivity analysis
  - Professional engineering reporting
  - Regulatory compliance documentation

**Use Cases**: Flood studies, dam design, bridge hydraulics, environmental impact

```bash
python3 examples/gvf_dam_backwater_analysis.py
```

### 5. **Channel Transitions** (`gvf_channel_transitions.py`)
**Advanced transition analysis**

- **Purpose**: Analysis of hydraulic transitions in channel systems
- **Dependencies**: None
- **Features**:
  - Bridge contractions and expansions
  - Slope changes (mild to steep, steep to mild)
  - Cross-section changes (rectangular to trapezoidal)
  - Culvert analysis
  - Energy loss calculations
  - Hydraulic jump identification
  - Design guidelines and recommendations

**Use Cases**: Bridge design, culvert sizing, channel modifications, energy dissipation

```bash
python3 examples/gvf_channel_transitions.py
```

## 🎯 Example Selection Guide

| **Your Goal** | **Recommended Example** | **Why** |
|---------------|------------------------|---------|
| **Learn GVF basics** | `gvf_simple_example.py` | Clean, easy to understand |
| **Understand theory** | `gvf_basic_usage.py` | Detailed explanations |
| **Automate analysis** | `gvf_profile_classification.py` | Professional classification |
| **Flood studies** | `gvf_dam_backwater_analysis.py` | Comprehensive flood analysis |
| **Bridge/culvert design** | `gvf_channel_transitions.py` | Transition analysis |
| **Quick validation** | `gvf_simple_example.py` | Fast, no dependencies |
| **Research/documentation** | `gvf_profile_classification.py` | Professional reporting |

## 🔧 Technical Features Demonstrated

### **Core GVF Capabilities**
- ✅ High-accuracy numerical integration (RK4, RKF45, Dormand-Prince)
- ✅ Adaptive step sizing for optimal performance
- ✅ Event detection (critical depth transitions, hydraulic jumps)
- ✅ Analytical validation and cross-checking
- ✅ Multiple boundary condition types
- ✅ Professional error handling

### **Profile Classification System**
- ✅ Automatic profile type identification (M1, M2, M3, S1, S2, S3, C1, C3, H2, H3, A2, A3)
- ✅ Slope classification (mild, steep, critical, horizontal, adverse)
- ✅ Flow regime analysis (subcritical, supercritical, critical, mixed)
- ✅ Engineering significance interpretation
- ✅ Curvature analysis and asymptotic behavior
- ✅ Multi-profile comparison capabilities

### **Engineering Applications**
- ✅ Dam backwater analysis and flood mapping
- ✅ Bridge hydraulics and clearance design
- ✅ Channel transition analysis
- ✅ Culvert sizing and inlet/outlet control
- ✅ Energy dissipation structure design
- ✅ Hydraulic jump location and characteristics
- ✅ Scour analysis and protection design

### **Channel Geometries Supported**
- ✅ Rectangular channels
- ✅ Trapezoidal channels
- ✅ Triangular channels
- ✅ Circular channels
- ✅ Parabolic channels
- ✅ Custom geometries (via base class)

### **Unit Systems**
- ✅ SI (metric) units - meters, m³/s, m/s
- ✅ US Customary units - feet, ft³/s, ft/s
- ✅ Automatic unit-aware calculations
- ✅ Seamless unit system switching

## 🚀 Getting Started

### **Prerequisites**
```bash
# Core functionality (always required)
pip install pyopenchannel

# Optional for advanced plotting
pip install matplotlib numpy
```

### **Quick Start**
1. **Start with the simple example**:
   ```bash
   python3 examples/gvf_simple_example.py
   ```

2. **Explore profile classification**:
   ```bash
   python3 examples/gvf_profile_classification.py
   ```

3. **Try channel transitions**:
   ```bash
   python3 examples/gvf_channel_transitions.py
   ```

### **Integration into Your Projects**
```python
import pyopenchannel as poc
from pyopenchannel.gvf import GVFSolver, BoundaryType, ProfileClassifier

# Set up your analysis
poc.set_unit_system(poc.UnitSystem.SI)
solver = GVFSolver()
classifier = ProfileClassifier()

# Define your channel and conditions
channel = poc.RectangularChannel(width=5.0)
discharge = 20.0  # m³/s
slope = 0.001     # 0.1%
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

# Classify the profile
profile = classifier.classify_profile(
    gvf_result=result,
    channel=channel,
    discharge=discharge,
    slope=slope,
    manning_n=manning_n
)

print(f"Profile type: {profile.profile_type.value}")
print(f"Flow regime: {profile.flow_regime.value}")
```

## 📊 Example Output Quality

All examples provide **professional-grade output** including:

- **Detailed hydraulic calculations** with engineering units
- **Profile classification** with engineering significance
- **Design recommendations** based on analysis results
- **Error handling** with meaningful messages
- **Performance metrics** (computation time, convergence)
- **Validation results** against analytical solutions
- **Professional formatting** suitable for reports

## 🎓 Educational Value

These examples serve as:

- **Learning tools** for hydraulic engineering students
- **Reference implementations** for researchers
- **Validation cases** for software development
- **Professional templates** for consulting engineers
- **Documentation examples** for technical writing

## 🔬 Advanced Features

### **Numerical Methods**
- Multiple integration algorithms with adaptive stepping
- Event detection for hydraulic phenomena
- Analytical validation against known solutions
- Robust error handling and convergence checking

### **Engineering Analysis**
- Automatic identification of hydraulic phenomena
- Energy and momentum analysis
- Scour potential assessment
- Design optimization recommendations

### **Professional Reporting**
- Formatted output suitable for engineering reports
- Comprehensive analysis summaries
- Design guidelines and recommendations
- Regulatory compliance documentation

## 📈 Performance

- **Fast computation**: Optimized numerical algorithms
- **Memory efficient**: Minimal memory footprint
- **Scalable**: Handles large channel systems
- **Robust**: Extensive error handling and validation

## 🤝 Contributing

Found an issue or have suggestions for additional examples?

1. **Report issues**: Use the GitHub issue tracker
2. **Suggest examples**: Request specific use cases
3. **Contribute code**: Submit pull requests with new examples
4. **Improve documentation**: Help make examples clearer

## 📞 Support

- **Documentation**: See main PyOpenChannel documentation
- **Examples**: All examples include detailed comments
- **Community**: Join discussions on GitHub
- **Professional support**: Contact the development team

---

**PyOpenChannel GVF Examples** - Professional hydraulic analysis made accessible.

*These examples demonstrate the power and flexibility of PyOpenChannel for real-world hydraulic engineering applications.*
