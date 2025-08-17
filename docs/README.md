# PyOpenChannel - GVF Documentation

Welcome to the comprehensive documentation for PyOpenChannel's Gradually Varied Flow (GVF) module. This documentation provides everything you need to effectively use the GVF system for professional hydraulic engineering applications.

## 📚 Documentation Structure

### 🚀 [User Guide](GVF_User_Guide.md)
**Complete guide for using the GVF module**
- Getting started and installation
- Core concepts and theory
- Step-by-step tutorials
- Best practices and troubleshooting
- Real-world examples and applications

**Perfect for:** New users, learning GVF theory, practical applications

### 📖 [API Reference](GVF_API_Reference.md)
**Detailed technical reference for all classes and methods**
- Complete class documentation
- Method signatures and parameters
- Data structures and enumerations
- Exception handling
- Code examples for each component

**Perfect for:** Developers, integration projects, detailed technical reference

### 💡 [Examples Guide](../examples/README_GVF_Examples.md)
**Comprehensive examples and tutorials**
- Ready-to-run example scripts
- Progressive learning path
- Professional applications
- Visualization and analysis

**Perfect for:** Hands-on learning, practical implementation, validation

## 🎯 Quick Start Guide

### 1. **New to GVF?** → Start Here
```bash
# Read the User Guide introduction
# Run the simple example
python3 examples/gvf_simple_example.py
```

### 2. **Want to Learn the API?** → API Reference
```bash
# Check the API Reference
# Try the basic usage example
python3 examples/gvf_basic_usage.py
```

### 3. **Need Professional Applications?** → Applications Module
```bash
# Explore the applications demo
python3 examples/gvf_applications_demo.py
```

### 4. **Working on Specific Projects?** → Specialized Examples
```bash
# Dam analysis
python3 examples/gvf_dam_backwater_analysis.py

# Channel transitions
python3 examples/gvf_channel_transitions.py
```

## 🏗️ System Overview

### Core Components

#### **GVF Solver**
- High-accuracy numerical integration
- Multiple Runge-Kutta methods
- Adaptive step sizing
- Event detection
- Analytical validation

#### **Profile Classification**
- Automatic profile type identification
- Engineering significance interpretation
- Multi-profile comparison
- Professional reporting

#### **Applications Module**
- Dam backwater analysis
- Bridge hydraulic design
- Chute energy dissipation
- Channel transitions
- Professional recommendations

#### **Numerical Methods**
- Runge-Kutta 4th order
- Runge-Kutta-Fehlberg 4(5)
- Dormand-Prince 5th order
- Finite difference methods
- Adaptive grids

## 🎓 Learning Path

### **Beginner Path**
1. **Read**: [User Guide - Introduction](GVF_User_Guide.md#introduction)
2. **Run**: `gvf_simple_example.py`
3. **Learn**: [Core Concepts](GVF_User_Guide.md#core-concepts)
4. **Practice**: [Basic Usage](GVF_User_Guide.md#basic-usage)

### **Intermediate Path**
1. **Explore**: [Profile Classification](GVF_User_Guide.md#profile-classification)
2. **Run**: `gvf_profile_classification.py`
3. **Study**: [API Reference - Core Classes](GVF_API_Reference.md#core-classes)
4. **Apply**: Different channel geometries and conditions

### **Advanced Path**
1. **Master**: [Applications Module](GVF_User_Guide.md#applications-module)
2. **Run**: `gvf_applications_demo.py`
3. **Implement**: Professional engineering applications
4. **Customize**: [Advanced Features](GVF_User_Guide.md#advanced-features)

### **Professional Path**
1. **Apply**: Real-world engineering projects
2. **Integrate**: With existing workflows
3. **Validate**: Against field data and standards
4. **Document**: Professional reports and compliance

## 🔧 Technical Capabilities

### **Numerical Accuracy**
- ✅ 5th-order Runge-Kutta integration
- ✅ Adaptive step sizing with error control
- ✅ Event detection for hydraulic phenomena
- ✅ Analytical validation against known solutions
- ✅ Professional-grade convergence criteria

### **Engineering Applications**
- ✅ Dam backwater analysis and flood mapping
- ✅ Bridge hydraulics and clearance design
- ✅ Spillway and energy dissipation analysis
- ✅ Channel transition and modification design
- ✅ Environmental impact assessment support

### **Profile Classification**
- ✅ Automatic identification of 13 profile types
- ✅ Slope classification (mild, steep, critical, horizontal, adverse)
- ✅ Flow regime analysis (subcritical, supercritical, critical)
- ✅ Engineering significance interpretation
- ✅ Professional documentation support

### **Channel Geometries**
- ✅ Rectangular channels
- ✅ Trapezoidal channels
- ✅ Triangular channels
- ✅ Circular channels
- ✅ Parabolic channels
- ✅ Custom geometries via base class

### **Unit Systems**
- ✅ SI (metric) units - meters, m³/s, m/s
- ✅ US Customary units - feet, ft³/s, ft/s
- ✅ Automatic unit-aware calculations
- ✅ Seamless unit system switching

## 📊 Professional Features

### **Design Standards**
- Multiple design criteria levels (Conservative, Standard, Optimized)
- Safety factors based on engineering practice
- Regulatory compliance support
- Professional documentation templates

### **Analysis Quality**
- Engineering-grade accuracy and validation
- Comprehensive error handling and reporting
- Professional visualization and documentation
- Industry-standard calculation methods

### **Integration Support**
- Clean, well-documented API
- Extensive examples and tutorials
- Professional technical support
- Open-source development model

## 🎯 Use Cases

### **Academic and Research**
- Hydraulic engineering education
- Research validation and development
- Thesis and dissertation projects
- Academic publication support

### **Professional Consulting**
- Flood risk assessment and mapping
- Infrastructure design and analysis
- Environmental impact studies
- Regulatory compliance documentation

### **Government and Agencies**
- Dam safety analysis
- Bridge and culvert design
- Flood control planning
- Environmental regulation enforcement

### **Software Development**
- Integration into larger hydraulic systems
- Custom application development
- API integration and automation
- Professional tool development

## 📞 Support and Resources

### **Documentation**
- [User Guide](GVF_User_Guide.md) - Complete usage guide
- [API Reference](GVF_API_Reference.md) - Technical reference
- [Examples](../examples/) - Ready-to-run examples
- [Troubleshooting](GVF_User_Guide.md#troubleshooting) - Common issues and solutions

### **Community**
- GitHub repository for issues and discussions
- Example contributions and improvements
- Community-driven enhancements
- Professional support options

### **Development**
- Open-source development model
- Professional code quality standards
- Comprehensive testing and validation
- Continuous improvement and updates

## 🚀 Getting Started

### **Installation**
```bash
pip install pyopenchannel
```

### **First Example**
```python
import pyopenchannel as poc
from pyopenchannel.gvf import GVFSolver, BoundaryType

# Set up analysis
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

# Analyze results
if result.success:
    print(f"Profile computed with {len(result.profile_points)} points")
    depths = [p.depth for p in result.profile_points]
    print(f"Depth range: {min(depths):.3f} - {max(depths):.3f} m")
```

### **Next Steps**
1. **Run examples**: Start with `gvf_simple_example.py`
2. **Read documentation**: Begin with the [User Guide](GVF_User_Guide.md)
3. **Explore applications**: Try the [Applications Demo](../examples/gvf_applications_demo.py)
4. **Apply to projects**: Use for real engineering applications

---

## 📋 Documentation Index

| Document | Purpose | Audience |
|----------|---------|----------|
| [User Guide](GVF_User_Guide.md) | Complete usage guide | All users |
| [API Reference](GVF_API_Reference.md) | Technical reference | Developers |
| [Examples Guide](../examples/README_GVF_Examples.md) | Practical examples | Practitioners |
| [Examples Index](../examples/GVF_Examples_Index.md) | Quick example reference | All users |

---

**PyOpenChannel GVF Module** - Professional hydraulic analysis made accessible.

*For questions, issues, or contributions, please visit the PyOpenChannel GitHub repository.*
