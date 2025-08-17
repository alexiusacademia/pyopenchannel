# PyOpenChannel Examples

This directory contains example scripts demonstrating the capabilities of the PyOpenChannel library for open channel flow analysis and design.

## Examples Overview

### 1. Basic Calculations (`basic_calculations.py`)

Demonstrates fundamental open channel flow calculations:

- **Rectangular Channel Analysis**: Normal depth, critical depth, flow properties
- **Trapezoidal Channel Analysis**: Geometric properties and flow calculations
- **Circular Channel Analysis**: Pipe flow calculations with partial filling
- **Energy Calculations**: Specific energy, alternate depths, critical conditions
- **Uniform Flow Analysis**: Complete flow state analysis with classification

**Key Concepts Covered:**
- Channel geometry calculations (area, wetted perimeter, hydraulic radius)
- Normal depth calculations using Manning's equation
- Critical depth calculations and flow regime classification
- Energy equation applications
- Froude number calculations and flow classification

### 2. Channel Design (`channel_design.py`)

Demonstrates channel design and optimization techniques:

- **Optimal Rectangular Design**: Hydraulically efficient rectangular sections
- **Optimal Trapezoidal Design**: Hydraulically efficient trapezoidal sections
- **Economic Channel Design**: Cost-optimized channel design considering excavation, lining, and land costs
- **Channel Sizing**: Sizing channels for specific discharge requirements
- **Side Slope Recommendations**: Appropriate side slopes for different soil types

**Key Concepts Covered:**
- Hydraulic efficiency optimization (minimum wetted perimeter)
- Economic optimization considering construction costs
- Channel sizing for capacity requirements
- Freeboard calculations
- Velocity limit checks for different channel materials
- Soil-based side slope selection

## Running the Examples

### Prerequisites

Make sure you have Python 3.12+ installed and the PyOpenChannel library is properly set up.

### Running Individual Examples

```bash
# Run basic calculations examples
python examples/basic_calculations.py

# Run channel design examples
python examples/channel_design.py
```

### Expected Output

Each example script will display:
- Input parameters and design requirements
- Calculated results with proper units
- Flow regime classification (subcritical, critical, supercritical)
- Design recommendations and warnings
- Comparisons between different design approaches

## Example Applications

### Urban Drainage
- Storm drain sizing
- Culvert design
- Highway drainage ditches

### Agricultural Engineering
- Irrigation canal design
- Drainage channel sizing
- Farm pond spillways

### Water Resources Engineering
- River channel analysis
- Flood control channels
- Water conveyance systems

### Environmental Engineering
- Constructed wetland channels
- Stormwater management facilities
- Stream restoration projects

## Key Learning Points

### Hydraulic Principles
1. **Flow Regimes**: Understanding subcritical, critical, and supercritical flow
2. **Energy Concepts**: Specific energy, alternate depths, and critical conditions
3. **Uniform Flow**: Normal depth calculations and Manning's equation applications
4. **Channel Efficiency**: Optimal sections for minimum excavation

### Design Considerations
1. **Hydraulic Efficiency**: Minimizing wetted perimeter for given area
2. **Economic Optimization**: Balancing hydraulic efficiency with construction costs
3. **Practical Constraints**: Velocity limits, freeboard requirements, soil stability
4. **Material Selection**: Appropriate roughness coefficients and side slopes

### Engineering Judgment
1. **Velocity Limits**: Preventing erosion and sedimentation
2. **Freeboard**: Safety margins for flow variations
3. **Constructability**: Practical dimensions and slopes
4. **Maintenance**: Access requirements and long-term performance

## Extending the Examples

You can modify these examples to:

1. **Analyze Different Channel Shapes**: Circular, parabolic, or custom geometries
2. **Explore Parameter Sensitivity**: Vary slope, roughness, or discharge
3. **Compare Design Alternatives**: Multiple channel types for same requirements
4. **Add Custom Cost Functions**: Include specific regional cost data
5. **Implement Gradually Varied Flow**: Water surface profile calculations

## Common Issues and Solutions

### Convergence Problems
- Ensure reasonable initial guesses for iterative calculations
- Check that flow conditions are physically possible
- Verify that channel geometry parameters are positive

### Unrealistic Results
- Check velocity limits for channel material
- Verify that slopes are appropriate for the application
- Ensure Manning's roughness coefficients are reasonable

### Design Constraints
- Consider maximum depth limitations
- Account for construction tolerances
- Include safety factors in design

## Further Reading

For more detailed information about open channel flow theory and design principles, refer to:

1. "Open Channel Hydraulics" by Ven Te Chow
2. "Open-Channel Flow" by M. Hanif Chaudhry  
3. "Hydraulic Engineering" by John A. Roberson
4. ASCE Manual of Practice for channel design
5. Local design standards and regulations

## Support

If you encounter issues or have questions about these examples, please:

1. Check the main library documentation
2. Review the source code comments
3. Examine the validation functions for input requirements
4. Consider the physical reasonableness of your inputs and results
