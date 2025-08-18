# PyOpenChannel Plots Directory

This directory automatically stores all generated plots from PyOpenChannel examples and analyses.

## File Naming Convention

All plots are saved with the following format:
```
{example_name}_{timestamp}.png
```

Where:
- `example_name`: The name of the example or analysis
- `timestamp`: Format `YYYYMMDD_HHMMSS` for unique identification

## Plot Quality

- **Resolution**: 300 DPI (publication quality)
- **Format**: PNG with transparent background handling
- **Layout**: Tight bounding box to minimize whitespace
- **Background**: White background for professional appearance

## Examples

- `ogee_diversion_dam_rigorous_analysis_*.png`: Complete rigorous analysis of ogee diversion dams
- `rigorous_spillway_demo_*.png`: Comprehensive spillway analysis demonstration
- `complete_gate_flow_analysis_*.png`: Gate flow analysis with multiple plots
- `rvf_submerged_ogee_weir_*.png`: Submerged weir flow analysis

## Usage

Plots are automatically saved when running any PyOpenChannel example that generates visualizations. No manual intervention required.

## Integration

The plot saving functionality is provided by `examples/plot_utils.py` which can be imported into any example script:

```python
from plot_utils import save_and_show_plot
save_and_show_plot("my_analysis_name")
```

This ensures consistent plot saving across all PyOpenChannel examples and maintains a complete record of all generated analyses.
