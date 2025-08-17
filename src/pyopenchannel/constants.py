"""
constants.py
Author: Alexius Academia
Date: 2025-08-17

Physical constants and default values for open channel flow calculations.

This module contains the following constants:
- Physical constants
- Manning's roughness coefficients
- Convergence criteria for iterative calculations
- Common channel side slopes
- Unit conversion factors
- Mathematical constants

The constants are used in the open channel flow calculations.
"""

import math

# Physical constants
GRAVITY = 9.81  # m/s² - Standard gravitational acceleration
WATER_DENSITY = 1000.0  # kg/m³ - Density of water at 20°C
KINEMATIC_VISCOSITY = 1.004e-6  # m²/s - Kinematic viscosity of water at 20°C

# Manning's roughness coefficients (typical values)
MANNING_COEFFICIENTS = {
    # Natural channels
    "earth_straight_clean": 0.025,
    "earth_winding_clean": 0.030,
    "earth_with_stones": 0.035,
    "earth_weedy": 0.040,
    "rock_smooth": 0.030,
    "rock_jagged": 0.040,
    
    # Artificial channels
    "concrete_smooth": 0.012,
    "concrete_rough": 0.017,
    "brick_good": 0.015,
    "brick_rough": 0.020,
    "steel_smooth": 0.012,
    "steel_corrugated": 0.025,
    "cast_iron": 0.013,
    "pvc_smooth": 0.010,
}

# Convergence criteria for iterative calculations
DEFAULT_TOLERANCE = 1e-6
MAX_ITERATIONS = 1000

# Common channel side slopes (horizontal:vertical)
SIDE_SLOPES = {
    "vertical": 0.0,
    "steep_rock": 0.25,
    "ordinary_rock": 0.5,
    "earth_firm": 1.0,
    "earth_ordinary": 1.5,
    "earth_sandy": 2.0,
    "earth_loose": 3.0,
}

# Unit conversion factors
CONVERSION_FACTORS = {
    # Length
    "ft_to_m": 0.3048,
    "in_to_m": 0.0254,
    "m_to_ft": 3.28084,
    "m_to_in": 39.3701,
    
    # Area
    "ft2_to_m2": 0.092903,
    "m2_to_ft2": 10.7639,
    
    # Flow rate
    "cfs_to_cms": 0.028317,
    "cms_to_cfs": 35.3147,
    "gpm_to_cms": 6.309e-5,
    "cms_to_gpm": 15850.3,
}

# Mathematical constants
PI = math.pi
E = math.e
SQRT_2 = math.sqrt(2)
SQRT_PI = math.sqrt(math.pi)
