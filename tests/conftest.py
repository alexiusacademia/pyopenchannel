"""
Pytest configuration and fixtures for PyOpenChannel tests.
"""

import pytest
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import the package to make it available for all tests
import pyopenchannel


@pytest.fixture
def rectangular_channel():
    """Fixture providing a standard rectangular channel."""
    return pyopenchannel.RectangularChannel(width=3.0)


@pytest.fixture
def trapezoidal_channel():
    """Fixture providing a standard trapezoidal channel."""
    return pyopenchannel.TrapezoidalChannel(bottom_width=2.0, side_slope=1.5)


@pytest.fixture
def triangular_channel():
    """Fixture providing a standard triangular channel."""
    return pyopenchannel.TriangularChannel(side_slope=2.0)


@pytest.fixture
def circular_channel():
    """Fixture providing a standard circular channel."""
    return pyopenchannel.CircularChannel(diameter=1.2)


@pytest.fixture
def standard_flow_parameters():
    """Fixture providing standard flow parameters for testing."""
    return {
        'discharge': 5.0,  # m³/s
        'slope': 0.001,    # dimensionless
        'manning_n': 0.025,
        'depth': 1.5,      # m
    }


@pytest.fixture
def tolerance():
    """Fixture providing standard tolerance for numerical comparisons."""
    return 1e-6


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add 'unit' marker to all tests by default
        if not any(marker.name in ['integration', 'slow'] for marker in item.iter_markers()):
            item.add_marker(pytest.mark.unit)


# Custom assertions for hydraulic calculations
class HydraulicAssertions:
    """Custom assertions for hydraulic calculations."""
    
    @staticmethod
    def assert_positive_depth(depth, name="depth"):
        """Assert that depth is positive."""
        assert depth > 0, f"{name} must be positive, got {depth}"
    
    @staticmethod
    def assert_reasonable_velocity(velocity, min_vel=0.1, max_vel=10.0):
        """Assert that velocity is within reasonable range."""
        assert min_vel <= velocity <= max_vel, \
            f"Velocity {velocity} m/s is outside reasonable range [{min_vel}, {max_vel}]"
    
    @staticmethod
    def assert_froude_number_range(froude_number, min_fr=0.1, max_fr=5.0):
        """Assert that Froude number is within reasonable range."""
        assert min_fr <= froude_number <= max_fr, \
            f"Froude number {froude_number} is outside reasonable range [{min_fr}, {max_fr}]"
    
    @staticmethod
    def assert_manning_equation_satisfied(discharge, area, hydraulic_radius, slope, manning_n, tolerance=1e-6):
        """Assert that Manning's equation is satisfied."""
        calculated_q = pyopenchannel.ManningEquation.discharge(
            area, hydraulic_radius, slope, manning_n
        )
        relative_error = abs(calculated_q - discharge) / discharge
        assert relative_error < tolerance, \
            f"Manning's equation not satisfied: calculated Q={calculated_q}, expected Q={discharge}"
    
    @staticmethod
    def assert_critical_flow_condition(discharge, area, top_width, tolerance=1e-3):
        """Assert that critical flow condition is satisfied."""
        lhs = discharge**2
        rhs = pyopenchannel.GRAVITY * (area**3) / top_width
        relative_error = abs(lhs - rhs) / lhs
        assert relative_error < tolerance, \
            f"Critical flow condition not satisfied: Q²={lhs}, gA³/T={rhs}"


@pytest.fixture
def hydraulic_assertions():
    """Fixture providing hydraulic assertion methods."""
    return HydraulicAssertions()


# Test data generators
@pytest.fixture
def channel_test_cases():
    """Fixture providing test cases for different channel types."""
    return [
        {
            'type': 'rectangular',
            'params': {'width': 3.0},
            'test_depth': 2.0,
            'expected_area': 6.0,
            'expected_perimeter': 7.0,
            'expected_top_width': 3.0,
        },
        {
            'type': 'trapezoidal',
            'params': {'bottom_width': 2.0, 'side_slope': 1.5},
            'test_depth': 2.0,
            'expected_area': 10.0,  # 2*(2 + 1.5*2) = 2*5 = 10
            'expected_top_width': 8.0,  # 2 + 2*1.5*2 = 2 + 6 = 8
        },
        {
            'type': 'triangular',
            'params': {'side_slope': 2.0},
            'test_depth': 2.0,
            'expected_area': 8.0,  # 2*2² = 8
            'expected_top_width': 8.0,  # 2*2*2 = 8
        },
    ]


@pytest.fixture
def flow_test_scenarios():
    """Fixture providing flow test scenarios."""
    return [
        {
            'name': 'mild_slope_subcritical',
            'channel_type': 'rectangular',
            'channel_params': {'width': 4.0},
            'discharge': 8.0,
            'slope': 0.0005,  # Mild slope
            'manning_n': 0.025,
            'expected_flow_type': 'subcritical',
        },
        {
            'name': 'steep_slope_supercritical',
            'channel_type': 'rectangular',
            'channel_params': {'width': 2.0},
            'discharge': 3.0,
            'slope': 0.01,  # Steep slope
            'manning_n': 0.015,
            'expected_flow_type': 'supercritical',
        },
        {
            'name': 'trapezoidal_normal',
            'channel_type': 'trapezoidal',
            'channel_params': {'bottom_width': 3.0, 'side_slope': 1.5},
            'discharge': 15.0,
            'slope': 0.002,
            'manning_n': 0.030,
            'expected_flow_type': 'subcritical',
        },
    ]


# Performance testing utilities
@pytest.fixture
def performance_timer():
    """Fixture for timing test execution."""
    import time
    
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = time.time()
        
        def stop(self):
            self.end_time = time.time()
        
        @property
        def elapsed(self):
            if self.start_time is None or self.end_time is None:
                return None
            return self.end_time - self.start_time
    
    return Timer()


# Error testing utilities
@pytest.fixture
def error_test_cases():
    """Fixture providing error test cases."""
    return {
        'negative_values': [-1.0, -0.5, -10.0],
        'zero_values': [0.0],
        'invalid_strings': ['invalid', 'abc', ''],
        'none_values': [None],
        'extreme_large': [1e10, 1e20],
        'extreme_small': [1e-10, 1e-20],
    }
