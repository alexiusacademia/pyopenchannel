"""
File: test_geometry.py
Author: Alexius Academia
Date: 2025-08-17

Tests for the geometry module.
"""

import pytest
import math
from pyopenchannel.geometry import (
    RectangularChannel,
    TrapezoidalChannel,
    TriangularChannel,
    CircularChannel,
    ParabolicChannel,
    create_channel,
)
from pyopenchannel.exceptions import InvalidGeometryError


class TestRectangularChannel:
    """Test rectangular channel geometry calculations."""
    
    def test_initialization(self):
        """Test channel initialization."""
        channel = RectangularChannel(width=3.0)
        assert channel.width == 3.0
    
    def test_invalid_width(self):
        """Test invalid width raises exception."""
        with pytest.raises(InvalidGeometryError):
            RectangularChannel(width=-1.0)
        
        with pytest.raises(InvalidGeometryError):
            RectangularChannel(width=0.0)
    
    def test_area_calculation(self):
        """Test area calculation."""
        channel = RectangularChannel(width=4.0)
        assert channel.area(depth=2.0) == 8.0
        assert channel.area(depth=1.5) == 6.0
    
    def test_wetted_perimeter(self):
        """Test wetted perimeter calculation."""
        channel = RectangularChannel(width=4.0)
        assert channel.wetted_perimeter(depth=2.0) == 8.0  # 4 + 2*2
        assert channel.wetted_perimeter(depth=1.0) == 6.0  # 4 + 2*1
    
    def test_top_width(self):
        """Test top width calculation."""
        channel = RectangularChannel(width=4.0)
        assert channel.top_width(depth=2.0) == 4.0
        assert channel.top_width(depth=1.0) == 4.0
    
    def test_hydraulic_radius(self):
        """Test hydraulic radius calculation."""
        channel = RectangularChannel(width=4.0)
        # R = A/P = (4*2)/(4+2*2) = 8/8 = 1.0
        assert channel.hydraulic_radius(depth=2.0) == 1.0
        # R = A/P = (4*1)/(4+2*1) = 4/6 = 2/3
        assert abs(channel.hydraulic_radius(depth=1.0) - 2/3) < 1e-10
    
    def test_hydraulic_depth(self):
        """Test hydraulic depth calculation."""
        channel = RectangularChannel(width=4.0)
        # D = A/T = (4*2)/4 = 2.0
        assert channel.hydraulic_depth(depth=2.0) == 2.0
        # D = A/T = (4*1)/4 = 1.0
        assert channel.hydraulic_depth(depth=1.0) == 1.0


class TestTrapezoidalChannel:
    """Test trapezoidal channel geometry calculations."""
    
    def test_initialization(self):
        """Test channel initialization."""
        channel = TrapezoidalChannel(bottom_width=2.0, side_slope=1.5)
        assert channel.bottom_width == 2.0
        assert channel.side_slope == 1.5
    
    def test_invalid_parameters(self):
        """Test invalid parameters raise exceptions."""
        with pytest.raises(InvalidGeometryError):
            TrapezoidalChannel(bottom_width=-1.0, side_slope=1.5)
        
        with pytest.raises(InvalidGeometryError):
            TrapezoidalChannel(bottom_width=2.0, side_slope=-0.5)
    
    def test_area_calculation(self):
        """Test area calculation."""
        channel = TrapezoidalChannel(bottom_width=2.0, side_slope=1.5)
        # A = y(b + my) = 2(2 + 1.5*2) = 2(2 + 3) = 10
        assert channel.area(depth=2.0) == 10.0
    
    def test_wetted_perimeter(self):
        """Test wetted perimeter calculation."""
        channel = TrapezoidalChannel(bottom_width=2.0, side_slope=1.5)
        # P = b + 2y*sqrt(1 + m²) = 2 + 2*2*sqrt(1 + 1.5²)
        expected = 2.0 + 2 * 2.0 * math.sqrt(1 + 1.5**2)
        assert abs(channel.wetted_perimeter(depth=2.0) - expected) < 1e-10
    
    def test_top_width(self):
        """Test top width calculation."""
        channel = TrapezoidalChannel(bottom_width=2.0, side_slope=1.5)
        # T = b + 2my = 2 + 2*1.5*2 = 2 + 6 = 8
        assert channel.top_width(depth=2.0) == 8.0


class TestTriangularChannel:
    """Test triangular channel geometry calculations."""
    
    def test_initialization(self):
        """Test channel initialization."""
        channel = TriangularChannel(side_slope=2.0)
        assert channel.side_slope == 2.0
    
    def test_invalid_side_slope(self):
        """Test invalid side slope raises exception."""
        with pytest.raises(InvalidGeometryError):
            TriangularChannel(side_slope=-1.0)
        
        with pytest.raises(InvalidGeometryError):
            TriangularChannel(side_slope=0.0)
    
    def test_area_calculation(self):
        """Test area calculation."""
        channel = TriangularChannel(side_slope=2.0)
        # A = my² = 2*2² = 8
        assert channel.area(depth=2.0) == 8.0
    
    def test_wetted_perimeter(self):
        """Test wetted perimeter calculation."""
        channel = TriangularChannel(side_slope=2.0)
        # P = 2y*sqrt(1 + m²) = 2*2*sqrt(1 + 4) = 4*sqrt(5)
        expected = 4.0 * math.sqrt(5.0)
        assert abs(channel.wetted_perimeter(depth=2.0) - expected) < 1e-10
    
    def test_top_width(self):
        """Test top width calculation."""
        channel = TriangularChannel(side_slope=2.0)
        # T = 2my = 2*2*2 = 8
        assert channel.top_width(depth=2.0) == 8.0


class TestCircularChannel:
    """Test circular channel geometry calculations."""
    
    def test_initialization(self):
        """Test channel initialization."""
        channel = CircularChannel(diameter=2.0)
        assert channel.diameter == 2.0
        assert channel.radius == 1.0
    
    def test_invalid_diameter(self):
        """Test invalid diameter raises exception."""
        with pytest.raises(InvalidGeometryError):
            CircularChannel(diameter=-1.0)
        
        with pytest.raises(InvalidGeometryError):
            CircularChannel(diameter=0.0)
    
    def test_full_pipe_area(self):
        """Test area calculation for full pipe."""
        channel = CircularChannel(diameter=2.0)
        # Full pipe: A = πr² = π*1² = π
        expected = math.pi
        assert abs(channel.area(depth=2.0) - expected) < 1e-10
    
    def test_half_full_pipe_area(self):
        """Test area calculation for half-full pipe."""
        channel = CircularChannel(diameter=2.0)
        # Half full: A = πr²/2 = π/2
        expected = math.pi / 2
        assert abs(channel.area(depth=1.0) - expected) < 1e-6
    
    def test_full_pipe_perimeter(self):
        """Test wetted perimeter for full pipe."""
        channel = CircularChannel(diameter=2.0)
        # Full pipe: P = πd = 2π
        expected = 2 * math.pi
        assert abs(channel.wetted_perimeter(depth=2.0) - expected) < 1e-10
    
    def test_half_full_pipe_perimeter(self):
        """Test wetted perimeter for half-full pipe."""
        channel = CircularChannel(diameter=2.0)
        # Half full: P = πr = π
        expected = math.pi
        assert abs(channel.wetted_perimeter(depth=1.0) - expected) < 1e-6
    
    def test_full_pipe_top_width(self):
        """Test top width for full pipe."""
        channel = CircularChannel(diameter=2.0)
        # Full pipe has no free surface
        assert channel.top_width(depth=2.0) == 0.0
    
    def test_half_full_pipe_top_width(self):
        """Test top width for half-full pipe."""
        channel = CircularChannel(diameter=2.0)
        # Half full: T = d = 2.0
        assert abs(channel.top_width(depth=1.0) - 2.0) < 1e-6
    
    def test_depth_exceeds_diameter(self):
        """Test behavior when depth exceeds diameter."""
        channel = CircularChannel(diameter=2.0)
        with pytest.raises(InvalidGeometryError):
            channel._central_angle(depth=3.0)


class TestParabolicChannel:
    """Test parabolic channel geometry calculations."""
    
    def test_initialization(self):
        """Test channel initialization."""
        channel = ParabolicChannel(shape_parameter=0.5)
        assert channel.shape_parameter == 0.5
    
    def test_invalid_shape_parameter(self):
        """Test invalid shape parameter raises exception."""
        with pytest.raises(InvalidGeometryError):
            ParabolicChannel(shape_parameter=-1.0)
        
        with pytest.raises(InvalidGeometryError):
            ParabolicChannel(shape_parameter=0.0)
    
    def test_area_calculation(self):
        """Test area calculation."""
        channel = ParabolicChannel(shape_parameter=0.5)
        # For parabola y = ax², A = (2/3) * T * y
        # T = 2*sqrt(y/a) = 2*sqrt(2/0.5) = 2*2 = 4
        # A = (2/3) * 4 * 2 = 16/3
        expected = 16.0 / 3.0
        assert abs(channel.area(depth=2.0) - expected) < 1e-10
    
    def test_top_width_calculation(self):
        """Test top width calculation."""
        channel = ParabolicChannel(shape_parameter=0.5)
        # T = 2*sqrt(y/a) = 2*sqrt(2/0.5) = 2*2 = 4
        assert abs(channel.top_width(depth=2.0) - 4.0) < 1e-10
    
    def test_zero_depth(self):
        """Test calculations at zero depth."""
        channel = ParabolicChannel(shape_parameter=0.5)
        
        # Test truly zero depth
        assert channel.area(depth=0.0) == 0.0
        assert channel.top_width(depth=0.0) == 0.0
        assert channel.wetted_perimeter(depth=0.0) == 0.0
        
        # Test very small depth
        assert channel.area(depth=1e-10) < 1e-6
        # For parabolic channel: top_width = 2*sqrt(depth/a)
        # With depth=1e-10 and a=0.5: top_width = 2*sqrt(2e-10) ≈ 2.83e-5
        assert channel.top_width(depth=1e-10) < 1e-4  # More realistic tolerance


class TestChannelFactory:
    """Test channel factory function."""
    
    def test_create_rectangular(self):
        """Test creating rectangular channel."""
        channel = create_channel('rectangular', width=3.0)
        assert isinstance(channel, RectangularChannel)
        assert channel.width == 3.0
    
    def test_create_trapezoidal(self):
        """Test creating trapezoidal channel."""
        channel = create_channel('trapezoidal', bottom_width=2.0, side_slope=1.5)
        assert isinstance(channel, TrapezoidalChannel)
        assert channel.bottom_width == 2.0
        assert channel.side_slope == 1.5
    
    def test_create_triangular(self):
        """Test creating triangular channel."""
        channel = create_channel('triangular', side_slope=2.0)
        assert isinstance(channel, TriangularChannel)
        assert channel.side_slope == 2.0
    
    def test_create_circular(self):
        """Test creating circular channel."""
        channel = create_channel('circular', diameter=1.5)
        assert isinstance(channel, CircularChannel)
        assert channel.diameter == 1.5
    
    def test_create_parabolic(self):
        """Test creating parabolic channel."""
        channel = create_channel('parabolic', shape_parameter=0.8)
        assert isinstance(channel, ParabolicChannel)
        assert channel.shape_parameter == 0.8
    
    def test_invalid_channel_type(self):
        """Test invalid channel type raises exception."""
        with pytest.raises(InvalidGeometryError):
            create_channel('invalid_type', width=3.0)
    
    def test_case_insensitive(self):
        """Test case insensitive channel type."""
        channel = create_channel('RECTANGULAR', width=3.0)
        assert isinstance(channel, RectangularChannel)
        
        channel = create_channel('Trapezoidal', bottom_width=2.0, side_slope=1.5)
        assert isinstance(channel, TrapezoidalChannel)


class TestGeometryEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_negative_depth(self):
        """Test negative depth raises exception."""
        channel = RectangularChannel(width=3.0)
        
        with pytest.raises(Exception):  # Should raise InvalidFlowConditionError
            channel.area(depth=-1.0)
    
    def test_zero_depth(self):
        """Test zero depth raises exception."""
        channel = RectangularChannel(width=3.0)
        
        with pytest.raises(Exception):  # Should raise InvalidFlowConditionError
            channel.area(depth=0.0)
    
    def test_very_small_depth(self):
        """Test very small depth calculations."""
        channel = RectangularChannel(width=3.0)
        small_depth = 1e-6
        
        area = channel.area(depth=small_depth)
        assert area == 3.0 * small_depth
        
        perimeter = channel.wetted_perimeter(depth=small_depth)
        assert abs(perimeter - (3.0 + 2 * small_depth)) < 1e-10
    
    def test_very_large_depth(self):
        """Test very large depth calculations."""
        channel = RectangularChannel(width=3.0)
        large_depth = 1e6
        
        area = channel.area(depth=large_depth)
        assert area == 3.0 * large_depth
        
        # Hydraulic radius should approach width/2 for very deep channels
        hydraulic_radius = channel.hydraulic_radius(depth=large_depth)
        assert abs(hydraulic_radius - 1.5) < 0.01  # Should be close to width/2
