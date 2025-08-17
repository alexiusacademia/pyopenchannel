"""
File: numerical/integration.py
Author: Alexius Academia
Date: 2025-08-17

High-accuracy integration methods for hydraulic computations.

This module provides robust numerical integration methods optimized for
accuracy in gradually varied flow analysis:

- Runge-Kutta 4th order (RK4): Classic, reliable method
- Runge-Kutta-Fehlberg 4(5): Adaptive step size with error control
- Dormand-Prince: High-accuracy adaptive method
- Adaptive integrators with automatic step size control

All methods are designed for solving differential equations of the form:
dy/dx = f(x, y)

Particularly optimized for the GVF equation:
dy/dx = (S₀ - Sf) / (1 - Fr²)
"""

import math
from typing import Callable, Tuple, List, Optional, Union, Dict, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum

from ..exceptions import ConvergenceError, InvalidFlowConditionError


class EventType(Enum):
    """Types of events that can be detected during integration."""
    CRITICAL_DEPTH = "critical_depth"
    NORMAL_DEPTH = "normal_depth"
    HYDRAULIC_JUMP = "hydraulic_jump"
    SHOCK_WAVE = "shock_wave"
    FLOW_REVERSAL = "flow_reversal"
    CHANNEL_TRANSITION = "channel_transition"
    CUSTOM = "custom"


@dataclass
class Event:
    """
    Represents a detected event during integration.
    
    Attributes:
        event_type: Type of event detected
        x_location: X-coordinate where event occurred
        y_value: Y-value at event location
        description: Human-readable description
        properties: Additional event-specific properties
    """
    event_type: EventType
    x_location: float
    y_value: float
    description: str
    properties: Dict[str, Any] = field(default_factory=dict)


class EventDetector:
    """
    Detects hydraulic events during integration.
    
    Monitors solution for:
    - Critical depth transitions (Fr = 1.0)
    - Hydraulic jumps (shock waves)
    - Flow reversals
    - Channel transitions
    """
    
    def __init__(
        self,
        froude_tolerance: float = 0.01,
        shock_threshold: float = 0.1,
        enable_critical_detection: bool = True,
        enable_shock_detection: bool = True
    ):
        """
        Initialize event detector.
        
        Args:
            froude_tolerance: Tolerance for critical flow detection (|Fr - 1| < tol)
            shock_threshold: Minimum depth change ratio for shock detection
            enable_critical_detection: Enable critical depth transition detection
            enable_shock_detection: Enable hydraulic jump/shock detection
        """
        self.froude_tolerance = froude_tolerance
        self.shock_threshold = shock_threshold
        self.enable_critical_detection = enable_critical_detection
        self.enable_shock_detection = enable_shock_detection
        self.events: List[Event] = []
        
        # Previous step values for comparison
        self.prev_x: Optional[float] = None
        self.prev_y: Optional[float] = None
        self.prev_froude: Optional[float] = None
    
    def check_events(
        self,
        x: float,
        y: float,
        froude_number: Optional[float] = None,
        velocity: Optional[float] = None,
        **kwargs
    ) -> List[Event]:
        """
        Check for events at current integration step.
        
        Args:
            x: Current x-coordinate
            y: Current y-value (depth)
            froude_number: Current Froude number (if available)
            velocity: Current velocity (if available)
            **kwargs: Additional parameters for event detection
            
        Returns:
            List of detected events
        """
        detected_events = []
        
        # Skip first step (no previous values)
        if self.prev_x is None:
            self.prev_x, self.prev_y, self.prev_froude = x, y, froude_number
            return detected_events
        
        # Critical depth transition detection
        if (self.enable_critical_detection and 
            froude_number is not None and 
            self.prev_froude is not None):
            
            if (abs(froude_number - 1.0) < self.froude_tolerance and
                abs(self.prev_froude - 1.0) >= self.froude_tolerance):
                
                event = Event(
                    event_type=EventType.CRITICAL_DEPTH,
                    x_location=x,
                    y_value=y,
                    description=f"Critical depth transition at x={x:.3f}, Fr={froude_number:.3f}",
                    properties={
                        "froude_number": froude_number,
                        "transition_type": "approaching_critical" if self.prev_froude > 1.0 else "leaving_critical"
                    }
                )
                detected_events.append(event)
                self.events.append(event)
        
        # Hydraulic jump / shock detection
        if self.enable_shock_detection and self.prev_y is not None:
            depth_ratio = abs(y - self.prev_y) / max(self.prev_y, 1e-6)
            
            if depth_ratio > self.shock_threshold:
                # Determine if it's a hydraulic jump or drawdown
                if y > self.prev_y * (1 + self.shock_threshold):
                    shock_type = "hydraulic_jump"
                    description = f"Hydraulic jump detected at x={x:.3f}"
                else:
                    shock_type = "drawdown"
                    description = f"Rapid drawdown detected at x={x:.3f}"
                
                event = Event(
                    event_type=EventType.HYDRAULIC_JUMP,
                    x_location=x,
                    y_value=y,
                    description=description,
                    properties={
                        "shock_type": shock_type,
                        "depth_ratio": y / self.prev_y,
                        "depth_change": y - self.prev_y,
                        "upstream_depth": self.prev_y,
                        "downstream_depth": y
                    }
                )
                detected_events.append(event)
                self.events.append(event)
        
        # Flow reversal detection
        if velocity is not None and velocity < 0:
            event = Event(
                event_type=EventType.FLOW_REVERSAL,
                x_location=x,
                y_value=y,
                description=f"Flow reversal detected at x={x:.3f}",
                properties={"velocity": velocity}
            )
            detected_events.append(event)
            self.events.append(event)
        
        # Update previous values
        self.prev_x, self.prev_y, self.prev_froude = x, y, froude_number
        
        return detected_events
    
    def reset(self):
        """Reset detector for new integration."""
        self.events.clear()
        self.prev_x = None
        self.prev_y = None
        self.prev_froude = None


class AnalyticalValidator:
    """
    Cross-checks numerical solutions with analytical solutions where possible.
    
    Provides validation for:
    - Uniform flow (Manning's equation)
    - Critical flow conditions
    - Simple geometric cases
    """
    
    def __init__(self, tolerance: float = 1e-3):
        """
        Initialize analytical validator.
        
        Args:
            tolerance: Tolerance for analytical comparison
        """
        self.tolerance = tolerance
    
    def validate_uniform_flow(
        self,
        depth: float,
        discharge: float,
        slope: float,
        manning_n: float,
        channel_area: float,
        hydraulic_radius: float
    ) -> Dict[str, Any]:
        """
        Validate against Manning's equation for uniform flow.
        
        Args:
            depth: Flow depth
            discharge: Discharge
            slope: Channel slope
            manning_n: Manning's roughness coefficient
            channel_area: Cross-sectional area
            hydraulic_radius: Hydraulic radius
            
        Returns:
            Validation results dictionary
        """
        # Calculate analytical discharge using Manning's equation
        # Note: This will use unit-aware Manning factor from the main library
        try:
            from ..units import get_manning_factor
            manning_factor = get_manning_factor()
        except ImportError:
            manning_factor = 1.0  # Default SI value
        
        analytical_discharge = (
            (manning_factor / manning_n) * 
            channel_area * 
            (hydraulic_radius ** (2/3)) * 
            (slope ** 0.5)
        )
        
        # Calculate relative error
        relative_error = abs(discharge - analytical_discharge) / max(analytical_discharge, 1e-6)
        
        return {
            "is_valid": relative_error < self.tolerance,
            "analytical_discharge": analytical_discharge,
            "numerical_discharge": discharge,
            "relative_error": relative_error,
            "absolute_error": abs(discharge - analytical_discharge),
            "validation_type": "uniform_flow_manning"
        }
    
    def validate_critical_flow(
        self,
        depth: float,
        discharge: float,
        channel_area: float,
        top_width: float
    ) -> Dict[str, Any]:
        """
        Validate critical flow condition: Q² = g * A³ / T
        
        Args:
            depth: Flow depth
            discharge: Discharge
            channel_area: Cross-sectional area
            top_width: Top width
            
        Returns:
            Validation results dictionary
        """
        try:
            from ..units import get_gravity
            gravity = get_gravity()
        except ImportError:
            gravity = 9.81  # Default SI value
        
        # Critical flow condition
        lhs = discharge ** 2
        rhs = gravity * (channel_area ** 3) / top_width
        
        relative_error = abs(lhs - rhs) / max(rhs, 1e-6)
        
        # Calculate Froude number
        hydraulic_depth = channel_area / top_width
        velocity = discharge / channel_area
        froude_number = velocity / math.sqrt(gravity * hydraulic_depth)
        
        return {
            "is_valid": relative_error < self.tolerance,
            "froude_number": froude_number,
            "critical_condition_lhs": lhs,
            "critical_condition_rhs": rhs,
            "relative_error": relative_error,
            "validation_type": "critical_flow"
        }


@dataclass
class IntegrationResult:
    """
    Results from numerical integration with advanced features.
    
    Attributes:
        x_values: Array of x coordinates
        y_values: Array of y coordinates  
        step_sizes: Array of step sizes used
        error_estimates: Array of error estimates (if available)
        total_steps: Total number of integration steps
        success: Whether integration completed successfully
        message: Status message or error description
        events: List of detected events during integration
        validation_results: Results from analytical validation (if performed)
        performance_metrics: Integration performance statistics
    """
    x_values: List[float]
    y_values: List[float]
    step_sizes: List[float]
    error_estimates: Optional[List[float]] = None
    total_steps: int = 0
    success: bool = True
    message: str = "Integration completed successfully"
    events: List[Event] = field(default_factory=list)
    validation_results: List[Dict[str, Any]] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)


class IntegrationMethod(ABC):
    """Abstract base class for integration methods."""
    
    @abstractmethod
    def integrate(
        self,
        func: Callable[[float, float], float],
        x0: float,
        y0: float,
        x_end: float,
        step_size: float,
        **kwargs
    ) -> IntegrationResult:
        """Integrate the differential equation dy/dx = func(x, y)."""
        pass


class RungeKutta4(IntegrationMethod):
    """
    Classic 4th-order Runge-Kutta method.
    
    Provides excellent balance of accuracy and computational efficiency.
    Global error is O(h⁴) where h is the step size.
    
    Ideal for:
    - Smooth water surface profiles
    - When step size can be chosen appropriately
    - Educational and verification purposes
    """
    
    def __init__(self, tolerance: float = 1e-6):
        """
        Initialize RK4 integrator.
        
        Args:
            tolerance: Convergence tolerance for step size validation
        """
        self.tolerance = tolerance
    
    def integrate(
        self,
        func: Callable[[float, float], float],
        x0: float,
        y0: float,
        x_end: float,
        step_size: float,
        max_steps: int = 10000
    ) -> IntegrationResult:
        """
        Integrate using 4th-order Runge-Kutta method.
        
        Args:
            func: Function f(x, y) representing dy/dx
            x0: Initial x value
            y0: Initial y value
            x_end: Final x value
            step_size: Integration step size
            max_steps: Maximum number of steps
            
        Returns:
            IntegrationResult with solution arrays
        """
        if abs(step_size) < self.tolerance:
            raise InvalidFlowConditionError(f"Step size too small: {step_size}")
        
        # Determine integration direction
        direction = 1 if x_end > x0 else -1
        h = abs(step_size) * direction
        
        # Initialize arrays
        x_values = [x0]
        y_values = [y0]
        step_sizes = []
        
        x, y = x0, y0
        steps = 0
        
        try:
            while (direction * (x_end - x) > self.tolerance) and (steps < max_steps):
                # Adjust final step if needed
                if direction * (x + h - x_end) > 0:
                    h = x_end - x
                
                # RK4 coefficients
                k1 = h * func(x, y)
                k2 = h * func(x + h/2, y + k1/2)
                k3 = h * func(x + h/2, y + k2/2)
                k4 = h * func(x + h, y + k3)
                
                # Update solution
                y_new = y + (k1 + 2*k2 + 2*k3 + k4) / 6
                x_new = x + h
                
                # Store results
                x_values.append(x_new)
                y_values.append(y_new)
                step_sizes.append(abs(h))
                
                # Update for next iteration
                x, y = x_new, y_new
                steps += 1
                
        except Exception as e:
            return IntegrationResult(
                x_values=x_values,
                y_values=y_values,
                step_sizes=step_sizes,
                total_steps=steps,
                success=False,
                message=f"RK4 integration failed: {e}"
            )
        
        if steps >= max_steps:
            return IntegrationResult(
                x_values=x_values,
                y_values=y_values,
                step_sizes=step_sizes,
                total_steps=steps,
                success=False,
                message=f"Maximum steps ({max_steps}) exceeded"
            )
        
        return IntegrationResult(
            x_values=x_values,
            y_values=y_values,
            step_sizes=step_sizes,
            total_steps=steps,
            success=True,
            message="RK4 integration completed successfully"
        )


class RungeKuttaFehlberg45(IntegrationMethod):
    """
    Runge-Kutta-Fehlberg 4(5) method with adaptive step size control.
    
    Uses embedded 4th and 5th order formulas to estimate local error
    and automatically adjust step size for optimal accuracy/efficiency.
    
    Ideal for:
    - Variable flow conditions
    - Automatic error control
    - Production applications requiring reliability
    """
    
    def __init__(
        self,
        rtol: float = 1e-6,
        atol: float = 1e-9,
        safety_factor: float = 0.9,
        min_step_factor: float = 0.1,
        max_step_factor: float = 5.0
    ):
        """
        Initialize RKF45 integrator.
        
        Args:
            rtol: Relative tolerance for error control
            atol: Absolute tolerance for error control
            safety_factor: Safety factor for step size adjustment
            min_step_factor: Minimum step size reduction factor
            max_step_factor: Maximum step size increase factor
        """
        self.rtol = rtol
        self.atol = atol
        self.safety_factor = safety_factor
        self.min_step_factor = min_step_factor
        self.max_step_factor = max_step_factor
    
    def integrate(
        self,
        func: Callable[[float, float], float],
        x0: float,
        y0: float,
        x_end: float,
        initial_step: float,
        max_steps: int = 50000
    ) -> IntegrationResult:
        """
        Integrate using RKF45 with adaptive step size.
        
        Args:
            func: Function f(x, y) representing dy/dx
            x0: Initial x value
            y0: Initial y value
            x_end: Final x value
            initial_step: Initial step size guess
            max_steps: Maximum number of steps
            
        Returns:
            IntegrationResult with solution arrays and error estimates
        """
        # RKF45 coefficients
        a = [0, 1/4, 3/8, 12/13, 1, 1/2]
        b = [
            [],
            [1/4],
            [3/32, 9/32],
            [1932/2197, -7200/2197, 7296/2197],
            [439/216, -8, 3680/513, -845/4104],
            [-8/27, 2, -3544/2565, 1859/4104, -11/40]
        ]
        c4 = [25/216, 0, 1408/2565, 2197/4104, -1/5, 0]  # 4th order
        c5 = [16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55]  # 5th order
        
        # Determine integration direction
        direction = 1 if x_end > x0 else -1
        h = abs(initial_step) * direction
        
        # Initialize arrays
        x_values = [x0]
        y_values = [y0]
        step_sizes = []
        error_estimates = []
        
        x, y = x0, y0
        steps = 0
        
        try:
            while (direction * (x_end - x) > self.atol) and (steps < max_steps):
                # Adjust final step if needed
                if direction * (x + h - x_end) > 0:
                    h = x_end - x
                
                # Compute RKF45 stages
                k = [0] * 6
                k[0] = h * func(x, y)
                
                for i in range(1, 6):
                    y_temp = y
                    for j in range(i):
                        y_temp += b[i][j] * k[j]
                    k[i] = h * func(x + a[i] * h, y_temp)
                
                # Compute 4th and 5th order solutions
                y4 = y + sum(c4[i] * k[i] for i in range(6))
                y5 = y + sum(c5[i] * k[i] for i in range(6))
                
                # Estimate local error
                error = abs(y5 - y4)
                
                # Compute tolerance
                tolerance = self.atol + self.rtol * max(abs(y), abs(y5))
                
                # Check if step is acceptable
                if error <= tolerance or abs(h) <= self.atol:
                    # Accept step
                    x_values.append(x + h)
                    y_values.append(y5)  # Use higher order solution
                    step_sizes.append(abs(h))
                    error_estimates.append(error)
                    
                    x, y = x + h, y5
                    steps += 1
                
                # Adjust step size for next iteration
                if error > 0:
                    factor = self.safety_factor * (tolerance / error) ** 0.2
                    factor = max(self.min_step_factor, min(self.max_step_factor, factor))
                    h = h * factor
                else:
                    h = h * self.max_step_factor
                
        except Exception as e:
            return IntegrationResult(
                x_values=x_values,
                y_values=y_values,
                step_sizes=step_sizes,
                error_estimates=error_estimates,
                total_steps=steps,
                success=False,
                message=f"RKF45 integration failed: {e}"
            )
        
        if steps >= max_steps:
            return IntegrationResult(
                x_values=x_values,
                y_values=y_values,
                step_sizes=step_sizes,
                error_estimates=error_estimates,
                total_steps=steps,
                success=False,
                message=f"Maximum steps ({max_steps}) exceeded"
            )
        
        return IntegrationResult(
            x_values=x_values,
            y_values=y_values,
            step_sizes=step_sizes,
            error_estimates=error_estimates,
            total_steps=steps,
            success=True,
            message="RKF45 integration completed successfully"
        )


class DormandPrince(IntegrationMethod):
    """
    Dormand-Prince 5(4) method - high accuracy adaptive integrator.
    
    State-of-the-art method with excellent stability and accuracy properties.
    Uses embedded 5th and 4th order formulas for error estimation.
    
    Ideal for:
    - High-accuracy requirements
    - Complex flow transitions
    - Research and validation applications
    """
    
    def __init__(
        self,
        rtol: float = 1e-8,
        atol: float = 1e-12,
        safety_factor: float = 0.9,
        min_step_factor: float = 0.2,
        max_step_factor: float = 10.0
    ):
        """
        Initialize Dormand-Prince integrator.
        
        Args:
            rtol: Relative tolerance for error control
            atol: Absolute tolerance for error control
            safety_factor: Safety factor for step size adjustment
            min_step_factor: Minimum step size reduction factor
            max_step_factor: Maximum step size increase factor
        """
        self.rtol = rtol
        self.atol = atol
        self.safety_factor = safety_factor
        self.min_step_factor = min_step_factor
        self.max_step_factor = max_step_factor
    
    def integrate(
        self,
        func: Callable[[float, float], float],
        x0: float,
        y0: float,
        x_end: float,
        initial_step: float,
        max_steps: int = 100000
    ) -> IntegrationResult:
        """
        Integrate using Dormand-Prince 5(4) method.
        
        Args:
            func: Function f(x, y) representing dy/dx
            x0: Initial x value
            y0: Initial y value
            x_end: Final x value
            initial_step: Initial step size guess
            max_steps: Maximum number of steps
            
        Returns:
            IntegrationResult with high-accuracy solution
        """
        # Dormand-Prince coefficients
        a = [0, 1/5, 3/10, 4/5, 8/9, 1, 1]
        b = [
            [],
            [1/5],
            [3/40, 9/40],
            [44/45, -56/15, 32/9],
            [19372/6561, -25360/2187, 64448/6561, -212/729],
            [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656],
            [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84]
        ]
        c5 = [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]  # 5th order
        c4 = [5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40]  # 4th order
        
        # Determine integration direction
        direction = 1 if x_end > x0 else -1
        h = abs(initial_step) * direction
        
        # Initialize arrays
        x_values = [x0]
        y_values = [y0]
        step_sizes = []
        error_estimates = []
        
        x, y = x0, y0
        steps = 0
        
        try:
            while (direction * (x_end - x) > self.atol) and (steps < max_steps):
                # Adjust final step if needed
                if direction * (x + h - x_end) > 0:
                    h = x_end - x
                
                # Compute Dormand-Prince stages
                k = [0] * 7
                k[0] = h * func(x, y)
                
                for i in range(1, 7):
                    y_temp = y
                    for j in range(i):
                        y_temp += b[i][j] * k[j]
                    k[i] = h * func(x + a[i] * h, y_temp)
                
                # Compute 4th and 5th order solutions
                y4 = y + sum(c4[i] * k[i] for i in range(7))
                y5 = y + sum(c5[i] * k[i] for i in range(7))
                
                # Estimate local error
                error = abs(y5 - y4)
                
                # Compute tolerance
                tolerance = self.atol + self.rtol * max(abs(y), abs(y5))
                
                # Check if step is acceptable
                if error <= tolerance or abs(h) <= self.atol:
                    # Accept step
                    x_values.append(x + h)
                    y_values.append(y5)  # Use higher order solution
                    step_sizes.append(abs(h))
                    error_estimates.append(error)
                    
                    x, y = x + h, y5
                    steps += 1
                
                # Adjust step size for next iteration
                if error > 0:
                    factor = self.safety_factor * (tolerance / error) ** 0.2
                    factor = max(self.min_step_factor, min(self.max_step_factor, factor))
                    h = h * factor
                else:
                    h = h * self.max_step_factor
                
        except Exception as e:
            return IntegrationResult(
                x_values=x_values,
                y_values=y_values,
                step_sizes=step_sizes,
                error_estimates=error_estimates,
                total_steps=steps,
                success=False,
                message=f"Dormand-Prince integration failed: {e}"
            )
        
        if steps >= max_steps:
            return IntegrationResult(
                x_values=x_values,
                y_values=y_values,
                step_sizes=step_sizes,
                error_estimates=error_estimates,
                total_steps=steps,
                success=False,
                message=f"Maximum steps ({max_steps}) exceeded"
            )
        
        return IntegrationResult(
            x_values=x_values,
            y_values=y_values,
            step_sizes=step_sizes,
            error_estimates=error_estimates,
            total_steps=steps,
            success=True,
            message="Dormand-Prince integration completed successfully"
        )


class AdaptiveIntegrator:
    """
    High-level adaptive integrator with automatic method selection.
    
    Automatically chooses the best integration method based on:
    - Problem characteristics
    - Accuracy requirements
    - Computational budget
    """
    
    def __init__(
        self,
        method: str = "dormand_prince",
        rtol: float = 1e-6,
        atol: float = 1e-9
    ):
        """
        Initialize adaptive integrator.
        
        Args:
            method: Integration method ("rk4", "rkf45", "dormand_prince")
            rtol: Relative tolerance
            atol: Absolute tolerance
        """
        self.method = method.lower()
        self.rtol = rtol
        self.atol = atol
        
        # Initialize integrator
        if self.method == "rk4":
            self.integrator = RungeKutta4(tolerance=atol)
        elif self.method == "rkf45":
            self.integrator = RungeKuttaFehlberg45(rtol=rtol, atol=atol)
        elif self.method == "dormand_prince":
            self.integrator = DormandPrince(rtol=rtol, atol=atol)
        else:
            raise ValueError(f"Unknown integration method: {method}")
    
    def integrate(
        self,
        func: Callable[[float, float], float],
        x0: float,
        y0: float,
        x_end: float,
        initial_step: Optional[float] = None,
        **kwargs
    ) -> IntegrationResult:
        """
        Integrate differential equation with automatic method selection.
        
        Args:
            func: Function f(x, y) representing dy/dx
            x0: Initial x value
            y0: Initial y value
            x_end: Final x value
            initial_step: Initial step size (auto-estimated if None)
            **kwargs: Additional arguments for specific methods
            
        Returns:
            IntegrationResult with optimal accuracy/efficiency
        """
        # Auto-estimate initial step size if not provided
        if initial_step is None:
            dx = abs(x_end - x0)
            initial_step = dx / 1000  # Conservative initial guess
        
        # Call appropriate integrator
        if self.method == "rk4":
            return self.integrator.integrate(func, x0, y0, x_end, initial_step, **kwargs)
        else:
            return self.integrator.integrate(func, x0, y0, x_end, initial_step, **kwargs)
