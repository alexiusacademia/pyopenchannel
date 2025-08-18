"""
File: numerical/fvm/schemes.py
Author: Alexius Academia
Date: 2025-08-17

Numerical schemes for FVM implementation.

This module provides shock-capturing numerical schemes for solving
the shallow water equations:
- Roe scheme (exact Riemann solver)
- HLL scheme (Harten-Lax-van Leer)
- HLLC scheme (with contact wave)
- Lax-Friedrichs scheme (simple and robust)

All schemes are designed to handle discontinuities and shocks
accurately while maintaining conservation properties.
"""

import numpy as np
import math
from typing import Tuple, Optional
from dataclasses import dataclass

from .core import (
    FVMScheme, 
    ConservativeVariables, 
    FluxVector,
    get_gravity
)
from ...exceptions import InvalidFlowConditionError


@dataclass
class RiemannState:
    """State for Riemann problem solution."""
    h: float      # Depth
    u: float      # Velocity
    c: float      # Wave speed
    
    @classmethod
    def from_conservative(cls, U: ConservativeVariables) -> 'RiemannState':
        """Create Riemann state from conservative variables."""
        h = U.h
        u = U.u
        c = math.sqrt(get_gravity() * h) if h > 1e-12 else 0.0
        return cls(h=h, u=u, c=c)


class LaxFriedrichsScheme(FVMScheme):
    """
    Lax-Friedrichs numerical scheme.
    
    Simple and robust first-order scheme with artificial diffusion.
    Good for initial testing and difficult cases.
    
    Formula: F_{i+1/2} = 0.5 * (F_L + F_R) - 0.5 * α * (U_R - U_L)
    where α is the maximum wave speed.
    """
    
    def __init__(self):
        """Initialize Lax-Friedrichs scheme."""
        super().__init__("Lax-Friedrichs")
    
    def calculate_flux(
        self, 
        U_left: ConservativeVariables, 
        U_right: ConservativeVariables
    ) -> FluxVector:
        """Calculate Lax-Friedrichs flux."""
        # Physical fluxes
        F_left = FluxVector.from_conservative(U_left)
        F_right = FluxVector.from_conservative(U_right)
        
        # Maximum wave speed
        alpha = self.calculate_max_wave_speed(U_left, U_right)
        
        # Lax-Friedrichs flux
        mass_flux = 0.5 * (F_left.mass_flux + F_right.mass_flux) - 0.5 * alpha * (U_right.h - U_left.h)
        momentum_flux = 0.5 * (F_left.momentum_flux + F_right.momentum_flux) - 0.5 * alpha * (U_right.hu - U_left.hu)
        
        return FluxVector(mass_flux=mass_flux, momentum_flux=momentum_flux)


class HLLScheme(FVMScheme):
    """
    HLL (Harten-Lax-van Leer) numerical scheme.
    
    Robust approximate Riemann solver that captures shocks well.
    More accurate than Lax-Friedrichs but diffusive at contact waves.
    
    Formula: F_{i+1/2} = (S_R * F_L - S_L * F_R + S_L * S_R * (U_R - U_L)) / (S_R - S_L)
    where S_L and S_R are left and right wave speeds.
    """
    
    def __init__(self):
        """Initialize HLL scheme."""
        super().__init__("HLL")
    
    def calculate_flux(
        self, 
        U_left: ConservativeVariables, 
        U_right: ConservativeVariables
    ) -> FluxVector:
        """Calculate HLL flux."""
        # Handle dry cells
        if U_left.h <= 1e-12 and U_right.h <= 1e-12:
            return FluxVector(mass_flux=0.0, momentum_flux=0.0)
        
        # Physical fluxes
        F_left = FluxVector.from_conservative(U_left)
        F_right = FluxVector.from_conservative(U_right)
        
        # Wave speeds
        S_left, S_right = self._estimate_wave_speeds(U_left, U_right)
        
        # HLL flux formula
        if S_left >= 0:
            # All waves moving right - use left flux
            return F_left
        elif S_right <= 0:
            # All waves moving left - use right flux
            return F_right
        else:
            # Waves on both sides - use HLL formula
            denominator = S_right - S_left
            
            mass_flux = (
                (S_right * F_left.mass_flux - S_left * F_right.mass_flux + 
                 S_left * S_right * (U_right.h - U_left.h)) / denominator
            )
            
            momentum_flux = (
                (S_right * F_left.momentum_flux - S_left * F_right.momentum_flux + 
                 S_left * S_right * (U_right.hu - U_left.hu)) / denominator
            )
            
            return FluxVector(mass_flux=mass_flux, momentum_flux=momentum_flux)
    
    def _estimate_wave_speeds(
        self, 
        U_left: ConservativeVariables, 
        U_right: ConservativeVariables
    ) -> Tuple[float, float]:
        """Estimate left and right wave speeds for HLL scheme."""
        # Left state
        if U_left.h > 1e-12:
            u_L = U_left.u
            c_L = math.sqrt(self.gravity * U_left.h)
        else:
            u_L = 0.0
            c_L = 0.0
        
        # Right state
        if U_right.h > 1e-12:
            u_R = U_right.u
            c_R = math.sqrt(self.gravity * U_right.h)
        else:
            u_R = 0.0
            c_R = 0.0
        
        # Simple wave speed estimates
        S_left = min(u_L - c_L, u_R - c_R)
        S_right = max(u_L + c_L, u_R + c_R)
        
        return (S_left, S_right)


class RoeScheme(FVMScheme):
    """
    Roe approximate Riemann solver.
    
    Exact Riemann solver using Roe averages. Provides excellent
    accuracy for smooth flows and good shock resolution.
    
    Can suffer from entropy violations in rare cases, but generally
    very reliable for hydraulic applications.
    """
    
    def __init__(self, entropy_fix: bool = True):
        """
        Initialize Roe scheme.
        
        Args:
            entropy_fix: Apply entropy fix for sonic points
        """
        super().__init__("Roe")
        self.entropy_fix = entropy_fix
        self.entropy_parameter = 0.1  # Parameter for entropy fix
    
    def calculate_flux(
        self, 
        U_left: ConservativeVariables, 
        U_right: ConservativeVariables
    ) -> FluxVector:
        """Calculate Roe flux."""
        # Handle dry cells
        if U_left.h <= 1e-12 and U_right.h <= 1e-12:
            return FluxVector(mass_flux=0.0, momentum_flux=0.0)
        
        # Physical fluxes
        F_left = FluxVector.from_conservative(U_left)
        F_right = FluxVector.from_conservative(U_right)
        
        # Roe averages
        roe_state = self._calculate_roe_averages(U_left, U_right)
        
        # Eigenvalues and eigenvectors
        eigenvalues = self._calculate_eigenvalues(roe_state)
        eigenvectors = self._calculate_eigenvectors(roe_state)
        
        # Wave strengths
        wave_strengths = self._calculate_wave_strengths(U_left, U_right, roe_state)
        
        # Apply entropy fix if needed
        if self.entropy_fix:
            eigenvalues = self._apply_entropy_fix(eigenvalues, U_left, U_right)
        
        # Roe flux: F = 0.5 * (F_L + F_R) - 0.5 * Σ |λ_k| α_k r_k
        flux_correction = np.zeros(2)
        
        for k in range(2):
            flux_correction += abs(eigenvalues[k]) * wave_strengths[k] * eigenvectors[:, k]
        
        mass_flux = 0.5 * (F_left.mass_flux + F_right.mass_flux) - 0.5 * flux_correction[0]
        momentum_flux = 0.5 * (F_left.momentum_flux + F_right.momentum_flux) - 0.5 * flux_correction[1]
        
        return FluxVector(mass_flux=mass_flux, momentum_flux=momentum_flux)
    
    def _calculate_roe_averages(
        self, 
        U_left: ConservativeVariables, 
        U_right: ConservativeVariables
    ) -> RiemannState:
        """Calculate Roe-averaged state."""
        # Handle dry cells
        if U_left.h <= 1e-12:
            return RiemannState.from_conservative(U_right)
        if U_right.h <= 1e-12:
            return RiemannState.from_conservative(U_left)
        
        # Roe averages for shallow water
        sqrt_h_L = math.sqrt(U_left.h)
        sqrt_h_R = math.sqrt(U_right.h)
        
        h_roe = 0.5 * (U_left.h + U_right.h)
        u_roe = (sqrt_h_L * U_left.u + sqrt_h_R * U_right.u) / (sqrt_h_L + sqrt_h_R)
        c_roe = math.sqrt(self.gravity * h_roe)
        
        return RiemannState(h=h_roe, u=u_roe, c=c_roe)
    
    def _calculate_eigenvalues(self, roe_state: RiemannState) -> np.ndarray:
        """Calculate eigenvalues (characteristic speeds)."""
        return np.array([
            roe_state.u - roe_state.c,  # λ₁ = u - c
            roe_state.u + roe_state.c   # λ₂ = u + c
        ])
    
    def _calculate_eigenvectors(self, roe_state: RiemannState) -> np.ndarray:
        """Calculate right eigenvectors."""
        return np.array([
            [1.0, 1.0],                                    # First row: [1, 1]
            [roe_state.u - roe_state.c, roe_state.u + roe_state.c]  # Second row: [u-c, u+c]
        ])
    
    def _calculate_wave_strengths(
        self, 
        U_left: ConservativeVariables, 
        U_right: ConservativeVariables, 
        roe_state: RiemannState
    ) -> np.ndarray:
        """Calculate wave strengths (α coefficients)."""
        # Differences in conservative variables
        delta_h = U_right.h - U_left.h
        delta_hu = U_right.hu - U_left.hu
        
        # Wave strengths for shallow water equations
        alpha_1 = 0.5 * (delta_hu - roe_state.u * delta_h - roe_state.c * delta_h) / roe_state.c
        alpha_2 = 0.5 * (delta_hu - roe_state.u * delta_h + roe_state.c * delta_h) / roe_state.c
        
        return np.array([alpha_1, alpha_2])
    
    def _apply_entropy_fix(
        self, 
        eigenvalues: np.ndarray, 
        U_left: ConservativeVariables, 
        U_right: ConservativeVariables
    ) -> np.ndarray:
        """Apply entropy fix to prevent entropy violations."""
        fixed_eigenvalues = eigenvalues.copy()
        
        # Calculate actual eigenvalues at left and right states
        if U_left.h > 1e-12:
            c_L = math.sqrt(self.gravity * U_left.h)
            lambda_L = [U_left.u - c_L, U_left.u + c_L]
        else:
            lambda_L = [0.0, 0.0]
        
        if U_right.h > 1e-12:
            c_R = math.sqrt(self.gravity * U_right.h)
            lambda_R = [U_right.u - c_R, U_right.u + c_R]
        else:
            lambda_R = [0.0, 0.0]
        
        # Apply entropy fix for each eigenvalue
        for k in range(2):
            if abs(eigenvalues[k]) < self.entropy_parameter:
                # Check for sonic point
                if lambda_L[k] * lambda_R[k] < 0:
                    # Sonic point detected - apply entropy fix
                    delta_lambda = max(0, lambda_R[k] - lambda_L[k])
                    fixed_eigenvalues[k] = 0.5 * (eigenvalues[k]**2 + delta_lambda**2) / delta_lambda
        
        return fixed_eigenvalues


class HLLCScheme(FVMScheme):
    """
    HLLC (HLL-Contact) numerical scheme.
    
    Extension of HLL that resolves contact discontinuities better.
    Less diffusive than HLL while maintaining robustness.
    
    Excellent choice for high-accuracy applications.
    """
    
    def __init__(self):
        """Initialize HLLC scheme."""
        super().__init__("HLLC")
    
    def calculate_flux(
        self, 
        U_left: ConservativeVariables, 
        U_right: ConservativeVariables
    ) -> FluxVector:
        """Calculate HLLC flux."""
        # Handle dry cells
        if U_left.h <= 1e-12 and U_right.h <= 1e-12:
            return FluxVector(mass_flux=0.0, momentum_flux=0.0)
        
        # Physical fluxes
        F_left = FluxVector.from_conservative(U_left)
        F_right = FluxVector.from_conservative(U_right)
        
        # Wave speeds
        S_left, S_right, S_star = self._estimate_wave_speeds_hllc(U_left, U_right)
        
        # HLLC flux formula
        if S_left >= 0:
            # All waves moving right
            return F_left
        elif S_right <= 0:
            # All waves moving left
            return F_right
        elif S_star >= 0:
            # Left intermediate state
            U_star_left = self._calculate_star_state(U_left, S_left, S_star)
            F_star_left = self._calculate_star_flux(F_left, U_left, U_star_left, S_left)
            return F_star_left
        else:
            # Right intermediate state
            U_star_right = self._calculate_star_state(U_right, S_right, S_star)
            F_star_right = self._calculate_star_flux(F_right, U_right, U_star_right, S_right)
            return F_star_right
    
    def _estimate_wave_speeds_hllc(
        self, 
        U_left: ConservativeVariables, 
        U_right: ConservativeVariables
    ) -> Tuple[float, float, float]:
        """Estimate wave speeds for HLLC scheme."""
        # Left and right states
        if U_left.h > 1e-12:
            u_L = U_left.u
            c_L = math.sqrt(self.gravity * U_left.h)
        else:
            u_L = 0.0
            c_L = 0.0
        
        if U_right.h > 1e-12:
            u_R = U_right.u
            c_R = math.sqrt(self.gravity * U_right.h)
        else:
            u_R = 0.0
            c_R = 0.0
        
        # Left and right wave speeds
        S_left = min(u_L - c_L, u_R - c_R)
        S_right = max(u_L + c_L, u_R + c_R)
        
        # Contact wave speed (star region)
        if abs(S_right - S_left) < 1e-12:
            S_star = 0.5 * (u_L + u_R)
        else:
            numerator = (
                U_right.hu - U_left.hu + 
                S_left * U_left.h - S_right * U_right.h
            )
            denominator = U_left.h - U_right.h + S_left * U_left.h / S_left - S_right * U_right.h / S_right
            
            if abs(denominator) > 1e-12:
                S_star = numerator / denominator
            else:
                S_star = 0.5 * (u_L + u_R)
        
        return (S_left, S_right, S_star)
    
    def _calculate_star_state(
        self, 
        U: ConservativeVariables, 
        S: float, 
        S_star: float
    ) -> ConservativeVariables:
        """Calculate intermediate (star) state."""
        if abs(S - S_star) < 1e-12:
            return U.copy()
        
        factor = U.h * (S - U.u) / (S - S_star)
        
        h_star = factor
        hu_star = factor * S_star
        
        return ConservativeVariables(h=h_star, hu=hu_star)
    
    def _calculate_star_flux(
        self, 
        F: FluxVector, 
        U: ConservativeVariables, 
        U_star: ConservativeVariables, 
        S: float
    ) -> FluxVector:
        """Calculate flux in star region."""
        mass_flux = F.mass_flux + S * (U_star.h - U.h)
        momentum_flux = F.momentum_flux + S * (U_star.hu - U.hu)
        
        return FluxVector(mass_flux=mass_flux, momentum_flux=momentum_flux)
