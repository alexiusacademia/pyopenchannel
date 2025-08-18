#!/usr/bin/env python3
"""
RVF FVM Implementation Comparison - PyOpenChannel

Author: Alexius Academia
Email: alexius.sayco.academia@gmail.com

This example demonstrates the difference between:
1. Current algebraic RVF approach
2. Proposed FVM approach for detailed RVF profiles

The FVM approach would provide much more accurate representation of:
- Shock structure within hydraulic jumps
- Velocity and pressure distributions
- Turbulent mixing zones
- Energy dissipation mechanisms
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pyopenchannel as poc
import numpy as np
import math

# Optional matplotlib import for plotting
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
    print("📊 Matplotlib available - FVM comparison plots will be generated")
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️  Matplotlib not available - analysis will run without plots")


class RVF_FVM_Solver:
    """
    Conceptual FVM solver for RVF analysis.
    
    This demonstrates what a full FVM implementation would provide
    for accurate hydraulic jump profile calculation.
    
    NOTE: This is a conceptual implementation showing the approach.
    A full FVM solver would require extensive development.
    """
    
    def __init__(self, nx=100, cfl=0.5):
        """
        Initialize FVM solver.
        
        Args:
            nx: Number of grid cells
            cfl: CFL number for stability
        """
        self.nx = nx
        self.cfl = cfl
        self.gamma = 1.4  # Ratio of specific heats (for compressible analogy)
        
    def solve_hydraulic_jump_fvm(self, channel, discharge, upstream_depth, 
                                downstream_depth, jump_length):
        """
        Solve hydraulic jump using FVM approach.
        
        This would solve the shallow water equations:
        ∂h/∂t + ∂(hu)/∂x = 0                    (Continuity)
        ∂(hu)/∂t + ∂(hu² + gh²/2)/∂x = -ghSf   (Momentum)
        
        Using finite volume discretization with shock-capturing schemes.
        """
        
        # Grid setup
        x = np.linspace(0, jump_length, self.nx)
        dx = x[1] - x[0]
        
        # Initialize conservative variables
        # U = [h, hu] where h=depth, u=velocity
        h = np.zeros(self.nx)
        hu = np.zeros(self.nx)
        
        # Initial conditions
        u1 = discharge / (channel.width * upstream_depth)
        u2 = discharge / (channel.width * downstream_depth)
        
        # Upstream conditions (supercritical)
        h[:self.nx//3] = upstream_depth
        hu[:self.nx//3] = upstream_depth * u1
        
        # Downstream conditions (subcritical)
        h[2*self.nx//3:] = downstream_depth
        hu[2*self.nx//3:] = downstream_depth * u2
        
        # Transition region (initial guess)
        transition_region = slice(self.nx//3, 2*self.nx//3)
        h_transition = np.linspace(upstream_depth, downstream_depth, 
                                 len(range(*transition_region.indices(self.nx))))
        h[transition_region] = h_transition
        
        # Velocity in transition (approximately conserve discharge)
        for i in range(self.nx//3, 2*self.nx//3):
            if h[i] > 0:
                hu[i] = discharge / channel.width
            else:
                hu[i] = 0
        
        # Time stepping parameters
        dt = self.cfl * dx / max(abs(hu/h) + np.sqrt(9.81 * h))
        max_time = 10.0  # Steady state time
        time = 0
        
        # Store results
        profiles = []
        times = []
        
        print(f"   🔧 FVM Setup: {self.nx} cells, dx={dx:.3f}m, dt={dt:.4f}s")
        print(f"   ⏱️  Solving to steady state (t_max={max_time}s)...")
        
        # Main time-stepping loop (conceptual)
        step = 0
        while time < max_time and step < 1000:
            
            # Calculate fluxes using Roe or HLL scheme
            fluxes_h, fluxes_hu = self._calculate_fluxes(h, hu, dx)
            
            # Update conservative variables
            h_new = h - dt/dx * fluxes_h
            hu_new = hu - dt/dx * fluxes_hu
            
            # Apply boundary conditions
            h_new[0] = upstream_depth
            hu_new[0] = upstream_depth * u1
            h_new[-1] = downstream_depth
            hu_new[-1] = downstream_depth * u2
            
            # Check for convergence
            residual = np.max(np.abs(h_new - h))
            
            if step % 100 == 0:
                print(f"   Step {step}: t={time:.3f}s, residual={residual:.2e}")
                profiles.append(h.copy())
                times.append(time)
            
            if residual < 1e-6:
                print(f"   ✅ Converged at step {step}, t={time:.3f}s")
                break
            
            # Update for next iteration
            h = h_new
            hu = hu_new
            time += dt
            step += 1
        
        # Calculate final velocities and flow properties
        u = np.where(h > 1e-6, hu / h, 0)
        froude = np.where(h > 1e-6, u / np.sqrt(9.81 * h), 0)
        
        # Calculate energy and momentum
        energy = h + u**2 / (2 * 9.81)
        momentum = hu * u + 9.81 * h**2 / 2
        
        return {
            'x': x,
            'depth': h,
            'velocity': u,
            'froude': froude,
            'energy': energy,
            'momentum': momentum,
            'profiles': profiles,
            'times': times,
            'converged': residual < 1e-6,
            'steps': step
        }
    
    def _calculate_fluxes(self, h, hu, dx):
        """
        Calculate numerical fluxes using Roe or HLL scheme.
        
        This is where the shock-capturing magic happens!
        """
        # Simplified flux calculation (conceptual)
        # Real implementation would use Roe, HLL, or HLLC schemes
        
        fluxes_h = np.zeros_like(h)
        fluxes_hu = np.zeros_like(hu)
        
        for i in range(1, len(h)-1):
            # Left and right states
            h_L, h_R = h[i-1], h[i]
            hu_L, hu_R = hu[i-1], hu[i]
            
            # Calculate fluxes (simplified Lax-Friedrichs)
            u_L = hu_L / h_L if h_L > 1e-6 else 0
            u_R = hu_R / h_R if h_R > 1e-6 else 0
            
            # Physical fluxes
            F1_L = hu_L
            F1_R = hu_R
            F2_L = hu_L * u_L + 9.81 * h_L**2 / 2
            F2_R = hu_R * u_R + 9.81 * h_R**2 / 2
            
            # Wave speeds
            c_L = np.sqrt(9.81 * h_L) if h_L > 0 else 0
            c_R = np.sqrt(9.81 * h_R) if h_R > 0 else 0
            lambda_max = max(abs(u_L) + c_L, abs(u_R) + c_R)
            
            # Lax-Friedrichs flux
            fluxes_h[i] = 0.5 * (F1_L + F1_R) - 0.5 * lambda_max * (h_R - h_L)
            fluxes_hu[i] = 0.5 * (F2_L + F2_R) - 0.5 * lambda_max * (hu_R - hu_L)
        
        return fluxes_h, fluxes_hu


def compare_rvf_methods():
    """Compare current algebraic method vs FVM approach."""
    
    print("🔬 RVF METHOD COMPARISON: Algebraic vs FVM")
    print("=" * 80)
    
    # Test conditions
    channel = poc.RectangularChannel(width=8.0)
    discharge = 25.0
    upstream_depth = 0.8
    downstream_depth = 3.2
    
    print(f"📊 TEST CONDITIONS:")
    print(f"   Channel width: {channel.width} m")
    print(f"   Discharge: {discharge} m³/s")
    print(f"   Upstream depth: {upstream_depth} m")
    print(f"   Downstream depth: {downstream_depth} m")
    
    # === METHOD 1: Current Algebraic Approach ===
    print(f"\n🔧 METHOD 1: CURRENT ALGEBRAIC APPROACH")
    print("-" * 60)
    
    rvf_solver = poc.RVFSolver()
    algebraic_result = rvf_solver.analyze_hydraulic_jump(
        channel=channel,
        discharge=discharge,
        upstream_depth=upstream_depth,
        tailwater_depth=downstream_depth
    )
    
    if algebraic_result.success:
        print(f"   ✅ Analysis: SUCCESS")
        print(f"   Jump type: {algebraic_result.jump_type.value}")
        print(f"   Jump length: {algebraic_result.jump_length:.1f} m")
        print(f"   Energy loss: {algebraic_result.energy_loss:.3f} m")
        print(f"   Efficiency: {algebraic_result.energy_efficiency:.1%}")
        print(f"   Method: Momentum equation + empirical correlations")
        print(f"   Profile: Linear interpolation between end states")
    
    # === METHOD 2: FVM Approach (Conceptual) ===
    print(f"\n🚀 METHOD 2: FVM APPROACH (CONCEPTUAL)")
    print("-" * 60)
    
    fvm_solver = RVF_FVM_Solver(nx=200, cfl=0.3)
    jump_length = algebraic_result.jump_length if algebraic_result.jump_length else 10.0
    
    fvm_result = fvm_solver.solve_hydraulic_jump_fvm(
        channel=channel,
        discharge=discharge,
        upstream_depth=upstream_depth,
        downstream_depth=downstream_depth,
        jump_length=jump_length
    )
    
    if fvm_result['converged']:
        print(f"   ✅ Analysis: SUCCESS")
        print(f"   Grid cells: {fvm_solver.nx}")
        print(f"   Convergence: {fvm_result['steps']} steps")
        print(f"   Method: Shallow water equations + shock-capturing")
        print(f"   Profile: Detailed flow structure resolved")
        
        # Calculate FVM-based properties
        fvm_energy_loss = fvm_result['energy'][0] - fvm_result['energy'][-1]
        print(f"   Energy loss (FVM): {fvm_energy_loss:.3f} m")
        
        # Find shock location
        depth_gradient = np.gradient(fvm_result['depth'])
        shock_idx = np.argmax(np.abs(depth_gradient))
        shock_location = fvm_result['x'][shock_idx]
        print(f"   Shock center: {shock_location:.1f} m")
        
        # Shock thickness
        shock_region = np.where(np.abs(depth_gradient) > 0.1 * np.max(np.abs(depth_gradient)))[0]
        if len(shock_region) > 0:
            shock_thickness = fvm_result['x'][shock_region[-1]] - fvm_result['x'][shock_region[0]]
            print(f"   Shock thickness: {shock_thickness:.2f} m")
    
    return algebraic_result, fvm_result


def create_comparison_plots(algebraic_result, fvm_result):
    """Create comparison plots between methods."""
    
    if not MATPLOTLIB_AVAILABLE:
        print("📊 Matplotlib not available - skipping plots")
        return
    
    print(f"\n📊 CREATING COMPARISON PLOTS...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('RVF Analysis: Algebraic vs FVM Comparison\nPyOpenChannel Professional Analysis', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Depth profiles
    ax1.set_title('Water Depth Profiles')
    
    # Algebraic approach (simplified)
    jump_length = algebraic_result.jump_length if algebraic_result.jump_length else 10.0
    x_alg = np.linspace(0, jump_length, 50)
    depth_alg = np.zeros_like(x_alg)
    
    for i, x in enumerate(x_alg):
        if x < jump_length * 0.2:
            depth_alg[i] = algebraic_result.upstream_depth
        elif x > jump_length * 0.8:
            depth_alg[i] = algebraic_result.downstream_depth
        else:
            # Linear transition
            progress = (x - jump_length * 0.2) / (jump_length * 0.6)
            depth_alg[i] = (algebraic_result.upstream_depth + 
                           (algebraic_result.downstream_depth - algebraic_result.upstream_depth) * progress)
    
    ax1.plot(x_alg, depth_alg, 'b-', linewidth=3, label='Algebraic (Current)', alpha=0.7)
    
    # FVM approach
    if fvm_result['converged']:
        ax1.plot(fvm_result['x'], fvm_result['depth'], 'r-', linewidth=2, 
                label='FVM (Detailed)', alpha=0.8)
    
    ax1.set_xlabel('Distance (m)')
    ax1.set_ylabel('Depth (m)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Velocity profiles
    ax2.set_title('Velocity Profiles')
    
    # Algebraic velocities
    vel_alg = np.where(depth_alg > 0, 25.0 / (8.0 * depth_alg), 0)
    ax2.plot(x_alg, vel_alg, 'b-', linewidth=3, label='Algebraic', alpha=0.7)
    
    # FVM velocities
    if fvm_result['converged']:
        ax2.plot(fvm_result['x'], fvm_result['velocity'], 'r-', linewidth=2, 
                label='FVM', alpha=0.8)
    
    ax2.set_xlabel('Distance (m)')
    ax2.set_ylabel('Velocity (m/s)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Froude number
    ax3.set_title('Froude Number Distribution')
    
    # Algebraic Froude
    froude_alg = vel_alg / np.sqrt(9.81 * depth_alg)
    ax3.plot(x_alg, froude_alg, 'b-', linewidth=3, label='Algebraic', alpha=0.7)
    
    # FVM Froude
    if fvm_result['converged']:
        ax3.plot(fvm_result['x'], fvm_result['froude'], 'r-', linewidth=2, 
                label='FVM', alpha=0.8)
    
    ax3.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Critical (Fr=1)')
    ax3.set_xlabel('Distance (m)')
    ax3.set_ylabel('Froude Number')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Energy grade line
    ax4.set_title('Energy Grade Line')
    
    # Algebraic energy
    energy_alg = depth_alg + vel_alg**2 / (2 * 9.81)
    ax4.plot(x_alg, energy_alg, 'b-', linewidth=3, label='Algebraic', alpha=0.7)
    
    # FVM energy
    if fvm_result['converged']:
        ax4.plot(fvm_result['x'], fvm_result['energy'], 'r-', linewidth=2, 
                label='FVM', alpha=0.8)
    
    ax4.set_xlabel('Distance (m)')
    ax4.set_ylabel('Energy (m)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    # Save plot
    plt.savefig('rvf_method_comparison.png', dpi=300, bbox_inches='tight')
    print(f"   ✅ Comparison plots saved as 'rvf_method_comparison.png'")
    plt.show()


def analyze_fvm_advantages():
    """Analyze the advantages of FVM for RVF."""
    
    print(f"\n🎯 FVM ADVANTAGES FOR RVF ANALYSIS:")
    print("=" * 80)
    
    advantages = [
        {
            "aspect": "Physical Accuracy",
            "current": "Simplified momentum balance",
            "fvm": "Full shallow water equations",
            "benefit": "Captures complex flow physics"
        },
        {
            "aspect": "Shock Resolution",
            "current": "Linear interpolation",
            "fvm": "Shock-capturing schemes",
            "benefit": "Accurate shock structure"
        },
        {
            "aspect": "Profile Detail",
            "current": "End-point calculation",
            "fvm": "Continuous profile",
            "benefit": "Internal flow structure"
        },
        {
            "aspect": "Turbulence",
            "current": "Empirical correlations",
            "fvm": "Resolved mixing zones",
            "benefit": "Better energy dissipation"
        },
        {
            "aspect": "Pressure Distribution",
            "current": "Hydrostatic assumption",
            "fvm": "Dynamic pressure effects",
            "benefit": "Accurate force calculations"
        },
        {
            "aspect": "Air Entrainment",
            "current": "Not modeled",
            "fvm": "Can be included",
            "benefit": "Complete jump physics"
        }
    ]
    
    print(f"{'Aspect':<20} | {'Current Method':<25} | {'FVM Method':<25} | {'Benefit':<30}")
    print("-" * 105)
    
    for adv in advantages:
        print(f"{adv['aspect']:<20} | {adv['current']:<25} | {adv['fvm']:<25} | {adv['benefit']:<30}")
    
    print(f"\n💡 WHEN TO USE FVM:")
    print(f"   ✅ Detailed hydraulic jump design")
    print(f"   ✅ Stilling basin optimization")
    print(f"   ✅ Energy dissipator design")
    print(f"   ✅ Research and development")
    print(f"   ✅ Complex geometry effects")
    print(f"   ✅ Unsteady flow analysis")
    
    print(f"\n⚖️  WHEN CURRENT METHOD IS SUFFICIENT:")
    print(f"   ✅ Preliminary design")
    print(f"   ✅ Overall energy balance")
    print(f"   ✅ Jump classification")
    print(f"   ✅ Routine engineering calculations")
    print(f"   ✅ Educational purposes")


def main():
    """Run RVF method comparison analysis."""
    
    print("🔬 RVF ANALYSIS: ALGEBRAIC vs FVM COMPARISON")
    print("=" * 90)
    print("Demonstrating accuracy differences for hydraulic jump analysis")
    print("Author: Alexius Academia")
    print("=" * 90)
    
    try:
        # Compare methods
        algebraic_result, fvm_result = compare_rvf_methods()
        
        # Create comparison plots
        create_comparison_plots(algebraic_result, fvm_result)
        
        # Analyze advantages
        analyze_fvm_advantages()
        
        print("\n" + "=" * 90)
        print("🎉 RVF METHOD COMPARISON COMPLETED!")
        print("=" * 90)
        
        print(f"\n🎯 CONCLUSION:")
        print(f"   • Current method: Good for engineering design")
        print(f"   • FVM approach: Superior for detailed analysis")
        print(f"   • Recommendation: Use FVM for critical applications")
        print(f"   • Implementation: Would require significant development")
        
    except Exception as e:
        print(f"\n❌ Analysis failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
