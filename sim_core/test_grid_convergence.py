"""
Test grid convergence for noPT simulations.

Runs the same noPT simulation with two different grid resolutions
(64 and 128) and compares the final total energy density.
"""

import numpy as np
from axion_sim import create_simulation_pt_at_zero


def run_nopt_test(Ngrid, H_PT=0.7, theta0=1.0):
    """
    Run a single noPT simulation and return final total energy density.

    Parameters
    ----------
    Ngrid : int
        Grid resolution
    H_PT : float
        Hubble parameter
    theta0 : float
        Initial axion field angle

    Returns
    -------
    float
        Final total energy density
    """
    end_time = 10.0 / H_PT

    sim, tau_final = create_simulation_pt_at_zero(
        H_PT=H_PT,
        beta=0.0,
        start_time=0.0,
        end_time=end_time,
        nucleation_mode='standard',
        Ngrid=Ngrid,
        num_tracers=1_000,  # irrelevant for noPT
        enable_nucleation=False,
        mb=1.0,  # background mass everywhere (no bubbles)
        theta0_initial=theta0,
        v_bubble=0.6,
        spatial_hash_cells=8,
        energy_save_interval=10,
        checkpoint_interval=0,
    )

    print(f"  Running noPT with Ngrid={Ngrid}...")
    results = sim.run_simulation(tau_final, progress_bar=True, save_interval=10)

    if results['energy_history']:
        final_energy = results['energy_history'][-1]
        rho_total = final_energy['total_energy']
        return rho_total
    else:
        raise RuntimeError("No energy history saved")


if __name__ == '__main__':
    print("=" * 70)
    print("Grid Convergence Test for noPT Simulations")
    print("=" * 70)

    H_PT = 0.7
    theta0 = 1.0

    print(f"\nTest parameters:")
    print(f"  θ₀ = {theta0}")
    print(f"  H_PT = {H_PT}")
    print(f"  end_time = {10.0/H_PT:.4f}")
    print()

    # Run with Ngrid=64
    print("Running Ngrid=64...")
    rho_64 = run_nopt_test(Ngrid=64, H_PT=H_PT, theta0=theta0)
    print(f"  Final total energy density (Ngrid=64): {rho_64:.6e}\n")

    # Run with Ngrid=128
    print("Running Ngrid=128...")
    rho_128 = run_nopt_test(Ngrid=128, H_PT=H_PT, theta0=theta0)
    print(f"  Final total energy density (Ngrid=128): {rho_128:.6e}\n")

    # Compare
    print("=" * 70)
    print("Comparison:")
    print("=" * 70)
    print(f"  rho(Ngrid=64):  {rho_64:.6e}")
    print(f"  rho(Ngrid=128): {rho_128:.6e}")

    relative_diff = abs(rho_128 - rho_64) / rho_64 * 100
    print(f"  Relative diff:  {relative_diff:.2f}%")

    if relative_diff < 2.0:
        print("\n✓ Grids are converged (< 2% difference). Use Ngrid=64 for speed.")
    elif relative_diff < 5.0:
        print("\n△ Modest difference (2-5%). Ngrid=128 may be safer.")
    else:
        print("\n✗ Significant difference (> 5%). Use Ngrid=128 or higher.")
    print("=" * 70)
