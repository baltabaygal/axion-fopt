"""
Analyze energy scaling from first principles.

Run noPT simulations at specific θ₀ values and trace the energy evolution
to understand the A ≈ 0.215 factor.
"""

import numpy as np
from axion_sim import create_simulation_pt_at_zero

def analyze_theta0(theta0, H_PT=0.7, Ngrid=64):
    """
    Run noPT simulation and analyze energy scaling.

    Returns dict with:
    - theta0_input
    - energy_at_start
    - energy_at_t_osc (if available)
    - energy_at_t_end
    - scale factors
    - redshift factors
    """

    # Time definitions
    M_phi = 1.0  # in code units
    t_osc = 3.0 / (2.0 * M_phi)
    t_end = 10.0 / H_PT

    # Create simulation: start at t=0, run to t_end
    sim, tau_final = create_simulation_pt_at_zero(
        H_PT=H_PT,
        beta=0.0,
        start_time=0.0,
        end_time=t_end,
        Ngrid=Ngrid,
        num_tracers=1_000,
        enable_nucleation=False,
        mb=1.0,
        theta0_initial=theta0,
        v_bubble=0.6,
        energy_save_interval=10,
    )

    # Run simulation
    results = sim.run_simulation(tau_final, progress_bar=False, save_interval=10)

    # Extract energy history
    energy_hist = results['energy_history']
    times_cosmic = results['times_cosmic']
    config = results['config']

    # Scale factors: a(t) = sqrt(2*H_PT*t)
    a_end = np.sqrt(2.0 * H_PT * t_end)
    a_osc = np.sqrt(2.0 * H_PT * t_osc)
    redshift_cube = (a_osc / a_end) ** 3

    # Find energy at closest times to t_osc and t_end
    times_cosmic_arr = np.array(times_cosmic)

    # Energy at start (first recorded)
    E_start = energy_hist[0]['total_energy']
    a_start = energy_hist[0]['scale_factor']

    # Energy at t_osc
    idx_osc = np.argmin(np.abs(times_cosmic_arr - t_osc))
    E_osc = energy_hist[idx_osc]['total_energy']
    a_osc_actual = energy_hist[idx_osc]['scale_factor']
    t_osc_actual = times_cosmic_arr[idx_osc]

    # Energy at t_end (last recorded)
    E_end = energy_hist[-1]['total_energy']
    a_end_actual = energy_hist[-1]['scale_factor']
    t_end_actual = times_cosmic_arr[-1]

    # Redshift energy from t_end back to t_osc
    E_redshifted = E_end / redshift_cube

    # What we extract from data:
    # rho_noPT = E_end * (a_osc/a_end)^3
    #
    # Define f_anh such that:
    # rho_noPT = A * f_anh * theta0^2 * (a_osc/a_end)^3
    #
    # For small theta0, f_anh ≈ 1, so:
    # E_end * (a_osc/a_end)^3 ≈ A * theta0^2 * (a_osc/a_end)^3
    # => A ≈ E_end / theta0^2

    A_naive = E_end / (theta0**2)

    # But also check what the actual potential energy is
    V_start = energy_hist[0]['potential_energy']
    V_osc = energy_hist[idx_osc]['potential_energy']
    V_end = energy_hist[-1]['potential_energy']

    K_start = energy_hist[0]['kinetic_energy']
    K_osc = energy_hist[idx_osc]['kinetic_energy']
    K_end = energy_hist[-1]['kinetic_energy']

    # For reference: theoretical potential energy at theta0 with m=1
    V_theory = 1.0 - np.cos(theta0)

    return {
        'theta0': theta0,
        'H_PT': H_PT,
        't_osc': t_osc,
        't_end': t_end,
        'a_osc': a_osc,
        'a_end': a_end,
        'redshift_cube': redshift_cube,
        'E_start': E_start,
        'E_osc': E_osc,
        'E_end': E_end,
        'E_redshifted': E_redshifted,
        'V_start': V_start,
        'V_osc': V_osc,
        'V_end': V_end,
        'K_start': K_start,
        'K_osc': K_osc,
        'K_end': K_end,
        'V_theory': V_theory,
        'A_naive': A_naive,
        't_osc_actual': t_osc_actual,
        't_end_actual': t_end_actual,
        'a_osc_actual': a_osc_actual,
        'a_end_actual': a_end_actual,
    }

if __name__ == '__main__':
    print("=" * 80)
    print("Energy Scaling Analysis: First Principles")
    print("=" * 80)

    # Test a few theta0 values
    test_theta0s = [0.5, 1.0, 1.5, 2.0]

    results = []
    for theta0 in test_theta0s:
        print(f"\nRunning θ₀ = {theta0:.4f}...")
        res = analyze_theta0(theta0)
        results.append(res)

        print(f"  Theory: V(θ₀) = 1 - cos(θ₀) = {res['V_theory']:.6f}")
        print(f"  Simulation:")
        print(f"    E_start  = {res['E_start']:.6e} (K={res['K_start']:.6e}, V={res['V_start']:.6e})")
        print(f"    E_osc    = {res['E_osc']:.6e} (K={res['K_osc']:.6e}, V={res['V_osc']:.6e})")
        print(f"    E_end    = {res['E_end']:.6e} (K={res['K_end']:.6e}, V={res['V_end']:.6e})")
        print(f"  Redshift factor: (a_osc/a_end)³ = {res['redshift_cube']:.6f}")
        print(f"  E_redshifted = E_end / (a_osc/a_end)³ = {res['E_redshifted']:.6e}")
        print(f"  A_naive = E_end / θ₀² = {res['A_naive']:.6f}")
        print(f"  Ratio E_end / V_theory = {res['E_end'] / res['V_theory']:.6f}")

    print("\n" + "=" * 80)
    print("Summary Table")
    print("=" * 80)
    print(f"{'θ₀':<10} {'V_theory':<15} {'E_end':<15} {'E_end/V_theory':<15} {'A_naive':<15}")
    print("-" * 80)
    for res in results:
        ratio = res['E_end'] / res['V_theory']
        print(f"{res['theta0']:<10.4f} {res['V_theory']:<15.6e} {res['E_end']:<15.6e} {ratio:<15.6f} {res['A_naive']:<15.6f}")

    # Key question: Is A_naive ≈ 0.215?
    print("\n" + "=" * 80)
    print("Analysis")
    print("=" * 80)
    avg_A = np.mean([r['A_naive'] for r in results])
    print(f"Average A_naive over test θ₀ values: {avg_A:.6f}")
    print(f"Expected from fit: A = 0.215412")
    print(f"Ratio: {avg_A / 0.215412:.6f}")

    # Also check the relationship with theta0^2
    print("\nChecking E_end vs θ₀²:")
    theta0_array = np.array([r['theta0'] for r in results])
    E_end_array = np.array([r['E_end'] for r in results])

    # Linear fit: E_end = C * theta0^2
    C = np.polyfit(theta0_array**2, E_end_array, 1)[0]
    print(f"E_end ≈ {C:.6f} * θ₀²")
    print(f"This matches A_naive if true: expected ~0.215")

    print("\n" + "=" * 80)
