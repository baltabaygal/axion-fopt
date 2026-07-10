"""
Comprehensive analysis: compare simulations with fine-grid data
and understand the A ≈ 0.215 factor from first principles.
"""

import numpy as np
from axion_sim import create_simulation_pt_at_zero
from pathlib import Path

print("=" * 90)
print("COMPREHENSIVE ENERGY ANALYSIS")
print("=" * 90)

# ============================================================================
# PART 1: Load and examine the fine-grid data
# ============================================================================
print("\nPART 1: Fine-grid data")
print("-" * 90)

fine_grid_file = Path("rho_noPT_fine_grid.txt")
if fine_grid_file.exists():
    arr = np.loadtxt(fine_grid_file)
    theta_data = arr[:, 0]
    H_data = arr[:, 1]
    rho_data = arr[:, 2]

    print(f"Loaded: {len(rho_data)} data points")
    print(f"θ₀ range: [{theta_data.min():.6f}, {theta_data.max():.6f}]")
    print(f"H_PT range: [{H_data.min():.6f}, {H_data.max():.6f}]")
    print(f"ρ range: [{rho_data.min():.6e}, {rho_data.max():.6e}]")

    # Group by unique theta0
    unique_theta = np.unique(theta_data)
    print(f"\nUnique θ₀ values: {len(unique_theta)}")

    # Sample a few theta0 values
    print("\nSample data points:")
    for theta_test in unique_theta[::5]:  # Every 5th unique theta
        mask = np.abs(theta_data - theta_test) < 1e-10
        rhos_at_theta = rho_data[mask]
        print(f"  θ₀ = {theta_test:.6f}: ρ values = {rhos_at_theta[:3]}...")
else:
    print(f"File not found: {fine_grid_file}")
    print("Creating it from scratch...")
    unique_theta = None

# ============================================================================
# PART 2: Run fresh simulations and compare
# ============================================================================
print("\n" + "=" * 90)
print("PART 2: Fresh simulations")
print("-" * 90)

H_PT = 0.7
M_phi = 1.0
t_osc = 3.0 / (2.0 * M_phi)
t_end = 10.0 / H_PT

print(f"Parameters:")
print(f"  H_PT = {H_PT}")
print(f"  M_phi = {M_phi}")
print(f"  t_osc = {t_osc:.6f}")
print(f"  t_end = {t_end:.6f}")

# Scale factors
a_osc = np.sqrt(2.0 * H_PT * t_osc)
a_end = np.sqrt(2.0 * H_PT * t_end)
redshift_cube = (a_osc / a_end) ** 3

print(f"  a_osc = {a_osc:.6f}")
print(f"  a_end = {a_end:.6f}")
print(f"  (a_osc/a_end)³ = {redshift_cube:.6f}")

# Simulate a few theta0 values
test_theta0s = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
sim_results = {}

print(f"\nRunning {len(test_theta0s)} simulations...")

for theta0 in test_theta0s:
    print(f"  θ₀ = {theta0:.4f}...", end=" ", flush=True)

    try:
        sim, tau_final = create_simulation_pt_at_zero(
            H_PT=H_PT,
            beta=0.0,
            start_time=0.0,
            end_time=t_end,
            Ngrid=64,
            num_tracers=1_000,
            enable_nucleation=False,
            mb=1.0,
            theta0_initial=theta0,
            v_bubble=0.6,
            energy_save_interval=10,
        )

        results = sim.run_simulation(tau_final, progress_bar=False, save_interval=10)

        energy_hist = results['energy_history']
        times_cosmic = results['times_cosmic']

        # Get energies at end
        E_end = energy_hist[-1]['total_energy']
        V_end = energy_hist[-1]['potential_energy']
        K_end = energy_hist[-1]['kinetic_energy']

        # Theoretical potential at theta0
        V_theory = 1.0 - np.cos(theta0)

        sim_results[theta0] = {
            'E_end': E_end,
            'V_end': V_end,
            'K_end': K_end,
            'V_theory': V_theory,
            'E_end_redshifted': E_end / redshift_cube,
        }

        print(f"E_end={E_end:.6e}")

    except Exception as e:
        print(f"FAILED: {str(e)[:50]}")

# ============================================================================
# PART 3: Compare and analyze the A factor
# ============================================================================
print("\n" + "=" * 90)
print("PART 3: Comparison and A-factor analysis")
print("-" * 90)

# If we have fine-grid data, compare
if fine_grid_file.exists():
    print("\nComparing fresh simulations with fine-grid data:")
    print(f"{'θ₀':<10} {'ρ_data':<15} {'E_end':<15} {'Ratio':<15}")
    print("-" * 60)

    for theta0 in test_theta0s:
        if theta0 in sim_results:
            # Find closest data point
            mask = np.abs(theta_data - theta0) < 0.01
            if np.any(mask):
                rho_closest = rho_data[mask].mean()
                E_end_sim = sim_results[theta0]['E_end']
                ratio = rho_closest / E_end_sim
                print(f"{theta0:<10.4f} {rho_closest:<15.6e} {E_end_sim:<15.6e} {ratio:<15.6f}")

# ============================================================================
# PART 4: Derive A from the data
# ============================================================================
print("\n" + "=" * 90)
print("PART 4: Deriving the A factor")
print("-" * 90)

# Theory: rho_noPT = A * f_anh * theta0^2 * (a_osc/a_end)^3
# For small theta0, f_anh ≈ 1, so:
# rho_noPT ≈ A * theta0^2 * (a_osc/a_end)^3

# If we fit the data directly:
# A = rho_noPT / [theta0^2 * (a_osc/a_end)^3]

if fine_grid_file.exists():
    # Take only data points with theta0 < 2.5 (avoid singularity near pi)
    mask_fit = theta_data < 2.5
    theta_fit = theta_data[mask_fit]
    rho_fit = rho_data[mask_fit]

    # Compute A for each point
    A_values = rho_fit / (theta_fit**2 * redshift_cube)

    print(f"\nA values from fine-grid data (θ₀ < 2.5):")
    print(f"  Mean A = {np.mean(A_values):.6f}")
    print(f"  Std A  = {np.std(A_values):.6f}")
    print(f"  Min A  = {np.min(A_values):.6f}")
    print(f"  Max A  = {np.max(A_values):.6f}")
    print(f"  Expected (from fit): A = 0.215412")

    # Plot A vs theta0 to see if it's constant or varying
    print(f"\nA vs θ₀:")
    for t in sorted(set(np.round(theta_fit, 2))):
        mask_t = np.abs(theta_fit - t) < 0.01
        if np.any(mask_t):
            A_at_t = A_values[mask_t]
            print(f"  θ₀ ≈ {t:.4f}: A = {np.mean(A_at_t):.6f} ± {np.std(A_at_t):.6f}")

# ============================================================================
# PART 5: Energy density scaling law
# ============================================================================
print("\n" + "=" * 90)
print("PART 5: Energy density scaling")
print("-" * 90)

if fine_grid_file.exists() and len(sim_results) > 0:
    print("\nFrom simulations: E_end vs θ₀")
    print(f"{'θ₀':<10} {'E_end':<15} {'E_end/θ₀²':<15} {'E_end/(1-cosθ₀)':<15}")
    print("-" * 70)

    for theta0 in sorted(sim_results.keys()):
        E_end = sim_results[theta0]['E_end']
        V_th = sim_results[theta0]['V_theory']
        ratio_t2 = E_end / (theta0**2)
        ratio_V = E_end / V_th
        print(f"{theta0:<10.4f} {E_end:<15.6e} {ratio_t2:<15.6f} {ratio_V:<15.6f}")

# ============================================================================
# PART 6: Mystery of the factor 27
# ============================================================================
print("\n" + "=" * 90)
print("PART 6: The missing factor")
print("-" * 90)

if len(sim_results) > 0:
    # From simulations: E_end / theta0^2 ≈ 0.008-0.010
    avg_E_over_t2 = np.mean([sim_results[t]['E_end'] / t**2 for t in sim_results])

    # From fit: A ≈ 0.215
    A_fit = 0.215412

    # The ratio
    factor = A_fit / avg_E_over_t2

    print(f"\nE_end / θ₀² from simulations: {avg_E_over_t2:.6f}")
    print(f"A from fit data: {A_fit:.6f}")
    print(f"Ratio (A / avg_E_ratio): {factor:.6f}")

    print(f"\nPossible explanations:")
    print(f"1. Time-averaging: simulation only saves at intervals")
    print(f"2. Spatial averaging: some grid points may not be counted correctly")
    print(f"3. Energy definition: kinetic vs potential split may be different")
    print(f"4. Redshifting formula: (a_osc/a_end)³ = {redshift_cube:.6f} may be wrong")
    print(f"5. Missing physics: oscillation dynamics, gradient energy, etc.")

print("\n" + "=" * 90)
