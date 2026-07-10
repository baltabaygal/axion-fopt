"""
Sweep theta0 for noPT simulations on a fine grid.

Generates 50 theta0 values from 0 to π, runs noPT simulations for each,
and records final total energy density. All other parameters fixed.
"""

import numpy as np
from axion_sim import create_simulation_pt_at_zero
from tqdm import tqdm


def run_nopt_simulation(H_PT, theta0, Ngrid=64):
    """
    Run a single noPT simulation and return final total energy density.

    Parameters
    ----------
    H_PT : float
        Hubble parameter
    theta0 : float
        Initial axion field angle
    Ngrid : int
        Grid resolution

    Returns
    -------
    float or None
        Final total energy density, or None if simulation failed
    """
    end_time = 10.0 / H_PT

    try:
        sim, tau_final = create_simulation_pt_at_zero(
            H_PT=H_PT,
            beta=0.0,
            start_time=0.0,
            end_time=end_time,
            nucleation_mode='standard',
            Ngrid=Ngrid,
            num_tracers=1_000,
            enable_nucleation=False,
            mb=1.0,
            theta0_initial=theta0,
            v_bubble=0.6,
            spatial_hash_cells=8,
            energy_save_interval=10,
            checkpoint_interval=0,
        )

        results = sim.run_simulation(tau_final, progress_bar=False, save_interval=10)

        if results['energy_history']:
            final_energy = results['energy_history'][-1]
            return float(final_energy['total_energy'])
        else:
            return None

    except Exception as e:
        print(f"    ERROR at θ₀={theta0:.6f}: {str(e)}")
        return None


if __name__ == '__main__':
    print("=" * 70)
    print("Fine Grid theta0 Sweep (noPT only, 3 runs per theta0)")
    print("=" * 70)

    # Fixed parameters
    H_PT = 0.7
    Ngrid = 64
    num_theta0 = 50
    num_runs = 3

    # Generate theta0 grid: 50 points from 0 to pi
    theta0_values = np.linspace(0, np.pi, num_theta0)

    print(f"\nParameters:")
    print(f"  H_PT = {H_PT}")
    print(f"  Ngrid = {Ngrid}")
    print(f"  theta0 range: [0, π]")
    print(f"  num_theta0 = {num_theta0}")
    print(f"  runs per theta0 = {num_runs}")
    print(f"  total simulations = {num_theta0 * num_runs}")
    print(f"  end_time = {10.0/H_PT:.4f}")
    print()

    results = []

    total_sims = num_theta0 * num_runs
    print(f"Running {total_sims} noPT simulations ({num_runs} runs × {num_theta0} theta0 values)...")

    with tqdm(total=total_sims, desc="Overall progress") as pbar:
        for theta0 in theta0_values:
            for run in range(num_runs):
                rho = run_nopt_simulation(H_PT, theta0, Ngrid=Ngrid)

                if rho is not None:
                    results.append((theta0, H_PT, rho))
                else:
                    print(f"    Skipped θ₀={theta0:.6f} run {run+1}/{num_runs}")

                pbar.update(1)

    # Save to txt file
    output_file = "rho_noPT_fine_grid.txt"
    print(f"\nSaving {len(results)} results to {output_file}...")

    with open(output_file, 'w') as f:
        for theta0, H_PT_val, rho in results:
            f.write(f"{theta0:.15e} {H_PT_val} {rho:.15e}\n")

    print(f"Done. Saved {len(results)}/{total_sims} results.")
    print(f"Output: {output_file}")

    # Print summary per theta0
    print(f"\nEnergy density statistics by theta0:")
    theta0_unique = np.unique([r[0] for r in results])

    all_rhos = []
    for theta0 in theta0_unique:
        rhos_at_theta0 = np.array([r[2] for r in results if np.isclose(r[0], theta0)])
        if len(rhos_at_theta0) > 0:
            all_rhos.extend(rhos_at_theta0)
            if len(rhos_at_theta0) > 1:
                print(f"  θ₀={theta0:.6f}: mean={rhos_at_theta0.mean():.6e}, std={rhos_at_theta0.std():.6e}")
            else:
                print(f"  θ₀={theta0:.6f}: {rhos_at_theta0[0]:.6e}")

    if all_rhos:
        all_rhos = np.array(all_rhos)
        print(f"\nOverall energy density statistics:")
        print(f"  min: {all_rhos.min():.6e}")
        print(f"  max: {all_rhos.max():.6e}")
        print(f"  mean: {all_rhos.mean():.6e}")
        print(f"  std: {all_rhos.std():.6e}")

    print("=" * 70)
