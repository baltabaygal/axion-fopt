import json
import math
import time
from pathlib import Path

import numpy as np
from axion_sim import create_simulation_pt_at_zero


def run_one(theta0, realization, H_PT, beta_over_H, v_bubble, end_time_factor=50.0):
    np.random.seed(realization)

    sim, tau_final = create_simulation_pt_at_zero(
        H_PT=H_PT,
        beta=beta_over_H * H_PT,
        start_time=0.0,
        end_time=end_time_factor / H_PT,
        nucleation_mode='standard',
        Ngrid=64,
        num_tracers=100_000,
        enable_nucleation=True,
        mb=0.0,
        theta0_initial=float(theta0),
        v_bubble=v_bubble,
        spatial_hash_cells=8,
        energy_save_interval=10,
        checkpoint_interval=0,
    )

    t0 = time.perf_counter()
    res = sim.run_simulation(tau_final, progress_bar=False, save_interval=10)
    wall = time.perf_counter() - t0

    e = res['energy_history'][-1]
    return {
        'theta0': float(theta0),
        'realization': int(realization),
        'final_total_energy': float(e['total_energy']),
        'num_bubbles': int(sim.bubble_manager.count),
        'wall_seconds': float(wall),
    }


def run_nopt(theta0, H_PT, v_bubble, end_time):
    sim, tau_f = create_simulation_pt_at_zero(
        H_PT=H_PT,
        beta=0.0,
        start_time=0.0,
        end_time=end_time,
        nucleation_mode='standard',
        Ngrid=64,
        num_tracers=1_000,
        enable_nucleation=False,
        mb=1.0,
        theta0_initial=float(theta0),
        v_bubble=v_bubble,
        spatial_hash_cells=8,
        energy_save_interval=10,
        checkpoint_interval=0,
    )
    res = sim.run_simulation(tau_f, progress_bar=False, save_interval=10)
    return float(res['energy_history'][-1]['total_energy'])


def main():
    H_PT = 0.05
    beta_over_H = 20.0
    v_bubble = 0.8
    n_theta = 5
    n_realizations = 5

    theta_vals = np.linspace(0.0, math.pi, n_theta)
    end_time = 50.0 / H_PT

    out_dir = Path('/Users/baltabay/Desktop/miniclusters/axion-fopt/sim_core/cosine_sweep_results')
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    total = n_theta * n_realizations
    idx = 0
    for th in theta_vals:
        for r in range(n_realizations):
            idx += 1
            print(f"[{idx}/{total}] theta={th:.10f}, r={r}")
            rows.append(run_one(th, r, H_PT, beta_over_H, v_bubble, end_time_factor=50.0))

    nopt = {}
    for th in theta_vals:
        print(f"noPT theta={th:.10f}")
        nopt[float(th)] = run_nopt(float(th), H_PT, v_bubble, end_time)

    summary = []
    for th in theta_vals:
        pt_runs = [x for x in rows if abs(x['theta0'] - th) < 1e-12]
        pt_arr = np.array([x['final_total_energy'] for x in pt_runs], dtype=float)
        rho0 = nopt[float(th)]
        ratios = pt_arr / rho0 if rho0 != 0 else np.full_like(pt_arr, np.nan)
        summary.append({
            'theta0': float(th),
            'rho_PT_mean': float(pt_arr.mean()),
            'rho_PT_std': float(pt_arr.std(ddof=0)),
            'rho_noPT': float(rho0),
            'mean_ratio': float(np.nanmean(ratios)),
            'std_ratio': float(np.nanstd(ratios)),
            'N_samples': int(len(pt_arr)),
        })

    payload = {
        'config': {
            'potential': 'cosine',
            'H_PT': H_PT,
            'beta_over_H': beta_over_H,
            'beta': beta_over_H * H_PT,
            'v_bubble': v_bubble,
            'Ngrid': 64,
            'num_tracers': 100_000,
            'end_time': end_time,
            'n_theta': n_theta,
            'n_realizations': n_realizations,
            'theta_values': [float(x) for x in theta_vals],
        },
        'per_run': rows,
        'per_theta_summary': summary,
    }

    jpath = out_dir / 'cosine_theta_ratio_H0p05_bH20_vw0p8.json'
    tpath = out_dir / 'cosine_theta_ratio_H0p05_bH20_vw0p8.tsv'
    with open(jpath, 'w') as f:
        json.dump(payload, f, indent=2)
    with open(tpath, 'w') as f:
        f.write('theta0\trho_PT_mean\trho_PT_std\trho_noPT\tmean_ratio\tstd_ratio\tN_samples\n')
        for s in summary:
            f.write(f"{s['theta0']:.10f}\t{s['rho_PT_mean']:.12e}\t{s['rho_PT_std']:.12e}\t{s['rho_noPT']:.12e}\t{s['mean_ratio']:.12e}\t{s['std_ratio']:.12e}\t{s['N_samples']}\n")

    print('\nSUMMARY')
    for s in summary:
        print(f"theta={s['theta0']:.10f} ratio_mean={s['mean_ratio']:.6f} ratio_std={s['std_ratio']:.6f}")
    print(f"JSON: {jpath}")
    print(f"TSV:  {tpath}")


if __name__ == '__main__':
    main()
