import json
import math
import os
import time
from pathlib import Path

import numpy as np
from axion_harmonic import create_simulation_pt_at_zero


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
        'final_kinetic_energy': float(e['kinetic_energy']),
        'final_gradient_energy': float(e['gradient_energy']),
        'final_potential_energy': float(e['potential_energy']),
        'num_bubbles': int(sim.bubble_manager.count),
        'wall_seconds': float(wall),
    }


def main():
    H_PT = 0.05
    beta_over_H = 20.0
    v_bubble = 0.8
    n_theta = 5
    n_realizations = 5

    theta_vals = np.linspace(0.0, math.pi, n_theta)

    out_dir = Path('/Users/baltabay/Desktop/miniclusters/axion-fopt/sim_core/harmonic_sweep_results')
    out_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 72)
    print('Harmonic theta sweep')
    print(f'H_PT={H_PT}  beta/H={beta_over_H}  v_w={v_bubble}')
    print(f'theta points={n_theta} in [0, pi], realizations per theta={n_realizations}')
    print('Ngrid=64, num_tracers=100000, end_time=50/H*')
    print('=' * 72)

    rows = []
    total = n_theta * n_realizations
    idx = 0

    for th in theta_vals:
        for r in range(n_realizations):
            idx += 1
            print(f"\\n[{idx}/{total}] theta={th:.10f}, r={r}")
            row = run_one(th, r, H_PT, beta_over_H, v_bubble, end_time_factor=50.0)
            rows.append(row)
            print(f"  E_final={row['final_total_energy']:.9e}, bubbles={row['num_bubbles']}, wall={row['wall_seconds']:.2f}s")

    # Aggregate by theta
    summary = []
    for th in theta_vals:
        rr = [x for x in rows if abs(x['theta0'] - th) < 1e-12]
        arr = np.array([x['final_total_energy'] for x in rr], dtype=float)
        summary.append({
            'theta0': float(th),
            'mean_final_total_energy': float(arr.mean()),
            'std_final_total_energy': float(arr.std(ddof=0)),
            'N_samples': int(len(arr)),
        })

    payload = {
        'config': {
            'H_PT': H_PT,
            'beta_over_H': beta_over_H,
            'beta': beta_over_H * H_PT,
            'v_bubble': v_bubble,
            'Ngrid': 64,
            'num_tracers': 100_000,
            'end_time': 50.0 / H_PT,
            'n_theta': n_theta,
            'n_realizations': n_realizations,
            'theta_values': [float(x) for x in theta_vals],
        },
        'per_run': rows,
        'per_theta_summary': summary,
    }

    ts = time.strftime('%Y%m%d_%H%M%S')
    jpath = out_dir / f'harmonic_theta_sweep_H0p05_bH20_vw0p8_{ts}.json'
    tpath = out_dir / f'harmonic_theta_sweep_H0p05_bH20_vw0p8_{ts}.tsv'

    with open(jpath, 'w') as f:
        json.dump(payload, f, indent=2)

    with open(tpath, 'w') as f:
        f.write('theta0\tmean_final_total_energy\tstd_final_total_energy\tN_samples\n')
        for s in summary:
            f.write(f"{s['theta0']:.10f}\t{s['mean_final_total_energy']:.12e}\t{s['std_final_total_energy']:.12e}\t{s['N_samples']}\n")

    print('\\n' + '=' * 72)
    print('Per-theta summary:')
    for s in summary:
        print(f"theta={s['theta0']:.10f}  mean={s['mean_final_total_energy']:.9e}  std={s['std_final_total_energy']:.9e}  N={s['N_samples']}")
    print('=' * 72)
    print(f'JSON: {jpath}')
    print(f'TSV : {tpath}')


if __name__ == '__main__':
    main()
